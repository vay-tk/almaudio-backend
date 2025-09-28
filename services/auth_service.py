from datetime import datetime, timedelta, timezone
from typing import Optional
import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
import os
from motor.motor_asyncio import AsyncIOMotorCollection
from bson import ObjectId
import uuid

from models.user_models import User, UserCreate, UserLogin

# Security
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthService:
    def __init__(self, users_collection: AsyncIOMotorCollection):
        self.users_collection = users_collection

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        return pwd_context.hash(password)

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    def verify_token(self, token: str) -> Optional[str]:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            email: str = payload.get("sub")
            if email is None:
                return None
            return email
        except jwt.PyJWTError:
            return None

    async def get_user_by_email(self, email: str) -> Optional[dict]:
        user = await self.users_collection.find_one({"email": email})
        return user

    async def get_user_by_username(self, username: str) -> Optional[dict]:
        user = await self.users_collection.find_one({"username": username})
        return user

    async def create_user(self, user_data: UserCreate) -> dict:
        # Check if user already exists
        existing_email = await self.get_user_by_email(user_data.email)
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        existing_username = await self.get_user_by_username(user_data.username)
        if existing_username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )

        # Hash password
        hashed_password = self.get_password_hash(user_data.password)
        
        # Create user document
        user_doc = {
            "id": str(uuid.uuid4()),
            "username": user_data.username,
            "email": user_data.email,
            "hashed_password": hashed_password,
            "created_at": datetime.now(timezone.utc),
            "last_login": None,
            "is_active": True
        }

        # Insert user
        result = await self.users_collection.insert_one(user_doc)
        user_doc["_id"] = result.inserted_id
        
        return user_doc

    async def authenticate_user(self, email: str, password: str) -> Optional[dict]:
        user = await self.get_user_by_email(email)
        if not user:
            return None
        if not self.verify_password(password, user["hashed_password"]):
            return None
        
        # Update last login
        await self.users_collection.update_one(
            {"_id": user["_id"]},
            {"$set": {"last_login": datetime.now(timezone.utc)}}
        )
        
        return user

    async def get_user_by_id(self, user_id: str) -> Optional[dict]:
        user = await self.users_collection.find_one({"id": user_id})
        return user

    def create_user_response(self, user_doc: dict) -> dict:
        # Remove sensitive fields
        user_data = {
            "id": user_doc["id"],
            "username": user_doc["username"],
            "email": user_doc["email"],
            "created_at": user_doc["created_at"],
            "last_login": user_doc.get("last_login"),
            "is_active": user_doc["is_active"]
        }
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = self.create_access_token(
            data={"sub": user_doc["email"]}, expires_delta=access_token_expires
        )
        
        return {
            "user": user_data,
            "access_token": access_token,
            "token_type": "bearer"
        }