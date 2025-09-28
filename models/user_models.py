from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(BaseModel):
    id: str
    username: str
    email: EmailStr
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True

class UserResponse(BaseModel):
    user: User
    access_token: str
    token_type: str = "bearer"

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class UserSession(BaseModel):
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    message_count: int = 0
    has_audio: bool = False
    analysis: Optional[dict] = None
    duration: Optional[float] = None