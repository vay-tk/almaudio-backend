import motor.motor_asyncio
from pymongo import MongoClient
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
import uuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
        db_name = os.getenv("MONGODB_DB_NAME", "ai_audio_analyzer")
        
        # Async MongoDB client
        self.client = motor.motor_asyncio.AsyncIOMotorClient(mongodb_url)
        self.db = self.client[db_name]
        
        # Collections
        self.users = self.db.users
        self.sessions = self.db.sessions
        self.chat_history = self.db.chat_history
        self.audio_analyses = self.db.audio_analyses
    
    async def initialize(self):
        """Initialize database and create indexes"""
        try:
            # Test connection
            await self.client.admin.command('ismaster')
            logger.info("📊 Connected to MongoDB successfully")
            
            # Create indexes
            await self.users.create_index("email", unique=True)
            await self.users.create_index("username", unique=True)
            await self.users.create_index("id", unique=True)
            
            await self.sessions.create_index("session_id", unique=True)
            await self.sessions.create_index("user_id")
            await self.sessions.create_index("created_at")
            
            await self.chat_history.create_index("session_id")
            await self.chat_history.create_index("timestamp")
            
            await self.audio_analyses.create_index("session_id", unique=True)
            await self.audio_analyses.create_index("user_id")
            
            logger.info("📊 Database indexes created successfully")
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            logger.info("📊 Running without persistent storage (in-memory only)")
            self.client = None
    
    async def create_session(self, user_id: str) -> str:
        """Create a new chat session for a user"""
        session_id = str(uuid.uuid4())
        
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.now(timezone.utc),
            "last_activity": datetime.now(timezone.utc),
            "message_count": 0,
            "has_audio": False,
            "status": "active"
        }
        
        try:
            if self.client:
                await self.sessions.insert_one(session_data)
            logger.info(f"📊 Created new session: {session_id} for user: {user_id}")
        except Exception as e:
            logger.error(f"Session creation error: {e}")
        
        return session_id
    
    async def save_analysis(self, session_id: str, user_id: str, analysis: Dict[str, Any]):
        """Save audio analysis results"""
        analysis_data = {
            "session_id": session_id,
            "user_id": user_id,
            "analysis": analysis,
            "timestamp": datetime.now(timezone.utc)
        }
        
        try:
            if self.client:
                await self.audio_analyses.replace_one(
                    {"session_id": session_id},
                    analysis_data,
                    upsert=True
                )
                
                # Update session with audio flag and duration
                await self.sessions.update_one(
                    {"session_id": session_id},
                    {
                        "$set": {
                            "has_audio": True,
                            "duration": analysis.get("duration"),
                            "last_activity": datetime.now(timezone.utc)
                        }
                    }
                )
            logger.info(f"📊 Saved analysis for session: {session_id}")
        except Exception as e:
            logger.error(f"Analysis save error: {e}")
    
    async def get_analysis(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get audio analysis for a session"""
        try:
            if self.client:
                result = await self.audio_analyses.find_one({"session_id": session_id})
                if result:
                    return result.get("analysis")
        except Exception as e:
            logger.error(f"Analysis retrieval error: {e}")
        
        return None
    
    async def save_chat_message(self, session_id: str, user_id: str, question: str, answer: str):
        """Save chat message to history"""
        message_data = {
            "session_id": session_id,
            "user_id": user_id,
            "question": question,
            "answer": answer,
            "timestamp": datetime.now(timezone.utc)
        }
        
        try:
            if self.client:
                await self.chat_history.insert_one(message_data)
                
                # Update session message count and last activity
                await self.sessions.update_one(
                    {"session_id": session_id},
                    {
                        "$inc": {"message_count": 1},
                        "$set": {"last_activity": datetime.now(timezone.utc)}
                    }
                )
            logger.info(f"📊 Saved chat message for session: {session_id}")
        except Exception as e:
            logger.error(f"Chat message save error: {e}")
    
    async def get_chat_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get chat history for a session"""
        try:
            if self.client:
                cursor = self.chat_history.find({"session_id": session_id}).sort("timestamp", 1)
                history = []
                async for message in cursor:
                    history.extend([
                        {
                            "type": "user",
                            "content": message["question"],
                            "timestamp": message["timestamp"]
                        },
                        {
                            "type": "assistant", 
                            "content": message["answer"],
                            "timestamp": message["timestamp"]
                        }
                    ])
                return history
        except Exception as e:
            logger.error(f"Chat history retrieval error: {e}")
        
        return []
    
    async def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all sessions for a user"""
        try:
            if self.client:
                cursor = self.sessions.find({"user_id": user_id}).sort("last_activity", -1)
                sessions = []
                async for session in cursor:
                    # Get analysis data if available
                    analysis = await self.get_analysis(session["session_id"])
                    session_data = {
                        "session_id": session["session_id"],
                        "created_at": session["created_at"],
                        "last_activity": session["last_activity"],
                        "message_count": session.get("message_count", 0),
                        "has_audio": session.get("has_audio", False),
                        "duration": session.get("duration"),
                        "analysis": analysis
                    }
                    sessions.append(session_data)
                return sessions
        except Exception as e:
            logger.error(f"User sessions retrieval error: {e}")
        
        return []
    
    async def delete_session(self, session_id: str, user_id: str):
        """Delete a session and all its data"""
        try:
            if self.client:
                # Delete from all collections
                await self.sessions.delete_one({"session_id": session_id, "user_id": user_id})
                await self.chat_history.delete_many({"session_id": session_id, "user_id": user_id})
                await self.audio_analyses.delete_one({"session_id": session_id, "user_id": user_id})
                
            logger.info(f"📊 Deleted session: {session_id} for user: {user_id}")
        except Exception as e:
            logger.error(f"Session deletion error: {e}")
    
    async def cleanup_old_sessions(self, days: int = 7):
        """Clean up sessions older than specified days"""
        try:
            if self.client:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
                
                # Find old sessions
                cursor = self.sessions.find({"last_activity": {"$lt": cutoff_date}})
                old_sessions = []
                async for session in cursor:
                    old_sessions.append(session)
                
                for session in old_sessions:
                    await self.delete_session(session["session_id"], session["user_id"])
                
                logger.info(f"📊 Cleaned up {len(old_sessions)} old sessions")
        except Exception as e:
            logger.error(f"Session cleanup error: {e}")
    
    async def verify_session_ownership(self, session_id: str, user_id: str) -> bool:
        """Verify that a session belongs to a user"""
        try:
            if self.client:
                session = await self.sessions.find_one({
                    "session_id": session_id,
                    "user_id": user_id
                })
                return session is not None
        except Exception as e:
            logger.error(f"Session ownership verification error: {e}")
        
        return False

# Database instance for singleton pattern
db_instance = None

async def get_database() -> DatabaseManager:
    global db_instance
    if db_instance is None:
        db_instance = DatabaseManager()
        await db_instance.initialize()
    return db_instance