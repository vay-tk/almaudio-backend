from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from datetime import datetime

class QuestionRequest(BaseModel):
    question: str
    sessionId: Optional[str] = None
    audioAnalysis: Optional[Dict[str, Any]] = None

class ChatMessage(BaseModel):
    id: int
    type: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    audioFeatures: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None

class ChatResponse(BaseModel):
    answer: str
    confidence: float
    audioFeatures: Optional[Dict[str, Any]] = None
    sessionId: Optional[str] = None

class ChatHistoryResponse(BaseModel):
    history: List[Dict[str, Any]]
    session_id: str
    total_messages: int

class SessionInfo(BaseModel):
    session_id: str
    created_at: datetime
    last_activity: datetime
    status: str
    message_count: int = 0
    has_audio: bool = False