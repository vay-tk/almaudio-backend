from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import os
import tempfile
import aiofiles
from dotenv import load_dotenv
from typing import Optional
import uuid
import asyncio

from services.audio_processor import AudioProcessor
from services.conversation_manager import ConversationManager
from services.database import DatabaseManager, get_database
from services.auth_service import AuthService
from models.audio_models import AudioAnalysisResponse
from models.chat_models import QuestionRequest, ChatResponse
from models.user_models import UserCreate, UserLogin, UserResponse

# Load environment variables
load_dotenv()

# Global services (will be initialized on startup)
audio_processor = None
conversation_manager = None
auth_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all services on startup"""
    global audio_processor, conversation_manager, auth_service
    
    print("🚀 Starting AI Audio Analyzer & Conversational ALM...")
    
    try:
        # Initialize database
        db_manager = await get_database()
        print("✅ Database connection established")
        
        # Initialize authentication service
        auth_service = AuthService(db_manager.users)
        print("✅ Authentication service initialized")
        
        # Initialize AI services
        audio_processor = AudioProcessor()
        conversation_manager = ConversationManager()
        print("✅ AI services initialized")
        
        print("🎉 AI Audio Analyzer backend started successfully!")
        print("📚 API Documentation: http://localhost:8000/docs")
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise e
    
    yield
    
    # Cleanup on shutdown
    print("🔄 Shutting down AI Audio Analyzer...")

app = FastAPI(
    title="AI Audio Analyzer & Conversational ALM",
    description="Production-ready AI Audio Analyzer with ChatGPT-style conversational interface",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        print(f"🔗 WebSocket connected for session: {session_id}")

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            print(f"🔌 WebSocket disconnected for session: {session_id}")

    async def send_personal_message(self, message: str, session_id: str):
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            await websocket.send_text(message)

manager = ConnectionManager()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Get current authenticated user"""
    if not auth_service:
        raise HTTPException(status_code=500, detail="Authentication service not initialized")
    
    token = credentials.credentials
    email = auth_service.verify_token(token)
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = await auth_service.get_user_by_email(email)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


@app.get("/")
async def root():
    return {
        "message": "AI Audio Analyzer & Conversational ALM API",
        "status": "active",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "auth": "/api/auth/*",
            "audio": "/api/upload-audio",
            "chat": "/api/ask-question",
            "websocket": "/ws/{session_id}"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "audio_processor": audio_processor is not None,
            "conversation_manager": conversation_manager is not None,
            "auth_service": auth_service is not None
        }
    }

# Authentication endpoints
@app.post("/api/auth/register", response_model=UserResponse)
async def register(user_data: UserCreate):
    """Register a new user"""
    try:
        if not auth_service:
            raise HTTPException(status_code=500, detail="Authentication service not available")
        
        user_doc = await auth_service.create_user(user_data)
        response = auth_service.create_user_response(user_doc)
        print(f"👤 New user registered: {user_data.email}")
        return response
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Registration error: {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/api/auth/login", response_model=UserResponse)
async def login(user_data: UserLogin):
    """Login user"""
    try:
        if not auth_service:
            raise HTTPException(status_code=500, detail="Authentication service not available")
        
        user = await auth_service.authenticate_user(user_data.email, user_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        response = auth_service.create_user_response(user)
        print(f"🔐 User logged in: {user_data.email}")
        return response
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Login error: {e}")
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

@app.get("/api/auth/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user information"""
    return {
        "id": current_user["id"],
        "username": current_user["username"],
        "email": current_user["email"],
        "created_at": current_user["created_at"],
        "last_login": current_user.get("last_login"),
        "is_active": current_user["is_active"]
    }

@app.post("/api/auth/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """Logout user"""
    print(f"👋 User logged out: {current_user['email']}")
    return {"message": "Logged out successfully"}

# Audio processing endpoints
@app.post("/api/upload-audio", response_model=AudioAnalysisResponse)
async def upload_audio(
    file: UploadFile = File(...), 
    current_user: dict = Depends(get_current_user)
):
    """Upload and analyze audio file with full AI processing"""
    if not audio_processor:
        raise HTTPException(status_code=500, detail="Audio processing service not available")
    
    try:
        print(f"🎵 Processing audio upload from user: {current_user['username']}")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_ext = file.filename.lower().split('.')[-1]
        supported_formats = ['mp3', 'wav', 'm4a', 'ogg', 'flac', 'aac', 'wma', 'webm']
        
        if file_ext not in supported_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported audio format. Supported: {', '.join(supported_formats)}"
            )
        
        # Check file size (50MB limit)
        file_size = 0
        content = await file.read()
        file_size = len(content)
        
        if file_size > 50 * 1024 * 1024:  # 50MB
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 50MB")
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        
        print(f"📁 Temporary file saved: {temp_path} ({file_size / 1024 / 1024:.1f}MB)")
        
        try:
            # Process audio with all AI models
            print("🤖 Starting comprehensive audio analysis...")
            analysis_result = await audio_processor.process_audio(temp_path)
            print("✅ Audio analysis completed successfully")
            
            # Create user session
            db_manager = await get_database()
            session_id = await db_manager.create_session(current_user["id"])
            
            # Save analysis to database
            await db_manager.save_analysis(session_id, current_user["id"], analysis_result)
            print(f"💾 Analysis saved for session: {session_id}")
            
            return AudioAnalysisResponse(
                sessionId=session_id,
                analysis=analysis_result
            )
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
                print(f"🗑️ Temporary file cleaned up: {temp_path}")
            except Exception as e:
                print(f"⚠️ Failed to clean up temp file: {e}")
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Audio processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")

# WebSocket endpoint for real-time streaming
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time audio streaming"""
    if not audio_processor:
        await websocket.close(code=1011, reason="Audio processing service not available")
        return
    
    await manager.connect(websocket, session_id)
    try:
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()
            
            # Process streaming audio
            result = await audio_processor.process_streaming_audio(data)
            
            # Send results back
            import json
            await manager.send_personal_message(json.dumps(result), session_id)
            
    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        manager.disconnect(session_id)

# Conversation endpoints
@app.post("/api/ask-question", response_model=ChatResponse)
async def ask_question(
    request: QuestionRequest, 
    current_user: dict = Depends(get_current_user)
):
    """Ask a question about analyzed audio with full conversational AI"""
    if not conversation_manager:
        raise HTTPException(status_code=500, detail="Conversation service not available")
    
    try:
        print(f"💬 Processing question from {current_user['username']}: {request.question[:50]}...")
        
        db_manager = await get_database()
        
        # Verify session ownership if session ID provided
        if request.sessionId:
            is_owner = await db_manager.verify_session_ownership(request.sessionId, current_user["id"])
            if not is_owner:
                raise HTTPException(status_code=403, detail="Access denied to this session")
        
        # Get chat history and audio analysis
        history = []
        audio_analysis = None
        
        if request.sessionId:
            history = await db_manager.get_chat_history(request.sessionId)
            audio_analysis = await db_manager.get_analysis(request.sessionId)
        
        # Use provided analysis if available (for new uploads)
        if request.audioAnalysis:
            audio_analysis = request.audioAnalysis
        
        print(f"🧠 Generating AI response with context (history: {len(history)} messages)")
        
        # Generate response using Gemini with full context
        response = await conversation_manager.generate_response(
            question=request.question,
            audio_analysis=audio_analysis,
            chat_history=history
        )
        
        # Save to chat history if session exists
        if request.sessionId:
            await db_manager.save_chat_message(
                request.sessionId, 
                current_user["id"], 
                request.question, 
                response.answer
            )
            print(f"💾 Chat message saved to session: {request.sessionId}")
        
        print(f"✅ Response generated successfully ({len(response.answer)} characters)")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Question processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Question processing failed: {str(e)}")

# Session management endpoints
@app.get("/api/session/{session_id}/history")
async def get_chat_history(session_id: str, current_user: dict = Depends(get_current_user)):
    """Get chat history for a session"""
    try:
        db_manager = await get_database()
        
        # Verify session ownership
        is_owner = await db_manager.verify_session_ownership(session_id, current_user["id"])
        if not is_owner:
            raise HTTPException(status_code=403, detail="Access denied to this session")
        
        history = await db_manager.get_chat_history(session_id)
        return {"history": history, "session_id": session_id, "total_messages": len(history)}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ History retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

@app.get("/api/session/{session_id}/analysis")
async def get_analysis(session_id: str, current_user: dict = Depends(get_current_user)):
    """Get audio analysis for a session"""
    try:
        db_manager = await get_database()
        
        # Verify session ownership
        is_owner = await db_manager.verify_session_ownership(session_id, current_user["id"])
        if not is_owner:
            raise HTTPException(status_code=403, detail="Access denied to this session")
        
        analysis = await db_manager.get_analysis(session_id)
        return {"analysis": analysis, "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Analysis retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analysis: {str(e)}")

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a session and all its data"""
    try:
        db_manager = await get_database()
        
        # Verify session ownership
        is_owner = await db_manager.verify_session_ownership(session_id, current_user["id"])
        if not is_owner:
            raise HTTPException(status_code=403, detail="Access denied to this session")
        
        await db_manager.delete_session(session_id, current_user["id"])
        print(f"🗑️ Session deleted: {session_id}")
        return {"message": "Session deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Session deletion error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")

# User management endpoints
@app.get("/api/user/sessions")
async def get_user_sessions(current_user: dict = Depends(get_current_user)):
    """Get all sessions for the current user"""
    try:
        db_manager = await get_database()
        sessions = await db_manager.get_user_sessions(current_user["id"])
        return {"sessions": sessions, "total": len(sessions)}
    except Exception as e:
        print(f"❌ User sessions retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get sessions: {str(e)}")

@app.delete("/api/user/sessions/{session_id}")
async def delete_user_session(session_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a user's session"""
    try:
        db_manager = await get_database()
        
        # Verify session ownership
        is_owner = await db_manager.verify_session_ownership(session_id, current_user["id"])
        if not is_owner:
            raise HTTPException(status_code=403, detail="Access denied to this session")
        
        await db_manager.delete_session(session_id, current_user["id"])
        print(f"🗑️ User session deleted: {session_id}")
        return {"message": "Session deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ User session deletion error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")

# Error handlers
@app.exception_handler(500)
async def internal_server_error(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )

@app.exception_handler(413)
async def request_entity_too_large(request, exc):
    return JSONResponse(
        status_code=413,
        content={"detail": "File too large. Maximum size is 50MB."}
    )

if __name__ == "__main__":
    # Check for required environment variables
    required_vars = ["GEMINI_API_KEY", "MONGODB_URL"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"⚠️ Warning: Missing environment variables: {', '.join(missing_vars)}")
        print("💡 Copy backend/.env.example to backend/.env and configure your API keys")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )