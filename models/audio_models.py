from pydantic import BaseModel
from typing import Dict, List, Any, Optional

class AudioAnalysisRequest(BaseModel):
    file_path: str
    session_id: Optional[str] = None

class SpeakerSegment(BaseModel):
    speaker: str
    start: float
    end: float
    confidence: float
    text: str = ""

class EmotionAnalysis(BaseModel):
    emotion: str
    confidence: float

class AcousticFeatures(BaseModel):
    rms_energy: float
    zcr: float
    spectral_centroid: float
    tempo: float
    mfcc_mean: List[float]
    avg_pitch: float
    loudness_db: float

class AudioAnalysis(BaseModel):
    transcript: str
    environmentalSounds: List[str]
    acousticFeatures: Dict[str, Any]
    speakerSegments: List[Dict[str, Any]]
    emotions: List[Dict[str, Any]]
    duration: float
    speakerCount: int
    quality: str

class AudioAnalysisResponse(BaseModel):
    sessionId: str
    analysis: Dict[str, Any]

class StreamingAudioData(BaseModel):
    audio_data: bytes
    timestamp: float
    session_id: str

class StreamingAnalysisResult(BaseModel):
    transcript: str
    energy: float
    timestamp: float
    session_id: str