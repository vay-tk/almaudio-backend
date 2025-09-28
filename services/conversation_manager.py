import google.generativeai as genai
import os
from typing import Dict, List, Any, Optional
import json
import asyncio
import logging

from models.chat_models import ChatResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationManager:
    def __init__(self):
        # Configure Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("⚠️ GEMINI_API_KEY not found in environment variables")
            self.model = None
        else:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
                logger.info("✅ Gemini 2.0 Flash API configured successfully")
            except Exception as e:
                logger.error(f"Gemini initialization error: {e}")
                self.model = None
    
    async def generate_response(
        self, 
        question: str, 
        audio_analysis: Optional[Dict[str, Any]] = None, 
        chat_history: List[Dict[str, Any]] = None
    ) -> ChatResponse:
        """Generate contextual response using Gemini 2.0 Flash API with real audio analysis"""
        try:
            if not self.model:
                return ChatResponse(
                    answer="I apologize, but the AI service is currently unavailable. Please ensure your GEMINI_API_KEY is configured correctly in the environment variables.",
                    confidence=0.0,
                    audioFeatures=audio_analysis
                )
            
            # Build comprehensive context prompt
            context = self._build_comprehensive_context(question, audio_analysis, chat_history)
            
            # Generate response
            response = await self._generate_gemini_response(context)
            
            # Calculate confidence based on available data
            confidence = self._calculate_confidence(audio_analysis, question, response)
            
            return ChatResponse(
                answer=response,
                confidence=confidence,
                audioFeatures=audio_analysis
            )
            
        except Exception as e:
            logger.error(f"Conversation generation error: {e}")
            return ChatResponse(
                answer=f"I encountered an error while processing your question: {str(e)}. Please try again or check your API configuration.",
                confidence=0.0,
                audioFeatures=audio_analysis
            )
    
    def _build_comprehensive_context(
        self, 
        question: str, 
        audio_analysis: Optional[Dict[str, Any]] = None, 
        chat_history: List[Dict[str, Any]] = None
    ) -> str:
        """Build comprehensive context prompt for Gemini with real audio data"""
        
        prompt = f"""You are an expert AI Audio Analyzer assistant with deep expertise in:
- Speech recognition and transcription analysis
- Environmental sound detection and classification  
- Acoustic feature analysis (pitch, tempo, energy, spectral characteristics)
- Speaker diarization and voice analysis
- Emotion detection in audio
- Audio quality assessment
- Contextual audio scene understanding

You provide detailed, accurate, and insightful responses about audio content based on real analysis data.

CURRENT USER QUESTION: {question}

"""
        
        # Add comprehensive audio analysis context if available
        if audio_analysis:
            prompt += f"""DETAILED AUDIO ANALYSIS RESULTS:

📝 **SPEECH TRANSCRIPTION:**
{audio_analysis.get('transcript', 'No speech detected in this audio')}

🔊 **ENVIRONMENTAL SOUNDS DETECTED:**
{self._format_environmental_sounds(audio_analysis.get('environmentalSounds', []))}

👥 **SPEAKER ANALYSIS:**
- Total Speakers: {audio_analysis.get('speakerCount', 0)}
- Audio Duration: {audio_analysis.get('duration', 0):.2f} seconds
- Audio Quality: {audio_analysis.get('quality', 'Unknown')}

{self._format_speaker_segments(audio_analysis.get('speakerSegments', []))}

🎵 **DETAILED ACOUSTIC FEATURES:**
{self._format_acoustic_features(audio_analysis.get('acousticFeatures', {}))}

😊 **EMOTIONAL ANALYSIS:**
{self._format_emotions(audio_analysis.get('emotions', []))}

"""
        
        # Add chat history context for follow-up questions
        if chat_history and len(chat_history) > 0:
            prompt += f"""
📚 **CONVERSATION HISTORY:**
"""
            # Include last 6 messages for context
            for msg in chat_history[-6:]:
                role = "User" if msg.get('type') == 'user' else "Assistant"
                content = msg.get('content', '')[:200] + "..." if len(msg.get('content', '')) > 200 else msg.get('content', '')
                prompt += f"{role}: {content}\n"
        
        prompt += f"""
🎯 **RESPONSE INSTRUCTIONS:**
1. Provide detailed, accurate insights based on the real audio analysis data above
2. Reference specific acoustic features, detected sounds, and speaker information when relevant
3. Use the conversation history to provide contextually aware responses
4. If asked about emotions, locations, situations, or scenes, analyze ALL available audio data
5. Be specific and cite actual measurements when possible (e.g., "The audio shows high energy at 0.045 RMS...")
6. If the question is about a follow-up topic, connect it to previous context
7. Provide actionable insights and explanations
8. Use clear, conversational language while maintaining technical accuracy
9. If asked about sounds not detected, explain what the analysis did find instead
10. Include confidence indicators when making interpretations

Please provide a comprehensive, contextually-aware response to the user's question based on all available audio analysis data and conversation history."""

        return prompt
    
    def _format_environmental_sounds(self, sounds: List[str]) -> str:
        """Format environmental sounds for context"""
        if not sounds:
            return "No specific environmental sounds detected"
        return f"Detected sounds: {', '.join(sounds)}"
    
    def _format_speaker_segments(self, segments: List[Dict[str, Any]]) -> str:
        """Format speaker segments for context"""
        if not segments:
            return "No speaker segments identified"
        
        result = "Speaker Timeline:\n"
        for i, segment in enumerate(segments[:5]):  # Show first 5 segments
            result += f"- {segment.get('speaker', 'Unknown')}: {segment.get('start', 0):.1f}s - {segment.get('end', 0):.1f}s (confidence: {segment.get('confidence', 0):.2f})\n"
        
        if len(segments) > 5:
            result += f"- And {len(segments) - 5} more segments...\n"
        
        return result
    
    def _format_acoustic_features(self, features: Dict[str, Any]) -> str:
        """Format acoustic features for context"""
        if not features:
            return "No acoustic features extracted"
        
        result = ""
        
        # Energy and dynamics
        if 'rms_energy' in features:
            result += f"- RMS Energy: {features['rms_energy']:.4f} (loudness indicator)\n"
        if 'loudness_db' in features:
            result += f"- Loudness: {features['loudness_db']:.1f} dB\n"
        
        # Spectral characteristics  
        if 'spectral_centroid_mean' in features:
            result += f"- Spectral Centroid: {features['spectral_centroid_mean']:.0f} Hz (brightness)\n"
        if 'spectral_rolloff_mean' in features:
            result += f"- Spectral Rolloff: {features['spectral_rolloff_mean']:.0f} Hz\n"
        
        # Pitch and tempo
        if 'avg_pitch' in features and features['avg_pitch'] > 0:
            result += f"- Average Pitch: {features['avg_pitch']:.0f} Hz\n"
        if 'tempo' in features:
            result += f"- Tempo: {features['tempo']:.0f} BPM\n"
        
        # Voice characteristics
        if 'zcr_mean' in features:
            result += f"- Zero Crossing Rate: {features['zcr_mean']:.4f} (voice quality indicator)\n"
        if 'harmonic_ratio' in features:
            result += f"- Harmonic Content: {features['harmonic_ratio']:.2f} (voice/music indicator)\n"
        
        return result if result else "Basic acoustic analysis completed"
    
    def _format_emotions(self, emotions: List[Dict[str, Any]]) -> str:
        """Format emotion analysis for context"""
        if not emotions:
            return "No emotional characteristics detected"
        
        result = "Detected emotional characteristics:\n"
        for emotion in emotions:
            emotion_name = emotion.get('emotion', 'Unknown')
            confidence = emotion.get('confidence', 0)
            result += f"- {emotion_name}: {confidence:.1%} confidence\n"
        
        return result
    
    async def _generate_gemini_response(self, prompt: str) -> str:
        """Generate response using Gemini API with proper async handling"""
        try:
            loop = asyncio.get_event_loop()
            
            def generate():
                generation_config = {
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'top_k': 40,
                    'max_output_tokens': 1000
                }
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                return response.text
            
            # Run in executor to avoid blocking
            response = await loop.run_in_executor(None, generate)
            
            logger.info(f"✅ Gemini response generated ({len(response)} characters)")
            return response
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._generate_intelligent_fallback(prompt)
    
    def _generate_intelligent_fallback(self, prompt: str) -> str:
        """Generate intelligent fallback response when Gemini is unavailable"""
        try:
            # Extract question and context
            lines = prompt.split('\n')
            question_line = [line for line in lines if line.startswith('CURRENT USER QUESTION:')]
            question = question_line[0].replace('CURRENT USER QUESTION:', '').strip() if question_line else "your question"
            
            # Check available analysis data
            has_transcript = 'SPEECH TRANSCRIPTION:' in prompt and 'No speech detected' not in prompt
            has_sounds = 'ENVIRONMENTAL SOUNDS DETECTED:' in prompt and 'No specific environmental sounds' not in prompt
            has_speakers = 'Total Speakers:' in prompt
            has_acoustic = 'DETAILED ACOUSTIC FEATURES:' in prompt
            has_emotions = 'EMOTIONAL ANALYSIS:' in prompt
            
            # Generate contextual response based on available data
            if not any([has_transcript, has_sounds, has_speakers, has_acoustic]):
                return """I'd be happy to help analyze your audio! However, I notice that no audio has been processed yet or the analysis is still in progress.

To get detailed insights, please:
1. Upload an audio file using the interface above
2. Wait for the analysis to complete
3. Ask your questions about the audio content

Once your audio is analyzed, I can provide detailed information about:
• Speech content and transcription
• Environmental sounds and background noise  
• Speaker identification and timeline
• Acoustic characteristics (pitch, tempo, energy)
• Emotional content and context
• Scene understanding and location insights

Please upload your audio file and try again!"""

            # Generate response based on available data
            response_parts = []
            
            if question.lower() in ['what is happening', 'what\'s happening', 'analyze this audio', 'tell me about this audio']:
                response_parts.append("Based on the audio analysis:")
                
                if has_transcript:
                    response_parts.append("📝 **Speech Content:** The audio contains spoken content that has been transcribed and analyzed.")
                
                if has_sounds:
                    response_parts.append("🔊 **Environmental Context:** Environmental sounds have been detected that provide context about the location and situation.")
                
                if has_speakers:
                    response_parts.append("👥 **Speaker Information:** Speaker analysis has identified different voices and their speaking patterns.")
                
                if has_acoustic:
                    response_parts.append("🎵 **Acoustic Properties:** Detailed acoustic analysis reveals information about the audio's characteristics including pitch, energy, and spectral content.")
                
                if has_emotions:
                    response_parts.append("😊 **Emotional Analysis:** Emotional characteristics have been detected based on vocal and acoustic patterns.")
                
                response_parts.append("\nFor more specific insights, please ask targeted questions like:")
                response_parts.append("• 'What emotions are present?'")
                response_parts.append("• 'Where might this be recorded?'") 
                response_parts.append("• 'What sounds are in the background?'")
                response_parts.append("• 'How many speakers are there?'")
                
            else:
                response_parts.append(f"I'd be happy to answer your question about '{question}'.")
                response_parts.append("\nBased on the available audio analysis data, I can provide insights about the content.")
                response_parts.append("\nPlease note: For the most detailed and accurate responses, ensure the Gemini API is properly configured with your API key.")
                
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Fallback generation error: {e}")
            return f"I apologize, but I'm having trouble processing your question about the audio analysis. Please ensure your API configuration is correct and try again."
    
    def _calculate_confidence(self, audio_analysis: Dict[str, Any], question: str, response: str) -> float:
        """Calculate response confidence based on available data quality"""
        try:
            confidence = 0.5  # Base confidence
            
            if audio_analysis:
                # Increase confidence based on available analysis components
                if audio_analysis.get('transcript'):
                    confidence += 0.15
                if audio_analysis.get('environmentalSounds'):
                    confidence += 0.1
                if audio_analysis.get('acousticFeatures'):
                    confidence += 0.1
                if audio_analysis.get('speakerSegments'):
                    confidence += 0.1
                if audio_analysis.get('duration', 0) > 1.0:  # Longer audio = better analysis
                    confidence += 0.05
            
            # Adjust based on response quality
            if len(response) > 100:  # Detailed response
                confidence += 0.05
            if 'API' not in response and 'error' not in response.lower():  # No API errors
                confidence += 0.05
                
            return min(confidence, 0.95)  # Cap at 95%
            
        except Exception:
            return 0.6