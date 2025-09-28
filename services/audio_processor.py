import whisper
import librosa
import numpy as np
import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional
import tempfile
import torch
from langdetect import detect, DetectorFactory
try:
    from polyglot.detect import Detector
    from polyglot.text import Text
    POLYGLOT_AVAILABLE = True
except ImportError:
    POLYGLOT_AVAILABLE = False
    
try:
    import pycld2 as cld2
    CLD2_AVAILABLE = True
except ImportError:
    CLD2_AVAILABLE = False

try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False


# Set seed for consistent language detection
DetectorFactory.seed = 0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.whisper_model = None
        self.speaker_pipeline = None
        
        # Initialize Whisper model
        self._initialize_whisper()
        
        # Initialize speaker diarization using pyannote.audio if available
        self._initialize_speaker_analysis()
    
    def _initialize_whisper(self):
        try:
            model_size = os.getenv("WHISPER_MODEL", "base")
            logger.info(f"🎤 Loading Whisper model: {model_size}")
            self.whisper_model = whisper.load_model(model_size, device="cpu")
            logger.info("✅ Whisper model loaded successfully with multi-language support")
        except Exception as e:
            logger.error(f"Whisper initialization error: {e}")
            logger.warning("⚠️ Whisper not available - transcription will be limited")
            self.whisper_model = None
    
    def _initialize_speaker_analysis(self):
        if PYANNOTE_AVAILABLE:
            try:
                self.speaker_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
                logger.info("✅ Pyannote.audio speaker diarization pipeline loaded")
            except Exception as e:
                logger.error(f"Pyannote.audio initialization error: {e}")
                self.speaker_pipeline = None
        else:
            logger.warning("⚠️ Pyannote.audio not available, speaker diarization will use fallback")
            self.speaker_pipeline = None
    
    async def process_audio(self, file_path: str) -> Dict[str, Any]:
        try:
            logger.info(f"🎵 Starting comprehensive audio analysis for: {file_path}")
            
            audio, sr = librosa.load(file_path, sr=None)
            duration = len(audio) / sr
            
            logger.info(f"📊 Audio loaded: {duration:.2f}s, {sr}Hz, {len(audio)} samples")
            
            tasks = [
                self._transcribe_audio(file_path),
                self._detect_environmental_sounds(audio, sr),
                self._extract_acoustic_features(audio, sr),
                self._perform_speaker_diarization(file_path),
                self._analyze_emotions(audio, sr)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            transcript = results[0] if not isinstance(results[0], Exception) else ""
            environmental_sounds = results[1] if not isinstance(results[1], Exception) else []
            acoustic_features = results[2] if not isinstance(results[2], Exception) else {}
            speaker_segments = results[3] if not isinstance(results[3], Exception) else []
            emotions = results[4] if not isinstance(results[4], Exception) else []
            
            unique_speakers = len(set(seg.get('speaker', 'Unknown') for seg in speaker_segments))
            quality = self._assess_audio_quality(acoustic_features)
            
            analysis_result = {
                "transcript": transcript,
                "environmentalSounds": environmental_sounds,
                "acousticFeatures": acoustic_features,
                "speakerSegments": speaker_segments,
                "emotions": emotions,
                "duration": float(duration),
                "speakerCount": unique_speakers,
                "quality": quality
            }
            
            logger.info("✅ Comprehensive audio analysis completed successfully")
            return analysis_result
        
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return {
                "transcript": "",
                "environmentalSounds": [],
                "acousticFeatures": {},
                "speakerSegments": [],
                "emotions": [],
                "duration": 0.0,
                "speakerCount": 0,
                "quality": "Unknown",
                "error": str(e)
            }
    
    def _detect_audio_languages(self, file_path: str) -> List[str]:
        detected_languages = []
        try:
            if self.whisper_model:
                logger.info("🔍 Using Whisper language detection...")
                audio = whisper.load_audio(file_path)
                audio = whisper.pad_or_trim(audio)
                mel = whisper.log_mel_spectrogram(audio).to(self.whisper_model.device)
                _, probs = self.whisper_model.detect_language(mel)
                sorted_langs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                detected_languages = [lang for lang, prob in sorted_langs if prob > 0.05]
                logger.info(f"🎯 Whisper detected languages: {detected_languages}")
        except Exception as e:
            logger.warning(f"Whisper language detection failed: {e}")
        
        if CLD2_AVAILABLE:
            try:
                if self.whisper_model:
                    result = self.whisper_model.transcribe(
                        file_path,
                        language=None,
                        temperature=0.0,
                        no_speech_threshold=0.1,
                        fp16=False
                    )
                    if result and result.get("text", "").strip():
                        text = result["text"].strip()
                        if len(text) > 10:
                            is_reliable, _, details = cld2.detect(text)
                            if is_reliable:
                                lang_code = details[0][1]
                                lang_mapping = {
                                    'en': 'en', 'hi': 'hi', 'es': 'es', 'fr': 'fr',
                                    'de': 'de', 'it': 'it', 'pt': 'pt', 'ru': 'ru',
                                    'ja': 'ja', 'ko': 'ko', 'zh': 'zh', 'ar': 'ar'
                                }
                                if lang_code in lang_mapping and lang_mapping[lang_code] not in detected_languages:
                                    detected_languages.append(lang_mapping[lang_code])
                                    logger.info(f"🔍 CLD2 detected: {lang_code}")
            except Exception as e:
                logger.warning(f"CLD2 language detection failed: {e}")

        if POLYGLOT_AVAILABLE:
            try:
                if self.whisper_model:
                    result = self.whisper_model.transcribe(
                        file_path,
                        language=None,
                        temperature=0.2,
                        no_speech_threshold=0.2,
                        fp16=False
                    )
                    if result and result.get("text", "").strip():
                        text = result["text"].strip()
                        if len(text) > 5:
                            detector = Detector(text)
                            if detector.reliable:
                                lang_code = detector.language.code
                                lang_mapping = {
                                    'en': 'en', 'hi': 'hi', 'es': 'es', 'fr': 'fr',
                                    'de': 'de', 'it': 'it', 'pt': 'pt', 'ru': 'ru'
                                }
                                if lang_code in lang_mapping and lang_mapping[lang_code] not in detected_languages:
                                    detected_languages.append(lang_mapping[lang_code])
                                    logger.info(f"🌐 Polyglot detected: {lang_code}")
            except Exception as e:
                logger.warning(f"Polyglot language detection failed: {e}")

        if not detected_languages:
            detected_languages = ['en', 'hi']  # fallback
            logger.info("🔄 Using fallback languages: English, Hindi")

        return detected_languages[:3]
    
    async def _transcribe_audio(self, file_path: str) -> str:
        loop = asyncio.get_event_loop()
        
        def transcribe():
            if self.whisper_model is None:
                logger.warning("⚠️ Whisper model not available")
                return ""
            try:
                detected_langs = self._detect_audio_languages(file_path)
                logger.info(f"Languages detected: {detected_langs}")
                transcripts = []
                for lang in detected_langs:
                    logger.info(f"🎤 Transcribing using language: {lang}")
                    result = self.whisper_model.transcribe(file_path, language=lang, fp16=False)
                    text = result.get('text', '').strip()
                    if text:
                        transcripts.append((lang, text))
                if transcripts:
                    best = max(transcripts, key=lambda x: len(x[1]))
                    logger.info(f"🎉 Best transcript chosen for language: {best[0]} with length {len(best[1])}")
                    return best[1]
                else:
                    return ""
            except Exception as e:
                logger.error(f"❌ Transcription error: {e}")
                return ""
        
        return await loop.run_in_executor(self.executor, transcribe)
    
    async def _detect_environmental_sounds(self, audio: np.ndarray, sr: int) -> List[str]:
        loop = asyncio.get_event_loop()
        
        def detect_sounds():
            try:
                logger.info("🔊 Analyzing environmental sounds...")
                sounds = []

                rms = librosa.feature.rms(y=audio)[0]
                avg_energy = np.mean(rms)

                spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
                avg_centroid = np.mean(spectral_centroids)

                zcr = librosa.feature.zero_crossing_rate(audio)[0]
                avg_zcr = np.mean(zcr)

                mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

                try:
                    tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
                except:
                    tempo = 0

                harmonic, percussive = librosa.effects.hpss(audio)
                harmonic_ratio = np.mean(np.square(harmonic)) / (np.mean(np.square(audio)) + 1e-6)
                percussive_ratio = np.mean(np.square(percussive)) / (np.mean(np.square(audio)) + 1e-6)

                if (0.01 < avg_zcr < 0.35 and avg_energy > 0.01 and 1000 < avg_centroid < 4000 and harmonic_ratio > 0.3):
                    sounds.append("Speech")

                if (harmonic_ratio > 0.6 and tempo > 60 and avg_energy > 0.02):
                    sounds.append("Music")

                if avg_centroid > 4000:
                    sounds.append("High-frequency sounds")

                if avg_centroid < 1000 and avg_energy > 0.015:
                    sounds.append("Low-frequency sounds")

                if percussive_ratio > 0.4:
                    sounds.append("Percussive sounds")

                if avg_energy < 0.01:
                    sounds.append("Quiet background")
                elif 0.01 <= avg_energy < 0.03:
                    sounds.append("Moderate background noise")
                elif avg_energy >= 0.03:
                    sounds.append("Loud environment")

                if (1000 < avg_centroid < 3000 and 0.01 < avg_energy < 0.05 and harmonic_ratio > 0.2):
                    sounds.append("Indoor environment")

                if (avg_centroid > 2000 and avg_energy > 0.02 and percussive_ratio < 0.3):
                    sounds.append("Outdoor environment")

                if (avg_centroid < 2000 and avg_energy > 0.03 and harmonic_ratio < 0.4):
                    sounds.append("Vehicle sounds")

                if (avg_zcr > 0.15 and avg_energy > 0.025 and np.std(spectral_centroids) > 500):
                    sounds.append("Crowd noise")

                sounds = list(set(sounds))[:8]
                if not sounds:
                    sounds = ["Ambient audio"]

                logger.info(f"🔊 Detected {len(sounds)} environmental sound categories")
                return sounds

            except Exception as e:
                logger.error(f"Environmental sound detection error: {e}")
                return ["Unknown audio content"]

        return await loop.run_in_executor(self.executor, detect_sounds)
    
    async def _extract_acoustic_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        
        def extract_features():
            try:
                logger.info("🎵 Extracting comprehensive acoustic features...")
                
                features = {}
                rms = librosa.feature.rms(y=audio)[0]
                features["rms_energy"] = float(np.mean(rms))
                features["rms_std"] = float(np.std(rms))

                zcr = librosa.feature.zero_crossing_rate(audio)[0]
                features["zcr_mean"] = float(np.mean(zcr))
                features["zcr_std"] = float(np.std(zcr))

                spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
                features["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
                features["spectral_centroid_std"] = float(np.std(spectral_centroids))

                spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
                features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))

                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
                features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))

                mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                for i in range(13):
                    features[f"mfcc_{i+1}_mean"] = float(np.mean(mfccs[i]))
                    features[f"mfcc_{i+1}_std"] = float(np.std(mfccs[i]))

                try:
                    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
                    features["tempo"] = float(tempo)
                except:
                    features["tempo"] = 0.0

                pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
                pitch_values = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)

                if pitch_values:
                    features["avg_pitch"] = float(np.mean(pitch_values))
                    features["pitch_std"] = float(np.std(pitch_values))
                else:
                    features["avg_pitch"] = 0.0
                    features["pitch_std"] = 0.0

                features["loudness_db"] = float(20 * np.log10(features["rms_energy"] + 1e-6))

                harmonic, percussive = librosa.effects.hpss(audio)
                features["harmonic_ratio"] = float(np.mean(np.square(harmonic)) / (np.mean(np.square(audio)) + 1e-6))
                features["percussive_ratio"] = float(np.mean(np.square(percussive)) / (np.mean(np.square(audio)) + 1e-6))

                onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
                features["onset_rate"] = float(len(onset_frames) / (len(audio) / sr))

                frame_length = 2048
                hop_length = 512
                frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
                voice_frames = 0

                for frame in frames.T:
                    frame_energy = np.sum(frame**2)
                    frame_zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))

                    if frame_energy > 0.01 and 0.01 < frame_zcr < 0.3:
                        voice_frames += 1

                features["voice_activity_ratio"] = float(voice_frames / len(frames.T))

                logger.info("✅ Acoustic feature extraction completed")
                return features

            except Exception as e:
                logger.error(f"Feature extraction error: {e}")
                return {"error": str(e)}

        return await loop.run_in_executor(self.executor, extract_features)
    
    async def _perform_speaker_diarization(self, file_path: str) -> List[Dict[str, Any]]:
        loop = asyncio.get_event_loop()
        
        def diarize():
            try:
                if self.speaker_pipeline is not None:
                    logger.info("👥 Running pyannote.audio speaker diarization...")
                    diarization = self.speaker_pipeline(file_path)
                    segments = []
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        segments.append({
                            "speaker": speaker,
                            "start": turn.start,
                            "end": turn.end,
                            "duration": turn.end - turn.start,
                            "confidence": 0.95,
                        })
                    logger.info(f"👥 Pyannote found {len(segments)} segments")
                    return segments
                else:
                    logger.warning("⚠️ Pyannote pipeline not available, using fallback")
                    return self._fallback_speaker_diarization(file_path)
            except Exception as e:
                logger.error(f"Speaker diarization error: {e}")
                return self._fallback_speaker_diarization(file_path)
        
        return await loop.run_in_executor(self.executor, diarize)

    def _fallback_speaker_diarization(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            logger.info("👥 Using enhanced energy-based speaker analysis fallback...")
            audio, sr = librosa.load(file_path, sr=16000)
            segments = []

            window_size = int(3 * sr)
            hop_length = int(1 * sr)

            features_list = []
            times = []

            for i in range(0, len(audio) - window_size + 1, hop_length):
                segment_audio = audio[i:i + window_size]

                mfcc = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=13)
                spectral_centroid = librosa.feature.spectral_centroid(y=segment_audio, sr=sr)
                rms = librosa.feature.rms(y=segment_audio)

                features = np.concatenate([
                    np.mean(mfcc, axis=1),
                    [np.mean(spectral_centroid)],
                    [np.mean(rms)]
                ])

                features_list.append(features)
                times.append(i / sr)

            if len(features_list) > 1:
                features_array = np.array(features_list)

                from scipy.stats import zscore
                features_normalized = zscore(features_array, axis=0)

                distances = []
                for i in range(1, len(features_normalized)):
                    dist = np.linalg.norm(features_normalized[i] - features_normalized[i-1])
                    distances.append(dist)

                threshold = np.mean(distances) + np.std(distances)
                speaker_changes = [0]

                for i, dist in enumerate(distances):
                    if dist > threshold:
                        speaker_changes.append(i + 1)

                for i, change_idx in enumerate(speaker_changes):
                    start_time = times[change_idx]
                    end_time = times[speaker_changes[i + 1]] if i + 1 < len(speaker_changes) else len(audio) / sr

                    segments.append({
                        "speaker": f"Speaker_{chr(65 + i)}",
                        "start": float(start_time),
                        "end": float(end_time),
                        "duration": float(end_time - start_time),
                        "confidence": 0.6
                    })
            return segments[:10]

        except Exception as e:
            logger.error(f"Fallback speaker analysis error: {e}")
            return []

    async def _analyze_emotions(self, audio: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        loop = asyncio.get_event_loop()
        
        def analyze():
            try:
                logger.info("😊 Analyzing emotional content...")
                emotions = []

                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
                rms = librosa.feature.rms(y=audio)[0]
                zcr = librosa.feature.zero_crossing_rate(audio)[0]
                tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)

                avg_rms = np.mean(rms)
                std_rms = np.std(rms)
                avg_centroid = np.mean(spectral_centroid)
                avg_zcr = np.mean(zcr)

                if avg_rms > 0.03 and avg_centroid > 2000 and tempo > 120:
                    emotions.append({"emotion": "Excited", "confidence": min(0.8, avg_rms * 20)})

                if avg_rms < 0.015 and avg_centroid < 1500:
                    emotions.append({"emotion": "Calm", "confidence": 0.7})

                if std_rms > avg_rms * 0.8:
                    emotions.append({"emotion": "Dynamic", "confidence": 0.6})

                if avg_zcr > 0.15:
                    emotions.append({"emotion": "Energetic", "confidence": min(0.75, avg_zcr * 4)})

                if avg_rms > 0.02 and avg_centroid < 1000:
                    emotions.append({"emotion": "Intense", "confidence": 0.65})

                if not emotions:
                    emotions.append({"emotion": "Neutral", "confidence": 0.5})

                emotions.sort(key=lambda x: x["confidence"], reverse=True)
                return emotions[:3]

            except Exception as e:
                logger.error(f"Emotion analysis error: {e}")
                return [{"emotion": "Unknown", "confidence": 0.0}]

        return await loop.run_in_executor(self.executor, analyze)

    def _assess_audio_quality(self, acoustic_features: Dict[str, Any]) -> str:
        try:
            if not acoustic_features or "rms_energy" not in acoustic_features:
                return "Unknown"
            rms = acoustic_features.get("rms_energy", 0)
            snr_estimate = acoustic_features.get("harmonic_ratio", 0.5)

            if rms > 0.05 and snr_estimate > 0.7:
                return "Excellent"
            elif rms > 0.02 and snr_estimate > 0.5:
                return "Good"
            elif rms > 0.01:
                return "Fair"
            else:
                return "Poor"
        except Exception:
            return "Unknown"
    
    async def process_streaming_audio(self, audio_data: bytes) -> Dict[str, Any]:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            result = await self.process_audio(temp_path)
            
            os.unlink(temp_path)
            
            return result
        
        except Exception as e:
            logger.error(f"Streaming audio processing error: {e}")
            return {
                "transcript": "",
                "environmentalSounds": [],
                "acousticFeatures": {},
                "speakerSegments": [],
                "emotions": [],
                "duration": 0.0,
                "speakerCount": 0,
                "quality": "Unknown",
                "error": str(e)
            }
