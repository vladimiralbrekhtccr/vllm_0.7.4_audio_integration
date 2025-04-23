# In vllm/multimodal/internavl_audio_parsers.py
from vllm.model_executor.models.internvl_whisper_processor import WhisperProcessor
import numpy as np
import torch
import os

# Add this class that's being imported in __init__.py
class AudioPlugin:
    """Plugin for audio processing in multimodal pipelines"""
    @staticmethod
    def register():
        """Register the audio plugin"""
        register_audio_parsers()

def register_audio_parsers():
    """Register parsers for audio inputs in multimodal pipelines"""
    from vllm.multimodal.parse import register_parser
    
    # Get the model path from environment or config
    whisper_model_path = os.environ.get("WHISPER_MODEL_PATH", None)
    
    # Lazily initialize the processor when needed
    whisper_processor = None
    
    def get_processor():
        nonlocal whisper_processor
        if whisper_processor is None:
            if whisper_model_path:
                whisper_processor = WhisperProcessor.from_pretrained(whisper_model_path)
            else:
                # Fallback to a default pretrained model if path not specified
                from transformers import WhisperProcessor
                whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        return whisper_processor
    
    @register_parser(item_type="audio")
    def parse_audio_data(item):
        """Process raw audio data into model-ready format"""
        processor = get_processor()
        
        if isinstance(item, (np.ndarray, list)):
            # This is raw audio data
            raw_audio = np.asarray(item, dtype=np.float32)
            
            # Process with Whisper processor
            processed_audio = processor(
                audio=raw_audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            )
            
            # Extract required fields
            return {
                "audio_features": processed_audio["input_features"],
                "audio_len_after_cnn": processed_audio["audio_len_after_cnn"],
                "audio_token_num": processed_audio["audio_token_num"],
            }
        
        # For already processed audio, just pass through
        return item