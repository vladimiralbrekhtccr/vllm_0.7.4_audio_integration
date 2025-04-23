import base64
import httpx
from pathlib import Path

API_URL = "http://localhost:6664/v1/chat/completions"
MODEL = "/scratch/vladimir_albrekht/projects/oylan_a_v_t/output/AVbaby_training_v1_train/checkpoint-1000"
HEADERS = {"Authorization": "Bearer token-abc123"}

# Path to a single audio file
AUDIO_PATH = Path("/scratch/vladimir_albrekht/projects/oylan_a_v_t/vllm_imp_debug/inference_vllm/__inference_and_co/assets/rustem_1.wav")

# Load and encode audio to base64
def load_audio_base64(path: Path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Prepare the OpenAI-compatible payload
def build_payload(audio_data: str):
    return {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Your name is Oylan, you are a useful multi-modal large language model developed by ISSAI, Kazakhstan."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Who is talking in this audio?"},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_data,
                            "format": "wav"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 512,
        "temperature": 0.1
    }

# Send a single request
def send_audio_request():
    audio_data = load_audio_base64(AUDIO_PATH)
    payload = build_payload(audio_data)
    
    with httpx.Client(timeout=60.0) as client:
        response = client.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        print("✅ Response:", content)

if __name__ == "__main__":
    send_audio_request()