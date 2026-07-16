from pyexpat import model
import requests
import json
import soundfile as sf
import numpy as np
from qwen_tts import Qwen3TTSModel
import sounddevice as sd

Question = input("Ask a question:  ")
Voice_instruction = ("Said in an angry Scottish accent")

TTSmodel = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    device_map="auto",
)


url = "http://localhost:11434/api/chat"

payload = {
    "model": "gemma4", 
    "messages": [
        {'role': 'system', 'content': 'You are a helpful assistant who speaks like a scottish person.'},
        {"role": "user", "content": Question}
        
        ]
}
response = requests.post(url, json=payload, stream=True)

if response.status_code == 200:
    print("Streaming response from Ollama:")
    for line in response.iter_lines(decode_unicode=True):
        if line:  
            try:
                json_data = json.loads(line)
                
                if "message" in json_data and "content" in json_data["message"]:
                    print(json_data["message"]["content"], end="")
            except json.JSONDecodeError:
                print(f"\nFailed to parse line: {line}")
    print()  
else:
    print(f"Error: {response.status_code}")
    print(response.text)
    wavs, sr = TTSmodel.generate_custom_voice(
        text=response.text,
        language="English", 
        speaker="Ryan",
        instruct=Voice_instruction, 

    )
    fs = sf.read("output_custom_voice.wav", dtype='float32')
    sf.write("output_custom_voice.wav", wavs[0], sr)
    sd.play(wavs[0], sr)
    sd.wait()

