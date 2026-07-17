from pyexpat import model
import requests
import json
import soundfile as sf
import numpy as np
from qwen_tts import Qwen3TTSModel
import sounddevice as sd
import sys
import time
import traceback
import ollama
from ollama import 

Question = input("Ask a question:  ")
Voice_instruction = ("Said in an angry Scottish accent")

sr = 16000 

TTSmodel = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    device_map="auto",
)


url = "http://localhost:11434/api/chat"

payload = {
    "model": "gemma4", 
    "messages": [
        {'role': 'system', 'content': 'You are a helpful assistant who speaks like a scottish person. Give answers under 3 lines.'},
        {"role": "user", "content": Question}
        
        ]
}
response_short = requests.post(url, json=payload, stream=True)

full_reply = ""

if response_short.status_code == 200:                     
    print("Streaming response from Ollama:")
    for line in response_short.iter_lines(decode_unicode=True):
        if line:  
            try:
                json_data = json.loads(line)
                
                if "message" in json_data and "content" in json_data["message"]:
                    chunk = json_data["message"]["content"]
                    print (chunk, end="")
                    full_reply += chunk

                if json_data.get("done"):
                    print("\nStreaming complete.")
                    break
    
            except json.JSONDecodeError:
                print(f"\nFailed to parse line: {line}")
    print()
    print("starting TTS generation...")

    start = time.time()

    wavs, sr = TTSmodel.generate_custom_voice(
        text=full_reply,
        language="English", 
        speaker="Ryan",
        instruct=Voice_instruction, 

    )

    audio_output = wavs[0].astype('float32')
    sd.play(audio_output, sr)
    sd.wait()
    print("Audio playback finished.")
    print(f"generation took {time.time() - start:.1f}s")
           
else:       
    print("Error: Failed to get a response from Ollama.")                                           
    print(f"Error: {response_short.status_code}")
    print(response_short.text)
    
    