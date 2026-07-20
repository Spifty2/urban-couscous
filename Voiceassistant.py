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
from llm_axe import OnlineAgent, OllamaChat, PdfReader, DataExtractor
import urllib.parse
import itertools
import sys
import time

def loading_animation():
    spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
    
    for _ in range(50):
        sys.stdout.write(f"\rLoading your data... {next(spinner)}")
        sys.stdout.flush()
        time.sleep(0.1)
        
    sys.stdout.write("\rLoading complete!      \n")
Question = input("Ask a question:  ")
Voice_instruction = ("Said in an angry Scottish accent")

Voice_toggle = input("Do you want the answer to be spoken? (yes/no):  ")

sr = 16000 

TTSmodel = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    device_map="auto",
)


url = "http://localhost:11434/api/chat"


llm = OllamaChat(model="gemma4")
searcher = OnlineAgent(llm)
de = DataExtractor(llm)


payload = {
    "model": "gemma4", 
    "messages": [
        {'role': 'system', 'content': ' You are a helpful assistant who speaks like a scottish person. Give answers under 3 lines. Give a possible google search to find more information labelled as "Google search". If you do not know the answer, say "I do not know" and give a possible google search to find more information labelled as "Google search"'},
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
    if Voice_toggle.lower() == "yes":
        print("Starting TTS generation...")
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
        print(f"Generation took {time.time() - start:.1f}s")
    else:
        print("TTS generation skipped as per user choice.")    
else:       
    print("Error: Failed to get a response from Ollama.")                                           
    print(f"Error: {response_short.status_code}")
    print(response_short.text)

print("Offline response complete")

Next_steps = input("Do you want to search the internet for more information? (yes/no):  ")

if Next_steps.lower() == "yes":

    info = full_reply
    web_search_question = de.ask(info, ["Google search:"])
    print("Searching the internet for more information on " + web_search_question + "...")
    search_url = "https://www.google.com/search?q=" + urllib.parse.quote(web_search_question)
    resp_internet = searcher.search(Question + " according to " + search_url)
    print("searching...")
    print(resp_internet)

else:
    print("No further internet search will be performed.")

    