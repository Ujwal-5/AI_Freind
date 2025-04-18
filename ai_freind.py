#!/usr/bin/env python3

import argparse
import os
import numpy as np
import speech_recognition as sr
import sounddevice as sd
import time
import threading
import queue
import subprocess
import re
import json
from datetime import datetime, timedelta
from sys import platform
from TTS.api import TTS
from pywhispercpp.model import Model
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import random

class AIFriend:
    def __init__(self, args):
        # Speech recognition settings
        self.energy_threshold = args.energy_threshold
        self.record_timeout = args.record_timeout
        self.phrase_timeout = args.phrase_timeout
        self.adaptive_energy = args.adaptive_energy
        self.silent_mode = args.silent_mode
        self.idle_timeout = args.idle_timeout
        
        # Initialize audio recording components
        self.setup_audio_recording()
        
        # Load whisper model for speech recognition
        whisper_model = args.model
        if whisper_model != "large" and not args.non_english:
            whisper_model = whisper_model + ".en"
        self.audio_model = Model(whisper_model, n_threads=args.threads)
        print(f"Loaded Whisper model: {whisper_model}")
        
        # Load Qwen model for conversation
        self.llm = Llama(
            model_path=args.llm_model,
            n_ctx=args.context_length,
            verbose=False
        )
        print(f"Loaded LLM model: {args.llm_model}")
        
        # Load TTS model
        self.tts = TTS(model_name=args.tts_model, progress_bar=False, gpu=args.use_gpu)
        self.tts_speaker = args.tts_speaker
        print(f"Loaded TTS model with speaker: {self.tts_speaker}")
        
        # Vector database for conversation memory
        self.setup_vector_database()
        
        # Conversation state
        self.last_interaction_time = datetime.now()
        self.conversation_history = []
        self.system_prompt = """You are a friendly and helpful AI assistant. 
You can have natural conversations and help with Linux commands.
Keep your responses concise and conversational.
If you're suggesting a Linux command, format it with [COMMAND] tags like: [COMMAND]ls -la[/COMMAND]
Don't produce code snippets other than simple Linux commands.
Be aware that you have limited context, so keep your responses focused."""
        
        # Command execution settings
        self.command_queue = queue.Queue()
        self.command_thread = threading.Thread(target=self.command_executor, daemon=True)
        self.command_thread.start()
        
        # Idle conversation settings
        self.idle_thread = threading.Thread(target=self.idle_conversation_monitor, daemon=True)
        self.idle_thread.start()
        
        # Dynamic energy threshold adjustment
        self.energy_adjustment_thread = None
        if self.adaptive_energy:
            self.energy_adjustment_thread = threading.Thread(target=self.adjust_energy_threshold, daemon=True)
            self.energy_adjustment_thread.start()

    def setup_audio_recording(self):
        """Set up the audio recording components"""
        # Thread safe Queue for passing data from the threaded recording callback
        self.data_queue = queue.Queue()
        # Bytes object which holds audio data for the current phrase
        self.phrase_bytes = bytes()
        # The last time a recording was retrieved from the queue
        self.phrase_time = None
        
        # Initialize the recognizer
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = self.energy_threshold
        self.recorder.dynamic_energy_threshold = False
        
        # Set up microphone
        if 'linux' in platform:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if 'pulse' in name.lower():
                    self.source = sr.Microphone(sample_rate=16000, device_index=index)
                    print(f"Using microphone: {name}")
                    break
            else:
                self.source = sr.Microphone(sample_rate=16000)
                print("Using default microphone")
        else:
            self.source = sr.Microphone(sample_rate=16000)
            print("Using default microphone")
        
        # Adjust for ambient noise
        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)

    def setup_vector_database(self):
        """Set up the vector database for conversation memory"""
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.vector_dimension)
        self.memory_data = []
        
        # Try to load existing memory
        try:
            if os.path.exists('conversation_memory.json'):
                with open('conversation_memory.json', 'r') as f:
                    self.memory_data = json.load(f)
                
                # Rebuild the index
                if self.memory_data:
                    embeddings = np.array([item['embedding'] for item in self.memory_data], dtype=np.float32)
                    self.index = faiss.IndexFlatL2(self.vector_dimension)
                    self.index.add(embeddings)
                    print(f"Loaded {len(self.memory_data)} memory items")
        except Exception as e:
            print(f"Failed to load conversation memory: {e}")
            self.memory_data = []
            self.index = faiss.IndexFlatL2(self.vector_dimension)

    def start_listening(self):
        """Start the background listening process"""
        # Create a background thread that will pass us raw audio bytes
        self.recorder.listen_in_background(self.source, self.record_callback, phrase_time_limit=self.record_timeout)
        print("Listening for speech...")
        
        # Main loop
        try:
            self.process_audio_loop()
        except KeyboardInterrupt:
            print("\nStopping AI Friend...")
            self.save_conversation_memory()

    def record_callback(self, _, audio):
        """Callback function to receive audio data when recordings finish"""
        data = audio.get_raw_data()
        self.data_queue.put(data)

    def process_audio_loop(self):
        """Main loop to process audio and handle conversation"""
        transcription = ['']
        
        # Welcome message
        welcome_message = "Hello! I'm your AI friend. How can I help you today?"
        self.speak(welcome_message)
        print(f"AI: {welcome_message}")
        
        while True:
            now = datetime.now()
            
            # Pull raw recorded audio from the queue
            if not self.data_queue.empty():
                phrase_complete = False
                
                # If enough time has passed between recordings, consider the phrase complete
                if self.phrase_time and now - self.phrase_time > timedelta(seconds=self.phrase_timeout):
                    self.phrase_bytes = bytes()
                    phrase_complete = True
                
                # This is the last time we received new audio data from the queue
                self.phrase_time = now
                
                # Combine audio data from queue
                audio_data = b''.join(self.data_queue.queue)
                self.data_queue.queue.clear()
                
                # Add the new audio data to the accumulated data for this phrase
                self.phrase_bytes += audio_data
                
                # Convert in-ram buffer to something the model can use directly
                audio_np = np.frombuffer(self.phrase_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Transcribe the audio
                segments = self.audio_model.transcribe(audio_np)
                text = ''.join([seg.text for seg in segments]).strip()
                
                # If text is empty, skip processing
                if not text:
                    time.sleep(0.1)
                    continue
                
                # If we detected a pause between recordings, add a new item to our transcription
                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text
                
                # Print updated transcription
                os.system('cls' if os.name=='nt' else 'clear')
                print("Transcription:")
                for line in transcription:
                    if line:
                        print(f"You: {line}")
                
                # Process the transcribed text if phrase is complete
                if phrase_complete and text:
                    self.process_user_input(text)
                    self.last_interaction_time = datetime.now()
            else:
                # Sleep to avoid CPU hogging
                time.sleep(0.1)

    def process_user_input(self, text):
        """Process the transcribed user input and generate a response"""
        # Check for context from vector database
        context = self.retrieve_relevant_context(text)
        
        # Prepare messages for LLM
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add context if available
        if context:
            context_prompt = "Here's some relevant information from previous conversations:\n" + context
            messages.append({"role": "system", "content": context_prompt})
        
        # Add current conversation history (limited to last 5 exchanges)
        for exchange in self.conversation_history[-5:]:
            messages.append({"role": "user", "content": exchange["user"]})
            messages.append({"role": "assistant", "content": exchange["assistant"]})
        
        # Add the new user message
        messages.append({"role": "user", "content": text})
        
        # Generate response from LLM
        try:
            response = ""
            for chunks in self.llm.create_chat_completion(
                messages=messages,
                max_tokens=500,
                temperature=0.7,
                stream=True
            ):
                part = chunks["choices"][0]["delta"].get("content", None)
                if part:
                    response += part
            
            # Store the conversation
            self.conversation_history.append({
                "user": text,
                "assistant": response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Store in vector database
            self.add_to_memory(text, response)
            
            # Check for commands in the response
            commands = re.findall(r'\[COMMAND\](.*?)\[/COMMAND\]', response)
            clean_response = re.sub(r'\[COMMAND\](.*?)\[/COMMAND\]', r'`\1`', response)
            
            # Speak the response
            print(f"AI: {clean_response}")
            if not self.silent_mode:
                self.speak(clean_response)
            
            # Execute commands if found
            for cmd in commands:
                cmd = cmd.strip()
                if cmd.startswith("sudo "):
                    print(f"\nThis command requires sudo permissions: {cmd}")
                    permission = input("Do you want to execute it? (y/n): ")
                    if permission.lower() == 'y':
                        self.command_queue.put(cmd)
                else:
                    self.command_queue.put(cmd)
                    
        except Exception as e:
            error_msg = f"Sorry, I encountered an error while generating a response: {str(e)}"
            print(f"AI: {error_msg}")
            if not self.silent_mode:
                self.speak("Sorry, I encountered an error while generating a response.")

    def speak(self, text):
        """Convert text to speech and play it"""
        try:
            # Remove code blocks and markdown for better speech
            clean_text = re.sub(r'`.*?`', '', text)
            clean_text = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_text)
            
            # Generate speech
            speech_waveform = self.tts.tts(text=clean_text, speaker=self.tts_speaker)
            # speech_waveform, sample_rate = 
            # , samplerate=sample_rate
            # Play the audio
            sd.play(speech_waveform, samplerate=22050)
            sd.wait()  # Wait until playback is finished
        except Exception as e:
            print(f"TTS Error: {e}")

    def add_to_memory(self, user_text, assistant_response):
        """Add conversation to vector memory"""
        # Create combined text for embedding
        combined_text = f"User: {user_text} Assistant: {assistant_response}"
        
        # Generate embedding
        embedding = self.embedding_model.encode(combined_text)
        
        # Store in memory
        memory_item = {
            "user_text": user_text,
            "assistant_response": assistant_response,
            "combined_text": combined_text,
            "embedding": embedding.tolist(),
            "timestamp": datetime.now().isoformat()
        }
        
        self.memory_data.append(memory_item)
        
        # Add to FAISS index
        self.index.add(np.array([embedding], dtype=np.float32))
        
        # Save memory periodically (every 10 items)
        if len(self.memory_data) % 10 == 0:
            self.save_conversation_memory()

    def retrieve_relevant_context(self, query_text, k=3):
        """Retrieve relevant context from vector database"""
        if not self.memory_data:
            return ""
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query_text)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        # Search for similar items
        distances, indices = self.index.search(query_embedding, k)
        
        # Filter results with a similarity threshold
        context_items = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.memory_data) and distances[0][i] < 15:  # Lower distance is better
                context_items.append(self.memory_data[idx]["combined_text"])
        
        return "\n".join(context_items)

    def save_conversation_memory(self):
        """Save conversation memory to disk"""
        try:
            with open('conversation_memory.json', 'w') as f:
                json.dump(self.memory_data, f)
            print(f"Saved {len(self.memory_data)} memory items")
        except Exception as e:
            print(f"Failed to save conversation memory: {e}")

    def command_executor(self):
        """Thread for executing Linux commands"""
        while True:
            try:
                # Get command from queue
                cmd = self.command_queue.get()
                
                print(f"\nExecuting command: {cmd}")
                
                # Execute the command
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                # Print the output
                if result.stdout:
                    print(f"Output:\n{result.stdout}")
                if result.stderr:
                    print(f"Error:\n{result.stderr}")
                
                # Mark task as done
                self.command_queue.task_done()
            except Exception as e:
                print(f"Command execution error: {e}")
            
            time.sleep(0.1)

    def idle_conversation_monitor(self):
        """Monitor idle time and initiate conversation when needed"""
        while True:
            time.sleep(5)  # Check every 5 seconds
            
            # Calculate idle time
            idle_time = (datetime.now() - self.last_interaction_time).total_seconds()
            
            # If idle for longer than the timeout, initiate conversation
            if idle_time > self.idle_timeout and not self.silent_mode:
                self.initiate_conversation()
                self.last_interaction_time = datetime.now()

    def initiate_conversation(self):
        """Initiate conversation after idle period"""
        # Generate a prompt based on history or use a generic one
        if self.conversation_history:
            # Pick a random topic from conversation history
            random_exchange = random.choice(self.conversation_history[-10:])
            user_topic = random_exchange["user"]
            
            prompts = [
                f"I was thinking about our conversation earlier. You mentioned {user_topic}. Would you like to continue talking about that?",
                f"I noticed we were discussing {user_topic} before. Did you have any more thoughts on that?",
                "Would you like me to help you with anything else today?",
                "Do you need any assistance with Linux commands or anything else?"
            ]
        else:
            prompts = [
                "It's been quiet for a while. Is there anything I can help you with?",
                "Do you need any assistance with something?",
                "I'm here if you need me. Would you like to chat about something?",
                "Need any help with Linux commands or want to just chat?"
            ]
        
        # Choose a random prompt
        prompt = random.choice(prompts)
        
        # Speak the prompt
        print(f"\nAI: {prompt}")
        self.speak(prompt)

    def adjust_energy_threshold(self):
        """Dynamically adjust energy threshold based on ambient noise"""
        while True:
            try:
                with self.source:
                    # Sample ambient noise
                    ambient = self.recorder.listen(self.source, timeout=1, phrase_time_limit=1)
                    
                    # Calculate energy level
                    raw_data = np.frombuffer(ambient.get_raw_data(), dtype=np.int16)
                    energy = np.mean(np.abs(raw_data))
                    
                    # Adjust threshold (base + margin)
                    new_threshold = int(energy * 1.2) + 50
                    
                    # Apply limits to prevent extreme values
                    new_threshold = max(50, min(4000, new_threshold))
                    
                    # Update threshold if it's significantly different
                    if abs(self.recorder.energy_threshold - new_threshold) > 20:
                        self.recorder.energy_threshold = new_threshold
                        print(f"Adjusted energy threshold to: {new_threshold}")
            except Exception as e:
                if "timeout" not in str(e).lower():
                    print(f"Energy adjustment error: {e}")
            
            time.sleep(60)  # Check every minute

def main():
    parser = argparse.ArgumentParser(description="AI Friend - Voice Assistant")
    
    # Speech recognition settings
    parser.add_argument("--model", default="tiny", 
                      choices=["tiny", "base", "small", "medium", "large"],
                      help="Whisper model to use")
    parser.add_argument("--non_english", action='store_true',
                      help="Don't use the English-specific model")
    parser.add_argument("--energy_threshold", default=100, type=int,
                      help="Energy level for mic to detect")
    parser.add_argument("--record_timeout", default=2, type=float,
                      help="How real-time the recording is in seconds")
    parser.add_argument("--phrase_timeout", default=3, type=float,
                      help="How much empty space between recordings before we consider it a new line")
    
    # AI model settings
    parser.add_argument("--llm_model", default="qwen2.5-1.5b-instruct-q8_0.gguf",
                      help="Path to the LLM model file")
    parser.add_argument("--context_length", default=512, type=int,
                      help="Context length for the LLM")
    parser.add_argument("--threads", default=4, type=int,
                      help="Number of threads to use for model inference")
    
    # TTS settings
    parser.add_argument("--tts_model", default="tts_models/en/vctk/vits",
                      help="TTS model to use")
    parser.add_argument("--tts_speaker", default="p276", 
                      help="Speaker voice for TTS")
    parser.add_argument("--use_gpu", action='store_true',
                      help="Use GPU for TTS if available")
    
    # Additional features
    parser.add_argument("--adaptive_energy", action='store_true',
                      help="Dynamically adjust energy threshold")
    parser.add_argument("--silent_mode", action='store_true',
                      help="Don't speak responses (text only)")
    parser.add_argument("--idle_timeout", default=300, type=int,
                      help="Time in seconds before initiating conversation after idle")
    
    args = parser.parse_args()
    
    # Print settings
    print("\n=== AI Friend Settings ===")
    print(f"Whisper Model: {args.model}")
    print(f"Energy Threshold: {args.energy_threshold}")
    print(f"Record Timeout: {args.record_timeout}s")
    print(f"Phrase Timeout: {args.phrase_timeout}s")
    print(f"LLM Model: {args.llm_model}")
    print(f"TTS Model: {args.tts_model} (Speaker: {args.tts_speaker})")
    print(f"Adaptive Energy: {'Enabled' if args.adaptive_energy else 'Disabled'}")
    print(f"Silent Mode: {'Enabled' if args.silent_mode else 'Disabled'}")
    print(f"Idle Timeout: {args.idle_timeout}s")
    print("==========================\n")
    
    # Initialize and start AI Friend
    ai_friend = AIFriend(args)
    ai_friend.start_listening()

if __name__ == "__main__":
    main()