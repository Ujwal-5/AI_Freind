# AI Friend - Voice-Enabled AI Assistant

AI Friend is a voice-activated assistant that combines speech recognition, natural language processing, and text-to-speech to create a seamless conversational experience. It can help with daily tasks, Linux commands, and engage in casual conversations.

## Features

- **Voice Recognition**: Uses Whisper.cpp for accurate speech-to-text conversion
- **Natural Conversations**: Powered by local LLM models like Qwen
- **Text-to-Speech**: Responds with natural-sounding speech
- **Command Execution**: Can execute Linux commands when requested
- **Conversation Memory**: Remembers previous interactions using vector embeddings
- **Adaptive Listening**: Adjusts to ambient noise levels
- **Idle Conversation**: Initiates conversations after periods of inactivity

## Installation

### Prerequisites

- Python 3.8+
- ffmpeg (for audio processing)
- Linux or macOS recommended (Windows supported with limitations)

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ai-friend.git
   cd ai-friend
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download required models:
   - A LLM model (Qwen2.5-1.5b-instruct recommended)
   - TTS model will be downloaded automatically on first run

## Usage

Basic usage:
```bash
python3 ai_friend.py
```

With options:
```bash
python3 ai_friend.py --model small --llm_model path/to/your/model.gguf --adaptive_energy
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | tiny | Whisper model to use (tiny, base, small, medium, large) |
| `--non_english` | False | Don't use the English-specific model |
| `--energy_threshold` | 100 | Energy level for mic to detect |
| `--record_timeout` | 2 | How real-time the recording is in seconds |
| `--phrase_timeout` | 3 | How much empty space between recordings before considering it a new line |
| `--llm_model` | qwen2.5-1.5b-instruct-q8_0.gguf | Path to the LLM model file |
| `--context_length` | 512 | Context length for the LLM |
| `--threads` | 4 | Number of threads to use for model inference |
| `--tts_model` | tts_models/en/vctk/vits | TTS model to use |
| `--tts_speaker` | p276 | Speaker voice for TTS |
| `--use_gpu` | False | Use GPU for TTS if available |
| `--adaptive_energy` | False | Dynamically adjust energy threshold |
| `--silent_mode` | False | Don't speak responses (text only) |
| `--idle_timeout` | 300 | Time in seconds before initiating conversation after idle |

## Interacting with AI Friend

1. Start the application
2. Wait for the welcome message
3. Speak naturally - AI Friend will transcribe your speech
4. It will respond both visually and audibly (unless in silent mode)
5. To run Linux commands, ask AI Friend to execute them for you

### Command Execution

When you need to execute a Linux command, simply ask AI Friend. It will format the command and execute it:

- "Can you show me the files in this directory?"
- "Please check how much disk space I have left"
- "Create a new directory called projects"

Commands requiring sudo will ask for confirmation before executing.

## Customization

### System Prompt

You can modify the system prompt in the code to change AI Friend's behavior and personality.

### Voice Selection

Change the TTS speaker voice with the `--tts_speaker` option. Available voices depend on the TTS model used.

## Troubleshooting

### Microphone Issues

If AI Friend can't hear you:
- Check your microphone settings
- Try increasing the `--energy_threshold` value
- Enable `--adaptive_energy` to automatically adjust to ambient noise

### Performance Issues

- Try a smaller Whisper model (`--model tiny` or `--model base`)
- Reduce context length (`--context_length 256`)
- Use a smaller LLM model

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) for speech recognition
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for LLM inference
- [Coqui TTS](https://github.com/coqui-ai/TTS) for text-to-speech capability
- All the open-source AI models that make this project possible
