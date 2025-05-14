# Local LLM Chat with Ollama

A simple web interface for interacting with Ollama's local LLM models.

## Prerequisites

1. Python 3.7+
2. [Ollama](https://ollama.ai/) installed and running locally

## Setup

1. Clone this repository
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Make sure Ollama is running locally (default: http://localhost:11434)
2. Start the web application:
   ```bash
   python main.py
   ```
3. Open your browser and navigate to `http://localhost:8000`

## Environment Variables

You can configure the application using the following environment variables:

- `OLLAMA_API_URL`: The URL of the Ollama API (default: `http://localhost:11434/api/generate`)
- `DEFAULT_MODEL`: The default model to use (default: `llama2`)

## Features

- Streamed responses for a more interactive experience
- Clean, responsive UI built with Tailwind CSS
- Support for multi-line messages (Shift+Enter for new line)
- Loading indicators during response generation
- Error handling for API connectivity issues

## License

MIT
