# Voice Chat Assistant - Streamlit Application

A web-based voice chat application built with Streamlit that allows you to record audio, transcribe it, get intelligent responses from Azure OpenAI, and hear the responses spoken back to you.

## Architecture

The application uses a modular, extensible architecture designed for agent development and capability enhancement:

### Project Structure

```
solution/
├── voice_chat_app.py      # Main Streamlit UI application
├── agent.py               # VoiceAssistantAgent class (core AI logic)
├── speech_utils.py        # Speech recognition and synthesis utilities
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

### Module Descriptions

#### `agent.py` - VoiceAssistantAgent
The core agent module that handles all LLM interactions and conversation management.

**Key Classes:**
- `VoiceAssistantAgent`: Manages Azure OpenAI interactions, conversation history, and extensible capabilities

**Key Methods:**
- `get_response(user_text)`: Get AI response for user input
- `clear_conversation()`: Reset conversation while keeping system prompt
- `get_conversation_history()`: Get user/assistant messages (excludes system prompt)
- `set_system_prompt(prompt)`: Update the system prompt
- `set_temperature(temperature)`: Adjust LLM creativity (0-2)
- `add_capability()`: Placeholder for future capability extensions
- `process_with_tools()`: Placeholder for future tool/function calling

**Future Capabilities:**
- Function calling integration
- Tool/API integration
- Knowledge base integration
- Memory management
- Custom reasoning engines

#### `speech_utils.py` - Speech Services Utilities
Utility functions for audio processing using Azure Speech Services.

**Key Functions:**
- `get_speech_config()`: Initialize Azure Speech configuration
- `transcribe_audio(audio_bytes)`: Convert audio to text
- `synthesize_speech(text)`: Convert text to audio

#### `voice_chat_app.py` - Main UI Application
Streamlit web interface for the voice chat assistant.

**Layout:**
- **Left Column**: User audio inputs and transcriptions
- **Middle Column**: Chat conversation interface (using `st.chat_message()`)
- **Right Column**: Assistant audio responses with text

## Features

- 🎤 **Audio Recording**: Record voice messages directly from the web interface
- 📝 **Speech-to-Text**: Automatically transcribe audio using Azure Speech Services
- 💬 **AI Responses**: Get intelligent responses from Azure OpenAI GPT
- 🔊 **Text-to-Speech**: Hear the assistant's responses spoken aloud
- 💾 **Conversation History**: View complete conversation with audio playback
- 🔄 **Continuous Conversation**: Maintain context across multiple interactions
- 🏗️ **Modular Architecture**: Clean separation of concerns for easy agent development

## Prerequisites

- Python 3.8+
- Azure OpenAI account and credentials
- Azure Speech Services account and credentials

## Environment Setup

Before running the application, create a `.env` file in the project root directory with the following environment variables:

```
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://<your-resource-name>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-api-key>
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT=<your-deployment-name>

# Azure Speech Services Configuration
AZURE_SPEECH_KEY=<your-speech-key>
AZURE_SPEECH_REGION=<your-region>
AZURE_SPEECH_RECOGNITION_LANGUAGE=en-US
AZURE_SPEECH_VOICE_NAME=en-US-AvaMultilingualNeural

# Optional alternatives
AZURE_OPENAI_GPT51_KEY=<alternative-api-key>
AZURE_OPENAI_GPT51_DEPLOYMENT=<alternative-deployment-name>
AZURE_SPEECH_VOICE=en-US-AvaMultilingualNeural
```

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit app with:

```bash
streamlit run voice_chat_app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Usage

1. **Record Audio**: Click the microphone icon in the center column to start recording
2. **Send Message**: Click "📤 Send Audio Message" button to process your audio
3. **View Transcript**: The transcribed text appears in the chat
4. **Get Response**: The assistant's response is displayed with audio playback
5. **View History**: Left column shows all your audio inputs, right column shows all responses
6. **Clear Chat**: Click "🗑️ Clear Chat" to reset the conversation

## How It Works

1. **Audio Input**: Your voice is recorded through the browser's audio input
2. **Transcription**: Audio is sent to Azure Speech Services for transcription
3. **LLM Processing**: Transcribed text is sent to Azure OpenAI for processing
4. **Response Generation**: The AI generates a contextual response maintaining conversation history
5. **Speech Synthesis**: The response is converted back to audio using Azure Speech Services
6. **Output**: The response is played back in the right column and stored for replay

## Extending the Agent

To add new capabilities to the agent:

1. Add new methods to the `VoiceAssistantAgent` class in `agent.py`
2. Implement the capability logic (e.g., function calling, tool integration)
3. Call the new methods from the main UI or other modules

### Example: Adding a Capability

```python
# In agent.py
class VoiceAssistantAgent:
    def retrieve_knowledge(self, query: str) -> str:
        """Retrieve information from knowledge base"""
        # Implementation here
        pass
    
    def get_response(self, user_text: str) -> str:
        # Use capabilities in response generation
        relevant_knowledge = self.retrieve_knowledge(user_text)
        # Incorporate knowledge into LLM context
        ...
```

## Error Handling

The application includes comprehensive error handling for:
- Missing environment variables
- Audio processing failures
- API communication errors
- Speech recognition/synthesis issues
- File access issues

All errors are displayed in the UI for easy troubleshooting.

## Notes

- The application maintains conversation context throughout the session
- System prompts are configured to keep responses concise and natural for voice conversations
- Audio processing is done in-memory for security and performance
- The application supports multiple languages through speech configuration
- Temporary audio files are automatically cleaned up after processing

## Troubleshooting

### No audio input available
- Ensure your microphone is properly connected and configured
- Check browser permissions for microphone access

### Transcription errors
- Verify your Azure Speech Services credentials are correct
- Check that the speech region matches your resource location

### API errors
- Ensure all required environment variables are set
- Verify your Azure credentials have sufficient permissions
- Check that your API keys haven't expired

### Agent capabilities not working
- Check the agent logs for detailed error messages
- Verify that all required APIs are properly configured
- Test capabilities individually in isolation

## Support

For issues with Azure services, refer to:
- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Azure Speech Services Documentation](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/)

## Development

The modular architecture makes it easy to:
- Add new agent capabilities
- Extend the UI with additional features
- Integrate with external APIs and services
- Implement advanced conversation strategies

For development, ensure you follow these patterns:
- Keep UI logic in `voice_chat_app.py`
- Keep agent logic in `agent.py`
- Keep utility functions in appropriate utility modules
- Use environment variables for all configuration
