import os
import streamlit as st
from dotenv import load_dotenv

from agent import VoiceAssistantAgent
from speech_utils import transcribe_audio, synthesize_speech

# Load environment variables
load_dotenv()


# Initialize Agent in session state
def init_session_state():
    if "agent" not in st.session_state:
        st.session_state.agent = VoiceAssistantAgent()
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "audio_inputs" not in st.session_state:
        st.session_state.audio_inputs = []
    if "audio_outputs" not in st.session_state:
        st.session_state.audio_outputs = []
    if "greeting_sent" not in st.session_state:
        st.session_state.greeting_sent = False


def get_llm_response(user_text: str) -> str:
    """Get response from the Voice Assistant Agent"""
    try:
        response = st.session_state.agent.get_response(user_text)
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"


def main():
    st.set_page_config(page_title="Voice Chat Assistant", layout="wide")
    st.title("🎤 Voice Chat Assistant")
    
    init_session_state()
    
    # Send greeting automatically on first load
    if not st.session_state.greeting_sent:
        greeting = st.session_state.agent.get_response("")
        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": greeting
        })
        # Generate audio for greeting
        try:
            audio_response = synthesize_speech(greeting)
            st.session_state.audio_outputs.append(audio_response)
        except Exception as e:
            st.warning(f"Could not generate audio for greeting: {str(e)}")
        st.session_state.greeting_sent = True
    
    # Create three columns for layout
    col_left, col_middle, col_right = st.columns([1, 2, 1])
    
    # MIDDLE COLUMN - Chat Interface
    with col_middle:
        st.header("💬 Conversation")
        
        # Display chat history
        for i, msg in enumerate(st.session_state.conversation_history):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        # Input section
        st.divider()
        st.subheader("Add Message")
        
        # Input method selection
        st.markdown("**Choose input method:**")
        input_method = st.radio(
            "Input method",
            options=["Audio", "Text"],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
        )

        col_input1, col_input2 = st.columns([1, 1])

        with col_input1:
            if input_method == "Audio":
                audio_data = st.audio_input("Record your message", label_visibility="collapsed")
                if audio_data is not None and st.button("📤 Send Audio Message"):
                    with st.spinner("Processing..."):
                        audio_bytes = audio_data.getvalue()
                        # Transcribe audio
                        with st.spinner("Transcribing..."):
                            user_text = transcribe_audio(audio_bytes)
                        if not user_text.startswith("Error"):
                            # Add user message to history
                            st.session_state.conversation_history.append({
                                "role": "user",
                                "content": user_text
                            })
                            st.session_state.audio_inputs.append(audio_bytes)
                            # Get LLM response
                            with st.spinner("Generating response..."):
                                response = get_llm_response(user_text)
                            # Add assistant message to history
                            st.session_state.conversation_history.append({
                                "role": "assistant",
                                "content": response
                            })
                            # Generate audio for response
                            with st.spinner("Generating audio..."):
                                try:
                                    audio_response = synthesize_speech(response)
                                    st.session_state.audio_outputs.append(audio_response)
                                except Exception as e:
                                    st.error(f"Error generating audio: {str(e)}")
                            st.rerun()
                        else:
                            st.error(user_text)
            else:
                user_text = st.text_input("Type your message", value="", key="text_input_field")
                if user_text and st.button("📤 Send Text Message"):
                    # Add user message to history
                    st.session_state.conversation_history.append({
                        "role": "user",
                        "content": user_text
                    })
                    # Get LLM response
                    with st.spinner("Generating response..."):
                        response = get_llm_response(user_text)
                    # Add assistant message to history
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    # Generate audio for response
                    with st.spinner("Generating audio..."):
                        try:
                            audio_response = synthesize_speech(response)
                            st.session_state.audio_outputs.append(audio_response)
                        except Exception as e:
                            st.error(f"Error generating audio: {str(e)}")
                    st.rerun()

        with col_input2:
            if st.button("🗑️ Clear Chat"):
                st.session_state.agent.clear_conversation()
                st.session_state.conversation_history = []
                st.session_state.audio_inputs = []
                st.session_state.audio_outputs = []
                st.rerun()
    
    # LEFT COLUMN - User Audio Inputs
    with col_left:
        st.header("🎤 User Inputs")
        
        if st.session_state.audio_inputs:
            st.subheader(f"Total: {len(st.session_state.audio_inputs)}")
            for i, audio_bytes in enumerate(st.session_state.audio_inputs):
                st.write(f"**Message {i + 1}**")
                st.audio(audio_bytes, format="audio/wav")
                
                # Show transcribed text
                if i < len(st.session_state.conversation_history):
                    # Find corresponding user message
                    user_count = 0
                    for msg in st.session_state.conversation_history:
                        if msg["role"] == "user":
                            if user_count == i:
                                st.caption(f"*{msg['content']}*")
                                break
                            user_count += 1
        else:
            st.info("No audio inputs yet. Record a message to start!")
    
    # RIGHT COLUMN - Assistant Audio Outputs
    with col_right:
        st.header("🔊 Responses")
        
        if st.session_state.audio_outputs:
            st.subheader(f"Total: {len(st.session_state.audio_outputs)}")
            for i, audio_bytes in enumerate(st.session_state.audio_outputs):
                st.write(f"**Response {i + 1}**")
                st.audio(audio_bytes, format="audio/wav")
                
                # Show response text
                if i < len(st.session_state.conversation_history):
                    # Find corresponding assistant message
                    assistant_count = 0
                    for msg in st.session_state.conversation_history:
                        if msg["role"] == "assistant":
                            if assistant_count == i:
                                st.caption(f"*{msg['content']}*")
                                break
                            assistant_count += 1
        else:
            st.info("No audio responses yet. Send a message to get started!")

    # Bottom section - Agent log stream
    st.divider()
    st.subheader("🧾 Agent Log Stream")
    logs = st.session_state.agent.get_logs(limit=300)
    if logs:
        st.text_area(
            "Agent logs",
            value="\n".join(logs),
            height=260,
            disabled=True,
            label_visibility="collapsed",
        )
    else:
        st.info("No agent logs yet.")


if __name__ == "__main__":
    main()
