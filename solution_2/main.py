import streamlit as st
import logging
from dotenv import load_dotenv
from agent.main_agent import TelecomSalesAgent

st.set_page_config(layout="wide")

# Custom Streamlit Logging Handler to push logs into the UI
class StreamlitLogHandler(logging.Handler):
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
        if "log_stream" not in st.session_state:
            st.session_state["log_stream"] = []

    def emit(self, record):
        # Filter out noisy logs and non-agent logs
        if record.name == "httpx" or record.name == "openai" or "Application starting" in record.getMessage():
            return
            
        log_msg = self.format(record)
        st.session_state["log_stream"].append(log_msg)
        self.placeholder.code("\n".join(st.session_state["log_stream"]), language="log")

# Layout: Two columns
st.title("🤖 Telecom Sales Assistant")
col_chat, col_logs = st.columns([1, 1])

with col_logs:
    st.subheader("System Logs")
    # A placeholder container for our logs
    log_placeholder = st.empty()
    
# Configure logging for the application safely (avoids duplicate handlers on reload)
if "logger_setup" not in st.session_state:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Attach our custom handler to the root logger so we get agent logs too
    root_logger = logging.getLogger()
    
    # Silence httpx and openai noisy logs entirely for the environment
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    
    # Remove any existing StreamlitLogHandlers to prevent duplicates during hot reloads
    root_logger.handlers = [h for h in root_logger.handlers if not isinstance(h, StreamlitLogHandler)]
    
    sl_handler = StreamlitLogHandler(log_placeholder)
    sl_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(sl_handler)
    
    st.session_state["logger_setup"] = True

# Ensure logs don't disappear on Streamlit ui re-renders
if "log_stream" in st.session_state:
    log_placeholder.code("\n".join(st.session_state["log_stream"]), language="log")

logger = logging.getLogger(__name__)

# Load environment variables from .env file at the very start
load_dotenv()
logger.info("Application starting up, loaded environment variables...")

with col_chat:
    st.subheader("Chat")

    # Initialize session state for the agent first
    if "agent" not in st.session_state:
        st.session_state["agent"] = TelecomSalesAgent()

    # Initialize session state for messages by asking the agent for the greeting
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": st.session_state["agent"].get_introduction()}]
        
    # We create a specific scrollable container for chat history so the prompt box 
    # Stays docked safely at the bottom instead of floating in the middle!
    chat_container = st.container(height=350, border=False)

    # Display chat history explicitly inside the container
    with chat_container:
        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

    # Accept user input (st.chat_input behaves best when outside the scroll area but inside the column)
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to session state
        st.session_state["messages"].append({"role": "user", "content": prompt})
        
        # Display the prompt right away inside the scrolling container
        with chat_container:
            st.chat_message("user").write(prompt)
            
            with st.spinner("Thinking..."):
                # Get response from the agent
                response = st.session_state["agent"].chat(prompt)
                
                # Add assistant message to session state and display it
                st.session_state["messages"].append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)
