import os
import json
import logging
from openai import AzureOpenAI
from prompts.sales_prompts import SYSTEM_PROMPT
from tools.registry import TOOLS_REGISTRY, TOOLS_SCHEMA

logger = logging.getLogger(__name__)

class TelecomSalesAgent:
    def __init__(self):
        # We try to use GPT-4.1 Nano first, and fall back to GPT-5.1 if it's set
        api_key = os.getenv("AZURE_OPENAI_GPT41_NANO_KEY") or os.getenv("AZURE_OPENAI_GPT51_KEY", "dummy_key")
        self.deployment_name = os.getenv("AZURE_OPENAI_GPT41_NANO_DEPLOYMENT") or os.getenv("AZURE_OPENAI_GPT51_DEPLOYMENT", "gpt-4.1-nano")
        
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://dummy.openai.azure.com/")
        )
        
        self.intro_message = "Hello! I'm Cathy, your sales assistant at A1 Austria—the national telecom provider with the best and fastest network! How can I help you today? To get started, could you please tell me your full name?"
        
        # Inject Cathy's persona into the overarching prompt
        cathy_persona = "\n\nCRITICAL PERSONA: Your name is Cathy. You work for A1 Austria, the national telecom provider with the best and fastest network. Be warm, welcoming, and proud of A1's network quality."
        
        # Initialize conversation memory
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT + cathy_persona},
            {"role": "assistant", "content": self.intro_message}
        ]
        logger.debug(f"Initialized TelecomSalesAgent with deployment: {self.deployment_name}")

    def get_introduction(self) -> str:
        """Returns the initial greeting for the UI."""
        return self.intro_message

    def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        logger.debug(f"Received user input: {user_input[:50]}...")
        logger.info("Sending message to LLM...")
        
        try:
            # We call the LLM and provide it with the tools
            logger.debug("Calling LLM with tools...")
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=self.messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto"
            )
            
            response_message = response.choices[0].message
            self.messages.append(response_message)
            logger.debug(f"Received LLM response message. Has tool calls: {bool(response_message.tool_calls)}")
            
            # Use tools if the LLM decides to call a function
            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    logger.info(f"LLM requested tool call: {function_name}")
                    function_args = json.loads(tool_call.function.arguments)
                    
                    if function_name in TOOLS_REGISTRY:
                        # 1. Execute the function locally
                        logger.debug(f"Executing tool {function_name} with args {function_args}")
                        function_to_call = TOOLS_REGISTRY[function_name]
                        function_response = function_to_call(**function_args)
                        
                        # 2. Append the function response
                        logger.debug(f"Received tool response for {function_name}")
                        self.messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                        })
                
                    else:
                        logger.warning(f"LLM requested unknown tool: {function_name}")
                
                # 3. Call LLM again to get the final response containing function outputs
                logger.debug("Calling LLM again with tool outputs...")
                second_response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=self.messages
                )
                
                final_reply = second_response.choices[0].message
                self.messages.append(final_reply)
                logger.info("Successfully returned final response after tool execution.")
                return final_reply.content
            
            # Simple text response
            logger.info("Successfully returned simple text response.")
            return response_message.content

        except Exception as e:
            logger.error(f"Error during chat interaction: {str(e)}", exc_info=True)
            return f"❌ Connection Error: {str(e)}\n\n(Please check your environment variables: `AZURE_OPENAI_GPT41_NANO_KEY`, `AZURE_OPENAI_ENDPOINT`, and `AZURE_OPENAI_API_VERSION`)"