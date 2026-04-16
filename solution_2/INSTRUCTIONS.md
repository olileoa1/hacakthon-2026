# 🤖 Telecom Sales Agent: AI Developer Instructions

## Project Overview
You are acting as a Senior AI Python Developer helping to build and maintain a multi-agent Telecom Sales application. 

The application uses an **Azure OpenAI `gpt-4`** Main Agent that interacts with users and executes backend logic using **Function Calling (Tools)**. 

## Architectural Rules
Do not suggest monolithic code. We strictly separate the LLM conversational logic from the backend execution tools to allow multiple colleagues to work simultaneously. 

### Directory Structure
Respect this workspace structure:
* `main.py`: App entry point. (Rarely modify)
* `agent/main_agent.py`: Contains the `TelecomSalesAgent` class and memory loop. (Rarely modify)
* `prompts/sales_prompts.py`: The System Prompt/Process Definition.
* `tools/`: The workspace for individual backend functions. **(Your primary focus)**
* `tools/registry.py`: The central hub where tools are bundled for the Main Agent.

---

## Standard Operating Procedure: Creating a New Tool

When asked to create or modify a tool (e.g., "Create a tool to check TV packages"), you must **always** provide two things in the specific tool file:

1. **The Python Execution Logic:** A strongly typed Python function that returns a JSON string.
2. **The JSON Schema:** The OpenAI-formatted function definition dictionary.

### Tool Template Example
```python
# tools/example_tools.py
import json

# 1. Execution Logic
def check_tv_package(package_name: str) -> str:
    """Check if a TV package is valid."""
    # ... logic ...
    return json.dumps({"status": "success", "price": 29.99})

# 2. LLM Schema
CHECK_TV_SCHEMA = {
    "type": "function",
    "function": {
        "name": "check_tv_package",
        "description": "Checks the price and validity of a TV package.",
        "parameters": {
            "type": "object",
            "properties": {
                "package_name": {
                    "type": "string", 
                    "description": "The name of the TV package."
                }
            },
            "required": ["package_name"]
        }
    }
}