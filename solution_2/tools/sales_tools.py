import json

# 1. Execution Logic
def check_tv_package(package_name: str) -> str:
    """Check if a TV package is valid and return its price."""
    # Mock logic
    package_name = package_name.lower().strip()
    if package_name in ["premium", "pro", "max"]:
        return json.dumps({"status": "success", "package": package_name, "price": 49.99, "channels": 200, "features": ["4K", "Sports", "Movies"]})
    elif package_name in ["basic", "starter", "standard"]:
        return json.dumps({"status": "success", "package": package_name, "price": 29.99, "channels": 100, "features": ["HD", "News", "Entertainment"]})
    else:
        return json.dumps({"status": "error", "message": f"TV package '{package_name}' not found. Available packages: Premium, Basic."})

def capture_customer_name(first_name: str, last_name: str) -> str:
    """Captures and saves the customer's full name."""
    print(f"\n---> [SYSTEM DATABASE] Saved Customer: {first_name} {last_name} <---\n")
    return json.dumps({"status": "success", "message": f"Successfully saved name: {first_name} {last_name}"})

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
                    "description": "The name of the TV package to check (e.g., Premium, Basic)."
                }
            },
            "required": ["package_name"]
        }
    }
}

CAPTURE_NAME_SCHEMA = {
    "type": "function",
    "function": {
        "name": "capture_customer_name",
        "description": "Saves the customer's full name into the system database. Call this ONLY when you have BOTH the first and last name.",
        "parameters": {
            "type": "object",
            "properties": {
                "first_name": {
                    "type": "string",
                    "description": "The customer's first name."
                },
                "last_name": {
                    "type": "string",
                    "description": "The customer's last name."
                }
            },
            "required": ["first_name", "last_name"]
        }
    }
}
