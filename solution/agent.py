"""Voice assistant with deterministic sales workflow tools.

This module keeps the existing chat-compatible interface while adding
workflow/state handling for the hackathon call process:
- basic customer identification
- customer DB checks
- address speed checks
- product recommendation from JSON catalog
- TV upsell prompt/decision
"""

import csv
from datetime import datetime
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import AzureOpenAI
from rapidfuzz import process as fuzz_process
from rapidfuzz.distance import Levenshtein


def require_env(name: str) -> str:
    """Get required environment variable"""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def first_env(*names: str) -> str:
    """Get first available environment variable from a list"""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    raise RuntimeError(f"Missing required environment variable. Tried: {', '.join(names)}")


class VoiceAssistantAgent:
    """
    Voice Assistant Agent with extensible capabilities.
    
    This agent manages conversations with Azure OpenAI and provides
    a clean interface for adding future capabilities such as function
    calling, memory management, knowledge base integration, etc.
    """
    
    def __init__(self, system_prompt_path: str = None):
        """
        Initialize the Voice Assistant Agent.
        
        Args:
            system_prompt_path: Path to system prompt file. If None, uses default from solution folder.
        """
        # Initialize logs FIRST (before any _log() calls)
        self.logs: List[str] = []
        self.max_log_entries = 500
        
        self.client = self._initialize_client()
        self.deployment = first_env("AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_GPT51_DEPLOYMENT")
        self.temperature = 0.4
        
        # Load system prompt from file or use fallback
        self.system_prompt = self._load_system_prompt(system_prompt_path)
        
        self.messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]

        self.state = ConversationState()
        self._log("INFO", "Agent instance created.")
        self._load_data_sources()
    
    def _load_system_prompt(self, prompt_path: str = None) -> str:
        """
        Load system prompt from file.
        
        Args:
            prompt_path: Optional custom path to system prompt file.
                        If None, uses default: solution/system_prompt.txt
        
        Returns:
            System prompt text
        """
        if prompt_path is None:
            # Default path: look for system_prompt.txt in solution folder
            project_root = Path(__file__).resolve().parent
            prompt_path = project_root / "system_prompt.txt"
        else:
            prompt_path = Path(prompt_path)
        
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()
            self._log("INFO", f"System prompt loaded from: {prompt_path}")
            return prompt_text
        except FileNotFoundError:
            error_msg = f"System prompt file not found: {prompt_path}"
            self._log("ERROR", error_msg)
            # Return a minimal fallback prompt
            fallback = (
                "You are a telecom sales voice assistant. "
                "Follow a structured workflow: greet, capture name, check customer status, capture age, "
                "assess needs, collect address, perform speedcheck, recommend product, negotiate, and upsell TV. "
                "Keep responses short and natural for spoken interaction."
            )
            self._log("INFO", "Using fallback system prompt.")
            return fallback
        except Exception as e:
            error_msg = f"Error reading system prompt file: {e}"
            self._log("ERROR", error_msg)
            fallback = (
                "You are a telecom sales voice assistant. "
                "Follow a structured workflow: greet, capture name, check customer status, capture age, "
                "assess needs, collect address, perform speedcheck, recommend product, negotiate, and upsell TV. "
                "Keep responses short and natural for spoken interaction."
            )
            self._log("INFO", "Using fallback system prompt.")
            return fallback
    
    def _initialize_client(self) -> AzureOpenAI:
        """Initialize and return Azure OpenAI client"""
        return AzureOpenAI(
            azure_endpoint=require_env("AZURE_OPENAI_ENDPOINT"),
            api_key=first_env("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_GPT51_KEY"),
            api_version=require_env("AZURE_OPENAI_API_VERSION"),
        )

    def _load_data_sources(self) -> None:
        """Load customer/address/product data into memory."""
        project_root = Path(__file__).resolve().parent.parent
        customers_path = project_root / "data" / "Customers_hack2026.csv"
        addresses_path = project_root / "data" / "Addresses_hack2026.csv"
        products_path = project_root / "data" / "products_all.json"

        self.customers_by_id: Dict[str, Dict[str, str]] = {}
        self.customers_by_name: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
        self.addresses: Dict[Tuple[str, str, str], Dict[str, int]] = {}
        self.products_all: List[Dict[str, object]] = []
        self.internet_products: List[Dict[str, object]] = []
        self.tv_products: List[Dict[str, object]] = []
        self.products_by_id: Dict[str, Dict[str, object]] = {}

        with customers_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=";"):
                subscriber_id = str(row["id"]).strip()
                first_name = str(row["first_name"]).strip()
                surname = str(row["surname"]).strip()
                customer = {
                    "id": subscriber_id,
                    "first_name": first_name,
                    "surname": surname,
                }
                self.customers_by_id[subscriber_id] = customer

                key = (first_name.lower(), surname.lower())
                self.customers_by_name.setdefault(key, []).append(customer)
            self._log("INFO", f"Loaded {len(self.customers_by_id)} customers.")

        with addresses_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=";"):
                plz = str(row["PLZ"]).strip()
                door_number = str(row["door_number"]).strip()
                street = self._normalize_street(str(row["street_name"]))
                self.addresses[(plz, street, door_number)] = {
                    "fix_max_speed": int(row["Fix_max_speed"]),
                    "cube_max_speed": int(row["Cube_max_speed"]),
                }
        self._log("INFO", f"Loaded {len(self.addresses)} addresses.")

        with products_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
            if isinstance(payload, list):
                self.products_all = payload
            else:
                self.products_all = payload.get("products", [])

        for product in self.products_all:
            product_id = str(product.get("product_id", "")).strip()
            if product_id:
                self.products_by_id[product_id] = product

            technology = str(product.get("technology", "")).strip().lower()
            if technology in {"fiber", "5g"}:
                self.internet_products.append(product)
            elif technology == "iptv":
                self.tv_products.append(product)

        self.internet_products = sorted(
            self.internet_products,
            key=lambda x: (self._price_months_7_to_24(x), self._download_speed(x)),
        )
        self.tv_products = sorted(self.tv_products, key=self._price_months_7_to_24)
        self._log(
            "INFO",
            (
                f"Loaded products: total={len(self.products_all)}, "
                f"internet={len(self.internet_products)}, tv={len(self.tv_products)}."
            ),
        )

    def _is_direct_stage_answer(self, stage: str, text: str) -> bool:
        """Return True when the user text likely answers the current stage question."""
        text = text.strip()
        if not text:
            return False

        stage_instruction = {
            "collect_name": "User should provide first name and surname.",
            "existing_customer": "User should answer yes or no.",
            "collect_identity": "User should provide subscriber number or full name.",
            "confirm_customer": "User should answer yes or no.",
            "collect_age": "User should provide age as number or approximate age.",
            "collect_users": "User should provide number of users.",
            "collect_address": "User should provide address details.",
            "confirm_address": "User should answer yes or no.",
            "offer_decision": "User should answer yes or no for the recommended offer.",
            "tv_upsell": "User should answer yes or no for TV upsell.",
        }.get(stage, "User should answer the requested question.")

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Classify if the user text DIRECTLY answers the current question. "
                            "Return JSON only with key is_answer (true/false). "
                            f"Current question expectation: {stage_instruction}"
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0,
                response_format={"type": "json_object"},
                max_completion_tokens=40,
            )
            parsed = json.loads((response.choices[0].message.content or "").strip())
            return bool(parsed.get("is_answer", False))
        except Exception as e:
            self._log("ERROR", f"Stage answer classification failed ({stage}): {str(e)}")
            return False
    
    def get_response(self, user_text: str) -> str:
        """
        Get a response from the agent for the given user text.
        
        This method maintains conversation history and context across
        multiple interactions.
        
        Args:
            user_text: The user's input message
            
        Returns:
            The agent's response text
        """
        try:
            # Add user message to conversation history
            self.messages.append({"role": "user", "content": user_text})
            self._log("INFO", f"User message received at stage='{self.state.stage}': {self._truncate(user_text)}")

            workflow_reply = self._run_sales_workflow(user_text)
            if workflow_reply:
                self.messages.append({"role": "assistant", "content": workflow_reply})
                self._log("INFO", f"Workflow reply returned at stage='{self.state.stage}'.")
                return workflow_reply
            
            # Get response from Azure OpenAI
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=self.messages,
                temperature=self.temperature,
            )
            
            # Extract and clean response
            reply = (response.choices[0].message.content or "").strip()
            if not reply:
                reply = "I heard you, but I could not generate a reply."
            
            # Add assistant response to conversation history
            self.messages.append({"role": "assistant", "content": reply})
            self._log("INFO", "LLM fallback response returned.")
            
            return reply
            
        except Exception as e:
            error_message = f"Error generating response: {str(e)}"
            self._log("ERROR", error_message)
            return error_message

    def _increment_stage_attempt(self) -> int:
        """Increment and return attempt counter for current stage."""
        stage = self.state.stage
        self.state.stage_attempts[stage] = self.state.stage_attempts.get(stage, 0) + 1
        return self.state.stage_attempts[stage]

    def _reset_stage_attempt(self) -> None:
        """Reset attempt counter for current stage."""
        self.state.stage_attempts[self.state.stage] = 0

    def _should_escalate(self, max_attempts: int = None) -> bool:
        """Check if current stage should escalate to agent."""
        if max_attempts is None:
            max_attempts = self.state.max_attempts_per_stage
        attempts = self.state.stage_attempts.get(self.state.stage, 0)
        return attempts >= max_attempts

    def _run_sales_workflow(self, user_text: str) -> Optional[str]:
        """Run deterministic process flow. Returns reply text if handled."""
        text = user_text.strip()

        if self.state.stage == "greeting":
            self.state.stage = "collect_name"
            self._log("INFO", "Stage transition: greeting -> collect_name")
            return (
                "Hello, you are connected to A1. My name is Cathy. "
                "How can I help you today—mobile, internet, TV, or something else?"
            )

        if self.state.stage == "collect_name":
            attempts = self._increment_stage_attempt()
            parsed_name = self._extract_name_llm(text)
            if parsed_name:
                self.store_name_surname(parsed_name[0], parsed_name[1])
                self._reset_stage_attempt()
                self.state.stage = "existing_customer"
                self._log("INFO", f"Name captured: {parsed_name[0]} {parsed_name[1]}. Stage transition: collect_name -> existing_customer")
                return f"Nice to meet you, {parsed_name[0]}. Are you already a customer with us?"

            if self._should_escalate(max_attempts=3):
                self._log("INFO", f"Name collection max attempts ({attempts}) reached. Escalating to agent.")
                self.state.stage = "escalate_to_agent"
                self._reset_stage_attempt()
                return (
                    "I'm having trouble capturing your name. "
                    "Let me connect you with an agent who can help. One moment..."
                )

            return "Sorry, I didn't catch that. Could you tell me your first name and surname?"

        if self.state.stage == "existing_customer":
            if not self._is_direct_stage_answer("existing_customer", text):
                self._log("INFO", "existing_customer received non-direct answer. Delegating to base LLM.")
                return None

            answer = self._extract_yes_no_llm(text)
            if answer is None:
                return "Please answer with yes or no: are you already a customer?"

            self.state.is_existing_customer = answer
            if answer:
                self.state.stage = "collect_identity"
                self._log("INFO", "Customer marked as existing. Stage transition: existing_customer -> collect_identity")
                return (
                    "Great. Please share your subscriber number, or your first name and surname."
                )

            self.state.stage = "collect_age"
            self._log("INFO", "Customer marked as new. Stage transition: existing_customer -> collect_age")
            return "Thanks. What is your age?"

        if self.state.stage == "collect_identity":
            provided_subscriber = self._extract_subscriber_number_llm(text)
            if provided_subscriber:
                self.provide_subscriber_number(provided_subscriber)

            parsed_name = self._extract_name_llm(text)
            if parsed_name:
                self.store_name_surname(parsed_name[0], parsed_name[1])

            lookup = self.check_customer_in_database()
            if not lookup["found"]:
                suggestions = self._suggest_customer_candidates(limit=3)
                if suggestions:
                    self.state.customer_candidates = suggestions
                    self.state.customer_candidate_index = 0
                    self.state.stage = "confirm_customer"
                    self._log(
                        "INFO",
                        f"Customer lookup not found exactly. Generated {len(suggestions)} fuzzy customer candidate(s).",
                    )
                    return self._build_customer_confirmation_prompt()

                self._log("INFO", "Customer lookup not found during identity collection.")
                return (
                    "I could not find a matching customer yet. "
                    "Please repeat subscriber number or full name and surname."
                )

            self.state.stage = "collect_age"
            self.state.customer_candidates = []
            self.state.customer_candidate_index = 0
            self._log("INFO", "Customer lookup successful. Stage transition: collect_identity -> collect_age")
            display_name = f"{self.state.first_name} {self.state.surname}".strip()
            return f"Thanks {display_name}. What is your age?"

        if self.state.stage == "confirm_customer":
            confirmed = self._extract_yes_no_llm(text)
            if confirmed is None:
                return "Please answer yes or no. Is this the correct customer?"

            if confirmed:
                idx = self.state.customer_candidate_index
                candidates = self.state.customer_candidates
                if idx < 0 or idx >= len(candidates):
                    self.state.stage = "collect_identity"
                    return "Please share your subscriber number, or your first name and surname."

                candidate = candidates[idx]
                self.state.subscriber_number = str(candidate["id"])
                self.state.first_name = str(candidate["first_name"])
                self.state.surname = str(candidate["surname"])
                self.state.customer_candidates = []
                self.state.customer_candidate_index = 0
                self.state.stage = "collect_age"
                self._log(
                    "INFO",
                    (
                        "Customer candidate confirmed: "
                        f"{self.state.first_name} {self.state.surname} (id={self.state.subscriber_number}). "
                        "Stage transition: confirm_customer -> collect_age"
                    ),
                )
                return f"Thanks {self.state.first_name} {self.state.surname}. What is your age?"

            self.state.customer_candidate_index += 1
            if self.state.customer_candidate_index < len(self.state.customer_candidates):
                self._log("INFO", "Customer candidate rejected. Proposing next candidate.")
                return self._build_customer_confirmation_prompt()

            self.state.customer_candidates = []
            self.state.customer_candidate_index = 0
            self.state.stage = "collect_identity"
            self._log("INFO", "All customer candidates rejected. Returning to collect_identity.")
            return "Understood. Please repeat your subscriber number, or your first name and surname."

        if self.state.stage == "collect_age":
            if not self._is_direct_stage_answer("collect_age", text):
                self._log("INFO", "collect_age received non-direct answer. Delegating to base LLM.")
                return None

            attempts = self._increment_stage_attempt()
            
            age = self._extract_age_llm(text)
            if age is not None and age > 0 and age <= 120:
                self.state.age = age
                self._reset_stage_attempt()
                
                if age < 18:
                    self._log("INFO", f"User age {age} is under 18. Switching to info-only mode.")
                    self.state.stage = "info_only"
                    return (
                        "Thank you for letting me know. To discuss plans and contracts, "
                        "we'd need a parent or guardian to be part of this conversation. "
                        "Would you like to involve them, or shall I connect you to our support team?"
                    )
                
                self.state.stage = "collect_users"
                self.state.age_explanation_given = False
                self._log("INFO", f"Age captured: {age}. Stage transition: collect_age -> collect_users")
                if age < 26:
                    return (
                        "Great, thanks. You may qualify for youth offers. "
                        "How many users will share the internet connection?"
                    )
                return "Thanks. How many users will share the internet connection?"

            if self._should_escalate(max_attempts=3):
                self._log("INFO", f"Age collection max attempts ({attempts}) reached. Using default (18+).")
                self._reset_stage_attempt()
                self.state.age = 18
                self.state.stage = "collect_users"
                return "No problem. Let's move forward. How many people will use the internet?"

            # Natural retry without attempt counter
            return "Sorry, I didn't catch that. Could you tell me your age?"

        if self.state.stage == "collect_users":
            if not self._is_direct_stage_answer("collect_users", text):
                self._log("INFO", "collect_users received non-direct answer. Delegating to base LLM.")
                return None

            attempts = self._increment_stage_attempt()
            
            users = self._extract_count_int(text)
            if users is not None and users > 0 and users <= 20:
                self.state.users = users
                self._reset_stage_attempt()
                self.state.required_speed_mbps = max(50, users * 50)
                self.state.stage = "collect_address"
                self._log(
                    "INFO",
                    (
                        f"Users captured: {users}. Required speed: {self.state.required_speed_mbps} Mbps. "
                        f"Transition: collect_users -> collect_address"
                    ),
                )
                return "Thanks. Where do you live? Just tell me your address naturally."

            if users is not None and (users <= 0 or users > 20):
                if self._should_escalate(max_attempts=2):
                    self._log("INFO", "User count out of valid range. Using default.")
                    self._reset_stage_attempt()
                    self.state.users = 1
                    self.state.required_speed_mbps = 50
                    self.state.stage = "collect_address"
                    return "Got it. Let's continue with your address."
                return "That seems high or low for a household. Could you give me a number between 1 and 20?"

            if self._should_escalate(max_attempts=3):
                self._log("INFO", "User count extraction max attempts. Using default.")
                self._reset_stage_attempt()
                self.state.users = 1
                self.state.required_speed_mbps = 50
                self.state.stage = "collect_address"
                return "No problem. Let's continue with your address."

            # Natural retry without attempt counter
            return "Sorry, I didn't quite catch that. How many people will use the internet in your home?"

        if self.state.stage == "collect_address":
            if not self._is_direct_stage_answer("collect_address", text):
                self._log("INFO", "collect_address received non-direct answer. Delegating to base LLM.")
                return None

            attempts = self._increment_stage_attempt()
            
            parsed_address = self._extract_address_with_llm(text)
            if not parsed_address:
                if self._should_escalate(max_attempts=4):
                    self._log("INFO", f"Address parsing max attempts ({attempts}) reached. Escalating to agent.")
                    self.state.stage = "escalate_to_agent"
                    self._reset_stage_attempt()
                    return (
                        "I'm having trouble capturing your address. "
                        "Let me connect you with an agent who can help. One moment..."
                    )
                return "Please share your address naturally, for example: 1020 Abbey Avenue 2."

            # Check which address fields are missing
            missing_fields = self._validate_address_completeness(parsed_address)
            if missing_fields:
                self._log("INFO", f"Address incomplete. Missing fields: {missing_fields}")
                if "plz" in missing_fields:
                    return "What is your postal code (PLZ)?"
                if "street_name" in missing_fields:
                    return "What is your street name?"
                if "door_number" in missing_fields:
                    return "What is your door or house number?"

            speedcheck = self.perform_speedcheck(
                plz=parsed_address["plz"],
                street_name=parsed_address["street_name"],
                door_number=parsed_address["door_number"],
            )
            if not speedcheck["found"]:
                suggestions = self._suggest_address_candidates(
                    plz=parsed_address["plz"],
                    street_name=parsed_address["street_name"],
                    door_number=parsed_address["door_number"],
                    limit=3,
                )
                if suggestions:
                    self.state.address_candidates = suggestions
                    self.state.address_candidate_index = 0
                    self.state.stage = "confirm_address"
                    self._log(
                        "INFO",
                        f"Speedcheck failed exactly. Generated {len(suggestions)} fuzzy address candidate(s).",
                    )
                    return self._build_address_confirmation_prompt()

                if self._should_escalate(max_attempts=4):
                    self._log("INFO", f"Address lookup max attempts ({attempts}) reached. Escalating to agent.")
                    self.state.stage = "escalate_to_agent"
                    self._reset_stage_attempt()
                    return (
                        "I'm unable to verify your address in our system. "
                        "Let me connect you with an agent. One moment..."
                    )

                self._log("INFO", f"Speedcheck failed: address not found (attempt {attempts}/4).")
                return "Could you please repeat your street name and door number?"

            self._reset_stage_attempt()
            self.state.address_candidates = []
            self.state.address_candidate_index = 0
            return self._recommendation_reply_from_current_state()

        if self.state.stage == "confirm_address":
            confirmed = self._extract_yes_no_llm(text)
            if confirmed is None:
                return "Please answer yes or no. Is this the correct address?"

            if confirmed:
                idx = self.state.address_candidate_index
                candidates = self.state.address_candidates
                if idx < 0 or idx >= len(candidates):
                    self.state.stage = "collect_address"
                    return "Please share your address again naturally."

                candidate = candidates[idx]
                self.state.plz = str(candidate["plz"])
                self.state.street_name = str(candidate["street_name"])
                self.state.door_number = str(candidate["door_number"])
                self.state.fix_max_speed = int(candidate["fix_max_speed"])
                self.state.cube_max_speed = int(candidate["cube_max_speed"])
                self.state.address_candidates = []
                self.state.address_candidate_index = 0
                self._log(
                    "INFO",
                    (
                        "Address candidate confirmed: "
                        f"{self.state.plz}, {self.state.street_name}, {self.state.door_number} "
                        f"(FIX={self.state.fix_max_speed}, CUBE={self.state.cube_max_speed})."
                    ),
                )
                return self._recommendation_reply_from_current_state()

            self.state.address_candidate_index += 1
            if self.state.address_candidate_index < len(self.state.address_candidates):
                self._log("INFO", "Address candidate rejected. Proposing next candidate.")
                return self._build_address_confirmation_prompt()

            self.state.address_candidates = []
            self.state.address_candidate_index = 0
            self.state.stage = "collect_address"
            self._log("INFO", "All address candidates rejected. Returning to collect_address.")
            return "Understood. Please repeat your address naturally, including postal code, street, and door number."

        if self.state.stage == "offer_decision":
            if not self._is_direct_stage_answer("offer_decision", text):
                self._log("INFO", "offer_decision received non-direct answer. Delegating to base LLM.")
                return None

            attempts = self._increment_stage_attempt()
            
            accepted = self._extract_yes_no_llm(text)
            if accepted is None:
                if self._should_escalate(max_attempts=2):
                    self._log("INFO", f"Offer decision max attempts ({attempts}) reached. Auto-accepting recommended product.")
                    self._reset_stage_attempt()
                    self.state.final_product_id = self.state.recommended_product_id
                    self.state.stage = "tv_upsell"
                    return (
                        "I'll move forward with the recommended offer. "
                        "I also have a TV subscription add-on for you. Would you like to add TV?"
                    )
                return "Sorry, I didn't catch that. Does this plan sound good to you?"

            if accepted:
                self._reset_stage_attempt()
                self.state.final_product_id = self.state.recommended_product_id
                self.state.stage = "tv_upsell"
                self._log("INFO", "Internet offer accepted. Stage transition: offer_decision -> tv_upsell")
                return (
                    "Perfect. I also have a TV subscription add-on for you. "
                    "Would you like to add TV to your package?"
                )

            self._reset_stage_attempt()
            self.state.negotiation_round += 1
            if self.state.negotiation_round >= 2:
                self._log("INFO", "Max negotiation rounds reached.")
                self.state.stage = "complete"
                return "I respect your decision. Is there anything else I can help you with today?"
            
            alternative = self._find_alternative_offer()
            if not alternative:
                self._log("INFO", "Offer rejected and no alternative eligible offer available.")
                return (
                    "Understood. This is currently the best matching package for your speed needs. "
                    "Would you like to proceed with it?"
                )

            alt_id = str(alternative["product_id"])
            self.state.recommended_product_id = alt_id
            self.state.offered_product_ids.append(alt_id)
            self._log("INFO", f"Alternative offer proposed: {alt_id}.")
            return (
                f"I can offer an alternative: {alternative.get('name', alt_id)} with "
                f"{self._download_speed(alternative)} Mbps for €{self._price_months_7_to_24(alternative)} from month 7. "
                "Would you like this one instead?"
            )

        if self.state.stage == "tv_upsell":
            if not self._is_direct_stage_answer("tv_upsell", text):
                self._log("INFO", "tv_upsell received non-direct answer. Delegating to base LLM.")
                return None

            attempts = self._increment_stage_attempt()
            
            tv_accepted = self._extract_yes_no_llm(text)
            if tv_accepted is None:
                if self._should_escalate(max_attempts=2):
                    self._log("INFO", f"TV upsell max attempts ({attempts}) reached. Skipping TV upsell.")
                    self._reset_stage_attempt()
                    self.state.tv_accepted = False
                    self.state.stage = "complete"
                    self._log("INFO", "Stage transition: tv_upsell -> complete (skipped)")
                    return (
                        "No problem, we keep internet only. "
                        "Thank you, I will now hand you over to an agent for final confirmation."
                    )
                return "Sorry, didn't catch that. Would you be interested in adding TV to your plan?"
            
            self._reset_stage_attempt()
            self.state.tv_accepted = bool(tv_accepted)
            self.state.stage = "complete"
            self._log("INFO", "Stage transition: tv_upsell -> complete")

            if tv_accepted:
                if self.tv_products:
                    tv_product = self.tv_products[0]
                    self.state.tv_product_id = str(tv_product.get("product_id", "")) or None
                    self._log("INFO", f"TV upsell accepted: {self.state.tv_product_id}")
                    return (
                        f"Excellent, I added {tv_product.get('name', 'a TV package')} for "
                        f"€{self._price_months_7_to_24(tv_product)} per month. "
                        "Thank you, I will now hand you over to an agent for final confirmation."
                    )
                return (
                    "Excellent, I added a TV package. "
                    "Thank you, I will now hand you over to an agent for final confirmation."
                )

            self._log("INFO", "TV upsell rejected.")
            return (
                "No problem, we keep internet only. "
                "Thank you, I will now hand you over to an agent for final confirmation."
            )

        if self.state.stage == "escalate_to_agent":
            return (
                "An agent will be with you momentarily. "
                "Thank you for your patience."
            )

        if self.state.stage == "complete":
            return (
                "Your request is already prepared for handoff. "
                "If you want to start a new request, please clear the chat."
            )

        return None

    def _extract_yes_no_llm(self, text: str) -> Optional[bool]:
        """Extract yes/no decision robustly using LLM structured output."""
        try:
            system_prompt = (
                "Determine if the following text means yes or no. "
                "Return a JSON object with a single key: answer (true/false/null). "
                "If the answer is ambiguous, set answer to null. "
                "Examples: 'yes please' -> {\"answer\": true}, 'not sure' -> {\"answer\": null}"
            )
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = (response.choices[0].message.content or "").strip()
            parsed = json.loads(content)
            answer = parsed.get("answer")
            if isinstance(answer, bool):
                self._log("INFO", f"LLM extracted yes/no: {answer}")
                return answer
            self._log("INFO", f"LLM could not extract yes/no from: {self._truncate(text)}")
            return None
        except Exception as e:
            self._log("ERROR", f"Yes/No extraction LLM failed: {str(e)}")
            return None

    def _extract_age_llm(self, text: str) -> Optional[int]:
        """Extract age robustly using LLM structured output."""
        try:
            system_prompt = (
                "Extract the age as an integer from the following text. "
                "Return a JSON object with a single key: age (integer or null). "
                "If no age is mentioned or it's invalid, set age to null. "
                "Accept approximate forms like 'around 30', 'mid twenties', 'early 40s'. "
                "Examples: 'I am 28' -> {\"age\": 28}, 'around thirty' -> {\"age\": 30}, 'I don't know' -> {\"age\": null}"
            )
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = (response.choices[0].message.content or "").strip()
            parsed = json.loads(content)
            age = parsed.get("age")
            if isinstance(age, int):
                self._log("INFO", f"LLM extracted age: {age}")
                return age
            self._log("INFO", f"LLM could not extract age from: {self._truncate(text)}")
            return None
        except Exception as e:
            self._log("ERROR", f"Age extraction LLM failed: {str(e)}")
            return None

    def _extract_first_int(text: str) -> Optional[int]:
        match = re.search(r"\b(\d{1,4})\b", text)
        if not match:
            return None
        return int(match.group(1))

    def _extract_count_int(self, text: str) -> Optional[int]:
        """
        Extract user count robustly using LLM structured output.
        Returns an integer if found, otherwise None.
        """
        try:
            system_prompt = (
                "Extract the number of users from the following text. "
                "Return a JSON object with a single key: users (integer or null). "
                "If no number is mentioned, set users to null. "
                "Examples: 'three users' -> {\"users\": 3}, 'I don't know' -> {\"users\": null}"
            )
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = (response.choices[0].message.content or "").strip()
            parsed = json.loads(content)
            users = parsed.get("users")
            if isinstance(users, int) and 0 < users < 21:
                self._log("INFO", f"LLM extracted user count: {users}")
                return users
            self._log("INFO", f"LLM could not extract user count from: {self._truncate(text)}")
            return None
        except Exception as e:
            self._log("ERROR", f"User count extraction LLM failed: {str(e)}")
            return None

    @staticmethod
    def _normalize_street(street_name: str) -> str:
        return re.sub(r"\s+", " ", street_name.strip().lower())

    def _extract_subscriber_number_llm(self, text: str) -> Optional[str]:
        """Extract subscriber number robustly using LLM structured output."""
        try:
            system_prompt = (
                "Extract the subscriber number or customer number from the following text. "
                "Return a JSON object with a single key: subscriber_number (string or null). "
                "If no number is mentioned, set subscriber_number to null. "
                "Examples: 'my number is 12345' -> {\"subscriber_number\": \"12345\"}, 'I don't have one' -> {\"subscriber_number\": null}"
            )
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = (response.choices[0].message.content or "").strip()
            parsed = json.loads(content)
            subscriber_number = parsed.get("subscriber_number")
            if isinstance(subscriber_number, str) and subscriber_number.strip():
                self._log("INFO", f"LLM extracted subscriber number: {subscriber_number}")
                return subscriber_number.strip()
            self._log("INFO", f"LLM could not extract subscriber number from: {self._truncate(text)}")
            return None
        except Exception as e:
            self._log("ERROR", f"Subscriber number extraction LLM failed: {str(e)}")
            return None

    def _extract_subscriber_number(self, text: str) -> Optional[str]:
        value = self._extract_first_int(text)
        if value is None:
            return None
        return str(value)

    def _extract_name_llm(self, text: str) -> Optional[Tuple[str, str]]:
        """Extract first name and surname robustly using LLM structured output."""
        try:
            system_prompt = (
                "Extract the first name and surname from the following text. "
                "Return a JSON object with keys: first_name (string or null), surname (string or null). "
                "If either name is missing, set it to null. "
                "Examples: 'I am John Smith' -> {\"first_name\": \"John\", \"surname\": \"Smith\"}, 'just call me John' -> {\"first_name\": \"John\", \"surname\": null}"
            )
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = (response.choices[0].message.content or "").strip()
            parsed = json.loads(content)
            first_name = parsed.get("first_name")
            surname = parsed.get("surname")
            if isinstance(first_name, str) and isinstance(surname, str):
                first_name = first_name.strip().capitalize() if first_name else None
                surname = surname.strip().capitalize() if surname else None
                if first_name and surname:
                    self._log("INFO", f"LLM extracted name: {first_name} {surname}")
                    return (first_name, surname)
            self._log("INFO", f"LLM could not extract full name from: {self._truncate(text)}")
            return None
        except Exception as e:
            self._log("ERROR", f"Name extraction LLM failed: {str(e)}")
            return None

    @staticmethod
    def _extract_name(text: str) -> Optional[Tuple[str, str]]:
        words = re.findall(r"[A-Za-zÀ-ÿ'-]+", text)
        if len(words) < 2:
            return None
        first_name = words[0].capitalize()
        surname = words[1].capitalize()
        return first_name, surname

    def _extract_address(self, text: str) -> Optional[Dict[str, str]]:
        # Format variant: "1020, Abbey Avenue, 2"
        pattern_csv = re.compile(
            r"(?P<plz>\d{4,5})\s*,\s*(?P<street>[A-Za-zÀ-ÿ'\- ]+?)\s*,\s*(?P<door>\d{1,6})"
        )
        match_csv = pattern_csv.search(text)
        if match_csv:
            return {
                "plz": match_csv.group("plz").strip(),
                "street_name": match_csv.group("street").strip(),
                "door_number": match_csv.group("door").strip(),
            }

        # Format variant: "Abbey Avenue 2, 1020"
        pattern_free = re.compile(
            r"(?P<street>[A-Za-zÀ-ÿ'\- ]+?)\s+(?P<door>\d{1,6})\s*,\s*(?P<plz>\d{4,5})"
        )
        match_free = pattern_free.search(text)
        if match_free:
            return {
                "plz": match_free.group("plz").strip(),
                "street_name": match_free.group("street").strip(),
                "door_number": match_free.group("door").strip(),
            }

        return None

    def _extract_address_with_llm(self, text: str) -> Optional[Dict[str, str]]:
        """Extract address via LLM structured JSON output: plz, street_name, door_number, city (optional)."""
        try:
            system_prompt = (
                "Extract an address from user text and return strict JSON only with keys: "
                "plz, street_name, door_number, city. "
                "Rules: "
                "- plz must be a postal code string (4-5 digits) or null. "
                "- street_name must be the street text without house number or city name, or null. "
                "- door_number must be the house number as string, or null. "
                "- city is the city/town name if mentioned, or null. "
                "- Do not add extra keys."
            )

            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )

            content = (response.choices[0].message.content or "").strip()
            if not content:
                self._log("INFO", "Address extraction LLM returned empty content.")
                return None

            parsed = json.loads(content)
            raw_plz = parsed.get("plz")
            raw_street = parsed.get("street_name")
            raw_door = parsed.get("door_number")

            plz = str(raw_plz).strip() if raw_plz is not None else ""
            street_name = str(raw_street).strip() if raw_street is not None else ""
            door_number = str(raw_door).strip() if raw_door is not None else ""

            plz_match = re.search(r"\b\d{4,5}\b", plz)
            door_match = re.search(r"\b\d{1,6}\b", door_number)

            if not (plz_match and street_name and door_match):
                self._log(
                    "INFO",
                    f"Address extraction incomplete from LLM output: {self._truncate(content)}",
                )
                return None

            result = {
                "plz": plz_match.group(0),
                "street_name": street_name,
                "door_number": door_match.group(0),
            }
            city = parsed.get("city")
            if city:
                result["city"] = str(city).strip()
            
            city_str = f", city: {result.get('city')}" if result.get('city') else ""
            self._log(
                "INFO",
                (
                    "Address extraction via LLM succeeded: "
                    f"{result['plz']}, {result['street_name']}, {result['door_number']}{city_str}"
                ),
            )
            return result
        except Exception as e:
            self._log("ERROR", f"Address extraction LLM failed: {str(e)}")
            return None

    def _validate_address_completeness(self, parsed_address: Dict[str, str]) -> List[str]:
        """
        Check if all required address fields are present and valid.
        Returns a list of missing field names, or empty list if complete.
        """
        missing = []
        
        plz = parsed_address.get("plz", "").strip() if parsed_address else ""
        street_name = parsed_address.get("street_name", "").strip() if parsed_address else ""
        door_number = parsed_address.get("door_number", "").strip() if parsed_address else ""
        
        if not plz or not re.search(r"\d{4,5}", plz):
            missing.append("plz")
        if not street_name:
            missing.append("street_name")
        if not door_number or not re.search(r"\d{1,6}", door_number):
            missing.append("door_number")
        
        return missing

    def _find_alternative_offer(self) -> Optional[Dict[str, object]]:
        offered_ids = set(self.state.offered_product_ids)
        for product_id in self.state.current_eligible_offer_ids:
            if product_id in offered_ids:
                continue
            product = self.products_by_id.get(product_id)
            if not product:
                continue
            if product_id not in offered_ids:
                return product
        return None

    @staticmethod
    def _download_speed(product: Dict[str, object]) -> int:
        speed = product.get("speed", {})
        if isinstance(speed, dict):
            value = speed.get("download_mbps")
            if value is None:
                return 0
            return int(value)
        return 0

    @staticmethod
    def _price_months_7_to_24(product: Dict[str, object]) -> float:
        pricing = product.get("pricing", {})
        if isinstance(pricing, dict):
            value = pricing.get("months_7_to_24", 0)
            return float(value or 0)
        return 0.0

    # Tools
    def store_name_surname(self, first_name: str, surname: str) -> Dict[str, object]:
        self.state.first_name = first_name.strip().capitalize()
        self.state.surname = surname.strip().capitalize()
        self._log("INFO", f"Stored customer name: {self.state.first_name} {self.state.surname}")
        return {
            "success": True,
            "first_name": self.state.first_name,
            "surname": self.state.surname,
        }

    def provide_subscriber_number(self, subscriber_number: str) -> Dict[str, object]:
        self.state.subscriber_number = str(subscriber_number).strip()
        self._log("INFO", f"Stored subscriber number: {self.state.subscriber_number}")
        return {
            "success": True,
            "subscriber_number": self.state.subscriber_number,
        }

    def check_customer_in_database(self) -> Dict[str, object]:
        if self.state.subscriber_number:
            customer = self.customers_by_id.get(self.state.subscriber_number)
            if customer:
                self.state.first_name = customer["first_name"]
                self.state.surname = customer["surname"]
                self._log("INFO", "Customer lookup succeeded by subscriber number.")
                return {"found": True, "match_type": "id", "customer": customer}

        if self.state.first_name and self.state.surname:
            key = (self.state.first_name.lower(), self.state.surname.lower())
            matches = self.customers_by_name.get(key, [])
            if len(matches) == 1:
                self.state.subscriber_number = matches[0]["id"]
                self._log("INFO", "Customer lookup succeeded by name/surname.")
                return {"found": True, "match_type": "name", "customer": matches[0]}
            if len(matches) > 1:
                self._log("INFO", "Customer lookup ambiguous: multiple name/surname matches.")
                return {"found": False, "reason": "multiple_matches", "matches": matches}

        self._log("INFO", "Customer lookup failed.")
        return {"found": False, "reason": "not_found"}

    def _suggest_customer_candidates(self, limit: int = 3) -> List[Dict[str, object]]:
        """Suggest closest customer records from DB based on current subscriber/name input."""
        scores_by_id: Dict[str, float] = {}
        candidates_by_id: Dict[str, Dict[str, str]] = {}

        try:
            if self.state.subscriber_number:
                subscriber_query = str(self.state.subscriber_number).strip()
                id_keys = list(self.customers_by_id.keys())
                for candidate_id, score, _ in fuzz_process.extract(
                    subscriber_query,
                    id_keys,
                    scorer=Levenshtein.normalized_similarity,
                    score_cutoff=0.70,
                    limit=limit,
                ):
                    customer = self.customers_by_id.get(candidate_id)
                    if not customer:
                        continue
                    scores_by_id[candidate_id] = max(scores_by_id.get(candidate_id, 0.0), float(score))
                    candidates_by_id[candidate_id] = customer

            if self.state.first_name and self.state.surname:
                name_query = f"{self.state.first_name.strip().lower()} {self.state.surname.strip().lower()}"
                name_choices: Dict[str, str] = {}
                for customer in self.customers_by_id.values():
                    candidate_name = f"{customer['first_name'].strip().lower()} {customer['surname'].strip().lower()}"
                    name_choices[str(customer["id"])] = candidate_name

                for _, score, candidate_id in fuzz_process.extract(
                    name_query,
                    name_choices,
                    scorer=Levenshtein.normalized_similarity,
                    score_cutoff=0.70,
                    limit=limit,
                ):
                    candidate_id = str(candidate_id)
                    customer = self.customers_by_id.get(candidate_id)
                    if not customer:
                        continue
                    scores_by_id[candidate_id] = max(scores_by_id.get(candidate_id, 0.0), float(score))
                    candidates_by_id[candidate_id] = customer

            ranked_ids = sorted(scores_by_id.keys(), key=lambda cid: scores_by_id[cid], reverse=True)
            suggestions: List[Dict[str, object]] = []
            for customer_id in ranked_ids[:limit]:
                customer = candidates_by_id[customer_id]
                suggestions.append(
                    {
                        "id": str(customer["id"]),
                        "first_name": str(customer["first_name"]),
                        "surname": str(customer["surname"]),
                        "score": round(scores_by_id[customer_id], 4),
                    }
                )

            return suggestions
        except Exception as e:
            self._log("ERROR", f"Customer fuzzy suggestion failed: {str(e)}")
            return []

    def _build_customer_confirmation_prompt(self) -> str:
        if not self.state.customer_candidates:
            return "Please share your subscriber number, or your first name and surname."

        idx = self.state.customer_candidate_index
        if idx < 0 or idx >= len(self.state.customer_candidates):
            return "Please share your subscriber number, or your first name and surname."

        candidate = self.state.customer_candidates[idx]
        return (
            "I found a close customer match: "
            f"{candidate['first_name']} {candidate['surname']} with subscriber number {candidate['id']}. "
            "Is this correct?"
        )

    def perform_speedcheck(self, plz: str, street_name: str, door_number: str) -> Dict[str, object]:
        normalized_street = self._normalize_street(street_name)
        key = (str(plz).strip(), normalized_street, str(door_number).strip())
        address_row = self.addresses.get(key)
        
        if not address_row:
            self._log("INFO", f"Address lookup failed for {plz}, {street_name}, {door_number}.")
            return {"found": False}

        self.state.plz = key[0]
        self.state.street_name = street_name.strip()
        self.state.door_number = key[2]
        self.state.fix_max_speed = int(address_row["fix_max_speed"])
        self.state.cube_max_speed = int(address_row["cube_max_speed"])
        self._log(
            "INFO",
            (
                f"Speedcheck success for {self.state.plz}, {self.state.street_name}, {self.state.door_number}: "
                f"FIX={self.state.fix_max_speed}, CUBE={self.state.cube_max_speed}"
            ),
        )
        return {
            "found": True,
            "plz": self.state.plz,
            "street_name": self.state.street_name,
            "door_number": self.state.door_number,
            "fix_max_speed": self.state.fix_max_speed,
            "cube_max_speed": self.state.cube_max_speed,
        }

    def _suggest_address_candidates(
        self,
        plz: str,
        street_name: str,
        door_number: str,
        limit: int = 3,
    ) -> List[Dict[str, object]]:
        """Suggest closest address records from DB for confirmation."""
        try:
            plz_key = str(plz).strip()
            door_key = str(door_number).strip()
            normalized_query_street = self._normalize_street(street_name)

            scoped_choices: List[Tuple[str, str, str, Dict[str, int]]] = []
            fallback_choices: List[Tuple[str, str, str, Dict[str, int]]] = []
            for (candidate_plz, candidate_street, candidate_door), row in self.addresses.items():
                entry = (candidate_plz, candidate_street, candidate_door, row)
                if candidate_plz == plz_key and candidate_door == door_key:
                    scoped_choices.append(entry)
                elif candidate_plz == plz_key:
                    fallback_choices.append(entry)

            search_space = scoped_choices if scoped_choices else fallback_choices
            if not search_space:
                return []

            search_strings: List[str] = []
            for candidate_plz, candidate_street, candidate_door, _ in search_space:
                if scoped_choices:
                    search_strings.append(candidate_street)
                else:
                    search_strings.append(f"{candidate_street} {candidate_door}")

            query = normalized_query_street if scoped_choices else f"{normalized_query_street} {door_key}"
            matches = fuzz_process.extract(
                query,
                search_strings,
                scorer=Levenshtein.normalized_similarity,
                score_cutoff=0.70,
                limit=limit,
            )

            suggestions: List[Dict[str, object]] = []
            for _, score, index in matches:
                candidate_plz, candidate_street, candidate_door, row = search_space[index]
                suggestions.append(
                    {
                        "plz": candidate_plz,
                        "street_name": candidate_street,
                        "door_number": candidate_door,
                        "fix_max_speed": int(row["fix_max_speed"]),
                        "cube_max_speed": int(row["cube_max_speed"]),
                        "score": round(float(score), 4),
                    }
                )

            return suggestions
        except Exception as e:
            self._log("ERROR", f"Address fuzzy suggestion failed: {str(e)}")
            return []

    def _build_address_confirmation_prompt(self) -> str:
        if not self.state.address_candidates:
            return "Please share your address again naturally."

        idx = self.state.address_candidate_index
        if idx < 0 or idx >= len(self.state.address_candidates):
            return "Please share your address again naturally."

        candidate = self.state.address_candidates[idx]
        return (
            "I found a close address match: "
            f"{candidate['plz']} {candidate['street_name']} {candidate['door_number']}. "
            "Is this correct?"
        )

    def _recommendation_reply_from_current_state(self) -> str:
        recommendation = self.recommend_product(self.state.required_speed_mbps or 0)
        self.state.stage = "offer_decision"
        self._log("INFO", "Stage transition: collect_address/confirm_address -> offer_decision")
        if not recommendation["found"]:
            self._log("INFO", "No eligible internet product found for required speed and coverage.")
            return (
                f"Your required speed is {self.state.required_speed_mbps} Mbps, "
                "which is above our current catalog. "
                "I can connect you to an agent for a custom offer."
            )

        product = recommendation["product"]
        product_id = str(product["product_id"])
        product_name = str(product.get("name", product_id))
        technology = str(product.get("technology", ""))
        speed_mbps = self._download_speed(product)
        price_eur = self._price_months_7_to_24(product)
        self.state.recommended_product_id = str(product_id)
        self.state.current_eligible_offer_ids = recommendation["eligible_product_ids"]
        self.state.offered_product_ids.append(str(product_id))
        self._log(
            "INFO",
            (
                f"Recommended product: {product_id} ({technology}), "
                f"{speed_mbps} Mbps, €{price_eur}."
            ),
        )

        return (
            f"Based on {self.state.users} users and your address coverage, "
            f"I recommend {product_name} ({technology}) with {speed_mbps} Mbps for €{price_eur} from month 7. "
            "Would you like to take this offer?"
        )

    def _fuzzy_match_address(self, plz: str, normalized_street: str, door_number: str) -> Optional[Dict[str, int]]:
        """
        Fuzzy match street name against addresses in the same PLZ/door_number.
        Returns address_row if high-confidence match found (>80%), otherwise None.
        """
        try:
            plz_key = str(plz).strip()
            door_key = str(door_number).strip()
            
            # Find all addresses with matching PLZ and door_number
            candidates = []
            candidate_streets = []
            for (p, street, d), row in self.addresses.items():
                if p == plz_key and d == door_key:
                    candidates.append(row)
                    candidate_streets.append(street)
            
            if not candidates:
                self._log("INFO", f"No address candidates found for PLZ={plz_key}, door={door_key}")
                return None
            
            # Fuzzy match the street name
            best_match = fuzz_process.extractOne(
                normalized_street,
                candidate_streets,
                scorer=Levenshtein.normalized_similarity,
                score_cutoff=0.80
            )
            
            if best_match:
                matched_street, score = best_match
                matched_idx = candidate_streets.index(matched_street)
                result = candidates[matched_idx]
                self._log("INFO", f"Fuzzy match found: '{normalized_street}' -> '{matched_street}' (score: {score:.2f})")
                return result
            
            self._log("INFO", f"No fuzzy match found for street '{normalized_street}' (threshold: 0.80)")
            return None
        except Exception as e:
            self._log("ERROR", f"Fuzzy address matching failed: {str(e)}")
            return None

    def recommend_product(self, required_speed_mbps: int) -> Dict[str, object]:
        # Rule: choose the cheapest product meeting required speed and available coverage,
        # respecting technology-specific coverage (Fiber->FIX, 5G->Cube).
        eligible = []
        for product in self.internet_products:
            technology = str(product.get("technology", "")).strip().lower()
            speed_mbps = self._download_speed(product)

            if technology == "fiber":
                coverage_limit = int(self.state.fix_max_speed or 0)
            elif technology == "5g":
                coverage_limit = int(self.state.cube_max_speed or 0)
            else:
                continue

            if speed_mbps >= required_speed_mbps and speed_mbps <= coverage_limit:
                eligible.append(product)

        if not eligible:
            self._log(
                "INFO",
                (
                    f"No eligible product for required={required_speed_mbps}, "
                    f"FIX={int(self.state.fix_max_speed or 0)}, CUBE={int(self.state.cube_max_speed or 0)}"
                ),
            )
            return {
                "found": False,
                "required_speed_mbps": required_speed_mbps,
                "coverage_limit_fix_mbps": int(self.state.fix_max_speed or 0),
                "coverage_limit_cube_mbps": int(self.state.cube_max_speed or 0),
            }

        sorted_eligible = sorted(
            eligible,
            key=lambda p: (self._price_months_7_to_24(p), self._download_speed(p)),
        )
        product = sorted_eligible[0]
        return {
            "found": True,
            "product": product,
            "eligible_product_ids": [str(p.get("product_id", "")) for p in sorted_eligible if p.get("product_id")],
        }

    def try_upsell_tv(self, user_text: str) -> Dict[str, object]:
        accepted = self._yes_no(user_text)
        self.state.tv_accepted = bool(accepted)
        if not accepted:
            self.state.tv_product_id = None
            self._log("INFO", "TV upsell not accepted by customer.")
            return {"accepted": False, "tv_product": None}

        if not self.tv_products:
            self.state.tv_product_id = None
            self._log("INFO", "TV upsell accepted, but no TV products loaded.")
            return {"accepted": False, "tv_product": None}

        tv_product = self.tv_products[0]
        self.state.tv_product_id = str(tv_product.get("product_id", "")) or None
        self._log("INFO", f"TV upsell accepted. Selected TV product: {self.state.tv_product_id}")
        return {"accepted": True, "tv_product": tv_product}

    @staticmethod
    def _truncate(text: str, max_length: int = 180) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        if len(normalized) <= max_length:
            return normalized
        return normalized[: max_length - 3] + "..."

    def _log(self, level: str, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp} [{level}] {message}"
        self.logs.append(line)
        if len(self.logs) > self.max_log_entries:
            self.logs = self.logs[-self.max_log_entries :]

    def get_logs(self, limit: int = 200) -> List[str]:
        if limit <= 0:
            return []
        return self.logs[-limit:]

    def clear_logs(self) -> None:
        self.logs = []
    
    def clear_conversation(self) -> None:
        """Clear conversation history while keeping the system prompt"""
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]
        self.state = ConversationState()
        self.clear_logs()
        self._log("INFO", "Conversation and logs cleared.")
    
    def load_system_prompt_from_file(self, prompt_path: str) -> None:
        """
        Load and set system prompt from file.
        
        Args:
            prompt_path: Path to system prompt file
        """
        prompt_text = self._load_system_prompt(prompt_path)
        self.set_system_prompt(prompt_text)
    
    def set_system_prompt(self, prompt: str) -> None:
        """
        Update the system prompt for the agent.
        
        Args:
            prompt: New system prompt text
        """
        self.system_prompt = prompt
        # Update system message in conversation history
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = prompt
        self._log("INFO", "System prompt updated.")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the current conversation history (excluding system prompt).
        
        Returns:
            List of user and assistant messages
        """
        # Return only user and assistant messages, not system prompt
        return [msg for msg in self.messages if msg["role"] != "system"]
    
    def set_temperature(self, temperature: float) -> None:
        """
        Set the temperature for LLM responses.
        
        Args:
            temperature: Temperature value between 0 and 2
        """
        if 0 <= temperature <= 2:
            self.temperature = temperature
        else:
            raise ValueError("Temperature must be between 0 and 2")
    
    # Future capability placeholders
    def add_capability(self, capability_name: str, capability_func) -> None:
        """
        Placeholder for adding custom capabilities to the agent.
        
        This will be used to extend the agent with function calling,
        tool integration, knowledge base access, etc.
        
        Args:
            capability_name: Name of the capability
            capability_func: Function implementing the capability
        """
        # TODO: Implement capability management
        pass
    
    def process_with_tools(self, user_text: str, tools: List = None) -> str:
        """
        Placeholder for processing requests with tool integration.
        
        This method will enable the agent to use external tools and
        function calling in the future.
        
        Args:
            user_text: The user's input message
            tools: List of available tools
            
        Returns:
            The agent's response, potentially using tools
        """
        # TODO: Implement tool/function calling
        # For now, just use standard get_response
        return self.get_response(user_text)


@dataclass
class ConversationState:
    """Simple state object for deterministic workflow progression."""

    stage: str = "greeting"
    is_existing_customer: Optional[bool] = None
    
    stage_attempts: Dict[str, int] = field(default_factory=dict)
    max_attempts_per_stage: int = 3

    first_name: Optional[str] = None
    surname: Optional[str] = None
    subscriber_number: Optional[str] = None
    customer_candidates: List[Dict[str, object]] = field(default_factory=list)
    customer_candidate_index: int = 0

    age: Optional[int] = None
    age_explanation_given: bool = False
    users: Optional[int] = None
    required_speed_mbps: Optional[int] = None

    plz: Optional[str] = None
    street_name: Optional[str] = None
    door_number: Optional[str] = None
    address_candidates: List[Dict[str, object]] = field(default_factory=list)
    address_candidate_index: int = 0

    fix_max_speed: Optional[int] = None
    cube_max_speed: Optional[int] = None

    recommended_product_id: Optional[str] = None
    final_product_id: Optional[str] = None
    offered_product_ids: List[str] = field(default_factory=list)
    current_eligible_offer_ids: List[str] = field(default_factory=list)

    tv_accepted: bool = False
    tv_product_id: Optional[str] = None
