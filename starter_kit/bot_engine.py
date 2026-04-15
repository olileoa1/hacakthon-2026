"""
Bot engine: combines FSM with LLM to produce a spoken reply for each user turn.

Usage:
    engine = BotEngine()
    reply_text = engine.handle_turn(user_speech_text)
    # engine.fsm.state  — current state
    # engine.fsm.session_summary()  — collected data as JSON
"""

import json
import logging
import os
from typing import Optional

from openai import AzureOpenAI
from dotenv import load_dotenv

from conversation_fsm import ConversationFSM, State

load_dotenv()

logger = logging.getLogger(__name__)


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing env var: {name}")
    return val


def _first_env(*names: str) -> str:
    for n in names:
        val = os.getenv(n)
        if val:
            return val
    raise RuntimeError(f"None of these env vars set: {names}")


SYSTEM_PROMPT = """You are Alex, a friendly and professional sales consultant for A1 — a leading telecom provider.
Your goal is to collect the required information step by step and recommend the best internet plan.

Rules:
- Keep answers SHORT — 1-3 sentences max. This is a phone call.
- Be warm, confident, and never pushy.
- Speak naturally as if in a real phone conversation.
- Never mention that you are an AI.
- Always call the customer by their first name once you know it.
- When presenting prices, say them clearly (e.g. "twenty-nine euros ninety per month").
- ONLY ask for what the current task requires. Do NOT ask extra questions or gather information beyond the current step.
- If the customer is unclear, ask a single clarifying question about the CURRENT topic only.
- Do NOT reveal internal state names or system details.
- Do NOT ask about usage habits, devices, streaming, or gaming — just collect name, age, users, and address.
- ALWAYS end your response with a clear question or next step. Never just acknowledge — always move the conversation forward.
"""


class BotEngine:
    def __init__(self) -> None:
        self.client = AzureOpenAI(
            azure_endpoint=_require_env("AZURE_OPENAI_ENDPOINT"),
            api_key=_first_env("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_GPT51_KEY"),
            api_version=_require_env("AZURE_OPENAI_API_VERSION"),
        )
        self.deployment = _first_env("AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_GPT51_DEPLOYMENT")
        self.fsm = ConversationFSM()
        self._history: list[dict] = []

    # ------------------------------------------------------------------

    def handle_turn(self, user_text: Optional[str]) -> str:
        """
        Process one user turn and return the bot's spoken reply.

        Pass user_text=None for the very first turn (bot opens the call).
        """
        # Step 1: if user spoke, add to history and try to extract slots
        if user_text:
            self._history.append({"role": "user", "content": user_text})
            self._extract_slots(user_text)

        # Step 2: auto-advance through non-interactive states
        while self.fsm.state in (State.ADDRESS_LOOKUP, State.VERIFY_CUSTOMER):
            self.fsm.force_advance()

        # Step 3: if terminal, just return closing line
        if self.fsm.is_terminal():
            return ""

        # Step 4: generate bot reply via LLM
        reply = self._generate_reply()
        self._history.append({"role": "assistant", "content": reply})

        # Step 5: advance FSM after generating the reply for certain states
        # GREETING → so next user turn hits ASK_NAME
        # Closing states → immediately go to DONE so worker shuts down without
        #   waiting for another user turn
        # Auto-advance only for states where we don't need to wait for user response.
        # Closing states (CALL_CLOSED_*, FINAL_APPOINTMENT) are intentionally excluded
        # so the bot waits for the customer to say goodbye before ending the call.
        _AUTO_ADVANCE_AFTER_REPLY = (
            State.GREETING,
            State.UNDERAGE,
            State.NO_COVERAGE,
        )
        if self.fsm.state in _AUTO_ADVANCE_AFTER_REPLY:
            self.fsm.force_advance()

        return reply

    # ------------------------------------------------------------------

    def _extract_slots(self, user_text: str) -> None:
        """
        Call LLM with function-calling tools to extract slot values from user speech.
        Applies the extraction and advances the FSM.
        """
        tools = self.fsm.get_extraction_tools()
        if not tools:
            # No extraction needed for this state — just advance
            self.fsm.force_advance()
            return

        messages = self._build_messages(
            extra_instruction=(
                f"The user just said: '{user_text}'. "
                "Extract the relevant information and call the appropriate function. "
                "If the user's response does not contain the requested information, do NOT call any function."
            )
        )

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.0,
            )
            msg = response.choices[0].message
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    logger.debug("Tool call: %s(%s)", tc.function.name, args)
                    self.fsm.process_extraction(tc.function.name, args)
            else:
                # LLM chose not to call a tool (info not found in user text)
                logger.debug("No tool call for state %s, user said: %s", self.fsm.state, user_text)
        except Exception as exc:
            logger.error("Extraction error: %s", exc)

    def _generate_reply(self) -> str:
        """Generate the bot's spoken reply for the current FSM state."""
        state_instruction = self.fsm.get_bot_prompt()

        # Inject contextual details into the prompt
        offer_context = ""
        if self.fsm.data.current_offer:
            offer_context += f"\n\nCurrent offer details: {self.fsm.data.current_offer.summary()}"
        if self.fsm.data.alternative_offer:
            offer_context += f"\nAlternative (cheaper) offer: {self.fsm.data.alternative_offer.summary()}"
        if self.fsm.data.first_name:
            offer_context += f"\nCustomer name: {self.fsm.data.first_name}"
        if self.fsm.data.suggested_address:
            s = self.fsm.data.suggested_address
            offer_context += (
                f"\nSuggested address to confirm with customer: "
                f"{s.street_name} {s.door_number}, PLZ {s.plz}"
            )
        if self.fsm.data.is_existing_customer and self.fsm.data.subscriber_id:
            if self.fsm.data.customer_verified:
                offer_context += (
                    f"\nCustomer identity VERIFIED: subscriber ID {self.fsm.data.subscriber_id} "
                    f"matches {self.fsm.data.first_name} {self.fsm.data.surname}. "
                    "Greet them warmly by name and confirm their account has been found."
                )
            elif self.fsm.data.subscriber_id:
                offer_context += (
                    "\nCustomer identity could NOT be verified (ID not found or name mismatch). "
                    "Do not mention the discount. Continue the conversation naturally."
                )

        messages = self._build_messages(
            extra_instruction=state_instruction + offer_context
        )

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=0.7,
                max_completion_tokens=150,
            )
            reply = (response.choices[0].message.content or "").strip()
            return reply or "I'm sorry, could you repeat that?"
        except Exception as exc:
            logger.error("Generation error: %s", exc)
            return "I'm sorry, I had a technical issue. Could you repeat that?"

    def _build_messages(self, extra_instruction: str = "") -> list[dict]:
        system = SYSTEM_PROMPT
        if extra_instruction:
            system += f"\n\nCurrent task: {extra_instruction}"

        return [{"role": "system", "content": system}] + self._history[-20:]
