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
- Use the customer's first name sparingly — at most once every 3–4 turns. Do NOT put their name in every sentence.
- When presenting prices, ALWAYS spell them out in words (e.g. "twenty-nine euros ninety cents per month", NOT "29.90 euros"). Never read decimal points or digits.
- ONLY ask for what the current task requires. Do NOT ask extra questions or gather information beyond the current step.
- Never repeat a pleasantry you already used earlier in the call (e.g. say "nice to meet you" only once, "thanks for calling" only once).
- If the customer is unclear, ask a single clarifying question about the CURRENT topic only.
- Do NOT reveal internal state names or system details.
- Do NOT ask about usage habits, devices, streaming, or gaming — just collect name, age, users, and address.
- ALWAYS end your response with a clear question or next step. Never just acknowledge — always move the conversation forward.
"""


def _is_content_filter_error(exc: Exception) -> bool:
    """Return True if the exception is an Azure content filter 400 error."""
    msg = str(exc)
    return "content_filter" in msg or "ResponsibleAIPolicyViolation" in msg


_REPEAT_MARKERS = (
    "what", "pardon", "repeat", "say that again", "didn't catch",
    "didn't hear", "could you say", "once more", "sorry", "come again",
    "i can't hear", "can't understand", "not understand",
)


def _is_repeat_request(text: str) -> bool:
    """Return True if the user is asking the bot to repeat itself."""
    t = text.lower().strip().rstrip("?.!")
    # Very short utterances like "what?" or "sorry?" are almost always repeat requests
    if len(t.split()) <= 3 and any(m in t for m in _REPEAT_MARKERS):
        return True
    # Longer phrases that explicitly ask to repeat
    if any(m in t for m in ("repeat", "say that again", "didn't catch", "didn't hear",
                             "could you say", "once more", "come again")):
        return True
    return False


def _is_transient_error(exc: Exception) -> bool:
    """Return True for transient server/network errors worth a single retry."""
    msg = str(exc).lower()
    for marker in ("timeout", "timed out", "connection", "temporarily",
                   "server_error", "service unavailable",
                   " 500", " 502", " 503", " 504"):
        if marker in msg:
            return True
    return False


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
            # Cap history memory — keep only the last 50 turns
            if len(self._history) > 50:
                self._history = self._history[-50:]

            # If the user is asking to repeat, re-read the last bot reply without advancing FSM
            if _is_repeat_request(user_text):
                last_bot = next(
                    (m["content"] for m in reversed(self._history[:-1]) if m["role"] == "assistant"),
                    None,
                )
                if last_bot:
                    logger.info("Repeat request detected — re-reading last reply")
                    return last_bot

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
            State.CALL_CLOSED_SUCCESS,
            State.CALL_CLOSED_AGENT,
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

        extra_instr = (
            f"The user just said: '{user_text}'. "
            "Extract the relevant information and call the appropriate function. "
            "If the user's response does not contain the requested information, do NOT call any function."
        )

        response = self._call_llm_with_retries(
            extra_instruction=extra_instr,
            tools=tools,
            temperature=0.0,
        )
        if response is None:
            return

        msg = response.choices[0].message
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    logger.warning("Malformed tool args: %s", tc.function.arguments)
                    continue
                logger.debug("Tool call: %s(%s)", tc.function.name, args)
                self.fsm.process_extraction(tc.function.name, args)
        else:
            logger.debug("No tool call for state %s, user said: %s", self.fsm.state, user_text)

    def _generate_reply(self) -> str:
        """Generate the bot's spoken reply for the current FSM state."""
        state_instruction = self.fsm.get_bot_prompt()
        offer_context = self._build_offer_context()

        response = self._call_llm_with_retries(
            extra_instruction=state_instruction + offer_context,
            tools=None,
            temperature=0.7,
            max_completion_tokens=150,
        )
        if response is None:
            return "I'm sorry, I had a technical issue. Could you repeat that?"

        # Consume one-shot flags only after a successful reply is generated.
        if self.fsm.data.alternative_just_accepted:
            self.fsm.data.alternative_just_accepted = False

        reply = (response.choices[0].message.content or "").strip()
        return reply or "I'm sorry, could you repeat that?"

    def _build_offer_context(self) -> str:
        """Assemble the per-turn context block injected into the system prompt."""
        d = self.fsm.data
        parts: list[str] = []
        if d.current_offer:
            parts.append(f"\nCurrent offer details: {d.current_offer.summary()}")
        if d.alternative_offer:
            parts.append(f"Alternative (cheaper) offer: {d.alternative_offer.summary()}")
        if d.appointment_set:
            parts.append(f"appointment_set: true — agreed slot: {d.appointment_time or 'TBD'}")
        if d.appointment_specialist_callback:
            parts.append("appointment_specialist_callback: true — specialist will call customer to arrange time")
        if d.appointment_set and self.fsm.state == State.TV_UPSELL:
            parts.append("appointment_confirmed: true")
        if d.alternative_just_accepted:
            parts.append(
                "IMPORTANT: The customer just accepted the downgraded plan above. "
                "First briefly confirm the new plan (one sentence), then IMMEDIATELY pitch the TV addon as a separate offer. "
                "Do not close the call — the TV offer is the next step."
            )
        if d.suggested_address:
            s = d.suggested_address
            parts.append(f"Suggested address to confirm with customer: {s.street_name} {s.door_number}, PLZ {s.plz}")
        if d.is_existing_customer and d.subscriber_id:
            if d.customer_verified:
                parts.append(
                    f"Customer identity VERIFIED: subscriber ID {d.subscriber_id} "
                    f"matches {d.first_name} {d.surname}. "
                    "Greet them warmly by name and confirm their account has been found."
                )
            else:
                parts.append(
                    "Customer identity could NOT be verified (ID not found or name mismatch). "
                    "Do not mention the discount. Continue the conversation naturally."
                )
        return ("\n" + "\n".join(parts)) if parts else ""

    def _call_llm_with_retries(
        self,
        extra_instruction: str,
        tools: Optional[list[dict]],
        temperature: float,
        max_completion_tokens: Optional[int] = None,
    ):
        """
        Call the LLM with up to 3 attempts, progressively shortening history on content-filter
        errors and retrying once on transient network/server errors. Returns the response or None.
        """
        attempts = [
            self._build_messages(extra_instruction=extra_instruction),
            self._build_messages_short(extra_instruction=extra_instruction),
            self._build_messages_minimal(extra_instruction=extra_instruction),
        ]
        transient_retries_left = 1
        attempt = 0
        while attempt < len(attempts):
            msg_list = attempts[attempt]
            kwargs = {
                "model": self.deployment,
                "messages": msg_list,
                "temperature": temperature,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            if max_completion_tokens:
                kwargs["max_completion_tokens"] = max_completion_tokens

            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                if _is_content_filter_error(exc) and attempt < len(attempts) - 1:
                    logger.warning("Content filter (attempt %d), retrying with shorter history", attempt + 1)
                    attempt += 1
                    continue
                if _is_transient_error(exc) and transient_retries_left > 0:
                    logger.warning("Transient LLM error, retrying once: %s", exc)
                    transient_retries_left -= 1
                    continue  # retry same attempt
                logger.error("LLM call failed: %s", exc)
                return None
        return None

    def _build_messages(self, extra_instruction: str = "") -> list[dict]:
        return self._build_messages_with_history(extra_instruction, self._history[-20:])

    def _build_messages_short(self, extra_instruction: str = "") -> list[dict]:
        """Last 4 history messages — first retry when content filter fires."""
        return self._build_messages_with_history(extra_instruction, self._history[-4:])

    def _build_messages_minimal(self, extra_instruction: str = "") -> list[dict]:
        """No history at all — last resort when content filter fires on short history too."""
        return self._build_messages_with_history(extra_instruction, [])

    def _build_messages_with_history(self, extra_instruction: str, history: list[dict]) -> list[dict]:
        system = SYSTEM_PROMPT
        if extra_instruction:
            system += f"\n\nCurrent task: {extra_instruction}"
        return [{"role": "system", "content": system}] + history
