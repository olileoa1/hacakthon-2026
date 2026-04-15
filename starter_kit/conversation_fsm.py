"""
Conversation FSM for the A1 voice sales bot.

States follow the flow in first_page.png (data collection) and
second_page.png (negotiation).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from data_service import (AddressInfo, CustomerInfo, find_similar_address,
                          lookup_address, lookup_customer_by_id, lookup_customer_by_name)
from offer_engine import Offer, ProductType, add_tv, build_alternative_offer, build_offer, recommend_product

logger = logging.getLogger(__name__)


class State(Enum):
    GREETING = auto()
    ASK_EXISTING_CUSTOMER = auto()  # moved first — determines next path
    ASK_NAME = auto()
    ASK_SUBSCRIBER_ID = auto()      # only for existing customers
    VERIFY_CUSTOMER = auto()        # auto: lookup by ID, fuzzy-match name
    ASK_AGE = auto()
    ASK_USERS = auto()
    ASK_ADDRESS_PLZ = auto()
    ASK_ADDRESS_STREET = auto()
    ASK_ADDRESS_DOOR = auto()
    ADDRESS_LOOKUP = auto()
    RECOMMEND = auto()
    NEGOTIATE_MAIN = auto()       # present main offer, wait for ACCEPT/REJECT
    ALTERNATIVE_OFFER = auto()    # cheaper plan after main offer rejected
    TV_UPSELL = auto()            # offer TV addon
    TV_FINAL_OFFER = auto()       # final TV offer (after first reject)
    SCHEDULE_APPOINTMENT = auto() # FIX: propose tech appointment
    FINAL_APPOINTMENT = auto()    # FIX accepted + appointment set
    FINAL_OFFER = auto()          # last chance offer (voice discount)
    CALL_CLOSED_SUCCESS = auto()
    CALL_CLOSED_AGENT = auto()    # hand off to human agent
    CUSTOMER_UNVERIFIED = auto()  # ID/name mismatch — re-ask if really an A1 customer
    CONFIRM_ADDRESS = auto()      # fuzzy match found — ask customer to confirm
    NO_COVERAGE = auto()          # address not found in service area
    UNDERAGE = auto()             # customer under 18
    DONE = auto()


@dataclass
class SessionData:
    # Phase 1
    first_name: str = ""
    surname: str = ""
    is_existing_customer: bool = False
    subscriber_id: str = ""           # ID given by customer for verification
    customer_verified: bool = False   # True when ID + name matched in DB
    customer_info: Optional[CustomerInfo] = None
    age: Optional[int] = None
    num_users: Optional[int] = None
    plz: str = ""
    street_name: str = ""
    door_number: str = ""
    address_info: Optional[AddressInfo] = None
    suggested_address: Optional[AddressInfo] = None  # fuzzy match candidate

    # Phase 2
    recommended_product: Optional[ProductType] = None
    current_offer: Optional[Offer] = None
    alternative_offer: Optional[Offer] = None    # cheaper plan shown after main rejection
    main_offer_accepted: Optional[bool] = None   # None = not yet decided
    tv_accepted: Optional[bool] = None
    appointment_set: bool = False
    call_closed_successfully: bool = False
    voice_discount_offered: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_name": self.first_name,
            "surname": self.surname,
            "is_existing_customer": self.is_existing_customer,
            "customer_id": self.customer_info.customer_id if self.customer_info else None,
            "age": self.age,
            "num_users": self.num_users,
            "address": {
                "plz": self.plz,
                "street": self.street_name,
                "door": self.door_number,
            },
            "fix_max_speed": self.address_info.fix_max_speed if self.address_info else None,
            "cube_max_speed": self.address_info.cube_max_speed if self.address_info else None,
            "recommended_product": self.recommended_product,
            "offer": self.current_offer.summary() if self.current_offer else None,
            "main_offer_accepted": self.main_offer_accepted,
            "tv_accepted": self.tv_accepted,
            "appointment_set": self.appointment_set,
            "call_closed_successfully": self.call_closed_successfully,
        }


class ConversationFSM:
    def __init__(self) -> None:
        self.state = State.GREETING
        self.data = SessionData()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_bot_prompt(self) -> str:
        """Return the instruction string for the LLM for the current state."""
        return _STATE_PROMPTS[self.state]

    def get_extraction_tools(self) -> list[dict] | None:
        """Return OpenAI function-calling tools for the current state, or None."""
        return _STATE_TOOLS.get(self.state)

    def process_extraction(self, tool_name: str, tool_args: dict) -> None:
        """Apply extracted slot values from LLM function call."""
        handler = _EXTRACTION_HANDLERS.get(tool_name)
        if handler:
            handler(self, tool_args)
        self._try_advance()

    def force_advance(self) -> None:
        """Advance state even when extraction found nothing (fallback)."""
        self._try_advance()

    def is_terminal(self) -> bool:
        return self.state in (State.DONE,)

    def is_underage(self) -> bool:
        return self.data.age is not None and self.data.age < 18

    def session_summary(self) -> str:
        return json.dumps(self.data.to_dict(), ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Internal transitions
    # ------------------------------------------------------------------

    def _try_advance(self) -> None:  # noqa: C901
        s = self.state
        d = self.data

        if s == State.GREETING:
            self.state = State.ASK_EXISTING_CUSTOMER

        elif s == State.ASK_EXISTING_CUSTOMER:
            # is_existing_customer is set — always advance to ASK_NAME
            self.state = State.ASK_NAME

        elif s == State.ASK_NAME:
            if d.first_name and d.surname:
                if d.is_existing_customer:
                    self.state = State.ASK_SUBSCRIBER_ID
                else:
                    self.state = State.ASK_AGE

        elif s == State.ASK_SUBSCRIBER_ID:
            if d.subscriber_id:
                self.state = State.VERIFY_CUSTOMER
                self._do_verify_customer()

        elif s == State.VERIFY_CUSTOMER:
            # Auto state — always move to ASK_AGE after verification attempt
            self.state = State.ASK_AGE

        elif s == State.ASK_AGE:
            if d.age is not None:
                if d.age < 18:
                    self.state = State.UNDERAGE
                else:
                    self.state = State.ASK_USERS

        elif s == State.ASK_USERS:
            if d.num_users is not None:
                self.state = State.ASK_ADDRESS_PLZ

        elif s == State.ASK_ADDRESS_PLZ:
            if d.plz:
                self.state = State.ASK_ADDRESS_STREET

        elif s == State.ASK_ADDRESS_STREET:
            if d.street_name:
                self.state = State.ASK_ADDRESS_DOOR

        elif s == State.ASK_ADDRESS_DOOR:
            if d.door_number:
                self.state = State.ADDRESS_LOOKUP
                self._do_address_lookup()

        elif s == State.ADDRESS_LOOKUP:
            if d.address_info is not None:
                self.state = State.RECOMMEND
                self._do_recommend()
            elif d.suggested_address is not None:
                self.state = State.CONFIRM_ADDRESS
            else:
                self.state = State.NO_COVERAGE

        elif s == State.CONFIRM_ADDRESS:
            if d.address_info is not None:
                # Customer confirmed the suggested address
                self.state = State.RECOMMEND
                self._do_recommend()
            elif d.suggested_address is None:
                # Customer rejected — no coverage
                self.state = State.NO_COVERAGE

        elif s == State.RECOMMEND:
            # RECOMMEND doubles as the accept/reject state for the main offer.
            # The bot presents the offer and asks; the user's yes/no is extracted here.
            # Only advance once we have an explicit decision.
            if d.main_offer_accepted is True:
                if d.recommended_product == "FIX":
                    self.state = State.SCHEDULE_APPOINTMENT
                else:
                    self.state = State.TV_UPSELL
            elif d.main_offer_accepted is False:
                # Try to offer a cheaper alternative before going to TV upsell
                ai = d.address_info
                alt = build_alternative_offer(d.current_offer, ai.fix_max_speed, ai.cube_max_speed)
                if alt:
                    d.alternative_offer = alt
                    d.main_offer_accepted = None  # reset for ALTERNATIVE_OFFER decision
                    self.state = State.ALTERNATIVE_OFFER
                else:
                    self.state = State.TV_UPSELL
            # else: still waiting for decision — stay in RECOMMEND

        elif s == State.ALTERNATIVE_OFFER:
            if d.main_offer_accepted is True:
                # Switch current offer to the cheaper alternative
                d.current_offer = d.alternative_offer
                d.alternative_offer = None
                if d.recommended_product == "FIX":
                    self.state = State.SCHEDULE_APPOINTMENT
                else:
                    self.state = State.TV_UPSELL
            elif d.main_offer_accepted is False:
                d.alternative_offer = None
                self.state = State.TV_UPSELL
            # else: waiting

        elif s == State.NEGOTIATE_MAIN:
            # Legacy state kept for compatibility; skip straight to RECOMMEND logic
            if d.main_offer_accepted:
                if d.recommended_product == "FIX":
                    self.state = State.SCHEDULE_APPOINTMENT
                else:
                    self.state = State.TV_UPSELL
            else:
                self.state = State.TV_UPSELL

        elif s == State.TV_UPSELL:
            if d.tv_accepted is True:
                d.current_offer = add_tv(d.current_offer)
                d.call_closed_successfully = True
                self.state = State.CALL_CLOSED_SUCCESS
            elif d.tv_accepted is False:
                d.tv_accepted = None  # reset so TV_FINAL_OFFER waits for new answer
                if not d.voice_discount_offered:
                    self.state = State.TV_FINAL_OFFER
                else:
                    self.state = State.CALL_CLOSED_AGENT
            # else: None — still waiting

        elif s == State.TV_FINAL_OFFER:
            d.voice_discount_offered = True
            if d.tv_accepted is True:
                d.current_offer = add_tv(d.current_offer)
                d.call_closed_successfully = True
                self.state = State.CALL_CLOSED_SUCCESS
            elif d.tv_accepted is False:
                d.tv_accepted = None  # reset for FINAL_OFFER
                d.main_offer_accepted = None  # reset so FINAL_OFFER waits for a fresh decision
                self.state = State.FINAL_OFFER
            # else: waiting

        elif s == State.FINAL_OFFER:
            if d.main_offer_accepted is True:
                d.call_closed_successfully = True
                self.state = State.CALL_CLOSED_SUCCESS
            elif d.main_offer_accepted is False:
                self.state = State.CALL_CLOSED_AGENT
            # else: waiting

        elif s == State.SCHEDULE_APPOINTMENT:
            if d.appointment_set:
                self.state = State.FINAL_APPOINTMENT
            else:
                self.state = State.TV_UPSELL

        elif s == State.FINAL_APPOINTMENT:
            d.call_closed_successfully = True
            self.state = State.CALL_CLOSED_SUCCESS

        elif s in (State.CALL_CLOSED_SUCCESS, State.CALL_CLOSED_AGENT,
                   State.NO_COVERAGE, State.UNDERAGE):
            self.state = State.DONE

    def _do_address_lookup(self) -> None:
        # Step 1: exact match (100% — no confirmation needed)
        try:
            info = lookup_address(self.data.plz, self.data.street_name, self.data.door_number)
        except Exception as exc:
            logger.warning("Address lookup error: %s", exc)
            info = None

        if info:
            self.data.address_info = info
            logger.info("Exact address match found — proceeding without confirmation")
            return

        # Step 2: fuzzy match
        logger.info("Exact address not found: PLZ=%s street=%s door=%s — trying fuzzy match",
                    self.data.plz, self.data.street_name, self.data.door_number)
        try:
            result = find_similar_address(self.data.plz, self.data.street_name, self.data.door_number)
        except Exception as exc:
            logger.warning("Fuzzy address lookup error: %s", exc)
            result = None

        if result is None:
            # score < 60% — no coverage
            logger.warning("No fuzzy match found (score < 60%%) — going to NO_COVERAGE")
            return

        suggestion, sim_score = result
        logger.info("Fuzzy match: score=%.2f → %s %s %s",
                    sim_score, suggestion.plz, suggestion.street_name, suggestion.door_number)

        if sim_score >= 0.99:
            # Essentially 100% — skip confirmation, proceed directly
            self.data.address_info = suggestion
            logger.info("Near-perfect fuzzy match (score=%.2f) — skipping confirmation", sim_score)
        else:
            # > 60% but not 100% — ask customer to confirm
            self.data.suggested_address = suggestion

    def _do_verify_customer(self) -> None:
        """Look up customer by subscriber ID and fuzzy-match the provided name."""
        import difflib
        customer = lookup_customer_by_id(self.data.subscriber_id)
        if not customer:
            logger.warning("Subscriber ID %s not found in database", self.data.subscriber_id)
            self.data.is_existing_customer = False
            return

        # Fuzzy match: compare provided name against DB name
        provided = f"{self.data.first_name} {self.data.surname}".strip().lower()
        db_name  = f"{customer.first_name} {customer.surname}".strip().lower()
        similarity = difflib.SequenceMatcher(None, provided, db_name).ratio()
        logger.info("Customer verification: provided=%r db=%r similarity=%.2f",
                    provided, db_name, similarity)

        NAME_MATCH_THRESHOLD = 0.6
        if similarity >= NAME_MATCH_THRESHOLD:
            self.data.customer_info    = customer
            self.data.customer_verified = True
            # Use DB name as canonical (fixes STT mis-recognition)
            self.data.first_name = customer.first_name
            self.data.surname    = customer.surname
            logger.info("Customer verified: %s %s (id=%s)",
                        customer.first_name, customer.surname, customer.customer_id)
        else:
            logger.warning("Name mismatch — treating as unverified (similarity=%.2f)", similarity)
            self.data.is_existing_customer = False
            self.data.customer_verified    = False

    def _do_recommend(self) -> None:
        ai = self.data.address_info
        age = self.data.age or 30
        product = recommend_product(age, ai.fix_max_speed, ai.cube_max_speed)
        self.data.recommended_product = product
        self.data.current_offer = build_offer(
            product,
            ai.fix_max_speed,
            ai.cube_max_speed,
            is_existing_customer=self.data.is_existing_customer,
        )


# ---------------------------------------------------------------------------
# Extraction handlers
# ---------------------------------------------------------------------------

def _handle_set_name(fsm: ConversationFSM, args: dict) -> None:
    fsm.data.first_name = args.get("first_name", "").strip()
    fsm.data.surname = args.get("surname", "").strip()
    # also try to match existing customer
    if fsm.data.first_name and fsm.data.surname:
        info = lookup_customer_by_name(fsm.data.first_name, fsm.data.surname)
        if info:
            fsm.data.customer_info = info
            fsm.data.is_existing_customer = True


def _handle_set_existing(fsm: ConversationFSM, args: dict) -> None:
    fsm.data.is_existing_customer = bool(args.get("is_existing_customer", False))


def _handle_set_age(fsm: ConversationFSM, args: dict) -> None:
    try:
        fsm.data.age = int(args["age"])
    except (KeyError, ValueError):
        pass


def _handle_set_users(fsm: ConversationFSM, args: dict) -> None:
    try:
        fsm.data.num_users = int(args["num_users"])
    except (KeyError, ValueError):
        pass


def _handle_set_plz(fsm: ConversationFSM, args: dict) -> None:
    fsm.data.plz = str(args.get("plz", "")).strip()


def _handle_set_street(fsm: ConversationFSM, args: dict) -> None:
    fsm.data.street_name = args.get("street_name", "").strip()


def _handle_set_door(fsm: ConversationFSM, args: dict) -> None:
    fsm.data.door_number = str(args.get("door_number", "")).strip()


def _handle_confirm_address(fsm: ConversationFSM, args: dict) -> None:
    if args.get("accepted"):
        # Use the suggested address as confirmed
        fsm.data.address_info = fsm.data.suggested_address
        fsm.data.suggested_address = None
    else:
        # Customer rejected suggestion — clear it to trigger NO_COVERAGE
        fsm.data.suggested_address = None


def _handle_set_subscriber_id(fsm: ConversationFSM, args: dict) -> None:
    fsm.data.subscriber_id = str(args.get("subscriber_id", "")).strip()


def _handle_main_decision(fsm: ConversationFSM, args: dict) -> None:
    if "accepted" in args:
        fsm.data.main_offer_accepted = bool(args["accepted"])


def _handle_tv_decision(fsm: ConversationFSM, args: dict) -> None:
    if "accepted" in args:
        fsm.data.tv_accepted = bool(args["accepted"])


def _handle_appointment(fsm: ConversationFSM, args: dict) -> None:
    fsm.data.appointment_set = args.get("accepted", False)


def _handle_final_decision(fsm: ConversationFSM, args: dict) -> None:
    fsm.data.main_offer_accepted = args.get("accepted", False)


_EXTRACTION_HANDLERS = {
    "confirm_address": _handle_confirm_address,
    "set_subscriber_id": _handle_set_subscriber_id,
    "set_name": _handle_set_name,
    "set_existing_customer": _handle_set_existing,
    "set_age": _handle_set_age,
    "set_users": _handle_set_users,
    "set_plz": _handle_set_plz,
    "set_street": _handle_set_street,
    "set_door": _handle_set_door,
    "set_main_decision": _handle_main_decision,
    "set_tv_decision": _handle_tv_decision,
    "set_appointment": _handle_appointment,
    "set_final_decision": _handle_final_decision,
}

# ---------------------------------------------------------------------------
# Per-state prompts
# ---------------------------------------------------------------------------

_STATE_PROMPTS: dict[State, str] = {
    State.GREETING: (
        "Greet the customer warmly. Say you are an A1 assistant and you're calling to help find the best internet plan. "
        "Ask if they are already an A1 customer."
    ),
    State.ASK_EXISTING_CUSTOMER: (
        "Ask the customer if they are already an A1 customer (mobile, TV, or internet). "
        "Mention that existing customers get a special voice-only discount of 5 EUR per month."
    ),
    State.ASK_NAME: (
        "Ask the customer for their first name and last name. "
        "Be friendly and concise. "
        "If you already have the first name but not the surname, ask specifically for the last name and offer to spell it out letter by letter if it helps. "
        "Accept any spelling attempt and confirm back what you heard."
    ),
    State.ASK_SUBSCRIBER_ID: (
        "The customer is an existing A1 customer. "
        "Ask them for their subscriber ID (customer number) so we can verify their account. "
        "It's the number on their A1 bill or in their A1 online account. Keep it brief. "
        "IMPORTANT: accept whatever number the customer gives — even a single digit. Do NOT question its length or format."
    ),
    State.VERIFY_CUSTOMER: (
        "Tell the customer you are verifying their account, ask them to wait a moment."
    ),
    State.ASK_AGE: (
        "Ask the customer how old they are. Keep it natural and brief."
    ),
    State.ASK_USERS: (
        "Ask how many people in the household will be using the internet."
    ),
    State.ASK_ADDRESS_PLZ: (
        "Ask for the customer's postal code (PLZ)."
    ),
    State.ASK_ADDRESS_STREET: (
        "Ask for the street name of the customer's address."
    ),
    State.ASK_ADDRESS_DOOR: (
        "Ask for the house or door number."
    ),
    State.ADDRESS_LOOKUP: (
        "Tell the customer you are checking the available speeds at their address. Ask them to wait a moment."
    ),
    State.RECOMMEND: (
        "Present the recommended offer enthusiastically. "
        "Mention the product name, speed, and monthly price clearly. "
        "If the customer is an existing customer, mention the 5 EUR/month voice-only discount. "
        "Ask clearly if they would like to proceed with this plan."
    ),
    State.ALTERNATIVE_OFFER: (
        "The customer declined the main offer. You have a more affordable alternative plan available. "
        "Present it positively — acknowledge their hesitation, then offer the cheaper plan as a great value option. "
        "Mention the plan name, speed, and monthly price clearly. "
        "The alternative offer details are in the context below. Ask if they'd like to go with this plan instead."
    ),
    State.NEGOTIATE_MAIN: (
        "The customer responded to the main offer. "
        "If they accepted, express enthusiasm and confirm the next step. "
        "If they rejected, acknowledge their concern, briefly highlight the value, and ask if they'd reconsider."
    ),
    State.TV_UPSELL: (
        "Offer the TV addon (Aria Box Cube) for an additional 26 EUR per month. "
        "Briefly describe the benefit: hundreds of TV channels, on-demand content. "
        "Ask if they would like to add it."
    ),
    State.TV_FINAL_OFFER: (
        "The customer declined the TV offer. Make one final attempt with extra enthusiasm — "
        "mention that it bundles perfectly with their internet plan and is great value. Ask once more."
    ),
    State.SCHEDULE_APPOINTMENT: (
        "Great news — the customer accepted the Fix internet plan! "
        "Explain that a technician needs to come to install the connection. "
        "Propose scheduling a technical appointment. Ask if they are okay with that."
    ),
    State.FINAL_APPOINTMENT: (
        "The appointment is confirmed. Thank the customer, confirm the details of their plan, "
        "and let them know a technician will be in touch to schedule a specific time. "
        "Close the call warmly."
    ),
    State.FINAL_OFFER: (
        "Make one last offer: remind the customer they can get a 5 EUR/month voice-only discount "
        "if they sign up today. Ask if this makes the offer more attractive."
    ),
    State.CALL_CLOSED_SUCCESS: (
        "The call ended successfully. Thank the customer by name, summarize what they've signed up for, "
        "and wish them a great day."
    ),
    State.CALL_CLOSED_AGENT: (
        "The customer was not ready to proceed. Thank them for their time, let them know an agent "
        "will follow up, and wish them a good day."
    ),
    State.CONFIRM_ADDRESS: (
        "We could not find the exact address the customer gave us, but we found a similar one in our system. "
        "Read the suggested address clearly (street name, house number, postal code) and ask the customer: "
        "'Did you mean [address]?' Wait for a yes or no answer. "
        "The suggested address details are provided in the context below."
    ),
    State.NO_COVERAGE: (
        "Unfortunately, the address the customer provided is not in our service area. "
        "Apologize sincerely and explain that A1 does not yet offer coverage at their address. "
        "Let them know we are continuously expanding and encourage them to check back in the future. "
        "Wish them a pleasant day and close the call warmly. Keep it brief — 2-3 sentences."
    ),
    State.UNDERAGE: (
        "The customer is under 18 years old and cannot sign a contract independently. "
        "Explain this politely — they need to be at least 18 to subscribe. "
        "Suggest they could ask a parent or guardian to call on their behalf. "
        "Thank them for their interest and wish them a great day. "
        "Keep it brief and friendly. Do NOT ask any question at the end — simply close the conversation warmly."
    ),
    State.DONE: (
        "The conversation is complete."
    ),
}

# ---------------------------------------------------------------------------
# Per-state function-calling tool schemas
# ---------------------------------------------------------------------------

_ACCEPT_REJECT_TOOL = lambda name: [  # noqa: E731
    {
        "type": "function",
        "function": {
            "name": name,
            "description": "Record whether the customer accepted or rejected the offer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "accepted": {"type": "boolean", "description": "True if customer accepted, False if rejected."}
                },
                "required": ["accepted"],
            },
        },
    }
]

_STATE_TOOLS: dict[State, list[dict]] = {
    State.ASK_NAME: [
        {
            "type": "function",
            "function": {
                "name": "set_name",
                "description": "Store the customer's first and last name. Accept any phonetic spelling or letter-by-letter spelling of the surname.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "first_name": {"type": "string"},
                        "surname": {"type": "string"},
                    },
                    "required": ["first_name", "surname"],
                },
            },
        }
    ],
    State.ASK_EXISTING_CUSTOMER: [
        {
            "type": "function",
            "function": {
                "name": "set_existing_customer",
                "description": "Record whether the customer is already an A1 customer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "is_existing_customer": {"type": "boolean"},
                    },
                    "required": ["is_existing_customer"],
                },
            },
        }
    ],
    State.ASK_SUBSCRIBER_ID: [
        {
            "type": "function",
            "function": {
                "name": "set_subscriber_id",
                "description": "Store the customer's subscriber/customer ID number. Accept ANY number the customer says, even a single digit like '1'. Do NOT validate length.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subscriber_id": {"type": "string", "description": "The customer's A1 subscriber ID. Can be any length — even 1 digit. Accept as-is."},
                    },
                    "required": ["subscriber_id"],
                },
            },
        }
    ],
    State.ASK_AGE: [
        {
            "type": "function",
            "function": {
                "name": "set_age",
                "description": "Store the customer's age.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "age": {"type": "integer"},
                    },
                    "required": ["age"],
                },
            },
        }
    ],
    State.ASK_USERS: [
        {
            "type": "function",
            "function": {
                "name": "set_users",
                "description": "Store the number of users in the household.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "num_users": {"type": "integer"},
                    },
                    "required": ["num_users"],
                },
            },
        }
    ],
    State.ASK_ADDRESS_PLZ: [
        {
            "type": "function",
            "function": {
                "name": "set_plz",
                "description": "Store the postal code.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "plz": {"type": "string"},
                    },
                    "required": ["plz"],
                },
            },
        }
    ],
    State.ASK_ADDRESS_STREET: [
        {
            "type": "function",
            "function": {
                "name": "set_street",
                "description": "Store the street name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "street_name": {"type": "string"},
                    },
                    "required": ["street_name"],
                },
            },
        }
    ],
    State.ASK_ADDRESS_DOOR: [
        {
            "type": "function",
            "function": {
                "name": "set_door",
                "description": "Store the door/house number.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "door_number": {"type": "string"},
                    },
                    "required": ["door_number"],
                },
            },
        }
    ],
    State.CONFIRM_ADDRESS: [
        {
            "type": "function",
            "function": {
                "name": "confirm_address",
                "description": "Record whether the customer confirmed or rejected the suggested address.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "accepted": {"type": "boolean", "description": "True if customer confirmed the address, False if rejected."}
                    },
                    "required": ["accepted"],
                },
            },
        }
    ],
    State.RECOMMEND: _ACCEPT_REJECT_TOOL("set_main_decision"),
    State.ALTERNATIVE_OFFER: _ACCEPT_REJECT_TOOL("set_main_decision"),
    State.NEGOTIATE_MAIN: _ACCEPT_REJECT_TOOL("set_main_decision"),
    State.TV_UPSELL: _ACCEPT_REJECT_TOOL("set_tv_decision"),
    State.TV_FINAL_OFFER: _ACCEPT_REJECT_TOOL("set_tv_decision"),
    State.SCHEDULE_APPOINTMENT: _ACCEPT_REJECT_TOOL("set_appointment"),
    State.FINAL_OFFER: _ACCEPT_REJECT_TOOL("set_final_decision"),
}
