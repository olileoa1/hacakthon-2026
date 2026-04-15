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

from data_service import AddressInfo, CustomerInfo, lookup_address, lookup_customer_by_name
from offer_engine import Offer, ProductType, add_tv, build_offer, recommend_product

logger = logging.getLogger(__name__)


class State(Enum):
    GREETING = auto()
    ASK_NAME = auto()
    ASK_EXISTING_CUSTOMER = auto()
    ASK_AGE = auto()
    ASK_USERS = auto()
    ASK_ADDRESS_PLZ = auto()
    ASK_ADDRESS_STREET = auto()
    ASK_ADDRESS_DOOR = auto()
    ADDRESS_LOOKUP = auto()
    RECOMMEND = auto()
    NEGOTIATE_MAIN = auto()       # present main offer, wait for ACCEPT/REJECT
    TV_UPSELL = auto()            # offer TV addon
    TV_FINAL_OFFER = auto()       # final TV offer (after first reject)
    SCHEDULE_APPOINTMENT = auto() # FIX: propose tech appointment
    FINAL_APPOINTMENT = auto()    # FIX accepted + appointment set
    FINAL_OFFER = auto()          # last chance offer (voice discount)
    CALL_CLOSED_SUCCESS = auto()
    CALL_CLOSED_AGENT = auto()    # hand off to human agent
    DONE = auto()


@dataclass
class SessionData:
    # Phase 1
    first_name: str = ""
    surname: str = ""
    is_existing_customer: bool = False
    customer_info: Optional[CustomerInfo] = None
    age: Optional[int] = None
    num_users: Optional[int] = None
    plz: str = ""
    street_name: str = ""
    door_number: str = ""
    address_info: Optional[AddressInfo] = None

    # Phase 2
    recommended_product: Optional[ProductType] = None
    current_offer: Optional[Offer] = None
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

    def session_summary(self) -> str:
        return json.dumps(self.data.to_dict(), ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Internal transitions
    # ------------------------------------------------------------------

    def _try_advance(self) -> None:  # noqa: C901
        s = self.state
        d = self.data

        if s == State.GREETING:
            self.state = State.ASK_NAME

        elif s == State.ASK_NAME:
            if d.first_name and d.surname:
                self.state = State.ASK_EXISTING_CUSTOMER

        elif s == State.ASK_EXISTING_CUSTOMER:
            self.state = State.ASK_AGE

        elif s == State.ASK_AGE:
            if d.age is not None:
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
            self.state = State.RECOMMEND
            self._do_recommend()

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
                self.state = State.TV_UPSELL
            # else: still waiting for decision — stay in RECOMMEND

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

        elif s in (State.CALL_CLOSED_SUCCESS, State.CALL_CLOSED_AGENT):
            self.state = State.DONE

    def _do_address_lookup(self) -> None:
        try:
            info = lookup_address(self.data.plz, self.data.street_name, self.data.door_number)
        except Exception as exc:
            logger.warning("Address lookup error: %s — using fallback", exc)
            info = None
        if info:
            self.data.address_info = info
        else:
            logger.warning("Address not found in CSV, using fallback speeds")
            from data_service import AddressInfo
            self.data.address_info = AddressInfo(
                plz=self.data.plz,
                door_number=self.data.door_number,
                street_name=self.data.street_name,
                fix_max_speed=250,
                cube_max_speed=130,
            )

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
        "Ask for their first and last name."
    ),
    State.ASK_NAME: (
        "Ask the customer for their first name and last name if not yet provided. "
        "Be friendly and concise."
    ),
    State.ASK_EXISTING_CUSTOMER: (
        "Ask the customer if they are already an A1 customer. "
        "Mention that existing customers get a special voice-only discount."
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
                "description": "Store the customer's first and last name.",
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
    State.RECOMMEND: _ACCEPT_REJECT_TOOL("set_main_decision"),
    State.NEGOTIATE_MAIN: _ACCEPT_REJECT_TOOL("set_main_decision"),
    State.TV_UPSELL: _ACCEPT_REJECT_TOOL("set_tv_decision"),
    State.TV_FINAL_OFFER: _ACCEPT_REJECT_TOOL("set_tv_decision"),
    State.SCHEDULE_APPOINTMENT: _ACCEPT_REJECT_TOOL("set_appointment"),
    State.FINAL_OFFER: _ACCEPT_REJECT_TOOL("set_final_decision"),
}
