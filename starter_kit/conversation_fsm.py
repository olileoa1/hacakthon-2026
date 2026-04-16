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
from offer_engine import Offer, ProductType, add_tv, build_alternative_offer, build_offer, min_speed_for_users, recommend_product

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
    SCHEDULE_APPOINTMENT = auto()       # FIX: ask customer for time preference
    PROPOSE_APPOINTMENT_SLOT = auto()   # FIX: propose a specific slot, iterate until agreed
    FINAL_APPOINTMENT = auto()          # FIX accepted + appointment set
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

    # Derived fields (set during _do_recommend)
    eligible_for_cube_excite: bool = False
    minimum_speed: int = 0

    # Phase 2
    recommended_product: Optional[ProductType] = None
    current_offer: Optional[Offer] = None
    alternative_offer: Optional[Offer] = None    # cheaper plan shown after main rejection
    alternative_just_accepted: bool = False      # True for one turn after alternative accepted
    main_offer_accepted: Optional[bool] = None   # None = not yet decided
    tv_accepted: Optional[bool] = None
    appointment_set: bool = False
    appointment_time: str = ""              # agreed slot, e.g. "Tuesday 10:00–12:00"
    preferred_appointment_time: str = ""   # customer's stated preference
    appointment_attempts: int = 0          # number of proposed slots rejected
    appointment_specialist_callback: bool = False  # True when fallback to specialist call
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
            "eligible_for_cube_excite": self.eligible_for_cube_excite,
            "minimum_speed": self.minimum_speed,
            "recommended_product": self.recommended_product,
            "offer": self.current_offer.summary() if self.current_offer else None,
            "main_offer_accepted": self.main_offer_accepted,
            "tv_accepted": self.tv_accepted,
            "appointment_set": self.appointment_set,
            "appointment_time": self.appointment_time,
            "appointment_specialist_callback": self.appointment_specialist_callback,
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
        return self.state == State.DONE

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
            # Only advance once the extraction handler has explicitly set the flag.
            # _handle_set_existing marks the slot as resolved via _existing_customer_answered.
            if getattr(d, "_existing_customer_answered", False):
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
                if d.door_number:
                    # Customer gave street + door together — skip ASK_ADDRESS_DOOR
                    self.state = State.ADDRESS_LOOKUP
                    self._do_address_lookup()
                else:
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
                d.alternative_just_accepted = True
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
                if d.recommended_product == "FIX":
                    self.state = State.SCHEDULE_APPOINTMENT
                else:
                    d.call_closed_successfully = True
                    self.state = State.CALL_CLOSED_SUCCESS
            elif d.tv_accepted is False:
                d.tv_accepted = None  # reset so TV_FINAL_OFFER waits for new answer
                self.state = State.TV_FINAL_OFFER
            # else: None — still waiting

        elif s == State.TV_FINAL_OFFER:
            d.voice_discount_offered = True
            if d.tv_accepted is True:
                d.current_offer = add_tv(d.current_offer)
                if d.recommended_product == "FIX":
                    self.state = State.SCHEDULE_APPOINTMENT
                else:
                    d.call_closed_successfully = True
                    self.state = State.CALL_CLOSED_SUCCESS
            elif d.tv_accepted is False:
                d.tv_accepted = None  # reset for FINAL_OFFER
                d.main_offer_accepted = None  # reset so FINAL_OFFER waits for a fresh decision
                self.state = State.FINAL_OFFER
            # else: waiting

        elif s == State.FINAL_OFFER:
            if d.main_offer_accepted is True:
                if d.recommended_product == "FIX":
                    self.state = State.SCHEDULE_APPOINTMENT
                else:
                    d.call_closed_successfully = True
                    self.state = State.CALL_CLOSED_SUCCESS
            elif d.main_offer_accepted is False:
                self.state = State.CALL_CLOSED_AGENT
            # else: waiting

        elif s == State.SCHEDULE_APPOINTMENT:
            # Once we have the customer's time preference, move to slot proposing
            if d.preferred_appointment_time:
                self.state = State.PROPOSE_APPOINTMENT_SLOT

        elif s == State.PROPOSE_APPOINTMENT_SLOT:
            if d.appointment_set or d.appointment_specialist_callback:
                d.call_closed_successfully = True
                self.state = State.CALL_CLOSED_SUCCESS
            # else: stay — keep proposing (handled by appointment_attempts counter)

        elif s == State.FINAL_APPOINTMENT:
            self.state = State.TV_UPSELL

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
        num_users = self.data.num_users or 1
        min_spd = min_speed_for_users(num_users)
        self.data.minimum_speed = min_spd
        self.data.eligible_for_cube_excite = age < 26
        product = recommend_product(age, ai.fix_max_speed, ai.cube_max_speed)
        self.data.recommended_product = product
        self.data.current_offer = build_offer(
            product,
            ai.fix_max_speed,
            ai.cube_max_speed,
            is_existing_customer=self.data.is_existing_customer,
            min_speed=min_spd,
        )


# ---------------------------------------------------------------------------
# Extraction handlers
# ---------------------------------------------------------------------------

def _reconstruct_spelled(text: str) -> str:
    """
    If text looks like spelled-out letters ('Z A K H A R E N K A' or 'Z-A-K-H-A-R-E-N-K-A'),
    join them into a word. Otherwise return text unchanged.
    """
    import re
    # Normalise separators
    cleaned = re.sub(r"[-,\s]+", " ", text.strip().upper())
    tokens = cleaned.split()
    # All single letters → it's spelled out
    if tokens and all(len(t) == 1 and t.isalpha() for t in tokens):
        return "".join(tokens).capitalize()
    return text.strip()


def _fuzzy_match_name_to_db(first_name: str, surname: str):
    """
    Try to find the best matching customer in DB using fuzzy name matching.
    Returns CustomerInfo if similarity >= 0.75, else None.
    """
    import difflib
    from data_service import _ensure_loaded, _customers
    _ensure_loaded()
    query = f"{first_name} {surname}".strip().lower()
    best_sim = 0.0
    best_row = None
    for row in _customers:
        db_name = f"{row.get('first_name', '')} {row.get('surname', '')}".strip().lower()
        sim = difflib.SequenceMatcher(None, query, db_name).ratio()
        if sim > best_sim:
            best_sim = sim
            best_row = row
    if best_sim >= 0.75 and best_row:
        from data_service import CustomerInfo
        return CustomerInfo(
            customer_id=best_row["id"],
            first_name=best_row["first_name"],
            surname=best_row["surname"],
        )
    return None


def _handle_set_name(fsm: ConversationFSM, args: dict) -> None:
    first = _reconstruct_spelled(args.get("first_name", ""))
    surname = _reconstruct_spelled(args.get("surname", ""))
    fsm.data.first_name = first
    fsm.data.surname = surname

    if first and surname:
        # Exact match first
        info = lookup_customer_by_name(first, surname)
        if not info:
            # Fuzzy match — helps when STT slightly mishears the name
            info = _fuzzy_match_name_to_db(first, surname)
        if info:
            # Use canonical DB spelling
            fsm.data.first_name = info.first_name
            fsm.data.surname = info.surname
            fsm.data.customer_info = info
            if fsm.data.is_existing_customer:
                # Pre-populate verified if names match well (still requires ID check)
                pass
            logger.info("Name fuzzy-matched to DB: %r %r → %s %s",
                        first, surname, info.first_name, info.surname)


def _handle_set_existing(fsm: ConversationFSM, args: dict) -> None:
    fsm.data.is_existing_customer = bool(args.get("is_existing_customer", False))
    fsm.data._existing_customer_answered = True  # type: ignore[attr-defined]


def _handle_set_age(fsm: ConversationFSM, args: dict) -> None:
    try:
        age = int(args["age"])
    except (KeyError, ValueError, TypeError):
        return
    if 1 <= age <= 120:
        fsm.data.age = age
    else:
        logger.warning("Rejected out-of-range age: %s", age)


def _handle_set_users(fsm: ConversationFSM, args: dict) -> None:
    try:
        fsm.data.num_users = int(args["num_users"])
    except (KeyError, ValueError):
        pass


def _handle_set_plz(fsm: ConversationFSM, args: dict) -> None:
    fsm.data.plz = str(args.get("plz", "")).strip()


def _handle_set_street(fsm: ConversationFSM, args: dict) -> None:
    import re
    fsm.data.street_name = args.get("street_name", "").strip()
    door = args.get("door_number", "").strip()
    if door:
        # Strip leading noise like "#" or stray letters from STT (e.g. "S6" → "6")
        door = re.sub(r'^[^0-9]*(\d)', r'\1', re.sub(r'^[^0-9A-Za-z]+', '', door))
        fsm.data.door_number = door


def _handle_set_door(fsm: ConversationFSM, args: dict) -> None:
    import re
    raw = str(args.get("door_number", "")).strip()
    # Strip leading non-digit noise STT sometimes adds (e.g. "S6" → "6", "#6" → "6")
    cleaned = re.sub(r'^[^0-9A-Za-z]+', '', raw)  # remove leading symbols like #
    cleaned = re.sub(r'^[^0-9]*(\d)', r'\1', cleaned)  # strip leading letters like "S"
    fsm.data.door_number = cleaned or raw


def _handle_confirm_address(fsm: ConversationFSM, args: dict) -> None:
    if args.get("accepted"):
        # Use the suggested address as confirmed
        fsm.data.address_info = fsm.data.suggested_address
        fsm.data.suggested_address = None
    else:
        # Customer rejected suggestion — clear it to trigger NO_COVERAGE
        fsm.data.suggested_address = None


def _handle_set_subscriber_id(fsm: ConversationFSM, args: dict) -> None:
    if args.get("not_available"):
        # Customer doesn't remember — skip verification, drop existing-customer status.
        fsm.data.is_existing_customer = False
        fsm.data.customer_verified = False
        fsm.data.subscriber_id = ""
        # Sentinel so FSM advances past ASK_SUBSCRIBER_ID to ASK_AGE.
        fsm.data.subscriber_id = "SKIPPED"
        return
    fsm.data.subscriber_id = str(args.get("subscriber_id", "")).strip()


def _handle_main_decision(fsm: ConversationFSM, args: dict) -> None:
    if "accepted" in args:
        fsm.data.main_offer_accepted = bool(args["accepted"])


def _handle_tv_decision(fsm: ConversationFSM, args: dict) -> None:
    if "accepted" in args:
        fsm.data.tv_accepted = bool(args["accepted"])


def _handle_appointment_preference(fsm: ConversationFSM, args: dict) -> None:
    fsm.data.preferred_appointment_time = args.get("preference", "").strip()


def _handle_appointment_slot(fsm: ConversationFSM, args: dict) -> None:
    accepted = args.get("accepted", False)
    if accepted:
        fsm.data.appointment_set = True
        fsm.data.appointment_time = args.get("slot", "")
    else:
        fsm.data.appointment_attempts += 1
        if fsm.data.appointment_attempts >= 5:
            fsm.data.appointment_specialist_callback = True


def _handle_appointment(fsm: ConversationFSM, args: dict) -> None:
    fsm.data.appointment_set = args.get("accepted", False)


def _handle_final_decision(fsm: ConversationFSM, args: dict) -> None:
    fsm.data.main_offer_accepted = args.get("accepted", False)


_EXTRACTION_HANDLERS = {
    "set_appointment_preference": _handle_appointment_preference,
    "set_appointment_slot": _handle_appointment_slot,
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
        "The customer has just called in. Answer the call warmly and professionally. "
        "Introduce yourself: your name is Alex, you work for A1. "
        "Ask how you can help them today. Keep it to 1–2 sentences."
    ),
    State.ASK_EXISTING_CUSTOMER: (
        "The customer just told you what they need. Acknowledge their request briefly in one short phrase "
        "(e.g. 'Great, I can help you with that!' or 'Perfect, let's get you set up!'). "
        "Then ask if they are already an A1 customer (mobile, TV, or internet), and mention that "
        "existing A1 customers receive a special 5 EUR per month discount on their voice contract. "
        "Keep it to 2 sentences max."
    ),
    State.ASK_NAME: (
        "IMPORTANT: Do NOT say 'nice to meet you' at any point in this state — you don't know the customer's name yet. "
        "If this is the first ask: acknowledge the previous answer with one word ('Perfect!' / 'Great!') then ask for first and last name. "
        "If you have the first name but not the surname: ask ONLY for the last name, offer to spell it letter by letter. "
        "If a surname attempt was heard but sounds uncertain: repeat it back and ask to confirm ('Did I get that right?'). "
        "Accept any spelling and move on."
    ),
    State.ASK_SUBSCRIBER_ID: (
        "The customer is an existing A1 customer. "
        "Ask them for their subscriber ID (customer number) so we can verify their account. "
        "It's the number on their A1 bill or in their A1 online account. Keep it brief. "
        "IMPORTANT: accept whatever number the customer gives — even a single digit. Do NOT question its length or format. "
        "If the customer says they don't know or don't remember, reassure them it's not a problem — "
        "we'll simply continue without the loyalty discount — and move on."
    ),
    State.VERIFY_CUSTOMER: (
        "Tell the customer you are verifying their account, ask them to wait a moment."
    ),
    State.ASK_AGE: (
        "Greet the customer by name for the first time (e.g. 'Nice to meet you, [name]!'). "
        "Confirm the full name you heard: say 'I have you down as [first name] [surname] — is that correct?' "
        "Then ask how old they are. Keep it to 2 sentences max."
    ),
    State.ASK_USERS: (
        "Ask how many people in the household will be using the internet."
    ),
    State.ASK_ADDRESS_PLZ: (
        "Ask for the customer's postal code (PLZ) only — a 4–5 digit number. "
        "Do NOT greet again, do NOT re-introduce yourself, do NOT repeat any earlier pleasantry. "
        "One short question, nothing else."
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
        "If the customer is an existing A1 customer (voice_only_discount flag in offer context), "
        "mention that they also keep their 5 EUR/month discount on their existing voice contract — "
        "but do NOT subtract it from the internet plan price. "
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
        "Your response MUST start with the words: 'Before I let you go, I have one more offer for you.' "
        "Do not deviate from this opening. Do not say goodbye first. "
        "Offer A1 Xplore TV M: 65+ channels, 7-day replay, only 4.95 EUR/month first year "
        "(then 9.90 EUR/mo). End with: 'Would you like to add TV to your package?'"
    ),
    State.TV_FINAL_OFFER: (
        "The customer declined the TV offer. Make one final attempt. "
        "Emphasise the 7-day replay feature and that the first year is half price — just 4.95 EUR/month. "
        "That is less than two coffees for 65+ channels and full replay. Ask one last time."
    ),
    State.SCHEDULE_APPOINTMENT: (
        "The customer has finished the TV offer step. Now arrange the technician visit for their Glasfaser connection. "
        "Explain briefly: a technician needs to come to activate the fiber — it takes about 1–2 hours and is free. "
        "IMPORTANT: appointments are only available Monday to Friday. "
        "Ask what time of day works best for them (morning, afternoon, or evening) and which weekdays are convenient."
    ),
    State.PROPOSE_APPOINTMENT_SLOT: (
        "Based on the customer's stated time preference, propose ONE specific appointment slot. "
        "STRICT RULE: only propose weekdays (Monday–Friday). Never propose Saturday or Sunday. "
        "Use 2-hour time windows: morning (9:00–11:00 or 10:00–12:00), "
        "afternoon (13:00–15:00 or 14:00–16:00), evening (16:00–18:00 or 17:00–19:00). "
        "If the customer has already rejected previous slots (visible in conversation history), "
        "propose a different weekday or time window each time. "
        "If appointment_specialist_callback is true in context: do NOT propose another slot — instead "
        "tell the customer warmly that a specialist will contact them during the day to find a suitable time. "
        "Ask the customer if the proposed slot works for them."
    ),
    State.FINAL_APPOINTMENT: (
        "The installation appointment is confirmed. Tell the customer: "
        "a technician will contact them within 2 business days to agree on an exact time — "
        "the installation takes about 1–2 hours and is free of charge. "
        "End with 'While I have you on the line, I'd like to tell you about one more great option.' "
        "Do NOT close the call."
    ),
    State.FINAL_OFFER: (
        "Make one last offer. Remind the customer that the first 6 months are completely free — "
        "they pay nothing until month 7. "
        "If they are an existing A1 customer, also remind them of the 5 EUR/month loyalty discount. "
        "Ask if they would like to go ahead today."
    ),
    State.CALL_CLOSED_SUCCESS: (
        "The call ended successfully. Thank the customer by name and summarize what they've signed up for "
        "(plan name, speed, price from month 7, any TV addon if included). "
        "If appointment_set is true in context, remind them: 'A technician will call within 2 business days to arrange the installation.' "
        "End with a warm goodbye. Keep it to 3–4 sentences max."
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
                "description": (
                    "Store the customer's first and last name. "
                    "If the customer spelled letters one by one (e.g. 'Z A K H A R E N K A'), "
                    "reconstruct the word from those letters. "
                    "Accept any phonetic or letter-by-letter spelling."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "first_name": {
                            "type": "string",
                            "description": "First name. If spelled letter-by-letter, join the letters into a word.",
                        },
                        "surname": {
                            "type": "string",
                            "description": "Surname. If spelled letter-by-letter, join the letters into a word.",
                        },
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
                        "subscriber_id": {"type": "string", "description": "The customer's A1 subscriber ID. Can be any length — even 1 digit. Accept as-is. Omit if the customer says they do not know or do not remember."},
                        "not_available": {"type": "boolean", "description": "Set to true if the customer indicates they don't know, don't remember, or can't find their subscriber ID."},
                    },
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
                "description": (
                    "Store the street name. "
                    "If the customer also mentions a house/door number in the same utterance "
                    "(e.g. 'Abbey Avenue 6'), extract it into door_number as well."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "street_name": {"type": "string", "description": "Street name only, without the house number."},
                        "door_number": {"type": "string", "description": "House/door number if the customer said it together with the street. Omit if not provided."},
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
    State.SCHEDULE_APPOINTMENT: [
        {
            "type": "function",
            "function": {
                "name": "set_appointment_preference",
                "description": "Store the customer's stated time preference for the technician visit.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "preference": {
                            "type": "string",
                            "description": "Free-text description of customer's preference, e.g. 'mornings', 'Monday afternoon', 'evenings on weekdays'. Note: only weekdays are available.",
                        }
                    },
                    "required": ["preference"],
                },
            },
        }
    ],
    State.PROPOSE_APPOINTMENT_SLOT: [
        {
            "type": "function",
            "function": {
                "name": "set_appointment_slot",
                "description": (
                    "Record whether the customer accepted or rejected the proposed time slot. "
                    "If accepted, also store the agreed slot string."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "accepted": {
                            "type": "boolean",
                            "description": "True if customer accepted the proposed slot.",
                        },
                        "slot": {
                            "type": "string",
                            "description": "The agreed slot (e.g. 'Tuesday 10:00–12:00'). Required when accepted=true.",
                        },
                    },
                    "required": ["accepted"],
                },
            },
        }
    ],
    State.FINAL_OFFER: _ACCEPT_REJECT_TOOL("set_final_decision"),
}
