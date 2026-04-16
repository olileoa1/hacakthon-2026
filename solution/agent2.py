"""Refactored voice assistant agent (v2).

This module keeps deterministic workflow control but centralizes user-facing
messages and gracefully delegates non-direct answers to the base LLM using
chat history + system prompt.
"""

from __future__ import annotations

import json
from typing import Dict, Optional, Tuple

from agent import VoiceAssistantAgent as LegacyVoiceAssistantAgent


class VoiceAssistantAgent(LegacyVoiceAssistantAgent):
    """Refactored agent with stage contracts and centralized response catalog."""

    RESPONSE_CATALOG: Dict[str, str] = {
        "greeting": "Hello, you are connected to A1. My name is Cathy. How can I help you today—mobile, internet, TV, or something else?",
        "ask_surname_only": "Thanks {first_name}. Could you also share your surname?",
        "ask_name_full": "Could you please tell me your first name and surname?",
        "ask_existing_customer": "Nice to meet you, {first_name}. Are you already a customer with us?",
        "ask_existing_customer_generic": "Are you already a customer with us?",
        "ask_identity": "Great. Please share your subscriber number, or your first name and surname.",
        "ask_age": "Thanks. What is your age?",
        "ask_users": "How many users will share the internet connection?",
        "ask_address": "Where do you live? Just tell me your address naturally.",
        "ask_yes_no_existing": "Please answer yes or no: are you already a customer?",
        "ask_yes_no_customer_confirmation": "Please answer yes or no. Is this the correct customer?",
        "ask_yes_no_address_confirmation": "Please answer yes or no. Is this the correct address?",
        "retry_identity": "I could not find a matching customer yet. Please repeat subscriber number or full name and surname.",
        "retry_address": "Could you please repeat your address naturally, including postal code, street, and door number?",
        "retry_users_range": "That seems high or low for a household. Could you give me a number between 1 and 20?",
        "fallback_age_default": "No problem. Let's move forward. How many people will use the internet?",
        "fallback_users_default": "No problem. I'll continue with 1 user. Where do you live?",
        "handoff_name": "I'm having trouble capturing your name. Let me connect you with an agent who can help. One moment...",
        "handoff_address": "I'm unable to verify your address in our system. Let me connect you with an agent. One moment...",
        "under_18": "Thank you for letting me know. To discuss plans and contracts, we'd need a parent or guardian to be part of this conversation. Would you like to involve them, or shall I connect you to our support team?",
        "existing_yes": "Great. Please share your subscriber number, or your first name and surname.",
        "existing_no": "Thanks. What is your age?",
        "offer_yesno_retry": "Does this plan sound good to you?",
        "offer_alt": "I can offer an alternative: {name} with {speed} Mbps for €{price} from month 7. Would you like this one instead?",
        "offer_decline_done": "I respect your decision. Is there anything else I can help you with today?",
        "tv_ask": "Would you like to add TV to your package?",
        "tv_retry": "Would you be interested in adding TV to your plan?",
        "tv_declined": "No problem, we keep internet only. Thank you, I will now hand you over to an agent for final confirmation.",
        "tv_added": "Excellent, I added {name} for €{price} per month. Thank you, I will now hand you over to an agent for final confirmation.",
        "agent_wait": "An agent will be with you momentarily. Thank you for your patience.",
        "already_complete": "Your request is already prepared for handoff. If you want to start a new request, please clear the chat.",
    }

    def _say(self, key: str, **kwargs) -> str:
        template = self.RESPONSE_CATALOG.get(key, "")
        return template.format(**kwargs)

    def _mark_stage(self, stage: str) -> None:
        self.state.stage = stage

    def _delegate_non_direct(self, stage: str) -> Optional[str]:
        self._log("INFO", f"{stage} received non-direct answer. Delegating to base LLM.")
        return None

    def _extract_partial_name_llm(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract partial name (first and/or surname)."""
        try:
            system_prompt = (
                "Extract first name and surname from user text. "
                "Return strict JSON with keys first_name and surname (string or null). "
                "If only one is present, set the other to null."
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
            parsed = json.loads((response.choices[0].message.content or "").strip())
            first_name = parsed.get("first_name")
            surname = parsed.get("surname")
            first_name = first_name.strip().capitalize() if isinstance(first_name, str) and first_name.strip() else None
            surname = surname.strip().capitalize() if isinstance(surname, str) and surname.strip() else None
            return first_name, surname
        except Exception as e:
            self._log("ERROR", f"Partial name extraction failed: {str(e)}")
            return None, None

    def _answered(self, stage: str, text: str) -> bool:
        return self._is_direct_stage_answer(stage, text)

    def _run_sales_workflow(self, user_text: str) -> Optional[str]:
        text = user_text.strip()

        if self.state.stage == "greeting":
            self._mark_stage("collect_name")
            self._log("INFO", "Stage transition: greeting -> collect_name")
            return self._say("greeting")

        if self.state.stage == "collect_name":
            if not self._answered("collect_name", text):
                return self._delegate_non_direct("collect_name")

            attempts = self._increment_stage_attempt()
            full = self._extract_name_llm(text)
            if full:
                self.store_name_surname(full[0], full[1])
                self._reset_stage_attempt()
                self._mark_stage("existing_customer")
                self._log("INFO", f"Name captured: {full[0]} {full[1]}. Stage transition: collect_name -> existing_customer")
                return self._say("ask_existing_customer", first_name=full[0])

            first_name, surname = self._extract_partial_name_llm(text)
            if first_name and not surname:
                self.state.first_name = first_name
                self._log("INFO", f"Partial name captured (first_name={first_name}).")
                return self._say("ask_surname_only", first_name=first_name)

            if surname and not first_name and self.state.first_name:
                self.store_name_surname(self.state.first_name, surname)
                self._reset_stage_attempt()
                self._mark_stage("existing_customer")
                return self._say("ask_existing_customer", first_name=self.state.first_name)

            if self._should_escalate(max_attempts=3):
                self._reset_stage_attempt()
                self._mark_stage("escalate_to_agent")
                return self._say("handoff_name")

            return self._say("ask_name_full")

        if self.state.stage == "existing_customer":
            if not self._answered("existing_customer", text):
                return self._delegate_non_direct("existing_customer")

            answer = self._extract_yes_no_llm(text)
            if answer is None:
                return self._say("ask_yes_no_existing")

            self.state.is_existing_customer = answer
            if answer:
                self._mark_stage("collect_identity")
                self._log("INFO", "Customer marked as existing. Stage transition: existing_customer -> collect_identity")
                return self._say("existing_yes")

            self._mark_stage("collect_age")
            self._log("INFO", "Customer marked as new. Stage transition: existing_customer -> collect_age")
            return self._say("existing_no")

        if self.state.stage == "collect_identity":
            if not self._answered("collect_identity", text):
                return self._delegate_non_direct("collect_identity")

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
                    self._mark_stage("confirm_customer")
                    return self._build_customer_confirmation_prompt()
                return self._say("retry_identity")

            self.state.customer_candidates = []
            self.state.customer_candidate_index = 0
            self._mark_stage("collect_age")
            display_name = f"{self.state.first_name} {self.state.surname}".strip()
            return f"Thanks {display_name}. {self._say('ask_age')}"

        if self.state.stage == "confirm_customer":
            if not self._answered("confirm_customer", text):
                return self._delegate_non_direct("confirm_customer")

            confirmed = self._extract_yes_no_llm(text)
            if confirmed is None:
                return self._say("ask_yes_no_customer_confirmation")

            if confirmed:
                idx = self.state.customer_candidate_index
                candidates = self.state.customer_candidates
                if 0 <= idx < len(candidates):
                    candidate = candidates[idx]
                    self.state.subscriber_number = str(candidate["id"])
                    self.state.first_name = str(candidate["first_name"])
                    self.state.surname = str(candidate["surname"])
                self.state.customer_candidates = []
                self.state.customer_candidate_index = 0
                self._mark_stage("collect_age")
                return f"Thanks {self.state.first_name} {self.state.surname}. {self._say('ask_age')}"

            self.state.customer_candidate_index += 1
            if self.state.customer_candidate_index < len(self.state.customer_candidates):
                return self._build_customer_confirmation_prompt()

            self.state.customer_candidates = []
            self.state.customer_candidate_index = 0
            self._mark_stage("collect_identity")
            return self._say("retry_identity")

        if self.state.stage == "collect_age":
            if not self._answered("collect_age", text):
                return self._delegate_non_direct("collect_age")

            attempts = self._increment_stage_attempt()
            age = self._extract_age_llm(text)
            if age is not None and 0 < age <= 120:
                self.state.age = age
                self._reset_stage_attempt()
                if age < 18:
                    self._mark_stage("info_only")
                    return self._say("under_18")
                self._mark_stage("collect_users")
                return self._say("ask_users")

            if self._should_escalate(max_attempts=3):
                self._reset_stage_attempt()
                self.state.age = 18
                self._mark_stage("collect_users")
                return self._say("fallback_age_default")

            return self._delegate_non_direct("collect_age")

        if self.state.stage == "collect_users":
            if not self._answered("collect_users", text):
                return self._delegate_non_direct("collect_users")

            attempts = self._increment_stage_attempt()
            users = self._extract_count_int(text)
            if users is not None and 0 < users <= 20:
                self.state.users = users
                self.state.required_speed_mbps = max(50, users * 50)
                self._reset_stage_attempt()
                self._mark_stage("collect_address")
                return self._say("ask_address")

            if users is not None and (users <= 0 or users > 20):
                if self._should_escalate(max_attempts=2):
                    self._reset_stage_attempt()
                    self.state.users = 1
                    self.state.required_speed_mbps = 50
                    self._mark_stage("collect_address")
                    return self._say("fallback_users_default")
                return self._say("retry_users_range")

            if self._should_escalate(max_attempts=3):
                self._reset_stage_attempt()
                self.state.users = 1
                self.state.required_speed_mbps = 50
                self._mark_stage("collect_address")
                return self._say("fallback_users_default")

            return self._delegate_non_direct("collect_users")

        if self.state.stage == "collect_address":
            if not self._answered("collect_address", text):
                return self._delegate_non_direct("collect_address")

            attempts = self._increment_stage_attempt()
            parsed_address = self._extract_address_with_llm(text)
            if not parsed_address:
                if self._should_escalate(max_attempts=4):
                    self._reset_stage_attempt()
                    self._mark_stage("escalate_to_agent")
                    return self._say("handoff_address")
                return self._delegate_non_direct("collect_address")

            missing = self._validate_address_completeness(parsed_address)
            if missing:
                return self._delegate_non_direct("collect_address")

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
                    self._mark_stage("confirm_address")
                    return self._build_address_confirmation_prompt()

                if self._should_escalate(max_attempts=4):
                    self._reset_stage_attempt()
                    self._mark_stage("escalate_to_agent")
                    return self._say("handoff_address")

                return self._delegate_non_direct("collect_address")

            self._reset_stage_attempt()
            self.state.address_candidates = []
            self.state.address_candidate_index = 0
            return self._recommendation_reply_from_current_state()

        if self.state.stage == "confirm_address":
            if not self._answered("confirm_address", text):
                return self._delegate_non_direct("confirm_address")

            confirmed = self._extract_yes_no_llm(text)
            if confirmed is None:
                return self._say("ask_yes_no_address_confirmation")

            if confirmed:
                idx = self.state.address_candidate_index
                candidates = self.state.address_candidates
                if 0 <= idx < len(candidates):
                    candidate = candidates[idx]
                    self.state.plz = str(candidate["plz"])
                    self.state.street_name = str(candidate["street_name"])
                    self.state.door_number = str(candidate["door_number"])
                    self.state.fix_max_speed = int(candidate["fix_max_speed"])
                    self.state.cube_max_speed = int(candidate["cube_max_speed"])
                self.state.address_candidates = []
                self.state.address_candidate_index = 0
                return self._recommendation_reply_from_current_state()

            self.state.address_candidate_index += 1
            if self.state.address_candidate_index < len(self.state.address_candidates):
                return self._build_address_confirmation_prompt()

            self.state.address_candidates = []
            self.state.address_candidate_index = 0
            self._mark_stage("collect_address")
            return self._say("retry_address")

        if self.state.stage == "offer_decision":
            if not self._answered("offer_decision", text):
                return self._delegate_non_direct("offer_decision")

            attempts = self._increment_stage_attempt()
            accepted = self._extract_yes_no_llm(text)
            if accepted is None:
                if self._should_escalate(max_attempts=2):
                    self._reset_stage_attempt()
                    self.state.final_product_id = self.state.recommended_product_id
                    self._mark_stage("tv_upsell")
                    return self._say("tv_ask")
                return self._say("offer_yesno_retry")

            if accepted:
                self._reset_stage_attempt()
                self.state.final_product_id = self.state.recommended_product_id
                self._mark_stage("tv_upsell")
                return self._say("tv_ask")

            self._reset_stage_attempt()
            rounds = int(getattr(self.state, "negotiation_round", 0)) + 1
            setattr(self.state, "negotiation_round", rounds)
            if rounds >= 2:
                self._mark_stage("complete")
                return self._say("offer_decline_done")

            alternative = self._find_alternative_offer()
            if not alternative:
                self._mark_stage("complete")
                return self._say("offer_decline_done")

            alt_id = str(alternative["product_id"])
            self.state.recommended_product_id = alt_id
            self.state.offered_product_ids.append(alt_id)
            return self._say(
                "offer_alt",
                name=alternative.get("name", alt_id),
                speed=self._download_speed(alternative),
                price=self._price_months_7_to_24(alternative),
            )

        if self.state.stage == "tv_upsell":
            if not self._answered("tv_upsell", text):
                return self._delegate_non_direct("tv_upsell")

            attempts = self._increment_stage_attempt()
            tv_accepted = self._extract_yes_no_llm(text)
            if tv_accepted is None:
                if self._should_escalate(max_attempts=2):
                    self._reset_stage_attempt()
                    self.state.tv_accepted = False
                    self._mark_stage("complete")
                    return self._say("tv_declined")
                return self._say("tv_retry")

            self._reset_stage_attempt()
            self.state.tv_accepted = bool(tv_accepted)
            self._mark_stage("complete")

            if tv_accepted:
                if self.tv_products:
                    tv_product = self.tv_products[0]
                    self.state.tv_product_id = str(tv_product.get("product_id", "")) or None
                    return self._say(
                        "tv_added",
                        name=tv_product.get("name", "a TV package"),
                        price=self._price_months_7_to_24(tv_product),
                    )
                return self._say("tv_declined")

            return self._say("tv_declined")

        if self.state.stage == "info_only":
            return self._delegate_non_direct("info_only")

        if self.state.stage == "escalate_to_agent":
            return self._say("agent_wait")

        if self.state.stage == "complete":
            return self._say("already_complete")

        return None
