"""
CSV lookup service for customers and addresses.
"""

import csv
import difflib
import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@dataclass
class AddressInfo:
    plz: str
    door_number: str
    street_name: str
    fix_max_speed: int
    cube_max_speed: int


@dataclass
class CustomerInfo:
    customer_id: str
    first_name: str
    surname: str


def _load_addresses() -> list[dict]:
    path = os.path.join(DATA_DIR, "Addresses_hack2026.csv")
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter=";", skipinitialspace=True))


def _load_customers() -> list[dict]:
    path = os.path.join(DATA_DIR, "Customers_hack2026.csv")
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter=";", skipinitialspace=True))


_addresses: list[dict] = []
_customers: list[dict] = []


def _ensure_loaded() -> None:
    global _addresses, _customers
    if not _addresses:
        _addresses = _load_addresses()
    if not _customers:
        _customers = _load_customers()


_STREET_ABBREVS = {
    " ave": " avenue", " ave.": " avenue",
    " st": " street",  " st.": " street",
    " rd": " road",    " rd.": " road",
    " dr": " drive",   " dr.": " drive",
    " blvd": " boulevard",
    " ln": " lane",    " ln.": " lane",
    " ct": " court",   " ct.": " court",
    " pl": " place",   " pl.": " place",
}


def _normalize_street(name: str) -> str:
    name = name.strip().lower()
    for abbr, full in _STREET_ABBREVS.items():
        if name.endswith(abbr):
            name = name[: -len(abbr)] + full
            break
    return name


def lookup_address(plz: str, street_name: str, door_number: str) -> Optional[AddressInfo]:
    _ensure_loaded()
    plz = str(plz).strip()
    street_name = _normalize_street(street_name)
    door_number = str(door_number).strip()

    for row in _addresses:
        if (
            str(row["PLZ"]).strip() == plz
            and _normalize_street(row["street_name"]) == street_name
            and str(row["door_number"]).strip() == door_number
        ):
            return AddressInfo(
                plz=row["PLZ"],
                door_number=row["door_number"],
                street_name=row["street_name"],
                fix_max_speed=int(row["Fix_max_speed"]),
                cube_max_speed=int(row["Cube_max_speed"]),
            )
    return None


SIMILARITY_THRESHOLD = 0.45  # minimum score to suggest a fuzzy match


def find_similar_address(
    plz: str, street_name: str, door_number: str
) -> Optional[tuple["AddressInfo", float]]:
    """
    Find the closest address in the database when exact match fails.
    Returns (AddressInfo, similarity_score) or None if score < SIMILARITY_THRESHOLD.

    Similarity score is 0.0–1.0:
      - 1.0  → perfect street + door match within the PLZ
      - >0.6 → partial match (street similar, door may differ)
      - <0.6 → too different, treat as no coverage
    """
    _ensure_loaded()
    plz = str(plz).strip()
    norm_street = _normalize_street(street_name)
    door_number = str(door_number).strip()

    # Step 1: candidates in the same PLZ
    candidates = [r for r in _addresses if str(r["PLZ"]).strip() == plz]

    if not candidates:
        # No PLZ match — broaden search to all addresses
        candidates = _addresses

    if not candidates:
        return None

    # Step 2: score by street name similarity (street is primary, door is secondary)
    def score(row: dict) -> float:
        street_sim = difflib.SequenceMatcher(
            None, norm_street, _normalize_street(row["street_name"])
        ).ratio()
        door_sim = 1.0 if str(row["door_number"]).strip() == door_number else 0.0
        return street_sim * 0.8 + door_sim * 0.2

    best = max(candidates, key=score)
    best_score = score(best)

    if best_score < SIMILARITY_THRESHOLD:
        return None

    logger.debug("Fuzzy address match: score=%.2f → %s %s %s",
                 best_score, best["PLZ"], best["street_name"], best["door_number"])

    return AddressInfo(
        plz=best["PLZ"],
        door_number=best["door_number"],
        street_name=best["street_name"],
        fix_max_speed=int(best["Fix_max_speed"]),
        cube_max_speed=int(best["Cube_max_speed"]),
    ), best_score


def lookup_customer_by_id(customer_id: str) -> Optional[CustomerInfo]:
    """Look up a customer by their subscriber ID."""
    _ensure_loaded()
    customer_id = str(customer_id).strip()
    for row in _customers:
        if str(row["id"]).strip() == customer_id:
            return CustomerInfo(
                customer_id=row["id"],
                first_name=row["first_name"],
                surname=row["surname"],
            )
    return None


def lookup_customer_by_name(first_name: str, surname: str) -> Optional[CustomerInfo]:
    _ensure_loaded()
    first_name = first_name.strip().lower()
    surname = surname.strip().lower()
    for row in _customers:
        if row["first_name"].strip().lower() == first_name and row["surname"].strip().lower() == surname:
            return CustomerInfo(
                customer_id=row["id"],
                first_name=row["first_name"],
                surname=row["surname"],
            )
    return None
