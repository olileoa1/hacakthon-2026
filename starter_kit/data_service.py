"""
CSV lookup service for customers and addresses.
"""

import csv
import os
from dataclasses import dataclass
from typing import Optional


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
