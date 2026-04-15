"""
Business logic: choose Fix vs Cube, select the best plan, handle TV upsell.

Tariff data based on tariffs.png:
  FIX plans:  50, 250, 650, 1050 Mbps
  CUBE plans: 10, 30, 60, 140, 220, 310 Mbps   (illustrative tiers visible in image)
  TV addon:   +26 EUR / month (Aria Box Cube)
"""

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Tariff definitions
# ---------------------------------------------------------------------------

FIX_PLANS = [
    {"speed": 50,   "price": 19.90, "name": "Fix 50"},
    {"speed": 250,  "price": 29.90, "name": "Fix 250"},
    {"speed": 650,  "price": 39.90, "name": "Fix 650"},
    {"speed": 1050, "price": 49.90, "name": "Fix 1050"},
]

CUBE_PLANS = [
    {"speed": 10,  "price": 19.90, "name": "Cube 10"},
    {"speed": 30,  "price": 24.90, "name": "Cube 30"},
    {"speed": 60,  "price": 29.90, "name": "Cube 60"},
    {"speed": 140, "price": 34.90, "name": "Cube 140"},
    {"speed": 220, "price": 39.90, "name": "Cube 220"},
    {"speed": 310, "price": 44.90, "name": "Cube 310"},
]

TV_ADDON_PRICE = 26.0
TV_ADDON_NAME = "Aria Box Cube (TV)"

VOICE_ONLY_DISCOUNT = 5.0  # EUR/month discount for voice-only customers


ProductType = Literal["FIX", "CUBE"]


@dataclass
class Offer:
    product: ProductType
    plan_name: str
    speed: int
    price: float
    tv_included: bool = False
    voice_only_discount: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def effective_price(self) -> float:
        total = self.price
        if self.tv_included:
            total += TV_ADDON_PRICE
        if self.voice_only_discount:
            total -= VOICE_ONLY_DISCOUNT
        return total

    def summary(self) -> str:
        parts = [f"{self.plan_name} — {self.speed} Mbps — {self.price:.2f} EUR/month"]
        if self.voice_only_discount:
            parts.append(f"Voice-only discount: -{VOICE_ONLY_DISCOUNT:.2f} EUR")
        if self.tv_included:
            parts.append(f"+ {TV_ADDON_NAME}: +{TV_ADDON_PRICE:.2f} EUR")
        parts.append(f"Total: {self.effective_price:.2f} EUR/month")
        return " | ".join(parts)


def _best_plan(plans: list[dict], max_speed: int) -> dict:
    """Return the fastest plan that does not exceed max_speed."""
    eligible = [p for p in plans if p["speed"] <= max_speed]
    if not eligible:
        return plans[0]  # fallback to cheapest
    return max(eligible, key=lambda p: p["speed"])


def recommend_product(
    age: int,
    fix_max_speed: int,
    cube_max_speed: int,
) -> ProductType:
    """
    Level 1 logic from first_page.png:
      - age < 26  → Cube
      - mix speed (Fix) > 50 Mbps → Fix, else → Cube
    """
    if age < 26:
        return "CUBE"
    if fix_max_speed > 50:
        return "FIX"
    return "CUBE"


def build_offer(
    product: ProductType,
    fix_max_speed: int,
    cube_max_speed: int,
    is_existing_customer: bool = False,
) -> Offer:
    if product == "FIX":
        plan = _best_plan(FIX_PLANS, fix_max_speed)
    else:
        plan = _best_plan(CUBE_PLANS, cube_max_speed)

    offer = Offer(
        product=product,
        plan_name=plan["name"],
        speed=plan["speed"],
        price=plan["price"],
        voice_only_discount=is_existing_customer,
    )
    return offer


def add_tv(offer: Offer) -> Offer:
    """Return a copy of the offer with TV addon included."""
    import copy
    new_offer = copy.copy(offer)
    new_offer.tv_included = True
    new_offer.notes = list(offer.notes) + [f"TV addon ({TV_ADDON_NAME}) added"]
    return new_offer
