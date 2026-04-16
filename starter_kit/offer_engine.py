"""
Business logic: choose Fix vs Cube, select the best plan, handle TV upsell.

Tariff data (updated from product catalog):
  FIX (Fiber): 50, 100, 150, 250, 500, 1000 Mbps
  CUBE (5G):   50, 100, 150, 250, 500 Mbps + A1 Xcite (100 Mbps, no contract)
  TV addons:   Xplore TV S (3.90 EUR), Xplore TV M (4.95 EUR first year → 9.90 EUR)

Pricing shown is months 7–24 (ongoing price after free period).
Most plans: first 6 months free, then regular price kicks in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

# ---------------------------------------------------------------------------
# Tariff definitions
# ---------------------------------------------------------------------------

# FIX = Fiber (Glasfaser + Internet line)
FIX_PLANS = [
    {
        "product_id": "INT50",
        "name": "Internet 50",
        "speed": 50,
        "upload": 15,
        "price": 22.32,       # months 7-24 (20% discount)
        "price_full": 27.90,  # months 25+
        "setup_fee": 29.90,
        "promo": "First 6 months free, then 22.32 EUR/mo until month 24",
    },
    {
        "product_id": "INT100",
        "name": "Internet 100",
        "speed": 100,
        "upload": 20,
        "price": 29.90,
        "price_full": 29.90,
        "setup_fee": 29.90,
        "promo": "First 6 months free",
    },
    {
        "product_id": "INT150",
        "name": "Internet 150",
        "speed": 150,
        "upload": 30,
        "price": 29.90,
        "price_full": 29.90,
        "setup_fee": 29.90,
        "promo": "First 6 months free",
    },
    {
        "product_id": "GF250",
        "name": "Glasfaser Internet 250",
        "speed": 250,
        "upload": 100,
        "price": 29.90,
        "price_full": 29.90,
        "setup_fee": 29.90,
        "promo": "First 6 months free",
    },
    {
        "product_id": "GF500",
        "name": "Glasfaser Internet 500",
        "speed": 500,
        "upload": 100,
        "price": 43.92,       # months 7-24 (20% discount)
        "price_full": 54.90,  # months 25+
        "setup_fee": 29.90,
        "promo": "First 6 months free, then 20% discount until month 24",
    },
    {
        "product_id": "GF1000",
        "name": "Glasfaser Internet 1000",
        "speed": 1000,
        "upload": 250,
        "price": 74.90,
        "price_full": 74.90,
        "setup_fee": 29.90,
        "promo": "First 6 months free",
    },
]

# CUBE = 5G wireless home internet
CUBE_PLANS = [
    {
        "product_id": "CUBE50",
        "name": "Cube Internet 50",
        "speed": 50,
        "upload": 15,
        "price": 22.32,       # months 7-24 (20% discount)
        "price_full": 27.90,  # months 25+
        "setup_fee": 0.0,
        "promo": "First 6 months free, then 20% discount until month 24",
    },
    {
        "product_id": "XCITE",
        "name": "A1 Xcite Cube",
        "speed": 100,
        "upload": 20,
        "price": 22.90,
        "price_full": 22.90,
        "setup_fee": 0.0,
        "promo": "No contract, no setup fee, modem included",
    },
    {
        "product_id": "CUBE100",
        "name": "Cube Internet 100",
        "speed": 100,
        "upload": 20,
        "price": 29.90,
        "price_full": 29.90,
        "setup_fee": 0.0,
        "promo": "First 6 months free",
    },
    {
        "product_id": "CUBE150",
        "name": "Cube Internet 150",
        "speed": 150,
        "upload": 30,
        "price": 29.90,
        "price_full": 29.90,
        "setup_fee": 0.0,
        "promo": "First 6 months free",
    },
    {
        "product_id": "CUBE250",
        "name": "Cube Internet 250",
        "speed": 250,
        "upload": 50,
        "price": 29.90,
        "price_full": 29.90,
        "setup_fee": 0.0,
        "promo": "First 6 months free",
    },
    {
        "product_id": "CUBE500",
        "name": "Cube Internet 500",
        "speed": 500,
        "upload": 70,
        "price": 54.90,
        "price_full": 54.90,
        "setup_fee": 0.0,
        "promo": "First 6 months free",
    },
]

# TV addons
TV_PLANS = [
    {
        "product_id": "TVS",
        "name": "A1 Xplore TV S",
        "price": 3.90,
        "price_full": 3.90,
        "description": "65+ channels, TV box required",
    },
    {
        "product_id": "TVM",
        "name": "A1 Xplore TV M",
        "price": 4.95,        # months 1-12 (half price)
        "price_full": 9.90,   # months 13+
        "description": "7-day replay TV, 65+ channels — half price first year",
    },
]

# Default TV upsell: TV M (better value proposition for sales)
TV_ADDON_PRICE = TV_PLANS[1]["price"]        # 4.95 EUR/mo (first year)
TV_ADDON_PRICE_FULL = TV_PLANS[1]["price_full"]  # 9.90 EUR/mo after
TV_ADDON_NAME = TV_PLANS[1]["name"]          # A1 Xplore TV M

VOICE_ONLY_DISCOUNT = 5.0  # EUR/month discount for existing customers


def min_speed_for_users(num_users: int) -> int:
    """Minimum recommended speed (Mbps): 50 Mbps per user."""
    return max(50, (num_users or 1) * 50)

ProductType = Literal["FIX", "CUBE"]


# ---------------------------------------------------------------------------
# Offer dataclass
# ---------------------------------------------------------------------------

@dataclass
class Offer:
    product: ProductType
    product_id: str
    plan_name: str
    speed: int
    upload: int
    price: float          # ongoing price (months 7-24)
    price_full: float     # price after month 24
    setup_fee: float
    promo: str            # promotional description
    tv_included: bool = False
    tv_name: str = ""
    tv_price: float = 0.0
    voice_only_discount: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def effective_price(self) -> float:
        """Internet plan price only — voice discount applies to customer's voice contract, not here."""
        total = self.price
        if self.tv_included:
            total += self.tv_price
        return total

    def summary(self) -> str:
        parts = [
            f"{self.plan_name} — {self.speed} Mbps down / {self.upload} Mbps up"
            f" — {self.price:.2f} EUR/month (from month 7)",
            f"Promotion: {self.promo}",
        ]
        if self.setup_fee > 0:
            parts.append(f"Setup fee: {self.setup_fee:.2f} EUR (one-time)")
        if self.voice_only_discount:
            parts.append(
                f"Note: customer is an existing A1 client — they receive a {VOICE_ONLY_DISCOUNT:.0f} EUR/month "
                f"discount on their VOICE contract (not on this internet plan)"
            )
        if self.tv_included:
            parts.append(f"+ {self.tv_name}: +{self.tv_price:.2f} EUR/month (first year)")
        parts.append(f"Internet total from month 7: {self.effective_price:.2f} EUR/month")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Plan selection helpers
# ---------------------------------------------------------------------------

def _best_plan(plans: list[dict], max_speed: int, min_speed: int = 0) -> Optional[dict]:
    """Return the fastest plan within [min_speed, max_speed], or None if none qualify."""
    eligible = [p for p in plans if min_speed <= p["speed"] <= max_speed]
    if not eligible:
        # Fallback: ignore min_speed if no plan meets it (address can't deliver enough)
        eligible = [p for p in plans if p["speed"] <= max_speed]
    if not eligible:
        return None
    return max(eligible, key=lambda p: p["speed"])


def recommend_product(
    age: int,
    fix_max_speed: int,
    cube_max_speed: int,
) -> ProductType:
    """
    Product recommendation logic:
      - age < 26  → Cube if available, else Fix
      - Fix speed at address >= 50 Mbps → Fix (fiber preferred)
      - otherwise → Cube if available, else Fix
    """
    fix_available = _best_plan(FIX_PLANS, fix_max_speed) is not None
    cube_available = _best_plan(CUBE_PLANS, cube_max_speed) is not None

    if age < 26:
        return "CUBE" if cube_available else "FIX"
    if fix_max_speed >= 50 and fix_available:
        return "FIX"
    return "CUBE" if cube_available else "FIX"


def build_offer(
    product: ProductType,
    fix_max_speed: int,
    cube_max_speed: int,
    is_existing_customer: bool = False,
    min_speed: int = 0,
) -> Offer:
    if product == "FIX":
        plan = _best_plan(FIX_PLANS, fix_max_speed, min_speed)
        # Fallback to CUBE if no FIX plan fits the address speed
        if plan is None:
            product = "CUBE"
            plan = _best_plan(CUBE_PLANS, cube_max_speed, min_speed)
    else:
        plan = _best_plan(CUBE_PLANS, cube_max_speed, min_speed)
        # Fallback to FIX if no CUBE plan fits the address speed
        if plan is None:
            product = "FIX"
            plan = _best_plan(FIX_PLANS, fix_max_speed, min_speed)

    if plan is None:
        raise ValueError(
            f"No eligible plan found for fix_max_speed={fix_max_speed}, cube_max_speed={cube_max_speed}"
        )

    return Offer(
        product=product,
        product_id=plan["product_id"],
        plan_name=plan["name"],
        speed=plan["speed"],
        upload=plan["upload"],
        price=plan["price"],
        price_full=plan["price_full"],
        setup_fee=plan["setup_fee"],
        promo=plan["promo"],
        voice_only_discount=is_existing_customer,
    )


def build_alternative_offer(
    current_offer: Offer,
    fix_max_speed: int,
    cube_max_speed: int,
) -> Optional[Offer]:
    """
    Return the next cheaper plan of the same product type within address speed limits.
    Returns None if no cheaper option exists.
    """
    plans = FIX_PLANS if current_offer.product == "FIX" else CUBE_PLANS
    max_speed = fix_max_speed if current_offer.product == "FIX" else cube_max_speed

    eligible = [
        p for p in plans
        if p["speed"] <= max_speed and p["speed"] < current_offer.speed
    ]
    if not eligible:
        return None

    plan = max(eligible, key=lambda p: p["speed"])
    return Offer(
        product=current_offer.product,
        product_id=plan["product_id"],
        plan_name=plan["name"],
        speed=plan["speed"],
        upload=plan["upload"],
        price=plan["price"],
        price_full=plan["price_full"],
        setup_fee=plan["setup_fee"],
        promo=plan["promo"],
        voice_only_discount=current_offer.voice_only_discount,
    )


def add_tv(offer: Offer) -> Offer:
    """Return a copy of the offer with the TV M addon included."""
    import copy
    new_offer = copy.copy(offer)
    new_offer.tv_included = True
    new_offer.tv_name = TV_ADDON_NAME
    new_offer.tv_price = TV_ADDON_PRICE
    new_offer.notes = list(offer.notes) + [f"TV addon ({TV_ADDON_NAME}) added"]
    return new_offer
