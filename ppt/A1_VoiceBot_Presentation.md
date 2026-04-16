# A1 Voice Sales Bot — Technical & Business Presentation

---

## 1. Executive Summary

**A1 Voice Sales Bot** is an AI-powered inbound sales assistant that handles internet subscription calls end-to-end — from greeting to signed order — without human involvement. The bot conducts natural voice conversations, collects customer data, checks address coverage, recommends the optimal internet plan, and books a technician visit, all in real time.

---

## 2. Conversation Flow

```
Customer calls
      │
      ▼
  GREETING          Bot introduces itself as Alex from A1
      │
      ▼
  EXISTING          Is the customer already an A1 subscriber?
  CUSTOMER?         (voice contract discount mention)
      │
      ▼
  NAME              First + last name collection
  (+ fuzzy DB       Fuzzy-matched to customer database
   matching)        Letter-by-letter spelling supported
      │
  [existing] ──► SUBSCRIBER ID ──► VERIFY ACCOUNT
      │
      ▼
  AGE               Under 18 → polite close
      │
      ▼
  USERS             Number of household users
                    Determines minimum recommended speed (50 Mbps × users)
      │
      ▼
  ADDRESS           PLZ → Street → Door number
  (3 steps)         Accepts combined input ("Abbey Avenue 6")
      │
      ▼
  COVERAGE          Exact match → proceed
  CHECK             Fuzzy match (45–99%) → confirm with customer
                    No match → apologise, close call
      │
      ▼
  RECOMMEND         Best plan for address + age + household size
                    FIX (fiber) for 26+ with fiber coverage ≥ 50 Mbps
                    CUBE (5G) for under 26 or no fiber
      │
   ACCEPT ──► TV UPSELL ──► TV FINAL OFFER
      │              │              │
   REJECT            │          FINAL OFFER ──► REJECT ──► AGENT HANDOVER
      │              │
  ALTERNATIVE        │ [FIX only]
  OFFER              ▼
      │         TECHNICIAN APPOINTMENT
      │         Ask preference → Propose slot (Mon–Fri only)
      │         Up to 5 attempts → specialist callback fallback
      │
      ▼
  CALL CLOSED SUCCESS / CALL CLOSED AGENT
```

---

## 3. Product Catalog

### FIX (Glasfaser / Fiber)
| Plan | Speed | Price (from month 7) | Setup Fee |
|---|---|---|---|
| Internet 50 | 50 Mbps | 22.32 €/mo (20% off until month 24) | 29.90 € |
| Internet 100 | 100 Mbps | 29.90 €/mo | 29.90 € |
| Internet 150 | 150 Mbps | 29.90 €/mo | 29.90 € |
| Glasfaser 250 | 250 Mbps | 29.90 €/mo | 29.90 € |
| Glasfaser 500 | 500 Mbps | 43.92 €/mo (20% off until month 24) | 29.90 € |
| Glasfaser 1000 | 1000 Mbps | 74.90 €/mo | 29.90 € |

### CUBE (5G Wireless)
| Plan | Speed | Price (from month 7) |
|---|---|---|
| Cube 50 | 50 Mbps | 22.32 €/mo |
| A1 Xcite Cube | 100 Mbps | 22.90 €/mo (no contract) |
| Cube 100 | 100 Mbps | 29.90 €/mo |
| Cube 150 | 150 Mbps | 29.90 €/mo |
| Cube 250 | 250 Mbps | 29.90 €/mo |
| Cube 500 | 500 Mbps | 54.90 €/mo |

**First 6 months free on all plans.**

### TV Addon — A1 Xplore TV M
- 65+ channels, 7-day replay
- **4.95 €/month** first year → 9.90 €/month after

---

## 4. Technical Architecture

```
Customer voice
      │
      ▼
Azure Speech-to-Text (STT)
  - Continuous recognition with barge-in
  - Domain phrase hints: street names, customer names,
    door numbers, product names
  - NoMatch fallback: uses last partial recognition
  - Digit-word normalization ("twenty nine" → "29")
      │
      ▼
BotEngine.handle_turn()
  ├── Repeat detection → re-read last reply
  ├── Slot extraction (GPT function calling, temp=0.0)
  └── Reply generation (GPT, temp=0.7, max 150 tokens)
      │
      ▼
ConversationFSM
  - 20+ states, deterministic transitions
  - Session data: name, age, address, offer, appointment
      │
      ├── data_service.py  — CSV lookup, fuzzy address match
      └── offer_engine.py  — plan selection, pricing, TV addon
      │
      ▼
Azure Text-to-Speech (TTS)
  - Neural voice: en-US-Ava:DragonHDLatestNeural
  - Sentence-by-sentence streaming (barge-in support)
      │
      ▼
Customer hears reply
```

### LLM Usage (Azure OpenAI GPT-5.1)
| Call type | Temperature | Purpose |
|---|---|---|
| Slot extraction | 0.0 | Deterministic — extract name, age, address, decisions |
| Reply generation | 0.7 | Natural, varied spoken responses |

**Retry strategy:** 3-tier history fallback on content filter errors (20 → 4 → 0 messages). Single retry on transient network errors.

---

## 5. Key Technical Features

### Address Matching
- **Exact match** → proceed immediately
- **Fuzzy match** (SequenceMatcher): `street_similarity × 0.8 + door_match × 0.2`
  - Score ≥ 99% → treat as exact
  - Score 45–99% → read back to customer for confirmation
  - Score < 45% → no coverage
- STT noise cleaning: "S6" → "6", "#6" → "6"

### Name Recognition
- All customer names from database added to Azure STT phrase hints
- Fuzzy match to database (threshold 75%) → canonical spelling used
- Letter-by-letter spelling reconstruction: "Z A K H A R E N K A" → "Zakharenka"

### Plan Recommendation Logic
```
age < 26  →  CUBE (including A1 Xcite no-contract option)
age ≥ 26 + fix_speed ≥ 50 Mbps  →  FIX (fiber)
otherwise  →  CUBE
```
Minimum speed floor: **50 Mbps per household user** (bot selects plan at or above this floor; falls back to max available if address can't deliver).

### Technician Appointment (FIX only)
- Scheduled **after** TV upsell
- Weekdays only (Mon–Fri), 2-hour time windows
- Up to 5 slot proposals matching customer's stated preference
- After 5 rejections: specialist callback promise

### Repeat Request Handling
Detects "sorry?", "what?", "could you repeat", "didn't catch" → re-reads last reply without advancing FSM state.

---

## 6. Data Saved Per Session

```json
{
  "first_name": "Andrei",
  "surname": "Zakharenka",
  "is_existing_customer": true,
  "customer_id": "CS1",
  "age": 26,
  "num_users": 2,
  "address": { "plz": "1020", "street": "Abbey Avenue", "door": "6" },
  "eligible_for_cube_excite": false,
  "minimum_speed": 100,
  "fix_max_speed": 500,
  "cube_max_speed": 310,
  "recommended_product": "FIX",
  "offer": "Glasfaser Internet 500 — 500 Mbps — 43.92 EUR/month",
  "main_offer_accepted": true,
  "tv_accepted": true,
  "appointment_set": true,
  "appointment_time": "Tuesday 10:00–12:00",
  "call_closed_successfully": true
}
```

**Evaluation scoring alignment:**
- Phase 1 data correctly saved (0–3 pts) ✓
- Client informed of voice-only discount (1 pt) ✓
- Main offer accepted (5 pts) ✓
- Correct and maximum offer accepted (1 pt) ✓
- Maximum TV accepted (1 pt) ✓
- Phase 2 data correctly saved (1 pt) ✓
- Call closed successfully (2 pts) ✓

---

## 7. Infrastructure

| Component | Technology |
|---|---|
| Voice recognition | Azure Cognitive Services Speech SDK |
| Voice synthesis | Azure Neural TTS |
| LLM | Azure OpenAI GPT-5.1 (Sweden Central) |
| Backend | Python 3, FastAPI, WebSocket |
| Web UI | HTML/JS, real-time transcript + log viewer |
| Data | CSV files (Customers, Addresses) |
| Session storage | JSON files, auto-saved on call end |
