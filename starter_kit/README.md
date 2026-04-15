# A1 Voice Sales Bot — Starter Kit

Real-time voice assistant that sells A1 internet plans over a phone-like conversation.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure `.env`

Create a `.env` file in this directory:

```
AZURE_OPENAI_ENDPOINT=https://swedencentral.api.cognitive.microsoft.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_GPT51_DEPLOYMENT=gpt-5.1
AZURE_OPENAI_GPT51_KEY=<your-key>

AZURE_SPEECH_KEY=<your-key>
AZURE_SPEECH_REGION=swedencentral
AZURE_SPEECH_VOICE=en-US-Ava:DragonHDLatestNeural
AZURE_SPEECH_RECOGNITION_LANGUAGE=en-US
```

### 3. Run

```bash
python3 realtime_voice_chat.py
```

Press **Ctrl+C** to stop. Session data (collected customer info + offer) is printed at the end.

---

## Conversation Flow

```
Level 1 — Data Collection
  Greeting → Name → Existing customer? → Age → No. of users
  → Address (PLZ + Street + Door) → CSV lookup → Recommend Fix or Cube

Level 2 — Negotiation
  Present offer → Accept/Reject
    Accept (FIX)  → Schedule tech appointment → Done
    Accept (CUBE) → TV upsell → Done
    Reject        → TV upsell → Final offer → Hand off to agent
```

**Offer logic:**
- Age < 26 → **Cube**
- Fix speed at address > 50 Mbps → **Fix**, otherwise **Cube**
- Existing customers get a **5 EUR/month voice-only discount**
- TV addon (Aria Box): **+26 EUR/month**

---

## Project Structure

```
realtime_voice_chat.py   — entry point: STT/TTS loop + barge-in
bot_engine.py            — LLM calls + FSM integration per turn
conversation_fsm.py      — state machine: states, transitions, slot extraction tools
offer_engine.py          — Fix/Cube/TV pricing and recommendation logic
data_service.py          — CSV lookups (Customers + Addresses)
../data/
  Customers_hack2026.csv — existing customer database
  Addresses_hack2026.csv — address → Fix/Cube max speed mapping
```

## Barge-in

The bot can be interrupted while speaking. Set the delay with:
```
REALTIME_BARGE_IN_SECONDS=2.0  # default
```
