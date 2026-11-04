"""UK Desk tools: a small MCP server for the UK Desk agent (DTU course demo).

Tools
- classify       Typed decision questions -> answer + confidence, answered by Liquid AI's d1
                 decision model when D1_API_KEY is set (choice questions on the systemone API).
                 Without a key, or if d1 fails, transparent rules answer instead (engine = "rules").
- landed_cost    UK all-in prices per SKU: VAT, duty budget and price points. Plain code.
- check_order    One or more UK orders: the GBP 135 rule, duty, origin statement, ship-mode errors.
- stage3_check   Monthly break-even for moving bestsellers into a shared UK warehouse.
- memory_plan    Simple typed memory (rules, inspired by the Jev-Mem paper, no model): type a new
                 observation and link it to related memories (edge only if score >= 0.6).
                 Storage is the Memory tab in the sheet.
- memory_recall  Simple typed memory, read path: rank memories for a query.
- list_rules     The rules and example rates this server applies.

Environment variables (set them in Prefect Horizon):
  D1_API_KEY   your Liquid AI API key (leave empty to use the rules only)
  D1_API_URL   default https://api.liquid.ai/decisions/v1/systemone
  D1_MODEL     default d1

Scope: Great Britain (England, Scotland, Wales). Northern Ireland has different rules.
Rates marked EXAMPLE must be checked on the UK Trade Tariff before real use.
Deploy on Prefect Horizon with entrypoint server.py:mcp
"""

import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date as Date

import httpx
from fastmcp import FastMCP

mcp = FastMCP("uk-desk")

D1_API_URL = os.environ.get("D1_API_URL", "https://api.liquid.ai/decisions/v1/systemone")
D1_MODEL = os.environ.get("D1_MODEL", "d1")
D1_API_KEY = os.environ.get("D1_API_KEY", "")

# ---------------------------------------------------------------------------
# Rules (Great Britain). Sources: GOV.UK "Tax and customs for goods sent from abroad",
# HMRC VATREG37210, Royal Mail "Fee to pay", EU REX guidance. Simplified for a demo.
# ---------------------------------------------------------------------------
UK_VAT = 0.20
DK_VAT = 0.25
LOW_VALUE_LIMIT_GBP = 135.0      # <= 135: seller charges UK VAT at checkout, no customs duty
REX_LIMIT_EUR = 6000.0           # origin statement above this needs a REX number
CARRIER_FEE_GBP = 8.0            # Royal Mail handling fee charged to the customer (Parcelforce 12)
CARRIER_FEE_OVER_900_GBP = 25.0  # goods over GBP 900
MIN_NET_SHARE = 0.95             # brand keeps at least 95% of its Danish net price
EDGE_THRESHOLD = 0.6             # memory: link memories only if score >= 0.6 (as in the Jev-Mem paper)

# EXAMPLE third-country duty rates by HS heading (4 digits). 6110 = 12% checked for women's
# wool pullovers (6110119000); the others are typical rates. Confirm on the UK Trade Tariff.
EXAMPLE_DUTY_RATES = {
    "6110": 0.12, "6115": 0.12, "6116": 0.08, "6117": 0.12, "6214": 0.08,
    "6301": 0.12, "6302": 0.12, "6505": 0.02, "6911": 0.12, "6912": 0.05,
}
DEFAULT_DUTY_RATE = 0.12

# Price points ending in 9: 9, 19 ... 299, then 349, 399 ... 4999
PRICE_POINTS = [9 + 10 * i for i in range(30)] + [349 + 50 * i for i in range(94)]

EU_COUNTRIES = {
    "austria", "belgium", "bulgaria", "croatia", "cyprus", "czechia", "czech republic",
    "denmark", "estonia", "finland", "france", "germany", "greece", "hungary", "ireland",
    "italy", "latvia", "lithuania", "luxembourg", "malta", "netherlands", "poland",
    "portugal", "romania", "slovakia", "slovenia", "spain", "sweden",
}
NON_EU_COUNTRIES = {
    "china", "vietnam", "turkey", "bangladesh", "india", "pakistan", "peru", "cambodia",
    "indonesia", "sri lanka", "morocco", "tunisia", "united kingdom", "uk", "norway",
    "united states", "usa", "taiwan", "thailand", "myanmar", "ukraine",
}
CITY_TO_COUNTRY = {
    "porto": "portugal", "guimaraes": "portugal", "guimarães": "portugal", "lisbon": "portugal",
    "vilnius": "lithuania", "kaunas": "lithuania", "prato": "italy", "biella": "italy",
    "ningbo": "china", "shanghai": "china", "istanbul": "turkey", "izmir": "turkey",
    "dhaka": "bangladesh",
}
ORIGIN_VERBS = r"(?:made|knitted|knit|woven|sewn|spun|produced|manufactured|hand-poured|handmade|crafted|printed|thrown|dyed)"

OUT_OF_SCOPE = {
    "food": "food (import controls)", "coffee": "food (import controls)",
    "tea": "food (import controls)", "chocolate": "food (import controls)",
    "wine": "excise goods", "beer": "excise goods", "spirits": "excise goods",
    "cosmetics": "cosmetics (UK product rules)", "skincare": "cosmetics (UK product rules)",
    "supplements": "food supplements", "furniture": "bulky freight (not parcels)",
    "electronics": "electronics (UK product rules)",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _r2(x: float) -> float:
    return math.floor(x * 100 + 0.5) / 100


def _has(text: str, phrases) -> list:
    return [p for p in phrases if p in text]


def _duty_rate(code: str) -> float:
    digits = re.sub(r"\D", "", str(code or ""))
    return EXAMPLE_DUTY_RATES.get(digits[:4], DEFAULT_DUTY_RATE)


def _country(word: str) -> str:
    w = word.strip().lower()
    return CITY_TO_COUNTRY.get(w, w)


def _origins(text: str) -> tuple:
    """Countries named after an origin verb ('knitted in Porto'). Packing/design don't count."""
    low = text.lower()
    names = sorted(EU_COUNTRIES | NON_EU_COUNTRIES | set(CITY_TO_COUNTRY), key=len, reverse=True)
    pattern = ORIGIN_VERBS + r"(?:\s+and\s+" + ORIGIN_VERBS + r")?\s+in\s+(" + "|".join(re.escape(n) for n in names) + r")\b"
    found = [_country(m) for m in re.findall(pattern, low)]
    eu = sorted({c for c in found if c in EU_COUNTRIES})
    non = sorted({c for c in found if c in NON_EU_COUNTRIES})
    return eu, non


# ---------------------------------------------------------------------------
# classify: system-one decision questions
# ---------------------------------------------------------------------------
BLOCK_PHRASES = [
    "do not ship to the uk", "don't ship to the uk", "not ship to the united kingdom",
    "no shipping to the uk", "sender ikke til storbritannien", "sender desværre ikke til storbritannien",
    "ikke uden for eu", "kun inden for eu", "only within the eu", "eu only", "kun i danmark",
    "only ship to denmark", "sender ikke til england",
]
ALL_IN_PHRASES = [
    "duties paid", "all duties", "duties included", "include all duties", "includes all duties",
    "duties and taxes included", "taxes and duties included", "no extra charges", "ddp",
    "inkl. told", "told og moms er inkluderet", "ingen ekstra gebyrer",
]
FEES_PHRASES = [
    "paid by the customer", "responsibility of the customer", "collected by the carrier",
    "subject to import vat", "may be subject to", "handling fees", "dap", "betales af modtageren",
    "modtageren betaler", "customer pays", "charged on delivery",
]
WEAK_FEES_PHRASES = ["exclude local taxes", "exclude any local", "excluding local taxes", "excl. told"]
UK_WORDS = ["uk", "united kingdom", "storbritannien", "great britain", "england", "britain"]
SHIP_WORDS = ["ship to", "deliver", "delivered", "sender til", "leverer", "shipping to"]
COUNTRY_WORDS = ["denmark", "sweden", "norway", "finland", "germany", "danmark", "sverige", "norge", "tyskland"]


def _uk_shipping_status(inp: dict) -> dict:
    t = " " + str(inp.get("text", "")).lower() + " "
    blocks = _has(t, BLOCK_PHRASES)
    allin = _has(t, ALL_IN_PHRASES)
    fees = _has(t, FEES_PHRASES)
    weak = _has(t, WEAK_FEES_PHRASES)
    mentions_uk = any(re.search(r"\b" + re.escape(w) + r"\b", t) for w in UK_WORDS)
    if blocks:
        return dict(answer="blocks_uk", confidence=0.93, reasons=[f"says: '{blocks[0]}'"],
                    alternatives=[{"answer": "unclear", "confidence": 0.07}])
    if allin and fees:
        return dict(answer="unclear", confidence=0.55, reasons=["mentions both duties paid and customer-paid fees"],
                    alternatives=[{"answer": "ships_with_fees", "confidence": 0.25}, {"answer": "ships_all_in", "confidence": 0.2}])
    if allin:
        c = 0.95 if len(allin) >= 2 else 0.92
        return dict(answer="ships_all_in", confidence=c, reasons=[f"says: '{p}'" for p in allin],
                    alternatives=[{"answer": "unclear", "confidence": _r2(1 - c)}])
    if fees:
        c = 0.93 if len(fees) >= 2 else 0.9
        return dict(answer="ships_with_fees", confidence=c, reasons=[f"says: '{p}'" for p in fees],
                    alternatives=[{"answer": "unclear", "confidence": _r2(1 - c)}])
    if weak:
        return dict(answer="ships_with_fees", confidence=0.75, reasons=[f"says: '{weak[0]}' (customer probably pays duties)"],
                    alternatives=[{"answer": "unclear", "confidence": 0.25}])
    if not mentions_uk and _has(t, SHIP_WORDS) and _has(t, COUNTRY_WORDS):
        return dict(answer="blocks_uk", confidence=0.8, reasons=["lists countries it ships to and the UK is not one of them"],
                    alternatives=[{"answer": "unclear", "confidence": 0.2}])
    return dict(answer="unclear", confidence=0.4, reasons=["no clear statement about UK shipping or duties"], alternatives=[])


def _brand_fit(inp: dict) -> dict:
    cat = str(inp.get("category", "")).lower()
    for word, why in OUT_OF_SCOPE.items():
        if word in cat:
            return dict(answer=0.0, confidence=0.9, reasons=[f"category out of scope: {why}"], alternatives=[])
    status = str(inp.get("uk_status", "")).lower()
    base = {"blocks_uk": 0.45, "ships_with_fees": 0.5, "unclear": 0.25}.get(status)
    if status == "ships_all_in":
        return dict(answer=0.0, confidence=0.9, reasons=["already sells to the UK with duties included"], alternatives=[])
    if base is None:
        return dict(answer=0.0, confidence=0.5, reasons=[f"unknown uk_status '{status}'"], alternatives=[])
    eu, non = _origins(str(inp.get("made_in_notes", "")))
    if eu or non:
        share = len(eu) / (len(eu) + len(non))
        conf = 0.85
        origin_reason = f"made in EU: {', '.join(eu) or 'none'}; outside EU: {', '.join(non) or 'none'}"
    else:
        share, conf, origin_reason = 0.5, 0.6, "origin not stated"
    signals = [s for s in re.split(r"[;\n]", str(inp.get("uk_signals", ""))) if s.strip() and s.strip().lower() not in ("none", "-")]
    score = _r2(min(1.0, base + 0.3 * share + 0.1 * min(len(signals), 2)))
    reasons = [f"UK status {status} (+{base})", origin_reason + f" (+{_r2(0.3 * share)})",
               f"{len(signals)} UK signal(s) (+{_r2(0.1 * min(len(signals), 2))})"]
    return dict(answer=score, confidence=conf, reasons=reasons, alternatives=[])


FIBRES = {
    "merino": "wool", "lambswool": "wool", "wool": "wool", "cashmere": "cashmere",
    "alpaca": "fine_hair", "mohair": "fine_hair", "camel hair": "fine_hair", "yak": "fine_hair",
    "cotton": "cotton", "linen": "flax", "flax": "flax", "silk": "silk", "polyamide": "synthetic",
    "nylon": "synthetic", "polyester": "synthetic", "acrylic": "synthetic", "elastane": "synthetic",
    "viscose": "artificial", "lyocell": "artificial",
}


def _composition(text: str) -> list:
    comp = {}
    for pct, word in re.findall(r"(\d{1,3})\s*%\s*([a-z ]+?)(?=[,;/)]|\s\d|\s+and\b|$)", text):
        word = word.strip()
        fibre = next((f for k, f in FIBRES.items() if k in word), None)
        if fibre:
            comp[fibre] = comp.get(fibre, 0) + int(pct)
    return sorted(comp.items(), key=lambda kv: -kv[1])


def _customs_code(inp: dict) -> dict:
    text = (str(inp.get("product", "")) + ", " + str(inp.get("description", ""))).lower()
    comp = _composition(text)
    top = comp[0][0] if comp else None
    margin = (comp[0][1] - comp[1][1]) if len(comp) > 1 else 100
    knitted = bool(re.search(r"\bknit|knitted|rib-knit|crochet", text))
    woven = "woven" in text or "terry" in text
    reasons = [f"composition: {', '.join(f'{p}% {f}' for f, p in comp) or 'not stated'}"]

    def res(code, conf, alts=(), extra=None):
        r = list(reasons) + (extra or [])
        r.append("HS 6-digit; the customs lead confirms the full UK commodity code on the UK Trade Tariff")
        return dict(answer=code, confidence=conf, reasons=r,
                    alternatives=[{"answer": a, "confidence": c} for a, c in alts])

    if any(w in text for w in ("pullover", "sweater", "jumper", "cardigan", "roll-neck", "crew-neck", "turtleneck")):
        sub = {"wool": "6110.11", "cashmere": "6110.12", "fine_hair": "6110.19", "cotton": "6110.20",
               "synthetic": "6110.30"}.get(top, "6110.90")
        conf = 0.95 if knitted else 0.88
        if margin < 20 and len(comp) > 1:
            other = {"wool": "6110.11", "cashmere": "6110.12", "fine_hair": "6110.19", "cotton": "6110.20",
                     "synthetic": "6110.30"}.get(comp[1][0], "6110.90")
            return res(sub, 0.78, [(other, 0.22)], [f"{top} predominates by weight, but only by {margin} points: check the label or test report"])
        return res(sub, conf, [], ["knitted garment (chapter 61)" if knitted else "construction not stated; pullovers are normally knitted"])
    if any(w in text for w in ("scarf", "shawl", "muffler")):
        woven_sub = "6214.20" if top in ("wool", "cashmere", "fine_hair") else {"silk": "6214.10", "synthetic": "6214.30", "artificial": "6214.40"}.get(top, "6214.90")
        if knitted:
            return res("6117.10", 0.94, [], ["knitted scarf (chapter 61)"])
        if woven:
            return res(woven_sub, 0.93, [], ["woven scarf (chapter 62)"])
        return res(woven_sub, 0.55, [("6117.10", 0.45)], ["construction not stated: knitted (6117.10) or woven (6214) decides the chapter"])
    if any(w in text for w in ("beanie", "hat", "cap", "headgear")):
        return res("6505.00", 0.93 if knitted else 0.8, [], ["knitted headgear"])
    if "sock" in text:
        sub = {"wool": "6115.94", "cashmere": "6115.94", "fine_hair": "6115.94", "cotton": "6115.95", "synthetic": "6115.96"}.get(top, "6115.99")
        return res(sub, 0.93 if knitted else 0.85, [])
    if any(w in text for w in ("mitten", "glove")):
        sub = {"wool": "6116.91", "cashmere": "6116.91", "fine_hair": "6116.91", "cotton": "6116.92", "synthetic": "6116.93"}.get(top, "6116.99")
        return res(sub, 0.93 if knitted else 0.85, [])
    if any(w in text for w in ("blanket", "throw", "travelling rug")):
        sub = {"wool": "6301.20", "cashmere": "6301.20", "fine_hair": "6301.20", "cotton": "6301.30", "synthetic": "6301.40"}.get(top, "6301.90")
        return res(sub, 0.94, [])
    if "towel" in text and "terry" in text and top == "cotton":
        return res("6302.60", 0.94, [])
    if any(w in text for w in ("mug", "cup", "bowl", "plate", "tableware")):
        if "porcelain" in text:
            return res("6911.10", 0.9, [])
        if any(w in text for w in ("stoneware", "ceramic", "earthenware")):
            return res("6912.00", 0.92, [])
    return dict(answer="unknown", confidence=0.2, reasons=reasons + ["product type not in the demo code table"], alternatives=[])


def _origin_claim(inp: dict) -> dict:
    country = _country(str(inp.get("made_in", "")))
    decl = str(inp.get("origin_declaration", "")).lower().strip()
    on = str(inp.get("on_date") or Date.today().isoformat())
    until = str(inp.get("declaration_valid_until") or "")
    if country in NON_EU_COUNTRIES:
        return dict(answer="non_eu", confidence=0.98, reasons=[f"made in {country}: no EU preferential origin"], alternatives=[])
    if country not in EU_COUNTRIES:
        return dict(answer="unknown", confidence=0.5, reasons=[f"country '{country}' not recognised"], alternatives=[])
    if decl in ("missing", "", "none", "n/a"):
        return dict(answer="eu_missing_proof", confidence=0.95, reasons=[f"made in {country}, but no supplier's declaration on file"], alternatives=[])
    if until and until < on:
        return dict(answer="eu_proof_expired", confidence=0.95, reasons=[f"declaration expired on {until}"], alternatives=[])
    reasons = [f"made in {country}; supplier's declaration on file" + (f", valid until {until}" if until else "")]
    if until:
        days = (Date.fromisoformat(until) - Date.fromisoformat(on)).days
        if days <= int(inp.get("warning_days") or 30):
            reasons.append(f"WARNING: declaration expires in {days} days: ask the supplier for a renewal")
    return dict(answer="eu_proven", confidence=0.97, reasons=reasons, alternatives=[])


def _message_intent(inp: dict) -> dict:
    t = str(inp.get("text", "")).lower()
    extracted = {}
    m = re.search(r"£\s?(\d+(?:\.\d{1,2})?)", t)
    if m:
        extracted["amount_gbp"] = float(m.group(1))
        fx = inp.get("fx_dkk_per_gbp")
        if fx:
            extracted["amount_dkk"] = _r2(float(m.group(1)) * float(fx))
    if _has(t, ["northern ireland", "jersey", "guernsey", "isle of man"]):
        return dict(answer="scope_question", confidence=0.88, reasons=["asks about a destination outside Great Britain"], alternatives=[], extracted=extracted)
    fee = _has(t, ["handling fee", "import vat", "customs", "duty", "duties", "had to pay", "pay a fee", "charged", "extra charges", "before delivering"])
    if fee:
        c = 0.95 if len(fee) >= 2 else 0.85
        return dict(answer="customs_fee_complaint", confidence=c, reasons=[f"mentions: {', '.join(fee)}"],
                    alternatives=[{"answer": "other", "confidence": _r2(1 - c)}], extracted=extracted)
    ret = _has(t, ["return", "send back", "send it back", "exchange", "refund"])
    if ret:
        return dict(answer="return_request", confidence=0.9, reasons=[f"mentions: {', '.join(ret)}"], alternatives=[], extracted=extracted)
    if _has(t, ["where is my", "tracking", "not arrived", "hasn't arrived", "delivery date"]):
        return dict(answer="delivery_question", confidence=0.86, reasons=["asks about delivery status"], alternatives=[], extracted=extracted)
    return dict(answer="other", confidence=0.5, reasons=["no known intent"], alternatives=[], extracted=extracted)


def _return_condition(inp: dict) -> dict:
    t = str(inp.get("note", "")).lower()
    bad = _has(t, ["chip", "broken", "crack", "hole", "torn", "ripped", "heavy stain", "missing part", "mould", "mold"])
    good = _has(t, ["unused", "unworn", "original packaging", "tags attached", "as new", "sealed", "with tags"])
    minor = _has(t, ["small pull", "minor", "slight", "crease", "light mark", "small stain", "otherwise fine", "loose thread"])
    note = "demo reads the partner's note; a d1 image question would read the photo itself"
    if bad:
        return dict(answer="write_off", confidence=0.9, reasons=[f"partner notes: {', '.join(bad)}", note],
                    alternatives=[{"answer": "second", "confidence": 0.1}])
    if minor:
        return dict(answer="second", confidence=0.6, reasons=[f"minor issue: {', '.join(minor)}", note],
                    alternatives=[{"answer": "resellable", "confidence": 0.4}])
    if good:
        c = 0.95 if len(good) >= 2 else 0.86
        return dict(answer="resellable", confidence=c, reasons=[f"partner notes: {', '.join(good)}", note],
                    alternatives=[{"answer": "second", "confidence": _r2(1 - c)}])
    return dict(answer="second", confidence=0.4, reasons=["note does not describe the condition", note], alternatives=[])


def _reply_intent(inp: dict) -> dict:
    t = str(inp.get("text", "")).lower()
    if _has(t, ["unsubscribe", "remove me", "stop emailing", "don't contact", "do not contact"]):
        return dict(answer="unsubscribe", confidence=0.95, reasons=["asks not to be contacted"], alternatives=[])
    if _has(t, ["not interested", "no thanks", "no thank you", "not now", "nej tak", "ikke interesseret"]):
        return dict(answer="not_interested", confidence=0.92, reasons=["declines"], alternatives=[])
    pos = _has(t, ["interested", "sounds interesting", "let's talk", "lets talk", "call", "meeting", "tell me more", "show me", "next week", "lyder spændende"])
    if pos:
        return dict(answer="interested", confidence=0.9, reasons=[f"says: {', '.join(pos)}"], alternatives=[])
    if "?" in t:
        return dict(answer="question", confidence=0.8, reasons=["asks a question"], alternatives=[])
    return dict(answer="unclear", confidence=0.5, reasons=["no clear intent"], alternatives=[])


VAT_NUMBER_RE = re.compile(r"\bGB\s?(\d{3})\s?(\d{4})\s?(\d{2})\b", re.I)
SCAM_WORDS = ["tax refund", "you are due a refund", "rebate", "claim your refund", "card details",
              "bank details", "verify your account", "confirm your password", "suspended", "within 24 hours"]


def _sender_trusted(inp: dict) -> tuple[bool, str]:
    """Plain code: is the sender address on a trusted domain (gov.uk) or a configured demo sender?"""
    sender = str(inp.get("from", "")).strip().lower()
    m = re.search(r"[\w.+-]+@([\w-]+(?:\.[\w-]+)+)", sender)
    addr = m.group(0) if m else sender
    domain = m.group(1) if m else ""
    domains = [d.strip().lower() for d in re.split(r"[;,\s]+", str(inp.get("trusted_domains", "gov.uk"))) if d.strip()]
    extra = [s.strip().lower() for s in re.split(r"[;,\s]+", str(inp.get("trusted_senders", ""))) if s.strip()]
    ok = addr in extra or any(domain == d or domain.endswith("." + d) for d in domains)
    return ok, addr


def _hmrc_extract(inp: dict) -> dict:
    t = f"{inp.get('subject', '')} {inp.get('text', '')}"
    m = VAT_NUMBER_RE.search(t)
    return {"vat_number_candidate": f"GB{m.group(1)}{m.group(2)}{m.group(3)}"} if m else {}


def _hmrc_notice(inp: dict) -> dict:
    t = f"{inp.get('subject', '')} {inp.get('text', '')}".lower()
    ok, addr = _sender_trusted(inp)
    scam = _has(t, SCAM_WORDS)
    if not ok:
        return dict(answer="possible_phishing", confidence=0.95,
                    reasons=[f"sender {addr or 'unknown'} is not on a trusted HMRC domain"]
                    + ([f"typical scam wording: {', '.join(scam)}"] if scam else []), alternatives=[])
    if scam:
        return dict(answer="possible_phishing", confidence=0.8,
                    reasons=[f"trusted domain but typical scam wording: {', '.join(scam)}"], alternatives=[])
    if VAT_NUMBER_RE.search(t) or _has(t, ["vat registration number", "your vat number", "registration certificate",
                                            "you are now registered", "has been registered"]):
        return dict(answer="vat_number_issued", confidence=0.9, reasons=["mentions the VAT number or certificate"],
                    alternatives=[{"answer": "new_message_in_account", "confidence": 0.1}])
    if _has(t, ["more information", "further information", "we need", "please provide", "please send",
                "evidence", "supporting documents", "we could not", "we cannot process"]):
        return dict(answer="more_info_requested", confidence=0.88, reasons=["asks for information or evidence"],
                    alternatives=[{"answer": "new_message_in_account", "confidence": 0.12}])
    if _has(t, ["refused", "rejected", "cancelled", "not been able to register", "unable to register"]):
        return dict(answer="rejected_or_problem", confidence=0.85, reasons=["says the application has a problem"],
                    alternatives=[])
    if _has(t, ["we have received", "received your application", "acknowledg", "reference number"]):
        return dict(answer="application_received", confidence=0.86, reasons=["acknowledges the application"],
                    alternatives=[{"answer": "new_message_in_account", "confidence": 0.14}])
    if _has(t, ["new message", "sign in to", "online account", "business tax account", "secure message"]):
        return dict(answer="new_message_in_account", confidence=0.88,
                    reasons=["notice only: the content is in the HMRC online account"], alternatives=[])
    return dict(answer="other", confidence=0.5, reasons=["no clear HMRC notice type"], alternatives=[])


KINDS = {
    "uk_shipping_status": _uk_shipping_status,
    "brand_fit": _brand_fit,
    "customs_code": _customs_code,
    "origin_claim": _origin_claim,
    "message_intent": _message_intent,
    "return_condition": _return_condition,
    "reply_intent": _reply_intent,
    "hmrc_notice": _hmrc_notice,
}


# ---------------------------------------------------------------------------
# d1 (Liquid AI decision model): choice questions on the systemone API
# ---------------------------------------------------------------------------
D1_CHOICES = {
    "uk_shipping_status": (
        "How does this online store handle shipping to customers in the United Kingdom?",
        {
            "blocks_uk": "The store does not ship to the UK: it says so, or it only lists other countries or the EU.",
            "ships_with_fees": "The store ships to the UK, but the customer pays import VAT, duties or carrier fees on delivery.",
            "ships_all_in": "The store ships to the UK with all duties and taxes included: nothing to pay on delivery.",
            "unclear": "The text does not say how UK orders are handled.",
        },
    ),
    "message_intent": (
        "What does this customer message to an online store ask about?",
        {
            "customs_fee_complaint": "The customer was charged import VAT, duty, customs or a handling fee on delivery and complains.",
            "return_request": "The customer wants to return or exchange an item, or asks for a refund for a return.",
            "delivery_question": "The customer asks where the parcel is or when it will arrive.",
            "scope_question": "The customer asks about delivery to Northern Ireland, the Channel Islands or the Isle of Man.",
            "other": "Anything else.",
        },
    ),
    "return_condition": (
        "Based on the returns partner's inspection note, what should happen to this returned item?",
        {
            "resellable": "Unused or as new: it can be sold again at full price.",
            "second": "A small cosmetic issue: it can be sold as a second or after a minor fix.",
            "write_off": "Damaged or broken: it cannot be sold.",
        },
    ),
    "reply_intent": (
        "How did this prospect reply to our sales email?",
        {
            "interested": "Wants to talk, meet or learn more.",
            "question": "Asks a question before deciding.",
            "not_interested": "Declines.",
            "unsubscribe": "Asks not to be contacted again.",
            "unclear": "The intent is not clear.",
        },
    ),
    "hmrc_notice": (
        "This email came from HMRC (the UK tax authority) about a company's UK VAT registration. What is it about?",
        {
            "new_message_in_account": "A short notice that there is a new message or letter to read in the HMRC online account.",
            "more_info_requested": "HMRC asks for more information, documents or evidence before it can register the company.",
            "application_received": "HMRC confirms it received the VAT registration application, or gives a reference number.",
            "vat_number_issued": "The company is registered: the email gives or mentions the UK VAT number or certificate.",
            "rejected_or_problem": "HMRC refused, cancelled or cannot process the application.",
            "other": "Anything else, for example general news or a newsletter.",
        },
    ),
}

CODE_FAMILIES = {
    "pullover": (("pullover", "sweater", "jumper", "cardigan", "roll-neck", "crew-neck", "turtleneck"), {
        "6110.11": "Jerseys, pullovers, cardigans, knitted or crocheted, of wool",
        "6110.12": "Jerseys, pullovers, cardigans, knitted or crocheted, of Kashmir (cashmere) goats",
        "6110.19": "Jerseys, pullovers, cardigans, knitted or crocheted, of other fine animal hair",
        "6110.20": "Jerseys, pullovers, cardigans, knitted or crocheted, of cotton",
        "6110.30": "Jerseys, pullovers, cardigans, knitted or crocheted, of man-made fibres",
        "6110.90": "Jerseys, pullovers, cardigans, knitted or crocheted, of other textile materials",
    }),
    "scarf": (("scarf", "shawl", "muffler"), {
        "6117.10": "Shawls, scarves, mufflers, knitted or crocheted (any material)",
        "6214.10": "Shawls, scarves, mufflers, not knitted (woven), of silk",
        "6214.20": "Shawls, scarves, mufflers, not knitted (woven), of wool or fine animal hair",
        "6214.30": "Shawls, scarves, mufflers, not knitted (woven), of synthetic fibres",
        "6214.40": "Shawls, scarves, mufflers, not knitted (woven), of artificial fibres",
        "6214.90": "Shawls, scarves, mufflers, not knitted (woven), of other textile materials",
    }),
    "socks": (("sock",), {
        "6115.94": "Socks, knitted, of wool or fine animal hair",
        "6115.95": "Socks, knitted, of cotton",
        "6115.96": "Socks, knitted, of synthetic fibres",
        "6115.99": "Socks, knitted, of other textile materials",
    }),
    "gloves": (("mitten", "glove"), {
        "6116.91": "Gloves and mittens, knitted, of wool or fine animal hair",
        "6116.92": "Gloves and mittens, knitted, of cotton",
        "6116.93": "Gloves and mittens, knitted, of synthetic fibres",
        "6116.99": "Gloves and mittens, knitted, of other textile materials",
    }),
    "blanket": (("blanket", "throw", "travelling rug"), {
        "6301.20": "Blankets and travelling rugs, of wool or fine animal hair",
        "6301.30": "Blankets and travelling rugs, of cotton",
        "6301.40": "Blankets and travelling rugs, of synthetic fibres",
        "6301.90": "Other blankets and travelling rugs",
    }),
}
CODE_INSTRUCTIONS = ("Which HS subheading classifies this product? The fibre that predominates by weight decides "
                     "the material. Knitted or crocheted goods are in chapter 61, woven goods in chapter 62. "
                     "Do not assume anything the text does not state.")


def _d1_choice(instructions: str, options: dict, state: str) -> dict:
    body = {"model": D1_MODEL, "state": state,
            "questions": {"q": {"type": "choice", "instructions": instructions, "criteria": options}}}
    resp = httpx.post(D1_API_URL, json=body, timeout=20,
                      headers={"Authorization": f"Bearer {D1_API_KEY}", "Content-Type": "application/json"})
    resp.raise_for_status()
    data = resp.json()
    ans = (data.get("answers") or {}).get("q") or {}
    probs = {k: float(v) for k, v in (ans.get("probabilities") or {}).items()}
    choice = ans.get("choice") or (max(probs, key=probs.get) if probs else None)
    if choice is None:
        raise ValueError(f"unexpected d1 answer: {ans}")
    conf = float(ans.get("confidence", probs.get(choice, 0.0)))
    alts = [{"answer": k, "confidence": _r2(v)} for k, v in sorted(probs.items(), key=lambda kv: -kv[1]) if k != choice][:3]
    return dict(answer=choice, confidence=_r2(conf), alternatives=alts,
                reasons=[f"answered by {data.get('model', D1_MODEL)} (choice question)"])


def _d1_answer(kind: str, inp: dict):
    """Ask d1 when the kind is a judgment. Returns None when d1 is not used for this kind."""
    if kind in D1_CHOICES:
        instructions, options = D1_CHOICES[kind]
        if kind == "hmrc_notice":
            state = f"Subject: {inp.get('subject', '')}\n\n{inp.get('text', '')}"
        else:
            state = str(inp.get("text") or inp.get("note") or "")
        out = _d1_choice(instructions, options, state)
        if kind == "message_intent":  # amounts are read with code, not the model
            out["extracted"] = _message_intent(inp).get("extracted", {})
        return out
    if kind == "customs_code":
        text = (str(inp.get("product", "")) + ", " + str(inp.get("description", ""))).lower()
        for words, options in CODE_FAMILIES.values():
            if any(w in text for w in words):
                out = _d1_choice(CODE_INSTRUCTIONS, options, f"{inp.get('product', '')}: {inp.get('description', '')}")
                out["reasons"].append("HS 6-digit; the customs lead confirms the full UK commodity code on the UK Trade Tariff")
                return out
    return None  # brand_fit and origin_claim are rules on purpose; other products use the rules table


def _answer(kind: str, inp: dict, engine: str = "auto") -> dict:
    """d1 first (if a key is set and the kind is a judgment), rules otherwise or on any d1 error."""
    fn = KINDS.get(kind)
    if fn is None:
        return dict(answer="unknown", confidence=0.0, reasons=[f"unknown kind '{kind}'"], alternatives=[], engine="none")
    inp = inp or {}
    note = None
    if kind == "hmrc_notice":
        extracted = _hmrc_extract(inp)  # a VAT number is read with code, never by the model
        ok, _ = _sender_trusted(inp)
        rules_out = _hmrc_notice(inp)
        if not ok or rules_out["answer"] == "possible_phishing":  # sender and scam checks are plain code, before d1
            return {**rules_out, "extracted": extracted, "engine": "rules"}
        if engine != "rules" and D1_API_KEY:
            try:
                return {**_d1_answer(kind, inp), "extracted": extracted, "engine": "d1"}
            except Exception as exc:
                rules_out["reasons"].append(f"d1 unavailable ({type(exc).__name__}: {str(exc)[:120]}); rules used")
        return {**rules_out, "extracted": extracted, "engine": "rules"}
    if engine != "rules" and D1_API_KEY:
        try:
            out = _d1_answer(kind, inp)
            if out is not None:
                return {**out, "engine": "d1"}
        except Exception as exc:  # network, auth or format problem: fall back to the rules
            note = f"d1 unavailable ({type(exc).__name__}: {str(exc)[:120]}); rules used"
    out = fn(inp)
    if note:
        out["reasons"] = list(out.get("reasons", [])) + [note]
    return {**out, "engine": "rules"}


@mcp.tool
def classify(questions: list[dict], auto_threshold: float = 0.85, code_threshold: float = 0.90,
             engine: str = "auto") -> dict:
    """Answer typed decision questions with a confidence, using Liquid AI's d1 decision model.

    Send every question of one step in ONE call.
    engine: "auto" = d1 when a key is set (rules otherwise or on error); "rules" = rules only
    (reproducible classroom demo). Each result says which engine answered.
    questions: [{"id": "P01", "kind": "<kind>", "input": {...}}]
    kinds and their input:
      uk_shipping_status  {"text": shipping-page text}         -> blocks_uk | ships_with_fees | ships_all_in | unclear
      brand_fit           {"category", "made_in_notes", "uk_signals", "uk_status"} -> score 0..1
      customs_code        {"product", "description"}             -> HS 6-digit code
      origin_claim        {"made_in", "origin_declaration", "declaration_valid_until", "on_date", "warning_days"}
                                                                 -> eu_proven | eu_missing_proof | eu_proof_expired | non_eu
      message_intent      {"text", "fx_dkk_per_gbp"}             -> customs_fee_complaint | return_request | delivery_question | scope_question | other
                          (extracted.amount_gbp / amount_dkk when the message names an amount)
      return_condition    {"note", "photo_url"}                  -> resellable | second | write_off
      reply_intent        {"text"}                               -> interested | question | not_interested | unsubscribe | unclear
      hmrc_notice         {"from", "subject", "text", "trusted_domains", "trusted_senders"}
                                                                 -> new_message_in_account | more_info_requested | application_received
                                                                    | vat_number_issued | rejected_or_problem | other | possible_phishing
                          The sender and scam checks are plain code and run first: an untrusted sender is always
                          possible_phishing. extracted.vat_number_candidate is read with code when the email shows one.
    Each result has answer, confidence, alternatives, reasons and route ("auto" if confidence
    >= the threshold for that kind, else "human"). customs_code uses code_threshold.
    """
    qs = list(questions or [])
    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = list(pool.map(lambda q: _answer(q.get("kind", ""), q.get("input", {}), engine), qs))
    results = []
    for q, r in zip(qs, answers):
        kind = q.get("kind", "")
        threshold = code_threshold if kind == "customs_code" else auto_threshold
        if kind == "brand_fit":
            route = "auto" if r["confidence"] >= 0.6 else "human"
        else:
            route = "auto" if r["confidence"] >= threshold else "human"
        results.append({"id": q.get("id"), "kind": kind, **r, "route": route})
    return {"d1_configured": bool(D1_API_KEY), "model": D1_MODEL if D1_API_KEY else None, "results": results}


# ---------------------------------------------------------------------------
# landed_cost: prices (plain code)
# ---------------------------------------------------------------------------
def _nearest_point(x: float) -> int:
    return min(PRICE_POINTS, key=lambda p: (abs(p - x), -p))


def _next_point(p: int) -> int:
    later = [q for q in PRICE_POINTS if q > p]
    return later[0] if later else p + 50


def _keeps(price: float, duty_rate: float) -> float:
    return price / (1 + UK_VAT) * (1 - duty_rate)


def _price_one(s: dict, fx: float) -> dict:
    dk_price = float(s.get("dk_price_dkk") or 0)
    if dk_price <= 0:
        return {"sku": s.get("sku"), "error": "dk_price_dkk missing"}
    dk_net_gbp = dk_price / (1 + DK_VAT) / fx
    origin = str(s.get("origin_status") or "unknown")
    code = str(s.get("customs_code") or "")
    duty_rate = 0.0 if origin == "eu_proven" else _duty_rate(code)
    target = dk_net_gbp * (1 + duty_rate) * (1 + UK_VAT)
    price = _nearest_point(target)
    bumped = False
    while _keeps(price, duty_rate) < MIN_NET_SHARE * dk_net_gbp:
        price = _next_point(price)
        bumped = True
    net = price / (1 + UK_VAT)
    out = {
        "sku": s.get("sku"),
        "uk_price_gbp": price,
        "uk_vat_gbp": _r2(price - net),
        "duty_budget_gbp": _r2(net * duty_rate),
        "brand_keeps_gbp": _r2(_keeps(price, duty_rate)),
        "brand_keeps_dkk": _r2(_keeps(price, duty_rate) * fx),
        "danish_net_dkk": _r2(dk_price / (1 + DK_VAT)),
        "duty_rate_used": duty_rate,
        "flags": [],
    }
    cost = s.get("unit_cost_dkk")
    if cost not in (None, ""):
        keeps_dkk = _keeps(price, duty_rate) * fx
        out["margin_pct"] = _r2((keeps_dkk - float(cost)) / keeps_dkk)
    if str(s.get("code_status") or "") not in ("approved_auto", "approved_human"):
        out["flags"].append("customs code not approved yet: provisional price")
    if origin in ("eu_missing_proof", "eu_proof_expired"):
        alt = _price_one({**s, "origin_status": "eu_proven"}, fx)["uk_price_gbp"]
        out["price_if_origin_proven_gbp"] = alt
        out["flags"].append(f"origin proof missing: duty budgeted; price would be GBP {alt} with proof")
    elif origin == "non_eu":
        out["flags"].append("made outside the EU: duty applies when an order is over GBP 135; budgeted in the price")
    elif origin not in ("eu_proven",):
        out["flags"].append("origin unknown: duty budgeted")
    if duty_rate and code:
        out["flags"].append(f"duty rate {int(duty_rate * 100)}% is an EXAMPLE rate: confirm on the UK Trade Tariff")
    if bumped:
        out["flags"].append("rounded up: the nearest price point would leave the brand under 95% of its Danish net price")
    if dk_net_gbp > LOW_VALUE_LIMIT_GBP:
        fee = CARRIER_FEE_OVER_900_GBP if dk_net_gbp > 900 else CARRIER_FEE_GBP
        vat = dk_net_gbp * UK_VAT
        out["today_if_shipped_duties_unpaid"] = {
            "price_shown_gbp": _r2(dk_net_gbp),
            "paid_at_door_gbp": _r2(vat + fee),
            "customer_total_gbp": _r2(dk_net_gbp + vat + fee),
            "note": "20% VAT + Royal Mail GBP 8 fee at the door; plus duty if origin is not claimed",
        }
    return out


@mcp.tool
def landed_cost(skus: list[dict], fx_dkk_per_gbp: float = 8.6) -> dict:
    """UK all-in prices for many SKUs in one call. Never compute VAT, duty or prices yourself.

    skus: [{"sku", "dk_price_dkk" (Danish price incl. 25% VAT), "customs_code", "code_status",
            "origin_status" (eu_proven | eu_missing_proof | eu_proof_expired | non_eu), "unit_cost_dkk"}]
    Rule: all-in UK price = Danish net price in GBP, plus a duty budget for goods without
    proven EU origin, plus 20% UK VAT, rounded to the nearest price point ending in 9, but never
    leaving the brand under 95% of its Danish net price. Returns flags for anything provisional.
    """
    return {"fx_dkk_per_gbp": fx_dkk_per_gbp, "results": [_price_one(s, fx_dkk_per_gbp) for s in skus or []]}


# ---------------------------------------------------------------------------
# check_order
# ---------------------------------------------------------------------------
def _check_one(o: dict, eur_per_gbp: float) -> dict:
    lines = o.get("lines") or []
    net = sum(float(l.get("qty", 1)) * float(l.get("uk_price_gbp", 0)) / (1 + UK_VAT) for l in lines)
    over = net > LOW_VALUE_LIMIT_GBP
    duty, statement, issues = 0.0, False, []
    if over:
        for l in lines:
            origin = str(l.get("origin_status") or "unknown")
            value = float(l.get("qty", 1)) * float(l.get("uk_price_gbp", 0)) / (1 + UK_VAT)
            if origin == "eu_proven":
                statement = True
            else:
                d = value * _duty_rate(l.get("customs_code"))
                duty += d
                if origin in ("eu_missing_proof", "eu_proof_expired"):
                    issues.append(f"{l.get('sku')}: EU-made but no valid origin proof, so duty GBP {_r2(d)} will be charged")
    mode = str(o.get("ship_mode", "")).upper()
    status = "ok"
    if mode != "DDP":
        status = "error"
        issues.insert(0, "Shipped DAP: prices are all-in, so the carrier will bill the customer at the door. "
                         "Refund the charge and set the carrier default to DDP.")
    elif issues:
        status = "warning"
    if over:
        regime = "over GBP 135: import VAT and any duty at the border, paid by the brand under DDP"
        check = ("ok >£135: origin statement on invoice" if statement else "ok >£135") + (f", duty £{_r2(duty)}" if duty else ", no duty")
    else:
        regime = "GBP 135 or less: UK VAT charged at checkout, no customs duty; the UK VAT number must be on the customs data"
        check = "ok ≤£135: VAT at checkout, no duty"
    if status == "error":
        check = "ERROR: shipped DAP, customer will be billed"
    elif status == "warning":
        check = check + " (warning)"
    return {
        "order_id": o.get("order_id"),
        "consignment_net_gbp": _r2(net),
        "regime": regime,
        "duty_gbp": _r2(duty),
        "import_vat_gbp": _r2((net + duty) * UK_VAT) if over else 0.0,
        "origin_statement_needed": statement,
        "rex_needed": bool(statement and net * eur_per_gbp > REX_LIMIT_EUR),
        "status": status,
        "issues": issues,
        "customs_check": check,
    }


@mcp.tool
def check_order(orders: list[dict], eur_per_gbp: float = 1.15) -> dict:
    """Check many UK orders in one call (Great Britain only).

    orders: [{"order_id", "ship_mode" (DDP | DAP), "lines": [{"sku", "qty", "uk_price_gbp",
              "origin_status", "customs_code"}]}]   uk_price_gbp = the all-in price incl. VAT.
    Returns per order: consignment value excl. VAT, which regime applies (GBP 135 rule), duty,
    whether a statement on origin is needed on the invoice, whether REX is needed, issues, and
    a short customs_check text to write into the UK_Orders tab.
    """
    return {"results": [_check_one(o, eur_per_gbp) for o in orders or []]}


# ---------------------------------------------------------------------------
# stage3_check
# ---------------------------------------------------------------------------
@mcp.tool
def stage3_check(
    uk_orders_last_30_days: int,
    direct_parcel_cost_gbp: float,
    uk_parcel_cost_gbp: float,
    uk_pick_pack_gbp: float,
    uk_storage_gbp_month: float,
    pallet_cost_gbp_month: float,
    pallet_value_gbp: float = 0.0,
    eur_per_gbp: float = 1.15,
) -> dict:
    """Should the brand move its bestsellers into the shared UK warehouse (stage 3)?

    Saving per order = direct parcel from Denmark - (UK domestic parcel + pick & pack).
    Fixed monthly cost = storage + pallet freight and customs declaration.
    Recommends moving only when orders are at least 20% above break-even.
    """
    saving = direct_parcel_cost_gbp - (uk_parcel_cost_gbp + uk_pick_pack_gbp)
    fixed = uk_storage_gbp_month + pallet_cost_gbp_month
    if saving <= 0:
        return {"recommendation": "stay", "reason": "a UK parcel is not cheaper than a direct parcel", "saving_per_order_gbp": _r2(saving)}
    breakeven = math.ceil(fixed / saving)
    monthly = uk_orders_last_30_days * saving - fixed
    move = uk_orders_last_30_days >= breakeven * 1.2
    out = {
        "saving_per_order_gbp": _r2(saving),
        "fixed_cost_gbp_month": _r2(fixed),
        "breakeven_orders_per_month": breakeven,
        "uk_orders_last_30_days": uk_orders_last_30_days,
        "net_effect_gbp_month": _r2(monthly),
        "recommendation": "move bestsellers to the UK warehouse" if move else "stay with direct parcels",
        "reason": (f"{uk_orders_last_30_days} orders vs break-even {breakeven}: saves about GBP {_r2(monthly)} a month"
                   if move else f"{uk_orders_last_30_days} orders vs break-even {breakeven}: not yet worth it"),
        "note": "cost inputs are EXAMPLE values from Config; faster delivery and local stock are not priced in",
    }
    if move and pallet_value_gbp * eur_per_gbp > REX_LIMIT_EUR:
        out["rex_needed_before_first_pallet"] = True
    return out


# ---------------------------------------------------------------------------
# memory_plan / memory_recall (simple typed memory, rules only)
# ---------------------------------------------------------------------------
STOPWORDS = set("""a an the and or of to in on at for with from by is are was were be been being it its this
that these those as if then than so not no yes we you they he she i our your their his her them us me my
has have had do does did will would can could should may might about into over under after before only
also just very more most some any all each per via up out new one two og på til af er det en et vi ikke
med som der har kan""".split())
MONTHS = {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "january",
          "february", "march", "april", "june", "july", "august", "september", "october", "november", "december"}
TYPE_CUES = {
    "episodic": [r"\b\d{1,2} (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", r"\d{4}-\d{2}-\d{2}", r"\bwas\b",
                 r"\bwere\b", r"\barrived\b", r"\bcharged\b", r"\breplied\b", r"\bwrote\b", r"\bposted\b",
                 r"\bhappened\b", r"\bsent\b", r"\breceived\b", r"\bfixed\b", r"\breturned\b", r"\bdelayed\b",
                 r"\brefused\b", r"\bpitched\b", r"\bsigned\b", r"\btoday\b", r"\byesterday\b", r"\bfound\b", r"\bapproved\b"],
    "semantic": [r"\bis\b", r"\bare\b", r"\bmade in\b", r"\bknitted in\b", r"\bwoven\b", r"\bvalid until\b",
                 r"\busually\b", r"\btypically\b", r"\blead time\b", r"\bcosts?\b"],
    "procedural": [r"\bmust\b", r"\balways\b", r"\bnever\b", r"\bneeds?\b", r"\bgoes on\b", r"\bstep\b",
                   r"\brule\b", r"\bonly when\b", r"\brequires?\b", r"\bhas to\b", r"\bcheck\b", r"\bset\b"],
    "preference": [r"\bprefers?\b", r"\bwants?\b", r"\bwould rather\b", r"\blikes?\b", r"\bdoesn't want\b",
                   r"\bdoes not want\b", r"\basked us to\b", r"\bplease always\b"],
}
CAUSAL_CUES = ["because", "caused", "due to", "so that", "root cause", "led to", "therefore"]
TIME_WORDS = ["last time", "before", "since", "earlier", "previous", "again", "history", "when"]


def _tokens(text: str) -> set:
    words = re.findall(r"[£€]?\w[\w\-]*", str(text).lower())
    return {w for w in words if w not in STOPWORDS and w not in MONTHS and len(w) > 1 and not w.isdigit()}


def _entities(text: str) -> set:
    text = str(text)
    ents = set(re.findall(r"\b[A-Z]{1,3}-\d{2,4}\b", text))           # SKUs, orders, returns
    ents |= set(re.findall(r"£\s?\d+(?:\.\d+)?", text))                 # money
    for m in re.findall(r"\b([A-ZÆØÅ][\w&\-]+(?:\s+(?:&\s+)?[A-ZÆØÅ][\w\-]+)*)", text):
        if m.lower() not in MONTHS and m.lower() not in STOPWORDS and m not in ("On", "The", "Two", "A"):
            ents.add(m)
    return {e.lower() for e in ents}


def _overlap(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _type_scores(text: str) -> dict:
    low = str(text).lower()
    raw = {k: 0.2 + sum(1 for p in pats if re.search(p, low)) for k, pats in TYPE_CUES.items()}
    total = sum(raw.values())
    return {k: _r2(v / total) for k, v in raw.items()}


def _days_between(a: str, b: str) -> int | None:
    try:
        return abs((Date.fromisoformat(str(a)[:10]) - Date.fromisoformat(str(b)[:10])).days)
    except ValueError:
        return None


@mcp.tool
def memory_plan(observation: dict, candidates: list[dict]) -> dict:
    """Memory write path (rules, inspired by Jev-Mem): decide how a new observation is stored and linked.

    observation: {"text", "subject_id" (brand_id, prospect_id or desk), "source", "date"}
    candidates: existing Memory rows to compare with (same subject_id plus desk), each
                {"mem_id", "type", "text", "entities", "created_at"}.
    Returns the memory type with scores, entities, and relations (semantic, entity, temporal or
    causal) to existing memories whose score is at least 0.6, plus a ready-to-append row.
    """
    text = str(observation.get("text", ""))
    scores = _type_scores(text)
    mtype = max(scores, key=scores.get)
    ents = _entities(text)
    toks = _tokens(text)
    relations = []
    for c in candidates or []:
        ctext = str(c.get("text", ""))
        c_ents = {e.strip().lower() for e in re.split(r"[;,]", str(c.get("entities", ""))) if e.strip()} | _entities(ctext)
        sem = _overlap(toks, _tokens(ctext))
        ent = _overlap(ents, c_ents)
        score = 0.6 * sem + 0.4 * ent
        days = _days_between(observation.get("date", ""), c.get("created_at", ""))
        causal = any(k in text.lower() and k in ctext.lower() for k in CAUSAL_CUES)
        rel = "entity" if ent > sem else "semantic"
        if causal and score >= EDGE_THRESHOLD:
            rel = "causal"
        elif days is not None and days <= 14 and score >= EDGE_THRESHOLD and rel == "semantic":
            rel = "temporal"
        if score >= EDGE_THRESHOLD:
            relations.append({"to": c.get("mem_id"), "relation": rel, "score": _r2(score)})
    relations.sort(key=lambda r: -r["score"])
    row = {
        "subject_id": observation.get("subject_id", "desk"),
        "type": mtype,
        "text": text,
        "entities": "; ".join(sorted(ents)),
        "relations": "; ".join(f"{r['to']}:{r['relation']}:{r['score']}" for r in relations),
        "source": observation.get("source", ""),
        "created_at": observation.get("date", ""),
    }
    return {"engine": "rules", "type": mtype, "type_scores": scores, "entities": sorted(ents),
            "relations": relations, "edge_threshold": EDGE_THRESHOLD, "row_to_append": row}


@mcp.tool
def memory_recall(query: str, memories: list[dict], k: int = 3, on_date: str = "") -> dict:
    """Memory read path (rules): route a query to relation views and return the top-k memories.

    memories: Memory rows {"mem_id", "type", "text", "entities", "created_at", "valid_until"}.
    Views: semantic (always), entity (query names a brand, SKU, order or person), temporal
    ("last time", "before", "since"...), causal ("why", "cause"...). Expired memories
    (valid_until before on_date) are skipped. evidence_sufficient = top score >= 0.6.
    """
    q_toks, q_ents = _tokens(query), _entities(query)
    low = query.lower()
    views = ["semantic"]
    if q_ents:
        views.append("entity")
    if any(w in low for w in TIME_WORDS):
        views.append("temporal")
    if any(w in low for w in ("why", "cause", "because", "root")):
        views.append("causal")
    on = on_date or Date.today().isoformat()
    scored = []
    for m in memories or []:
        until = str(m.get("valid_until") or "")
        if until and until[:10] < on[:10]:
            continue
        mtext = str(m.get("text", ""))
        m_ents = {e.strip().lower() for e in re.split(r"[;,]", str(m.get("entities", ""))) if e.strip()} | _entities(mtext)
        s = _overlap(q_toks, _tokens(mtext))
        if "entity" in views:
            s = 0.7 * s + 0.3 * _overlap(q_ents, m_ents)
        if "temporal" in views and str(m.get("type")) == "episodic":
            s += 0.05
        if "causal" in views and any(c in mtext.lower() for c in CAUSAL_CUES):
            s += 0.05
        scored.append({"mem_id": m.get("mem_id"), "type": m.get("type"), "score": _r2(min(s, 1.0)), "text": mtext})
    scored.sort(key=lambda r: -r["score"])
    top = [r for r in scored[: max(k, 1)] if r["score"] > 0]
    return {"engine": "rules", "views_activated": views, "results": top,
            "evidence_sufficient": bool(top and top[0]["score"] >= EDGE_THRESHOLD)}


# ---------------------------------------------------------------------------
@mcp.tool
def list_rules() -> dict:
    """The rules and EXAMPLE rates this server applies (Great Britain only)."""
    return {
        "scope": "Great Britain (England, Scotland, Wales). Northern Ireland, Channel Islands and Isle of Man are out of scope.",
        "uk_vat": UK_VAT,
        "low_value_limit_gbp": LOW_VALUE_LIMIT_GBP,
        "at_or_below_limit": "seller must be UK VAT registered and charge UK VAT at checkout; no customs duty",
        "above_limit": "import VAT and duty at the border (paid by the brand under DDP); no duty on goods with EU preferential origin claimed by a statement on origin",
        "rex_limit_eur": REX_LIMIT_EUR,
        "carrier_fees_charged_to_customer_gbp": {"royal_mail": CARRIER_FEE_GBP, "parcelforce": 12.0, "goods_over_900": CARRIER_FEE_OVER_900_GBP},
        "example_duty_rates_by_heading": EXAMPLE_DUTY_RATES,
        "default_duty_rate": DEFAULT_DUTY_RATE,
        "pricing": "nearest price point ending in 9; brand keeps at least 95% of its Danish net price",
        "memory_edge_threshold": EDGE_THRESHOLD,
        "warning": "Simplified course-demo rules. EXAMPLE rates must be confirmed on the UK Trade Tariff.",
    }


if __name__ == "__main__":
    mcp.run()
