# uk-desk MCP server

Small MCP server for the **UK Desk** agent (DTU course demo).

| Tool | What it does | How it decides |
|---|---|---|
| `classify` | Answers typed decision questions with a confidence: a store's UK shipping status, customs code, customer-message intent, return condition, prospect reply intent, HMRC email type (`hmrc_notice`, with a plain-code phishing check first). Also brand fit and origin claim | **Liquid AI d1** (choice questions on the systemone API) when `D1_API_KEY` is set. Brand fit and origin claim are rules on purpose (a score formula and a legal rule). Without a key, or if d1 fails, rules answer, and each result says `engine: d1` or `engine: rules` |
| `landed_cost` | UK all-in prices per SKU (VAT, duty budget, price points) | Plain code: prices and taxes must be exact |
| `check_order` | Per order: the £135 rule, duty, statement on origin, REX, DAP mistakes | Plain code |
| `stage3_check` | Monthly break-even for the shared UK warehouse | Plain code |
| `memory_plan` / `memory_recall` | Simple typed memory (episodic, semantic, procedural, preference) with links when the score is 0.6 or more; storage is the `Memory` tab | Rules, inspired by the Jev-Mem paper (no model) |
| `list_rules` | Shows the rules and EXAMPLE rates | — |

Scope: Great Britain only. Rates marked EXAMPLE must be checked on the UK Trade Tariff before real use.

## Turn on d1

1. Create an account and an API key on Liquid AI's developer platform (docs.liquid.ai). There's no free tier, but the starting credit covers the demo: about $0.02 per million input tokens, so a whole demo costs well under one cent.
2. In Prefect Horizon, add these environment variables to the server:
   - `D1_API_KEY` = your key
   - `D1_MODEL` = `d1`
   - `D1_API_URL` only if your docs show a different address than `https://api.liquid.ai/decisions/v1/systemone`
3. Redeploy.

**If you get access another way:** AI/ML API (`https://api.aimlapi.com/v1/decisions`, model `liquid/d1`), OpenRouter and Vercel AI Gateway also serve d1. Set `D1_API_URL` and `D1_MODEL` to their values, and check their docs first, because the request format may differ from Liquid's own API.

**Check that it works:** the `classify` result shows `"d1_configured": true`, and each judgment says `"engine": "d1"`. If it says `rules` with a note "d1 unavailable (...)", the note tells you why (wrong key, URL or model name).

## Deploy on Prefect Horizon (same as site-weather)

1. Create a GitHub repository (e.g. `uk-desk-mcp`) and upload `server.py` and `requirements.txt`.
2. In Prefect Horizon, create a new server from the repository, with entrypoint `server.py:mcp`.
3. Add the environment variables above and deploy.
4. Copy the server URL that ends in `/mcp`.

## Connect in LangSmith Fleet

1. Connections → Add connection → **+ Add custom MCP**.
2. Name: `uk-desk`. URL: your Horizon `/mcp` URL. Authentication: **OAuth 2.1 (Auto)**. If you see "Bearer token required", this is the fix; or turn off authentication for the server in Horizon.
3. Set all seven tools to **Auto**. They only answer and calculate; they cannot send or change anything.

## Quick tests in the Fleet chat

```
Call classify with one question: id "t1", kind "uk_shipping_status",
input {"text": "Vi sender desværre ikke til Storbritannien efter Brexit."}
```
Expect `blocks_uk` and `engine: d1`. With `engine: rules` the confidence is exactly 0.93.

```
Call landed_cost with skus [{"sku": "NK-01", "dk_price_dkk": 1499, "origin_status": "eu_proven",
"customs_code": "6110.11", "code_status": "approved_auto"}] and fx_dkk_per_gbp 8.6
```
Expect `uk_price_gbp` 169, and today's comparison: £35.89 paid at the door.
