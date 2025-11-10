# Northstar Runbook

This runbook captures the operational knowledge required to run and maintain the nightly pipeline locally and in containerized environments.

## Environment variables and secrets

The application configuration is entirely driven by environment variables. All variables marked **required** must be present before the jobs will start; the configuration loader raises `MissingEnvironmentVariableError` when any required key is absent.【F:app/src/config.py†L10-L100】

| Variable | Required | Description |
| --- | --- | --- |
| `FINNHUB_TOKEN` | ✅ | API key for Finnhub. Mandatory for universe construction and quote enrichment.【F:app/src/config.py†L48-L99】 |
| `TWELVEDATA_KEY` | ➖ | Optional TwelveData API key used as a fallback data provider when present.【F:app/src/config.py†L48-L99】【F:app/src/data/universe.py†L75-L83】 |
| `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_DB`, `MYSQL_USER`, `MYSQL_PASSWORD` | ✅ | Connection details for the MySQL instance backing the `symbols`, `daily_metrics`, `valuations`, and `picks` tables.【F:app/src/config.py†L50-L94】【F:app/src/storage/db.py†L19-L49】 |
| `REDIS_URL` | ✅ | Redis connection string used by the caching and rate-limit helpers. Defaults to `redis://localhost:6379/0` if unset, but production should supply an explicit endpoint.【F:app/src/config.py†L55-L95】【F:app/src/data/cache.py†L13-L29】 |
| `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID` | ✅ | Credentials for the Telegram bot that receives the daily summary message.【F:app/src/config.py†L55-L99】【F:app/src/notify/telegram.py†L86-L99】 |
| `TIMEZONE` | ➖ | Olson timezone string for localized timestamps. Defaults to `America/Vancouver` when omitted.【F:app/src/config.py†L95-L99】 |
| `UNIVERSE_MIN_DOLLAR_VOL` | ➖ | Minimum trailing dollar volume used while filtering tradable symbols. Defaults to `300000` and must be numeric.【F:app/src/config.py†L78-L84】【F:app/src/data/universe.py†L174-L195】 |
| `ROTATION_SECTORS` | ✅ | Rotation schedule definition controlling sector inclusion during universe construction. Accepts JSON, newline, or delimiter-based formats (see “Rotating sectors & filters”).【F:app/src/config.py†L60-L100】【F:app/src/data/universe.py†L26-L72】 |
| `NIGHTLY_OUTPUT_DIR` | ➖ | Optional override for where nightly exports are written. Defaults to `./nightly_output` relative to the working directory.【F:app/src/jobs/nightly.py†L39-L40】【F:app/src/jobs/nightly.py†L321-L399】 |

**Secret management tips**

* Commit a `.env.sample` (without real secrets) alongside this runbook to document required keys for new environments.
* In Kubernetes, mirror these variables inside a Secret referenced by `CronJob` or `Deployment` manifests (see `app/k8s/cronjob-nightly.yaml`).
* When running locally, prefer an `.env` file loaded by your shell (`set -a; source .env; set +a`) so both Poetry and Docker invocations receive the same configuration.

## Running the nightly job locally

1. Install dependencies inside `app/` once: `cd app && poetry install`.
2. Export or source the environment variables above.
3. Execute the job with Poetry: `poetry run nightly`. This calls `src.jobs.nightly.main`, which orchestrates universe construction, feature engineering, CSV exports, and Telegram notifications for the current date.【F:app/pyproject.toml†L18-L25】【F:app/src/jobs/nightly.py†L321-L420】
4. Outputs land under `${NIGHTLY_OUTPUT_DIR:-./nightly_output}/YYYYMMDD/` as `daily_metrics.csv`, `valuations.csv`, and `picks.csv` by default.【F:app/src/jobs/nightly.py†L321-L420】

**Smoke-test checklist after local runs**

* Inspect the latest dated folder for non-empty CSVs.
* Tail the structured logs to confirm fetch batches completed and Telegram send succeeded (`event=nightly status=completed`).
* If Redis is running locally, verify the job didn’t leave stuck rate-limit keys by checking `redis-cli KEYS rate_limit:*`.

## Ad-hoc CLI commands

Northstar exposes a lightweight CLI for one-off calculations and integration tests. Ensure dependencies are installed (`cd app && poetry install`) and required environment variables are loaded before invoking the commands.

* Calculate a single ticker snapshot with Finnhub/TwelveData data and console output:

  ```bash
  poetry run python -m src.cli calc --symbol KXS.TO
  ```

* Run the same calculation while pushing the bullet summary to Telegram:

  ```bash
  poetry run calc --symbol AAPL --send-telegram
  ```

* Verify Telegram credentials independently of the calc workflow:

  ```bash
  poetry run tgtest --text "Ping"
  ```

The `calc` command accepts `--ds YYYY-MM-DD` to evaluate historical contexts and `--verbose` for raw payloads, while `tg-test` falls back to a green-check confirmation when `--text` is omitted.【F:app/src/cli.py†L37-L302】

## Running the nightly job in Docker

1. Build the image from the `app/` directory: `docker build -t northstar-nightly app` (the Dockerfile installs Poetry and sets the default command to `poetry run nightly`).【F:app/Dockerfile†L1-L33】
2. Prepare an environment file (for example `env/nightly.env`) containing the required variables listed earlier.
3. Launch the container, mounting a host directory for outputs:
   ```bash
   docker run --rm \
     --env-file env/nightly.env \
     -v "$PWD/nightly_output:/app/nightly_output" \
     northstar-nightly
   ```
   * Adjust the bind mount if you override `NIGHTLY_OUTPUT_DIR`—the directory must exist and be writable by the non-root `appuser` defined in the Dockerfile.【F:app/Dockerfile†L15-L32】【F:app/src/jobs/nightly.py†L321-L399】
4. Review container logs for `event=nightly status=completed` and confirm the host volume received the dated exports.

For Kubernetes or other schedulers, use the same image and propagate the environment variables via Secrets/ConfigMaps. Mount a persistent volume to preserve exports if needed.

## Rotating sectors and adjusting filters

*Rotation schedule (`ROTATION_SECTORS`)*

`ROTATION_SECTORS` controls which sectors are always included (`core`) and which rotate by weekday. Accepted formats include JSON objects with `core` + `rotation`, JSON arrays of arrays, or delimiter-based strings using `|`/`;` for groups and commas within groups. Empty tokens are ignored, and a simple comma-separated list is treated as individual groups.【F:app/src/data/universe.py†L26-L72】

Example definitions:

* JSON object: `{"core": ["Technology"], "rotation": [["Energy", "Utilities"], ["Financials"]]}`
* Pipe-delimited groups: `Energy,Utilities | Financials,Industrials | Healthcare`

During universe construction, the helper selects the current weekday’s rotation bucket while always including the `core` sectors, deduplicating symbols in order.【F:app/src/data/universe.py†L198-L220】

*Universe filters*

Symbols failing minimum price ($2), dollar volume (default $300k), or market-cap ($150M) checks are excluded. Adjust these criteria by raising or lowering `UNIVERSE_MIN_DOLLAR_VOL` or by modifying `_apply_filters` for deeper policy changes.【F:app/src/data/universe.py†L174-L195】 Keep in mind that loosening filters increases API usage and downstream load.

To test new rotation or filter settings safely:

1. Update the relevant environment variable(s) in a local `.env` file.
2. Run `poetry run nightly` and inspect the resulting `picks.csv` for sector representation.
3. Validate the `symbols` table (or CSV export) contains the expected sectors before promoting the change to production.

## Common errors and remediation

| Symptom | Likely cause | Mitigation |
| --- | --- | --- |
| `MissingEnvironmentVariableError: Environment variable 'MYSQL_HOST' is required but was not set.` | One or more required variables absent. | Verify your shell, `.env`, and Docker/Kubernetes manifests set all required keys. Re-run after exporting the missing variable.【F:app/src/config.py†L10-L100】 |
| `sqlalchemy.exc.OperationalError: (pymysql.err.OperationalError) (1045, "Access denied...")` | Invalid MySQL credentials or host unreachable. | Confirm `MYSQL_*` values, test connectivity with `mysql` CLI, and ensure the database allows connections from your runner IP. The connection URL is assembled in `create_engine`, so mismatched ports/users cause immediate auth failures.【F:app/src/storage/db.py†L19-L49】 |
| Burst of HTTP 429 responses from Finnhub/TwelveData | Upstream rate limits exceeded during symbol enrichment. | Reduce request concurrency by lowering the `TokenBucketLimiter` rate (default 5 ops/sec) or temporarily shrinking the rotation set. The limiter gates each API call in `fetch_symbol_metrics`, so even modest reductions help.【F:app/src/jobs/nightly.py†L76-L200】【F:app/src/jobs/nightly.py†L333-L355】 |
| `telegram.error.Unauthorized: Forbidden: bot was blocked by the user` or similar | Invalid Telegram bot token or chat ID mismatch. | Regenerate the bot token, ensure the chat ID corresponds to a user/group that has started the bot, and retry. Failures are logged in `send_daily_message` with `event=telegram_send status=failed`.【F:app/src/notify/telegram.py†L86-L99】 |
| Stale Redis data or connection errors | `REDIS_URL` incorrect or Redis unavailable. | Update the URL or start a local Redis instance. The cache helper lazily initializes a global client based on `REDIS_URL`.【F:app/src/data/cache.py†L19-L29】 |

## Backfilling a single day safely

Because `nightly.main()` always uses `date.today()`, run a controlled backfill by invoking the underlying helpers for a specific date instead of hacking system time. The snippet below executes the same steps for an arbitrary day while isolating outputs in a custom directory.

```bash
TARGET=2024-01-15
OUTPUT_ROOT="$PWD/backfill_output"
mkdir -p "$OUTPUT_ROOT"
poetry run python - <<'PY'
from datetime import date
from pathlib import Path
from app.src.jobs import nightly

run_date = date.fromisoformat("${TARGET}")
output_dir = Path("${OUTPUT_ROOT}") / run_date.strftime("%Y%m%d")
nightly._setup_logging()
nightly.ensure_schema(output_dir)
universe = nightly.build_universe(run_date)
if not universe:
    raise SystemExit("Universe build returned no symbols; aborting")
limiter = nightly.TokenBucketLimiter(rate=3)
fetched = [nightly.fetch_symbol_metrics(entry, limiter=limiter, seed=int(f"{run_date:%Y%m%d}{idx}"))
           for idx, entry in enumerate(universe)]
fetched = [row for row in fetched if row]
if not fetched:
    raise SystemExit("No symbol data fetched; aborting")
frame = nightly._prepare_feature_frame(fetched)
frame["price"] = frame["price"].astype(float)
quality = nightly.compute_quality_scores(frame[["roe", "free_cash_flow", "net_debt_ebitda", "interest_coverage", "margin_trend_5y"]])
frame["quality_score"] = quality
multiples = nightly.compute_multiple_scores(frame[["pe_ttm", "ps_ttm", "ev_ebit", "fcf_yield", "sector"]])
frame["multiple_score"] = multiples
frame["momentum_score"] = nightly.momentum_penalty(frame["price"], frame["sma100"], frame["sma200"], frame["drawdown_1m"])
composite = nightly.composite_score(frame)
frame["composite_score"] = composite["composite_score"]
frame["sector_median_ev_ebit"] = nightly._compute_sector_medians(frame)
daily_rows = nightly._build_daily_rows(frame, run_date)
valuation_rows = nightly._build_valuation_rows(frame, run_date)
(nightly.DEFAULT_OUTPUT_DIR / "dummy").exists()  # no-op; ensures module attributes stay referenced
nightly._write_csv(output_dir / "daily_metrics.csv", daily_rows)
nightly._write_csv(output_dir / "valuations.csv", valuation_rows)
picks_input = frame.reset_index()
picks, notes = nightly.select(picks_input)
picks_rows = [{
    "ds": run_date.isoformat(),
    "rank_order": rank,
    "symbol": row.get("symbol"),
    "target_price": float(row.get("target_price", float("nan"))),
    "stop_loss": float(row.get("stop_loss", float("nan"))),
    "notes_json": nightly.json.dumps(notes),
    "sent_telegram": 0,
} for rank, (_, row) in enumerate(picks.iterrows(), start=1)]
nightly._write_csv(output_dir / "picks.csv", picks_rows)
print(f"Backfill complete for {run_date} -> {output_dir}")
PY
```

Key safety notes:

* Use a distinct `OUTPUT_ROOT` to avoid clobbering current-day exports.【F:app/src/jobs/nightly.py†L321-L399】
* Lower the token bucket rate during backfills to stay within external API quotas.【F:app/src/jobs/nightly.py†L76-L200】
* Review the generated CSVs before manually loading them into MySQL. If they look correct, import them via the storage helpers (`insert_daily_metrics`, `insert_valuations`, `insert_picks`) or your ETL tooling.【F:app/src/storage/db.py†L52-L120】
* Skip the Telegram send to avoid retroactive notifications—this snippet omits `_send_telegram` entirely.【F:app/src/jobs/nightly.py†L317-L420】

With these procedures documented, future operators can confidently run the nightly pipeline, troubleshoot common issues, and perform ad-hoc backfills without rediscovering the workflow.
