# Form Lab — first implementation

Separate form-only training and prediction engine for Australia (`AU`), Great
Britain (`GB`), South Africa (`ZA`) and France (`FR`). Located entirely inside
`form_lab/`; the existing app, collectors and predictor weights are unchanged.

**Status:** the pipeline runs and is tested on synthetic races. It has not been
trained on the droplet's archive or deployed. No accuracy improvement is claimed.
The source available here is the `horserace` branch at
`9231bf6bfb042e29d047d8acf6fe298793496a55`, dated 7 September 2026. It is older
than the observed October droplet app. This is an integration candidate, not an
assertion that this repository matches the running server.

## What is implemented

- Strict full-field race contract, explicit availability evidence, stable IDs,
  separate labels, scratching/DNF/DQ/dead-heat handling and data-quality audit.
- One causal feature builder for historical replay and current cards. Horse
  history and connection statistics only incorporate results published before
  the prediction snapshot. Fixed feature allowlist excludes prices, Kash,
  market rank, tips, verdicts and source-generated predicted rankings.
- Regularised race-conditional logit baseline and **CatBoost QuerySoftMax**.
  The latter optimises the within-race winner distribution, not independent
  runner classifications. See [CatBoost's official objective definition](https://catboost.ai/docs/en/concepts/loss-functions-ranking#QuerySoftMax).
- Four chronological blocks: base training → blend weights → temperature
  calibration → untouched historical test. All runners and meetings stay in
  their block. Earlier test results can update form state once published;
  fitted parameters stay frozen throughout the test.
- A learned nonnegative two-model geometric blend and separately fitted
  temperature. No fixed 50/50 model weighting and no market component.
- Race log loss, race-summed Brier score, top-pick wins, winner mass covered by
  the top three, runner reliability bins and course/surface subgroups. Paired
  log-loss intervals resample whole meetings, and are withheld below 10 meetings.
- Native CatBoost model plus JSON coefficients/manifest; no pickle loading.
  Schema, feature list, input hash, training boundaries, package versions,
  parameter checksum and file checksum travel with the model.
- Exact Plackett–Luce top-three probabilities. These assume a strict complete
  order; they are **not** a separately trained place model, DNF model or bookmaker
  place probability. Winner dead heats use fractional winner mass in training.
  Place calibration and tied/non-completed finishing orders remain future work.
- Append-only SQLite snapshots/predictions, distinct historical replay versus
  genuinely pre-off forecast creation timestamps, and a separate Streamlit UI
  with venue/race tabs and descending combined probability.

**Research models still to add after the data audit:** the dynamic hierarchical
Bayesian horse-ability model with partial pooling for connections and suitability;
then a fitted correlated pace/performance model if forward tests justify it.
The linear baseline here is not represented as that Bayesian model. The first
experiment establishes an auditable benchmark before adding that complexity.

## Run and verify

From the repository root, using a **separate virtual environment**:

```bash
python3 -m venv .venv-form
.venv-form/bin/pip install -r form_lab/requirements-dev.txt
.venv-form/bin/python -m pytest form_lab/tests -q
.venv-form/bin/python -m form_lab --help
```

Tested with Python 3.12, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0 and CatBoost
1.2.10. Tests generate explicitly synthetic races, including complete
train/blend/calibration/test blocks; synthetic scores are not published as
racing performance. No trained racing weights are shipped.

## First droplet action: inventory, then map its real schema

The droplet terminal and historical files were not accessible during this build.
Run this **read-only, standard-library-only** inventory against the actual app
directory after placing this branch on the server:

```bash
python3 form_lab/inventory.py /actual/path/to/deployed/app > form-lab-inventory.json
```

It lists data files and SQLite table/column names, not row values or credentials.
It does not alter the existing database, restart services or trigger collection.
Review it alongside a small redacted raw race snapshot and a complete result
record. Those determine the deployed-schema adapter; do not guess table names.
Database copies must be consistent backups, including any WAL state.

## Existing collector findings

| Adapter in this branch | Reusable data | Integration gap |
|---|---|---|
| Ladbrokes | Cards, country/venue/date lookup, runners, draw, weights, declared connections | `Entry` drops entrant/horse IDs; parser chooses allocated/total weight without recording basis; does not preserve raw observation timestamps |
| Sporting Life | UK cards, official rating, claim, historical runs | Full stable identity mapping needs raw payloads; current lifetime aggregates must not become historical features; post-race ride status is not a pre-race scratching snapshot |
| PMU | France cards, discipline/source information, detailed past runs | Require explicit flat/jumps/trotting mapping; `tempsDuPremier` is winner time, not individual horse time; preserve source horse/race IDs |
| Winning Form | SA cards, merit ratings, mass, recent runs | `fetch()` searches current cover listings, not a demonstrated historical archive; parser includes source-predicted rankings which this engine excludes |

The old `common.RaceCard` has no persistent horse ID or general capture time.
`form_lab.bridge.from_legacy_card` therefore requires explicit race metadata and
verified per-runner IDs from the raw collector. It copies only approved fields.
It does **not** guess identities from horse names or turn days-ago strings into
precise result publication times. It omits ambiguous net carried weights unless
the caller supplies confirmed values. Dates alone are insufficient to sequence
same-day information safely.

The historical/live strategy should share the raw schema and feature code:

1. Reuse the deployed source clients and scheduling; capture immutable raw
   responses with source URL/ID, retrieval time and payload hash.
2. For current races, create canonical snapshots at the chosen prediction
   horizon, then again when the declared field, going or connections change.
   Keep earlier predictions; record which field each prediction saw.
3. For historical races, enumerate **all races and complete fields** from a
   confirmed historical endpoint/archive, then join official outcomes. Today's
   runners' past performances alone are a selected sample, not a training set.
4. Record snapshot evidence. If fetched after the event, use only documented
   source-as-of records or reconstruct features from earlier dated starts.
   Current career totals, updated ratings and post-race comments cannot be
   backdated. Mark unverifiable races unavailable for the experiment.
5. Feed the same canonical records into the engine. Add source-specific fixture
   tests when the actual deployed adapters are available.

Historical endpoint coverage and permissions still need checking against the
deployed collectors. This branch does not launch a speculative bulk scraper.

## Canonical data contract

UTF-8 JSONL: **one complete race per line**, one selected prediction snapshot per
race per experiment. Persistent identity namespaces should prevent collisions
between sources; keep a reviewed cross-source ID map for travelling horses.

| Level | Required data |
|---|---|
| Race identity | `schema_version: 1`, `race_id`, `meeting_id`, `course_id`, `country`, `discipline`, `distance_m`, `field_complete: true` |
| Timing/provenance | `scheduled_off`, `prediction_at`, `available_at`, `captured_at`, `availability_basis`, `source`, `source_ref` |
| Runner | `runner_id`, `horse_id`, `horse_name`, `status` (`active`/`scratched`); all declarations retained |
| Useful pre-race fields | Race surface, going, class; horse age/sex, draw, **net carried weight in kg**, official local rating, jockey/trainer IDs, gear |
| Attached prior start | Stable `race_id`, `off_at`, result `available_at`, country, discipline, distance, field size, status and official finishing position; optional course/surface/going/beaten lengths |
| Outcome | `result.available_at`, `result.status`, and every declared runner's ID, status and official finishing position |

All timestamps include a timezone. `available_at <= prediction_at < scheduled_off`.
For `availability_basis: archived_capture`, capture must also precede prediction.
For `source_asof`, a later retrieval is allowed **only with source evidence of
what the fields contained at the recorded earlier time**. Code checks timestamp
consistency; it cannot independently prove a supplied timestamp is genuine.
`available_at` covers all supplied pre-race fields, including attached history.

Optional unknown numeric values are JSON `null`, never zero. Surface/going/class
values stay native to the country; use explicit mappings with preserved raw
values upstream. `GB` does not include Ireland. Separate model bundles are
required for each country and discipline; no universal rating conversion.

Results have status `official`, `abandoned` or `void`. Outcome status is
`finished`, `dnf`, `dq` or `non_runner`; only finishers receive positive ranks.
Dead heats use competition ranks, e.g. `1,1,3`. Winners' **beaten** lengths are
zero; winning margin is a different quantity. A runner active in the snapshot
but later scratched excludes the entire race from this fixed-field training
experiment; it is not silently removed after seeing the result.

Time/sectional fields are deliberately absent from this first feature set until
the feed audit establishes individual versus winner times, measured versus
estimated times, section lengths and appropriate track/distance/going pars.
See `schema.py` for the exact allowlist, and `tests/helpers.py` for a complete
synthetic example. A field outside the contract raises an error rather than
becoming an accidental input feature.

## Audit and train real data

```bash
python -m form_lab audit /var/lib/form-lab/history.jsonl
python -m form_lab train /var/lib/form-lab/history.jsonl \
  --country AU --discipline flat \
  --blend-start 2026-07-01T00:00:00Z \
  --calibration-start 2026-08-01T00:00:00Z \
  --test-start 2026-09-01T00:00:00Z \
  --output /var/lib/form-lab/models/AU-flat-experiment-001
```

The dates above are proposed experiment boundaries, not evidence that the
archive is present or usable. Select boundaries once after coverage audit and
before inspecting final-test performance. Run separate country/discipline
experiments. Twenty races per block is an engineering minimum only; credible
country/subgroup comparisons need much larger samples and multiple subsequent
chronological test windows.

Outputs: `manifest.json`, `catboost.cbm`, `evaluation.json` and
`holdout_predictions.csv`. Existing output directories are refused. Do not
repeatedly adjust choices using the same held-out month. This initial command
implements one chronological experiment; expanding-window orchestration,
automatic model selection and continuous retraining are not enabled yet.

## Predict and record

Cards use the **same schema**, with `result` absent. Supply the same canonical
historical archive so horse and connection state can be rebuilt as of each card.

```bash
python -m form_lab predict /var/lib/form-lab/cards.jsonl \
  --model /var/lib/form-lab/models/AU-flat-experiment-001 \
  --history /var/lib/form-lab/history.jsonl \
  --ledger /var/lib/form-lab/form-lab.sqlite \
  --output /var/lib/form-lab/predictions-unique-run-id.jsonl
```

The model refuses country/discipline mismatches, target results in live input,
postdated snapshots and predictions earlier than its learned-information
cutoff. Each forecast includes model/input/feature hashes, per-model and combined
win probabilities, top-three estimates, missing-data counts and disagreement.
Explanations are contributions to the **linear baseline**, not causal claims or
CatBoost explanations. Displaying a forecast in Streamlit is read-only; the CLI
is the recording path. Prediction history is not yet a scoring/settlement daemon.

## Separate droplet deployment

Use `/opt/form-lab` for code and `/var/lib/form-lab` for the new data/models;
verify real paths and resource availability before installation. Use a dedicated
service user, virtual environment and local port. Example systemd and Nginx
fragments are in `deploy/`; they are not installed or active. Port 8517 and
`/formlab/` are proposed values. Check existing listeners and reverse-proxy
configuration first. Training defaults to two CatBoost threads; set BLAS thread
limits for the worker and benchmark memory/CPU before scheduling on the droplet.

Local dashboard command:

```bash
FORM_LAB_MODELS=/var/lib/form-lab/models \
FORM_LAB_CARDS=/var/lib/form-lab/cards.jsonl \
FORM_LAB_HISTORY=/var/lib/form-lab/history.jsonl \
python -m streamlit run form_lab/app.py --server.port=8517
```

Next integration milestone: obtain the actual deployed source/schema and a
consistent historical export; verify timestamp/identity coverage; implement
that export adapter; train the four country baselines; review untouched-period
results; then connect a separate prospective collection/prediction job. Only
after that baseline should we add and assess the dynamic Bayesian model.
