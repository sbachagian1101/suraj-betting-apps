"""Explicit bridge for common.RaceCard from the existing horserace collectors.

The old dataclasses discard persistent IDs and observation times. The caller
must preserve these from raw payloads and provide them here. Names/saddlecloths
are never promoted into longitudinal horse IDs. Current career/connection
aggregates, source predictions, tips, odds and free text are not copied.
"""
from __future__ import annotations

from copy import deepcopy

from . import SCHEMA_VERSION
from .schema import require, validate

COUNTRY_MAP = {"AUS": "AU", "UK": "GB", "SA": "ZA", "FR": "FR"}


def from_legacy_card(card, metadata: dict, identities: dict) -> dict:
    """identities maps saddlecloth strings to IDs and dated canonical history.

metadata contains the race-level contract (including discipline and availability
evidence). identities['3'] = {runner_id, horse_id, jockey_id, trainer_id,
horse_name, carried_weight_kg, history}. Require exact horse-name agreement with
the already verified ID match. A cross-source fuzzy match is not sufficient.
The caller supplies net carried weight because old feed weight/claim semantics
are inconsistent; absent confirmation it remains missing.
"""
    permitted = {"race_id", "meeting_id", "course_id", "scheduled_off", "prediction_at",
                 "available_at", "captured_at", "availability_basis", "source", "source_ref",
                 "field_complete", "discipline"}
    require(set(metadata) == permitted, f"metadata needs exactly {sorted(permitted)}")
    require(card.country in COUNTRY_MAP, f"Unsupported country {card.country}; Ireland needs a separate adapter/model")
    require(len({e.number for e in card.entries}) == len(card.entries), "Ambiguous saddlecloth mapping")
    require(set(identities) == {str(e.number) for e in card.entries}, "Identity map must cover exactly the full field")
    race = deepcopy(metadata)
    race.update(schema_version=SCHEMA_VERSION, country=COUNTRY_MAP[card.country],
                race_no=card.race_no, distance_m=card.dist_m, surface=card.surface,
                going=card.going, race_class=card.race_class, runners=[])
    for e in card.entries:
        identity = identities[str(e.number)]
        require(identity.get("horse_name") == e.name, f"Unverified ID/name match for runner {e.number}")
        require(not (set(identity) - {"runner_id", "horse_id", "horse_name", "jockey_id", "trainer_id",
                                      "carried_weight_kg", "history"}), "Unexpected identity metadata")
        runner = {k: deepcopy(identity.get(k)) for k in ("runner_id", "horse_id", "horse_name", "jockey_id", "trainer_id", "carried_weight_kg")}
        runner.update(number=e.number, status="scratched" if e.scratched else "active", age=e.age,
                      sex=e.sex, draw=e.barrier, official_rating=e.official_rating, gear=e.gear,
                      history=deepcopy(identity.get("history", [])))
        race["runners"].append(runner)
    return validate(race)
