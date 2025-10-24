"""Unit tests for the learning.patient_profiles helper module."""

from __future__ import annotations

import math
from typing import Sequence

import matplotlib
import pytest

from medvision_toolkit.learning.patient_profiles import (
    PatientProfile,
    calculate_bmi,
    categorize_pain_score,
    compute_vital_statistics,
    draft_handover_for_night_team,
    generate_handover_note,
    load_sample_patient,
    triage_pain_scores,
)

matplotlib.use("Agg")


@pytest.fixture()
def sample_profile() -> PatientProfile:
    """Provide a fresh sample patient profile for each test."""

    return load_sample_patient()


def test_calculate_bmi_valid_values() -> None:
    """BMI calculation should be accurate to two decimal places."""

    bmi = calculate_bmi(weight_kg=72.5, height_m=1.68)
    assert math.isclose(bmi, 25.69, rel_tol=0.0, abs_tol=0.01)


@pytest.mark.parametrize(
    "weight,height",
    [
        (0, 1.70),
        (-1, 1.70),
        (72.0, 0),
        (72.0, -1),
    ],
)
def test_calculate_bmi_invalid_inputs(weight: float, height: float) -> None:
    """Invalid heights or weights should raise a ValueError."""

    with pytest.raises(ValueError):
        calculate_bmi(weight_kg=weight, height_m=height)


def test_compute_vital_statistics(sample_profile: PatientProfile) -> None:
    """Vital statistics should include expected extrema and averages."""

    stats = compute_vital_statistics(sample_profile["vitals"])
    assert stats.latest_systolic_bp == sample_profile["vitals"]["systolic_bp"][-1]
    assert stats.highest_systolic_bp >= stats.lowest_systolic_bp
    assert stats.max_pain_score >= stats.min_pain_score


@pytest.mark.parametrize(
    "score,expected_phrase",
    [
        (9, "Severe pain"),
        (6, "High pain"),
        (4, "Moderate pain"),
        (1, "Minimal pain"),
    ],
)
def test_categorize_pain_score(score: int, expected_phrase: str) -> None:
    """Categorisation should map score ranges to the correct label."""

    message = categorize_pain_score(score)
    assert expected_phrase in message


def test_categorize_pain_score_rejects_negative() -> None:
    """Negative pain scores should fail fast."""

    with pytest.raises(ValueError):
        categorize_pain_score(-1)


def test_triage_pain_scores(sample_profile: PatientProfile) -> None:
    """The triage helper should return one message per score."""

    messages = triage_pain_scores(sample_profile["vitals"]["pain_scores"])
    assert len(messages) == len(sample_profile["vitals"]["pain_scores"])
    assert messages[0].startswith("Score")


def test_generate_handover_note(sample_profile: PatientProfile) -> None:
    """Handover note should stitch together structured data."""

    action_items: Sequence[str] = ("Reassess oxygen requirement", "Call PT")
    note = generate_handover_note(sample_profile, action_items)
    assert "Handover for" in note
    assert "Action Items for Next Shift" in note
    for item in action_items:
        assert item in note


def test_draft_handover_for_night_team(sample_profile: PatientProfile) -> None:
    """Default handover should include escalations when thresholds exceeded."""

    # Ensure thresholds trip by manipulating vitals.
    sample_profile["vitals"]["systolic_bp"][-1] = 150
    sample_profile["vitals"]["pain_scores"][-1] = 5

    note = draft_handover_for_night_team(sample_profile)
    assert "Recheck blood pressure in 1 hour." in note
    assert "Offer guided breathing before bedtime." in note
