"""
Patient profile helpers for the Project MedVision teaching notebooks.

This module provides typed data structures and high-level helper functions used
throughout the introductory Python notebook. The goal is to keep the student
experience focused on core programming concepts while this module handles
validation, formatting, and visualisation logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Final, List, Sequence, TypedDict

import matplotlib.pyplot as plt


class PatientDemographics(TypedDict):
    """Demographic information tracked for each patient."""

    full_name: str
    mrn: str
    age: int
    pronouns: str
    primary_language: str
    attending: str


class PatientVitals(TypedDict):
    """Vital trends captured over the most recent week of inpatient care."""

    systolic_bp: List[int]
    heart_rate: List[int]
    respiratory_rate: List[int]
    temperature_c: List[float]
    pain_scores: List[int]


class PatientMedicationPlan(TypedDict, total=False):
    """Scheduled and as-needed medications for the active encounter."""

    scheduled: List[str]
    prn: List[str]


class PatientProfile(TypedDict):
    """Top-level patient record surfaced to the student notebook."""

    demographics: PatientDemographics
    vitals: PatientVitals
    medications: PatientMedicationPlan
    allergies: List[str]
    primary_diagnosis: str
    consults: List[str]
    outstanding_tasks: List[str]


@dataclass(frozen=True)
class VitalStatistics:
    """Summary statistics for quick interpretation of vital sign trends."""

    latest_systolic_bp: int
    highest_systolic_bp: int
    lowest_systolic_bp: int
    average_systolic_bp: float
    max_pain_score: int
    min_pain_score: int


_SAMPLE_PROFILE: Final[PatientProfile] = {
    "demographics": {
        "full_name": "Jordan Nguyen",
        "mrn": "MV-102938",
        "age": 54,
        "pronouns": "they/them",
        "primary_language": "English",
        "attending": "Dr. Priya Patel",
    },
    "vitals": {
        "systolic_bp": [146, 142, 138, 135, 134, 131, 129],
        "heart_rate": [88, 86, 83, 82, 80, 78, 76],
        "respiratory_rate": [20, 20, 18, 18, 17, 17, 16],
        "temperature_c": [37.6, 37.3, 37.0, 36.9, 36.8, 36.7, 36.6],
        "pain_scores": [7, 6, 5, 4, 4, 3, 2],
    },
    "medications": {
        "scheduled": [
            "Metformin 1000 mg PO BID",
            "Lisinopril 10 mg PO daily",
            "Atorvastatin 20 mg PO nightly",
        ],
        "prn": ["Oxycodone 5 mg PO q4h PRN pain > 6", "Ondansetron 4 mg PO PRN nausea"],
    },
    "allergies": ["Penicillin"],
    "primary_diagnosis": "Community-acquired pneumonia with sepsis",
    "consults": ["Respiratory therapy", "Physical therapy"],
    "outstanding_tasks": [
        "Follow-up blood cultures",
        "Reassess oxygen requirement during ambulation",
    ],
}


def _copy_profile(profile: PatientProfile) -> PatientProfile:
    """Create a deep copy of the sample profile to avoid accidental mutation."""

    vitals = profile["vitals"]
    medications = profile["medications"]

    return {
        "demographics": {**profile["demographics"]},
        "vitals": {
            "systolic_bp": list(vitals["systolic_bp"]),
            "heart_rate": list(vitals["heart_rate"]),
            "respiratory_rate": list(vitals["respiratory_rate"]),
            "temperature_c": list(vitals["temperature_c"]),
            "pain_scores": list(vitals["pain_scores"]),
        },
        "medications": {
            "scheduled": list(medications.get("scheduled", [])),
            "prn": list(medications.get("prn", [])),
        },
        "allergies": list(profile["allergies"]),
        "primary_diagnosis": profile["primary_diagnosis"],
        "consults": list(profile["consults"]),
        "outstanding_tasks": list(profile["outstanding_tasks"]),
    }


def load_sample_patient() -> PatientProfile:
    """Return a fresh copy of the sample patient profile."""

    return _copy_profile(_SAMPLE_PROFILE)


def summarize_patient(profile: PatientProfile) -> str:
    """
    Create a short textual summary for morning rounds.

    Parameters
    ----------
    profile : PatientProfile
        Patient profile dictionary returned by :func:`load_sample_patient`.

    Returns
    -------
    str
        A formatted summary string.
    """

    demographics = profile["demographics"]
    latest_bp = profile["vitals"]["systolic_bp"][-1]
    latest_pain = profile["vitals"]["pain_scores"][-1]
    outstanding = len(profile["outstanding_tasks"])
    return (
        f"{demographics['full_name']} (MRN {demographics['mrn']}), "
        f"{demographics['age']} y/o, primary diagnosis: {profile['primary_diagnosis']}. "
        f"Latest systolic BP: {latest_bp} mmHg, pain score: {latest_pain}/10. "
        f"{outstanding} outstanding task(s) for the team."
    )


def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """
    Calculate the body mass index (BMI).

    Parameters
    ----------
    weight_kg : float
        Patient weight in kilograms. Must be a positive value.
    height_m : float
        Patient height in metres. Must be a positive value.

    Returns
    -------
    float
        The BMI value rounded to two decimal places.

    Raises
    ------
    ValueError
        If weight or height are not positive.
    """

    if weight_kg <= 0:
        raise ValueError("Weight must be positive.")
    if height_m <= 0:
        raise ValueError("Height must be positive.")
    bmi = weight_kg / (height_m * height_m)
    return round(bmi, 2)


def compute_vital_statistics(vitals: PatientVitals) -> VitalStatistics:
    """
    Compute quick statistics for systolic blood pressure and pain trends.

    Parameters
    ----------
    vitals : PatientVitals
        Vital signs dictionary from :func:`load_sample_patient`.

    Returns
    -------
    VitalStatistics
        Dataclass containing latest, highest, lowest, and average metrics.

    Raises
    ------
    ValueError
        If the systolic blood pressure or pain arrays are empty.
    """

    systolic = vitals["systolic_bp"]
    pain = vitals["pain_scores"]
    if not systolic:
        raise ValueError("Systolic blood pressure series cannot be empty.")
    if not pain:
        raise ValueError("Pain score series cannot be empty.")

    return VitalStatistics(
        latest_systolic_bp=systolic[-1],
        highest_systolic_bp=max(systolic),
        lowest_systolic_bp=min(systolic),
        average_systolic_bp=round(mean(systolic), 2),
        max_pain_score=max(pain),
        min_pain_score=min(pain),
    )


def categorize_pain_score(score: int) -> str:
    """
    Convert a numeric pain score into a triage-friendly label.

    Parameters
    ----------
    score : int
        Pain score on a 0-10 scale.

    Returns
    -------
    str
        A descriptive label paired with a recommended action.
    """

    if score < 0:
        raise ValueError("Pain score cannot be negative.")
    if score >= 8:
        return f"Score {score}: Severe pain. Escalate to attending immediately."
    if score >= 6:
        return f"Score {score}: High pain. Administer PRN medication."
    if score >= 3:
        return f"Score {score}: Moderate pain. Reassess in 30 minutes."
    return f"Score {score}: Minimal pain. Continue current plan."


def triage_pain_scores(pain_scores: Sequence[int]) -> List[str]:
    """
    Produce triage messages for a sequence of pain scores.

    Parameters
    ----------
    pain_scores : Sequence[int]
        Iterable of numeric pain scores.

    Returns
    -------
    list of str
        Triage messages generated by :func:`categorize_pain_score`.
    """

    return [categorize_pain_score(score) for score in pain_scores]


def plot_vital_trends(vitals: PatientVitals) -> None:
    """
    Plot systolic blood pressure and pain score trends.

    Parameters
    ----------
    vitals : PatientVitals
        Vital signs dictionary from :func:`load_sample_patient`.

    Notes
    -----
    The function uses Matplotlib to render inline plots. It raises any
    underlying Matplotlib exceptions to ensure failures are visible to the
    learner.
    """

    days = list(range(1, len(vitals["systolic_bp"]) + 1))
    fig, ax1 = plt.subplots(figsize=(8, 4))

    ax1.plot(
        days, vitals["systolic_bp"], marker="o", color="#1f77b4", label="Systolic BP"
    )
    ax1.set_xlabel("Hospital Day")
    ax1.set_ylabel("Systolic BP (mmHg)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_title("Vital Trends for the Past Week")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(
        days, vitals["pain_scores"], marker="s", color="#d62728", label="Pain score"
    )
    ax2.set_ylabel("Pain Score (0-10)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    plt.tight_layout()
    plt.show()


def generate_handover_note(profile: PatientProfile, action_items: Sequence[str]) -> str:
    """
    Create a structured handover note summarising the patient's status.

    Parameters
    ----------
    profile : PatientProfile
        Patient profile dictionary returned by :func:`load_sample_patient`.
    action_items : Sequence[str]
        Ordered list of outstanding tasks for the on-call team.

    Returns
    -------
    str
        A multi-line handover note ready to share with the next shift.
    """

    demographics = profile["demographics"]
    vitals = profile["vitals"]
    stats = compute_vital_statistics(vitals)
    meds = profile["medications"]

    action_section = "\n".join(f"- {item}" for item in action_items)
    if not action_section:
        action_section = "- No new actions; continue monitoring."

    scheduled_section = "\n".join(f"- {med}" for med in meds.get("scheduled", []))
    if not scheduled_section:
        scheduled_section = "- No scheduled medications in chart."

    prn_section = "\n".join(f"- {med}" for med in meds.get("prn", []))
    if not prn_section:
        prn_section = "- No PRN medications ordered."

    return (
        f"Handover for {demographics['full_name']} (MRN {demographics['mrn']})\n"
        f"Attending: {demographics['attending']} | Primary Dx: {profile['primary_diagnosis']}\n\n"
        f"Vitals Snapshot:\n"
        f"- Latest systolic BP: {stats.latest_systolic_bp} mmHg; "
        f"range {stats.lowest_systolic_bp}-{stats.highest_systolic_bp} mmHg\n"
        f"- Pain trend: peak {stats.max_pain_score}/10, current {vitals['pain_scores'][-1]}/10\n\n"
        f"Scheduled Medications:\n{scheduled_section}\n\n"
        f"PRN Medications:\n{prn_section}\n\n"
        f"Action Items for Next Shift:\n{action_section}"
    )


def draft_handover_for_night_team(profile: PatientProfile) -> str:
    """
    Generate a default handover note with standard night-shift safeguards.

    Parameters
    ----------
    profile : PatientProfile
        Patient profile dictionary returned by :func:`load_sample_patient`.

    Returns
    -------
    str
        Handover note string produced by :func:`generate_handover_note`.
    """

    vitals = profile["vitals"]
    stats = compute_vital_statistics(vitals)
    action_items = list(profile["outstanding_tasks"])

    if stats.latest_systolic_bp >= 140:
        action_items.append("Recheck blood pressure in 1 hour.")

    if vitals["pain_scores"][-1] >= 4:
        action_items.append("Offer guided breathing before bedtime.")

    return generate_handover_note(profile, action_items)
