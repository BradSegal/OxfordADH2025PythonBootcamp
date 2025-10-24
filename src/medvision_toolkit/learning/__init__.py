"""
Learning helpers for the introductory Python notebook.

The utilities exposed here provide typed data contracts and high-level helper
functions that power the pedagogical experience in the "Python First Dose"
notebook. All complex logic (validation, plotting, templating) lives in this
module so that students interact with a simplified, well-typed interface.
"""

from medvision_toolkit.learning.patient_profiles import (
    PatientDemographics,
    PatientMedicationPlan,
    PatientProfile,
    PatientVitals,
    VitalStatistics,
    calculate_bmi,
    categorize_pain_score,
    compute_vital_statistics,
    draft_handover_for_night_team,
    generate_handover_note,
    load_sample_patient,
    plot_vital_trends,
    summarize_patient,
    triage_pain_scores,
)
from medvision_toolkit.learning.imaging_helpers import (
    build_radiology_prompt,
    initialize_llava_engine,
    initialize_medgemma_engine,
    load_and_display_image,
    render_ai_report,
    summarize_patient_for_imaging,
)

__all__ = [
    "PatientDemographics",
    "PatientMedicationPlan",
    "PatientProfile",
    "PatientVitals",
    "VitalStatistics",
    "calculate_bmi",
    "categorize_pain_score",
    "compute_vital_statistics",
    "draft_handover_for_night_team",
    "generate_handover_note",
    "build_radiology_prompt",
    "initialize_llava_engine",
    "initialize_medgemma_engine",
    "load_and_display_image",
    "render_ai_report",
    "summarize_patient_for_imaging",
    "load_sample_patient",
    "plot_vital_trends",
    "summarize_patient",
    "triage_pain_scores",
]
