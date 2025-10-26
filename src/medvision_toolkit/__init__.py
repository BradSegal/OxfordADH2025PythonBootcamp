"""
MedVision Toolkit - AI-powered medical imaging analysis for educational purposes.

This package provides simplified interfaces to state-of-the-art vision-language
models for analyzing medical images, specifically designed for teaching Python
and AI to medical students with zero programming experience.

Main Components
---------------
RadiologyAI : class
    Production-grade AI engine using Google's MedGemma-4B model for medical
    image analysis with radiological expertise.

LlavaAI : class
    Experimental AI engine using LLaVA-1.5-7B model for general-purpose
    vision-language tasks and benchmarking exercises.

Examples
--------
>>> from medvision_toolkit import RadiologyAI
>>> ai = RadiologyAI()
>>> report = ai.analyze("chest_xray.jpg", "Describe any abnormalities")
"""

from medvision_toolkit.learning import (
    PatientDemographics,
    PatientMedicationPlan,
    PatientProfile,
    PatientVitals,
    VitalStatistics,
    build_radiology_prompt,
    calculate_bmi,
    categorize_pain_score,
    compute_vital_statistics,
    draft_handover_for_night_team,
    generate_handover_note,
    initialize_llava_engine,
    initialize_medgemma_engine,
    load_sample_patient,
    load_and_display_image,
    plot_vital_trends,
    render_ai_report,
    stream_ai_report,
    summarize_patient_for_imaging,
    summarize_patient,
    triage_pain_scores,
)
from medvision_toolkit.radiology_helpers import (
    LlavaAI,
    RadiologyAI,
    llama_cpp_has_cuda_support,
)

__version__ = "0.1.0"
__all__ = [
    "RadiologyAI",
    "LlavaAI",
    "PatientDemographics",
    "PatientMedicationPlan",
    "PatientProfile",
    "PatientVitals",
    "VitalStatistics",
    "build_radiology_prompt",
    "calculate_bmi",
    "categorize_pain_score",
    "compute_vital_statistics",
    "draft_handover_for_night_team",
    "generate_handover_note",
    "initialize_llava_engine",
    "initialize_medgemma_engine",
    "load_sample_patient",
    "load_and_display_image",
    "plot_vital_trends",
    "render_ai_report",
    "stream_ai_report",
    "summarize_patient_for_imaging",
    "summarize_patient",
    "triage_pain_scores",
    "llama_cpp_has_cuda_support",
]
