"""
High-level helpers for the imaging lab notebook.

These utilities wrap the underlying MedVision AI engines and provide
pedagogically friendly functions for Jupyter notebooks. They hide low-level
details such as HTTP fetching, persona prompt construction, and report
rendering.
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Optional

import requests
from IPython.display import Markdown, display  # type: ignore
from PIL import Image

from medvision_toolkit.learning.patient_profiles import PatientProfile
from medvision_toolkit.radiology_helpers import (
    DEFAULT_MAX_IMAGE_EDGE,
    LlavaAI,
    RadiologyAI,
)

logger = logging.getLogger(__name__)
_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0"}


def initialize_medgemma_engine(
    backend: str = "gguf", max_image_edge: int = DEFAULT_MAX_IMAGE_EDGE
) -> RadiologyAI:
    """
    Initialise the MedGemma-backed radiology engine.

    Parameters
    ----------
    backend : str, optional
        Either ``\"gguf\"`` (default) for llama.cpp or ``\"transformers\"`` for the
        Hugging Face backend. Any unexpected value raises ``ValueError``.

    max_image_edge : int, optional
        Maximum size for the image's longest edge in pixels.

    Returns
    -------
    RadiologyAI
        Ready-to-use engine instance.
    """

    backend_normalised = backend.lower()
    if backend_normalised not in {"gguf", "transformers"}:
        raise ValueError("backend must be either 'gguf' or 'transformers'.")
    logger.info("Initialising RadiologyAI with %s backend.", backend_normalised)
    return RadiologyAI(
        backend=backend_normalised,
        max_image_edge=max_image_edge,
    )


def initialize_llava_engine() -> LlavaAI:
    """
    Initialise the experimental LLaVA engine used for benchmarking.

    Returns
    -------
    LlavaAI
        Experimental vision-language engine.
    """

    logger.info("Initialising LlavaAI benchmark engine.")
    return LlavaAI()


def load_and_display_image(image_source: str, timeout: int = 30) -> Image.Image:
    """
    Fetch an image from disk or URL, display it inline, and return the PIL image.

    Parameters
    ----------
    image_source : str
        Local file path or HTTP/HTTPS URL.
    timeout : int, optional
        Network timeout in seconds when fetching remote images.

    Returns
    -------
    PIL.Image.Image
        RGB image ready for downstream analysis.
    """

    if image_source.startswith("http"):
        logger.info("Fetching remote image: %s", image_source)
        response = requests.get(
            image_source, stream=True, headers=_HTTP_HEADERS, timeout=timeout
        )
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        logger.info("Opening local image: %s", image_source)
        image = Image.open(image_source).convert("RGB")

    display(image)
    return image


def _strip_assistant_prefix(report_text: str) -> str:
    """
    Remove leading chat prefixes produced by some model backends.

    Parameters
    ----------
    report_text : str
        Raw text returned by an AI engine.

    Returns
    -------
    str
        Cleaned report text.
    """

    cleaned = report_text.strip()
    if cleaned.startswith("assistant\n"):
        return cleaned.split("assistant\n", maxsplit=1)[-1].strip()
    return cleaned


def render_ai_report(report_text: str) -> None:
    """
    Display an AI-generated report using Markdown for readability.

    Parameters
    ----------
    report_text : str
        Raw text returned by an AI engine.
    """

    cleaned = _strip_assistant_prefix(report_text)
    display(Markdown(cleaned))


def build_radiology_prompt(
    profile: PatientProfile,
    clinical_focus: str,
    question: str,
    persona: Optional[str] = None,
) -> str:
    """
    Build a context-rich prompt for the radiology engine.

    Parameters
    ----------
    profile : PatientProfile
        Structured patient record created in Notebook 01.
    clinical_focus : str
        Short phrase describing what the clinician is most concerned about.
    question : str
        The direct question to pose to the model about the image.
    persona : str, optional
        Specialist persona to emphasise (e.g., ``\"radiologist\"``).

    Returns
    -------
    str
        Prompt ready to send to :meth:`RadiologyAI.analyze`.
    """

    demographics = profile["demographics"]
    vitals = profile["vitals"]
    latest_bp = vitals["systolic_bp"][-1]
    latest_pain = vitals["pain_scores"][-1]
    persona_instruction = (
        f"You are acting as a {persona}. "
        if persona
        else "You are the covering physician. "
    )

    prompt_lines = [
        persona_instruction,
        f"Patient: {demographics['full_name']} ({demographics['age']} y/o, MRN {demographics['mrn']}).",
        f"Primary diagnosis: {profile['primary_diagnosis']}.",
        f"Clinical focus today: {clinical_focus}.",
        f"Latest vitals — systolic BP {latest_bp} mmHg, pain score {latest_pain}/10.",
        "",
        question.strip(),
    ]
    return "\n".join(prompt_lines)
