"""Unit tests for imaging helper utilities."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from medvision_toolkit.learning.imaging_helpers import (
    build_radiology_prompt,
    initialize_llava_engine,
    initialize_medgemma_engine,
    load_and_display_image,
    render_ai_report,
)
from medvision_toolkit.learning.patient_profiles import load_sample_patient


@patch("medvision_toolkit.learning.imaging_helpers.RadiologyAI")
def test_initialize_medgemma_engine_uses_backend(mock_radiology_ai: MagicMock) -> None:
    """Initialisation helper should forward the backend argument."""

    instance = MagicMock()
    mock_radiology_ai.return_value = instance

    engine = initialize_medgemma_engine(backend="transformers")

    mock_radiology_ai.assert_called_once_with(backend="transformers")
    assert engine is instance


def test_initialize_medgemma_engine_rejects_invalid_backend() -> None:
    """Unexpected backend names should raise immediately."""

    with pytest.raises(ValueError):
        initialize_medgemma_engine("quantum")


@patch("medvision_toolkit.learning.imaging_helpers.LlavaAI")
def test_initialize_llava_engine(mock_llava_ai: MagicMock) -> None:
    """Benchmark helper should proxy directly to LlavaAI."""

    instance = MagicMock()
    mock_llava_ai.return_value = instance

    engine = initialize_llava_engine()

    mock_llava_ai.assert_called_once_with()
    assert engine is instance


@patch("medvision_toolkit.learning.imaging_helpers.display")
@patch("medvision_toolkit.learning.imaging_helpers.requests.get")
def test_load_and_display_image_remote(
    mock_get: MagicMock, mock_display: MagicMock
) -> None:
    """Remote image sources should trigger an HTTP fetch."""

    buffer = BytesIO()
    Image.new("RGB", (32, 32), "white").save(buffer, format="PNG")
    buffer.seek(0)

    response = MagicMock()
    response.content = buffer.read()
    mock_get.return_value = response

    image = load_and_display_image("https://example.com/test.png")

    mock_get.assert_called_once()
    mock_display.assert_called_once()
    assert image.mode == "RGB"


@patch("medvision_toolkit.learning.imaging_helpers.display")
def test_load_and_display_image_local(mock_display: MagicMock, tmp_path: Path) -> None:
    """Local image paths should load without network calls."""

    image_path = tmp_path / "test.png"
    Image.new("RGB", (16, 16), "blue").save(image_path)

    image = load_and_display_image(str(image_path))

    mock_display.assert_called_once()
    assert image.mode == "RGB"


@patch("medvision_toolkit.learning.imaging_helpers.Markdown")
@patch("medvision_toolkit.learning.imaging_helpers.display")
def test_render_ai_report_strips_prefix(
    mock_display: MagicMock, mock_markdown: MagicMock
) -> None:
    """Render helper should remove `assistant` prefixes for readability."""

    render_ai_report("assistant\nReport body")

    mock_markdown.assert_called_once_with("Report body")
    mock_display.assert_called_once_with(mock_markdown.return_value)


def test_build_radiology_prompt_includes_patient_context() -> None:
    """Prompt builder should embed demographics and vitals in the template."""

    profile = load_sample_patient()
    prompt = build_radiology_prompt(
        profile=profile,
        clinical_focus="Evaluate consolidation changes",
        question="What has improved since the last imaging study?",
        persona="radiologist",
    )

    assert profile["demographics"]["full_name"] in prompt
    assert "radiologist" in prompt
    assert "Evaluate consolidation changes" in prompt
