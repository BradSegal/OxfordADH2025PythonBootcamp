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
    stream_ai_report,
    summarize_patient_for_imaging,
)
from medvision_toolkit.radiology_helpers import DEFAULT_MAX_IMAGE_EDGE, RadiologyAI
from medvision_toolkit.learning.patient_profiles import load_sample_patient


@patch("medvision_toolkit.learning.imaging_helpers.RadiologyAI")
def test_initialize_medgemma_engine_uses_backend(mock_radiology_ai: MagicMock) -> None:
    """Initialisation helper should forward the backend argument."""

    instance = MagicMock()
    mock_radiology_ai.return_value = instance

    engine = initialize_medgemma_engine(backend="transformers", max_image_edge=512)

    mock_radiology_ai.assert_called_once()
    kwargs = mock_radiology_ai.call_args.kwargs
    assert kwargs["backend"] == "transformers"
    assert kwargs["max_image_edge"] == 512
    assert kwargs["use_sampling"] is None
    assert kwargs["sampling_temperature"] is None
    assert kwargs["sampling_top_p"] is None
    assert kwargs["sampling_top_k"] is None
    assert engine is instance


def test_initialize_medgemma_engine_rejects_invalid_backend() -> None:
    """Unexpected backend names should raise immediately."""

    with pytest.raises(ValueError):
        initialize_medgemma_engine("quantum")


@patch("medvision_toolkit.learning.imaging_helpers.RadiologyAI")
def test_initialize_medgemma_engine_uses_default_edge_limit(
    mock_radiology_ai: MagicMock,
) -> None:
    """Default invocation should apply the library edge constraint."""

    instance = MagicMock()
    mock_radiology_ai.return_value = instance

    initialize_medgemma_engine()

    mock_radiology_ai.assert_called_once()
    kwargs = mock_radiology_ai.call_args.kwargs
    assert kwargs["backend"] == "gguf"
    assert kwargs["max_image_edge"] == DEFAULT_MAX_IMAGE_EDGE


@patch("medvision_toolkit.learning.imaging_helpers.RadiologyAI")
def test_initialize_medgemma_engine_sampling_overrides(
    mock_radiology_ai: MagicMock,
) -> None:
    """Sampling preferences should propagate to the engine factory."""

    instance = MagicMock()
    mock_radiology_ai.return_value = instance

    engine = initialize_medgemma_engine(
        backend="transformers",
        use_sampling=True,
        sampling_temperature=0.6,
        sampling_top_p=0.8,
        sampling_top_k=50,
    )

    kwargs = mock_radiology_ai.call_args.kwargs
    assert kwargs["use_sampling"] is True
    assert kwargs["sampling_temperature"] == 0.6
    assert kwargs["sampling_top_p"] == 0.8
    assert kwargs["sampling_top_k"] == 50
    assert engine is instance


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


@patch("medvision_toolkit.learning.imaging_helpers.Markdown")
@patch("medvision_toolkit.learning.imaging_helpers.display")
def test_render_ai_report_strips_model_prefix(
    mock_display: MagicMock, mock_markdown: MagicMock
) -> None:
    """Render helper should also handle `model` prefixes."""

    render_ai_report("model\nReport body")

    mock_markdown.assert_called_once_with("Report body")
    mock_display.assert_called_once_with(mock_markdown.return_value)


@patch("medvision_toolkit.learning.imaging_helpers.render_ai_report")
def test_stream_ai_report_forwards_sampling(mock_render: MagicMock) -> None:
    """Sampling arguments should be forwarded to the engine analyze call."""

    engine = MagicMock(spec=RadiologyAI)
    engine.backend = "transformers"
    engine.analyze.return_value = "Report"

    result = stream_ai_report(
        engine,
        image_path_or_url="img.png",
        prompt="Describe",
        persona="radiologist",
        use_sampling=True,
        temperature=0.65,
        top_p=0.85,
        top_k=40,
    )

    engine.analyze.assert_called_once_with(
        image_path_or_url="img.png",
        prompt="Describe",
        persona="radiologist",
        use_sampling=True,
        temperature=0.65,
        top_p=0.85,
        top_k=40,
    )
    mock_render.assert_called_once_with("Report")
    assert result == "Report"


def test_build_radiology_prompt_includes_patient_context() -> None:
    """Prompt builder should embed demographics and vitals in the template."""

    profile = load_sample_patient()
    prompt = build_radiology_prompt(
        profile=profile,
        clinical_focus="Evaluate consolidation changes",
        question="What has improved since the last imaging study?",
        persona="radiologist",
    )

    assert "Jordan Nguyen" in prompt
    assert "radiologist" in prompt
    assert "Evaluate consolidation changes" in prompt


def test_summarize_patient_for_imaging_truncates() -> None:
    """Imaging summary should stay within the configured length constraint."""

    profile = load_sample_patient()
    summary = summarize_patient_for_imaging(profile)

    assert len(summary) <= 200
    assert "Jordan Nguyen" in summary
    assert "Tasks:" in summary


def test_build_radiology_prompt_collapses_whitespace() -> None:
    """Whitespace in focus/question should be normalised."""

    profile = load_sample_patient()
    prompt = build_radiology_prompt(
        profile=profile,
        clinical_focus="Line one\nLine two",
        question="What findings?\nBe specific.",
    )

    assert "Line one Line two" in prompt
    assert "What findings? Be specific." in prompt
