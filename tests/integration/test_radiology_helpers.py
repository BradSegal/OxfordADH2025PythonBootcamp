"""
Integration tests for the radiology_helpers module.

These tests download and use real models from Hugging Face Hub.
They are marked with @pytest.mark.api and should be run separately
from fast unit tests.

Run with: pytest -m api
"""

import os
from typing import Optional

import pytest
import requests
from huggingface_hub import HfFolder
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
)

from medvision_toolkit.radiology_helpers import RadiologyAI

# Public chest X-ray URL for testing
# This is from the NIH Chest X-ray dataset, publicly available
TEST_XRAY_URL = (
    "https://staticnew-prod.topdoctors.co.uk/files/Image/large/"
    "5ad0cd79-11d0-47b7-840b-123525bbab96.jpg"
)

HF_TOKEN_ENV_KEYS = ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN")
HAS_HF_TOKEN = (
    any(os.environ.get(key) for key in HF_TOKEN_ENV_KEYS)
    or HfFolder.get_token() is not None
)
SKIP_REASON = (
    "Hugging Face token with access to google/medgemma-4b-it is required for API tests."
)


@pytest.mark.api
@pytest.mark.slow
@pytest.mark.skipif(not HAS_HF_TOKEN, reason=SKIP_REASON)
class TestRadiologyAIIntegration:
    """Integration tests for RadiologyAI with real model inference."""

    @staticmethod
    def _create_engine() -> RadiologyAI:
        """Initialise the MedGemma engine or skip if access is still gated."""
        try:
            return RadiologyAI()
        except (
            GatedRepoError,
            HfHubHTTPError,
            LocalEntryNotFoundError,
            OSError,
        ) as exc:
            chain = []
            current: Optional[BaseException] = exc
            while current is not None:
                chain.append(str(current))
                cause = current.__cause__
                current = cause if isinstance(cause, BaseException) else None
            message = " ".join(chain).lower()
            if any(keyword in message for keyword in ("gated", "403", "forbidden")):
                pytest.skip(
                    "MedGemma gated-access repository is not available for this token. "
                    "Visit https://huggingface.co/google/medgemma-4b-it and ensure "
                    "the token has 'Access public gated repos' enabled."
                )
            raise

    def test_radiology_ai_full_pipeline(self):
        """
        Test the complete RadiologyAI pipeline with real model download and inference.

        This test:
        1. Downloads the MedGemma-4B model (several GB, may take minutes)
        2. Loads a real chest X-ray from a public URL
        3. Runs inference to generate a medical report
        4. Validates that the response is meaningful

        Note: This test requires significant GPU memory (recommended: T4 or better)
        """
        # Initialize the AI engine (this downloads the model on first run)
        ai_engine = self._create_engine()

        if getattr(ai_engine, "backend", "transformers") == "gguf":
            assert hasattr(ai_engine, "llm")
        else:
            assert ai_engine.model is not None
            assert ai_engine.processor is not None
            assert ai_engine.device in ["cuda", "cpu"]

        # Run inference on a real chest X-ray
        prompt = "Describe any abnormalities or notable findings in this chest X-ray."
        response = ai_engine.analyze(TEST_XRAY_URL, prompt, persona="radiologist")

        # Validate the response
        assert isinstance(response, str)
        assert len(response) > 50  # Should be a substantive response
        assert not response.startswith(
            "An error occurred"
        )  # Should not be an error message

        print("\n--- Integration Test Result ---")
        print(f"Prompt: {prompt}")
        print(f"Response length: {len(response)} characters")
        print(f"Response preview: {response[:200]}...")

    def test_radiology_ai_with_different_personas(self):
        """Test that different personas produce valid responses."""
        ai_engine = self._create_engine()

        personas_to_test = ["radiologist", "cardiologist", "pulmonologist"]

        for persona in personas_to_test:
            response = ai_engine.analyze(
                TEST_XRAY_URL,
                "What do you observe in this image?",
                persona=persona,
            )

            assert isinstance(response, str)
            assert len(response) > 20
            assert not response.startswith("An error occurred")

            print(f"\n--- Persona: {persona} ---")
            print(f"Response: {response[:150]}...")

    def test_radiology_ai_error_handling_bad_url(self):
        """Bad URLs should raise clear HTTP errors (fail fast)."""
        ai_engine = self._create_engine()

        bad_url = "http://example.com/nonexistent-image.jpg"
        with pytest.raises(requests.exceptions.HTTPError):
            ai_engine.analyze(bad_url, "Analyze this image.")
