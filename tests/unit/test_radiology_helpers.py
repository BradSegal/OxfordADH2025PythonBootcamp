"""
Unit tests for the radiology_helpers module.

These tests use mocked transformers library components to validate the logic
of RadiologyAI and LlavaAI classes without downloading multi-gigabyte models.
"""

import base64
import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
import torch

from medvision_toolkit.radiology_helpers import (
    DEFAULT_CONTEXT_TOKENS,
    DEFAULT_MAX_IMAGE_EDGE,
    LlavaAI,
    RadiologyAI,
)


class TestRadiologyAI:
    """Unit tests for the RadiologyAI class."""

    @patch("medvision_toolkit.radiology_helpers.AutoProcessor.from_pretrained")
    @patch(
        "medvision_toolkit.radiology_helpers.AutoModelForImageTextToText.from_pretrained"
    )
    def test_init_success(self, mock_model_loader, mock_processor_loader):
        """Test that RadiologyAI initializes correctly with mocked dependencies."""
        # Configure the mock instances
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_model_loader.return_value = mock_model
        mock_processor_loader.return_value = mock_processor

        # Initialize the class
        ai_engine = RadiologyAI(backend="transformers")

        # Assert the loaders were called with correct parameters
        mock_model_loader.assert_called_once_with(
            "google/medgemma-4b-it",
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        mock_processor_loader.assert_called_once_with(
            "google/medgemma-4b-it", use_fast=True
        )

        # Assert the instance attributes are set correctly
        assert ai_engine.model == mock_model
        assert ai_engine.processor == mock_processor
        assert ai_engine.device in ["cuda", "cpu"]

    @patch("medvision_toolkit.radiology_helpers.AutoProcessor.from_pretrained")
    @patch(
        "medvision_toolkit.radiology_helpers.AutoModelForImageTextToText.from_pretrained"
    )
    def test_init_with_custom_model_id(self, mock_model_loader, mock_processor_loader):
        """Test RadiologyAI initialization with a custom model ID."""
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_model_loader.return_value = mock_model
        mock_processor_loader.return_value = mock_processor

        custom_model = "custom/model-id"
        ai_engine = RadiologyAI(model_id=custom_model, backend="transformers")

        mock_model_loader.assert_called_once()
        assert mock_model_loader.call_args[0][0] == custom_model
        assert ai_engine.model == mock_model
        assert ai_engine.processor == mock_processor

    @patch("medvision_toolkit.radiology_helpers.AutoProcessor.from_pretrained")
    @patch(
        "medvision_toolkit.radiology_helpers.AutoModelForImageTextToText.from_pretrained"
    )
    def test_init_failure_raises_exception(
        self, mock_model_loader, mock_processor_loader
    ):
        """Test that initialization failures raise exceptions (fail loudly)."""
        # Simulate a loading failure
        mock_model_loader.side_effect = RuntimeError("GPU out of memory")

        # Assert that the exception is raised
        with pytest.raises(RuntimeError, match="GPU out of memory"):
            RadiologyAI(backend="transformers")

    @patch("medvision_toolkit.radiology_helpers.AutoProcessor.from_pretrained")
    @patch(
        "medvision_toolkit.radiology_helpers.AutoModelForImageTextToText.from_pretrained"
    )
    @patch("medvision_toolkit.radiology_helpers.requests.get")
    def test_load_image_from_url(
        self, mock_requests_get, mock_model_loader, mock_processor_loader
    ):
        """Test loading an image from a URL."""
        # Setup mocks for initialization
        mock_model_loader.return_value = MagicMock()
        mock_processor_loader.return_value = MagicMock()

        # Create a mock image response
        test_image = Image.new("RGB", (100, 100), color="red")
        img_byte_arr = BytesIO()
        test_image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        mock_response = MagicMock()
        mock_response.content = img_byte_arr.read()
        mock_requests_get.return_value = mock_response

        # Initialize and test
        ai_engine = RadiologyAI(backend="transformers")
        loaded_image = ai_engine._load_image("http://example.com/test.jpg")

        # Assertions
        mock_requests_get.assert_called_once_with(
            "http://example.com/test.jpg",
            stream=True,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        mock_response.raise_for_status.assert_called_once()
        assert isinstance(loaded_image, Image.Image)
        assert loaded_image.mode == "RGB"

    @patch("medvision_toolkit.radiology_helpers.AutoProcessor.from_pretrained")
    @patch(
        "medvision_toolkit.radiology_helpers.AutoModelForImageTextToText.from_pretrained"
    )
    def test_load_image_from_local_path(self, mock_model_loader, mock_processor_loader):
        """Test loading an image from a local file path."""
        # Setup mocks for initialization
        mock_model_loader.return_value = MagicMock()
        mock_processor_loader.return_value = MagicMock()

        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            test_image = Image.new("RGB", (100, 100), color="blue")
            test_image.save(tmp_file, format="PNG")
            tmp_path = tmp_file.name

        try:
            ai_engine = RadiologyAI(backend="transformers")
            loaded_image = ai_engine._load_image(tmp_path)

            assert isinstance(loaded_image, Image.Image)
            assert loaded_image.mode == "RGB"
        finally:
            Path(tmp_path).unlink()

    @patch("medvision_toolkit.radiology_helpers.AutoProcessor.from_pretrained")
    @patch(
        "medvision_toolkit.radiology_helpers.AutoModelForImageTextToText.from_pretrained"
    )
    def test_load_image_file_not_found(self, mock_model_loader, mock_processor_loader):
        """Test that loading a non-existent file raises FileNotFoundError."""
        mock_model_loader.return_value = MagicMock()
        mock_processor_loader.return_value = MagicMock()

        ai_engine = RadiologyAI(backend="transformers")

        with pytest.raises(FileNotFoundError):
            ai_engine._load_image("/path/to/nonexistent/file.jpg")

    @patch("medvision_toolkit.radiology_helpers.AutoProcessor.from_pretrained")
    @patch(
        "medvision_toolkit.radiology_helpers.AutoModelForImageTextToText.from_pretrained"
    )
    def test_load_image_resizes_large_inputs(
        self, mock_model_loader, mock_processor_loader, tmp_path: Path
    ):
        """Images larger than the max edge should be resized proportionally."""
        mock_model_loader.return_value = MagicMock()
        mock_processor_loader.return_value = MagicMock()

        large_image_path = tmp_path / "large.png"
        Image.new("RGB", (1024, 512), color="green").save(large_image_path)

        ai_engine = RadiologyAI(backend="transformers", max_image_edge=256)
        loaded_image = ai_engine._load_image(str(large_image_path))

        assert max(loaded_image.size) == 256
        assert loaded_image.size[0] == 256
        assert loaded_image.size[1] < 256

    @patch("medvision_toolkit.radiology_helpers.AutoProcessor.from_pretrained")
    @patch(
        "medvision_toolkit.radiology_helpers.AutoModelForImageTextToText.from_pretrained"
    )
    def test_load_image_keeps_smaller_inputs(
        self, mock_model_loader, mock_processor_loader, tmp_path: Path
    ):
        """Images already below the threshold should remain unchanged."""
        mock_model_loader.return_value = MagicMock()
        mock_processor_loader.return_value = MagicMock()

        small_image_path = tmp_path / "small.png"
        Image.new("RGB", (64, 128), color="purple").save(small_image_path)

        ai_engine = RadiologyAI(backend="transformers", max_image_edge=256)
        loaded_image = ai_engine._load_image(str(small_image_path))

        assert loaded_image.size == (64, 128)

    @patch("medvision_toolkit.radiology_helpers.AutoProcessor.from_pretrained")
    @patch(
        "medvision_toolkit.radiology_helpers.AutoModelForImageTextToText.from_pretrained"
    )
    def test_init_default_max_image_edge(
        self, mock_model_loader, mock_processor_loader
    ):
        """Default configuration should use the library-provided limit."""
        mock_model_loader.return_value = MagicMock()
        mock_processor_loader.return_value = MagicMock()

        ai_engine = RadiologyAI(backend="transformers")

        assert ai_engine.max_image_edge == DEFAULT_MAX_IMAGE_EDGE

    @patch("medvision_toolkit.radiology_helpers.AutoProcessor.from_pretrained")
    @patch(
        "medvision_toolkit.radiology_helpers.AutoModelForImageTextToText.from_pretrained"
    )
    def test_init_rejects_invalid_max_image_edge(
        self, mock_model_loader, mock_processor_loader
    ):
        """Non-positive edge limits should raise immediately."""
        mock_model_loader.return_value = MagicMock()
        mock_processor_loader.return_value = MagicMock()

        with pytest.raises(ValueError):
            RadiologyAI(backend="transformers", max_image_edge=0)

    @patch("medvision_toolkit.radiology_helpers.AutoProcessor.from_pretrained")
    @patch(
        "medvision_toolkit.radiology_helpers.AutoModelForImageTextToText.from_pretrained"
    )
    @patch("medvision_toolkit.radiology_helpers.RadiologyAI._load_image")
    def test_analyze_prompt_formatting(
        self, mock_load_image, mock_model_loader, mock_processor_loader
    ):
        """Test that the analyze method correctly formats prompts and personas."""
        # Setup mocks
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_model_loader.return_value = mock_model
        mock_processor_loader.return_value = mock_processor

        # Mock image loading
        mock_image = Image.new("RGB", (100, 100))
        mock_load_image.return_value = mock_image

        # Mock the processor and model response
        mock_inputs = MagicMock()
        mock_processor.return_value = mock_inputs
        mock_inputs.to.return_value = mock_inputs

        mock_output = MagicMock()
        mock_model.generate.return_value = mock_output

        canned_response = (
            "assistant\nThis is a test report showing normal findings.<end_of_turn>"
        )
        mock_processor.decode.return_value = canned_response
        mock_processor.apply_chat_template.return_value = "formatted_prompt"

        # Initialize and analyze
        ai_engine = RadiologyAI(backend="transformers")
        test_url = "http://example.com/test.jpg"
        test_prompt = "Describe this X-ray."
        test_persona = "cardiologist"

        result = ai_engine.analyze(test_url, test_prompt, persona=test_persona)

        # Assertions
        mock_load_image.assert_called_once_with(test_url)

        mock_processor.apply_chat_template.assert_called_once()
        conversation = mock_processor.apply_chat_template.call_args[0][0]
        assert len(conversation) == 2
        assert conversation[0]["role"] == "system"
        assert f"expert {test_persona}" in conversation[0]["content"]
        assert conversation[1]["role"] == "user"
        assert conversation[1]["content"][0]["type"] == "image"
        assert conversation[1]["content"][1]["type"] == "text"
        assert test_prompt in conversation[1]["content"][1]["text"]

        processor_call_args = mock_processor.call_args
        assert processor_call_args[1]["text"] == "formatted_prompt"

        # Verify model.generate was called
        mock_model.generate.assert_called_once()

        # Verify the response was correctly extracted
        assert result == "This is a test report showing normal findings."

    @patch("medvision_toolkit.radiology_helpers.AutoProcessor.from_pretrained")
    @patch(
        "medvision_toolkit.radiology_helpers.AutoModelForImageTextToText.from_pretrained"
    )
    @patch("medvision_toolkit.radiology_helpers.RadiologyAI._load_image")
    def test_analyze_default_persona(
        self, mock_load_image, mock_model_loader, mock_processor_loader
    ):
        """Test that the default persona is 'radiologist'."""
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_model_loader.return_value = mock_model
        mock_processor_loader.return_value = mock_processor

        mock_load_image.return_value = Image.new("RGB", (100, 100))

        mock_inputs = MagicMock()
        mock_processor.return_value = mock_inputs
        mock_inputs.to.return_value = mock_inputs
        mock_model.generate.return_value = MagicMock()
        mock_processor.decode.return_value = "assistant\nTest"
        mock_processor.apply_chat_template.return_value = "formatted_prompt"

        ai_engine = RadiologyAI(backend="transformers")
        ai_engine.analyze("test.jpg", "Test prompt")

        # Check that the default persona was used in the chat template
        mock_processor.apply_chat_template.assert_called_once()
        conversation = mock_processor.apply_chat_template.call_args[0][0]
        assert "expert radiologist" in conversation[0]["content"]

    @patch("medvision_toolkit.radiology_helpers.AutoProcessor.from_pretrained")
    @patch(
        "medvision_toolkit.radiology_helpers.AutoModelForImageTextToText.from_pretrained"
    )
    @patch("medvision_toolkit.radiology_helpers.RadiologyAI._load_image")
    def test_analyze_raises_when_image_load_fails(
        self, mock_load_image, mock_model_loader, mock_processor_loader
    ):
        """analyze should bubble up file-system errors for missing images."""
        mock_model_loader.return_value = MagicMock()
        mock_processor_loader.return_value = MagicMock()
        mock_load_image.side_effect = FileNotFoundError("missing file")

        ai_engine = RadiologyAI(backend="transformers")

        with pytest.raises(FileNotFoundError, match="missing file"):
            ai_engine.analyze("/tmp/missing.jpg", "Describe this image.")

    @patch("medvision_toolkit.radiology_helpers.requests.get")
    @patch("llama_cpp.Llama.from_pretrained")
    @patch("llama_cpp.llama_chat_format.Llava15ChatHandler.from_pretrained")
    @patch(
        "medvision_toolkit.radiology_helpers.torch.cuda.is_available", return_value=True
    )
    def test_analyze_with_llama_cpp_backend(
        self,
        mock_cuda_available,
        mock_chat_handler_loader,
        mock_llama_loader,
        mock_requests_get,
    ):
        """Verify the GGUF backend formats messages correctly."""
        mock_chat_handler = MagicMock()
        mock_chat_handler_loader.return_value = mock_chat_handler
        mock_llama = MagicMock()
        mock_llama.create_chat_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "assistant\nGGUF response with detail.<end_of_turn>"
                    }
                }
            ]
        }
        mock_llama_loader.return_value = mock_llama

        test_image = Image.new("RGB", (64, 64), color="white")
        buffer = BytesIO()
        test_image.save(buffer, format="PNG")
        buffer.seek(0)
        image_bytes = buffer.getvalue()

        mock_response = MagicMock()
        mock_response.content = image_bytes
        mock_requests_get.return_value = mock_response

        ai_engine = RadiologyAI(backend="gguf")
        result = ai_engine.analyze(
            "http://example.com/test.jpg", "Explain this image", persona="tester"
        )

        mock_chat_handler_loader.assert_called_once_with(
            repo_id="unsloth/medgemma-4b-it-GGUF",
            filename="mmproj-F16.gguf",
            verbose=False,
        )
        mock_llama_loader.assert_called_once_with(
            repo_id="unsloth/medgemma-4b-it-GGUF",
            filename="medgemma-4b-it-Q4_0.gguf",
            verbose=False,
            n_ctx=DEFAULT_CONTEXT_TOKENS,
            n_gpu_layers=-1,
            chat_handler=mock_chat_handler,
        )

        # Ensure create_chat_completion received an inline image payload
        called_messages = mock_llama.create_chat_completion.call_args[1]["messages"]
        assert called_messages[0]["role"] == "system"
        assert "tester" in called_messages[0]["content"]
        assert called_messages[1]["role"] == "user"
        content_items = called_messages[1]["content"]
        image_item = next(
            item for item in content_items if item.get("type") == "image_url"
        )
        data_url = image_item["image_url"]
        assert isinstance(data_url, str)
        assert data_url.startswith("data:image/png;base64,")
        encoded_payload = data_url.split("data:image/png;base64,", maxsplit=1)[-1]
        assert base64.b64decode(encoded_payload) == image_bytes
        stop_markers = mock_llama.create_chat_completion.call_args[1]["stop"]
        assert "\nUSER:" in stop_markers
        assert "USER:" in stop_markers
        assert result == "GGUF response with detail."

    @patch("medvision_toolkit.radiology_helpers.requests.get")
    @patch("llama_cpp.Llama.from_pretrained")
    @patch("llama_cpp.llama_chat_format.Llava15ChatHandler.from_pretrained")
    @patch(
        "medvision_toolkit.radiology_helpers.torch.cuda.is_available", return_value=False
    )
    def test_analyze_with_llama_cpp_streaming(
        self,
        mock_cuda_available,
        mock_chat_handler_loader,
        mock_llama_loader,
        mock_requests_get,
    ):
        """Streaming mode should emit chunks and return the full sanitized report."""

        mock_chat_handler = MagicMock()
        mock_chat_handler_loader.return_value = mock_chat_handler

        def fake_stream(*args, **kwargs):
            assert kwargs["stream"] is True
            yield {"choices": [{"delta": {"content": "assistant\nFirst part. "}}]}
            yield {"choices": [{"delta": {"content": "Second part.<end_of_turn>"}}]}

        mock_llama = MagicMock()
        mock_llama.create_chat_completion.side_effect = fake_stream
        mock_llama_loader.return_value = mock_llama

        test_image = Image.new("RGB", (32, 32), color="white")
        buf = BytesIO()
        test_image.save(buf, format="PNG")
        buf.seek(0)
        mock_response = MagicMock()
        mock_response.content = buf.read()
        mock_requests_get.return_value = mock_response

        captured_chunks: list[str] = []
        ai_engine = RadiologyAI(backend="gguf")
        final_report = ai_engine.analyze(
            "http://example.com/stream.jpg",
            "Provide details.",
            stream_callback=captured_chunks.append,
        )

        assert captured_chunks == [
            "assistant\nFirst part. ",
            "Second part.<end_of_turn>",
        ]
        assert final_report == "First part. Second part."

    @patch("medvision_toolkit.radiology_helpers.AutoProcessor.from_pretrained")
    @patch(
        "medvision_toolkit.radiology_helpers.AutoModelForImageTextToText.from_pretrained"
    )
    def test_default_stop_tokens_include_user_marker(
        self, mock_model_loader, mock_processor_loader
    ):
        """RadiologyAI should include USER markers in default stop tokens."""
        mock_model_loader.return_value = MagicMock()
        mock_processor_loader.return_value = MagicMock()

        ai_engine = RadiologyAI(backend="transformers")
        assert "\nUSER:" in ai_engine.stop_tokens
        assert "USER:" in ai_engine.stop_tokens

    @patch("medvision_toolkit.radiology_helpers.AutoProcessor.from_pretrained")
    @patch(
        "medvision_toolkit.radiology_helpers.AutoModelForImageTextToText.from_pretrained"
    )
    def test_sanitize_output_truncates_followup_user_block(
        self, mock_model_loader, mock_processor_loader
    ):
        """_sanitize_output should trim any follow-up USER turns."""
        mock_model_loader.return_value = MagicMock()
        processor = MagicMock()
        mock_processor_loader.return_value = processor

        ai_engine = RadiologyAI(backend="transformers")
        raw = (
            "assistant\nInitial findings.\nUSER: Provide additional context.\n"
            "ASSISTANT: Irrelevant follow up."
        )
        cleaned = ai_engine._sanitize_output(raw)
        assert "USER:" not in cleaned
        assert cleaned == "Initial findings."
        assert "Explain this image" in text_item["text"]

        assert result == "GGUF response with detail."


class TestLlavaAI:
    """Unit tests for the LlavaAI class."""

    @patch("medvision_toolkit.radiology_helpers.AutoProcessor.from_pretrained")
    @patch(
        "medvision_toolkit.radiology_helpers.LlavaForConditionalGeneration.from_pretrained"
    )
    def test_init_success(self, mock_model_loader, mock_processor_loader):
        """Test that LlavaAI initializes correctly with mocked dependencies."""
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_model_loader.return_value = mock_model
        mock_processor_loader.return_value = mock_processor

        ai_engine = LlavaAI()

        mock_model_loader.assert_called_once_with(
            "llava-hf/llava-1.5-7b-hf",
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        mock_processor_loader.assert_called_once_with(
            "llava-hf/llava-1.5-7b-hf", use_fast=True
        )

        assert ai_engine.model == mock_model
        assert ai_engine.processor == mock_processor
        assert ai_engine.device in ["cuda", "cpu"]

    @patch("medvision_toolkit.radiology_helpers.AutoProcessor.from_pretrained")
    @patch(
        "medvision_toolkit.radiology_helpers.LlavaForConditionalGeneration.from_pretrained"
    )
    def test_init_failure_raises_exception(
        self, mock_model_loader, mock_processor_loader
    ):
        """Test that initialization failures raise exceptions (fail loudly)."""
        mock_model_loader.side_effect = RuntimeError("Network error")

        with pytest.raises(RuntimeError, match="Network error"):
            LlavaAI()

    @patch("medvision_toolkit.radiology_helpers.AutoProcessor.from_pretrained")
    @patch(
        "medvision_toolkit.radiology_helpers.LlavaForConditionalGeneration.from_pretrained"
    )
    @patch("medvision_toolkit.radiology_helpers.LlavaAI._load_image")
    def test_analyze_prompt_formatting(
        self, mock_load_image, mock_model_loader, mock_processor_loader
    ):
        """Test that LlavaAI correctly formats prompts using its chat template."""
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_model_loader.return_value = mock_model
        mock_processor_loader.return_value = mock_processor

        mock_load_image.return_value = Image.new("RGB", (100, 100))

        # Mock the processor methods
        mock_processor.apply_chat_template.return_value = "formatted_prompt"
        mock_inputs = MagicMock()
        mock_processor.return_value = mock_inputs
        mock_inputs.to.return_value = mock_inputs
        mock_model.generate.return_value = MagicMock()
        mock_processor.decode.return_value = "ASSISTANT: This is a test response."

        ai_engine = LlavaAI()
        result = ai_engine.analyze("test.jpg", "What is this?", persona="cardiologist")

        # Verify chat template was applied
        mock_processor.apply_chat_template.assert_called_once()
        template_call_args = mock_processor.apply_chat_template.call_args
        conversation = template_call_args[0][0]

        assert len(conversation) == 1
        assert conversation[0]["role"] == "user"
        assert len(conversation[0]["content"]) == 2
        assert conversation[0]["content"][0]["type"] == "image"
        assert conversation[0]["content"][1]["type"] == "text"
        text_prompt = conversation[0]["content"][1]["text"]
        assert "cardiologist" in text_prompt
        assert "What is this?" in text_prompt

        # Verify response extraction
        assert result == "This is a test response."

    @patch("medvision_toolkit.radiology_helpers.AutoProcessor.from_pretrained")
    @patch(
        "medvision_toolkit.radiology_helpers.LlavaForConditionalGeneration.from_pretrained"
    )
    @patch("medvision_toolkit.radiology_helpers.LlavaAI._load_image")
    def test_analyze_response_without_assistant_marker(
        self, mock_load_image, mock_model_loader, mock_processor_loader
    ):
        """Test response extraction when ASSISTANT: marker is not present."""
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_model_loader.return_value = mock_model
        mock_processor_loader.return_value = mock_processor

        mock_load_image.return_value = Image.new("RGB", (100, 100))
        mock_processor.apply_chat_template.return_value = "formatted"
        mock_inputs = MagicMock()
        mock_processor.return_value = mock_inputs
        mock_inputs.to.return_value = mock_inputs
        mock_model.generate.return_value = MagicMock()

        # Response without ASSISTANT: marker
        mock_processor.decode.return_value = "Just a plain response."

        ai_engine = LlavaAI()
        result = ai_engine.analyze("test.jpg", "Test")

        assert result == "Just a plain response."

    @patch("medvision_toolkit.radiology_helpers.AutoProcessor.from_pretrained")
    @patch(
        "medvision_toolkit.radiology_helpers.LlavaForConditionalGeneration.from_pretrained"
    )
    @patch("medvision_toolkit.radiology_helpers.LlavaAI._load_image")
    def test_analyze_raises_when_image_missing(
        self, mock_load_image, mock_model_loader, mock_processor_loader
    ):
        """Ensure LlavaAI.analyze does not swallow file errors."""
        mock_model_loader.return_value = MagicMock()
        mock_processor_loader.return_value = MagicMock()
        mock_load_image.side_effect = FileNotFoundError("missing")

        ai_engine = LlavaAI()

        with pytest.raises(FileNotFoundError, match="missing"):
            ai_engine.analyze("missing.jpg", "Describe this.")

    @patch("medvision_toolkit.radiology_helpers.AutoProcessor.from_pretrained")
    @patch(
        "medvision_toolkit.radiology_helpers.LlavaForConditionalGeneration.from_pretrained"
    )
    def test_load_image_from_url(self, mock_model_loader, mock_processor_loader):
        """Test that LlavaAI can load images from URLs."""
        mock_model_loader.return_value = MagicMock()
        mock_processor_loader.return_value = MagicMock()

        with patch("medvision_toolkit.radiology_helpers.requests.get") as mock_get:
            test_image = Image.new("RGB", (100, 100), color="green")
            img_byte_arr = BytesIO()
            test_image.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)

            mock_response = MagicMock()
            mock_response.content = img_byte_arr.read()
            mock_get.return_value = mock_response

            ai_engine = LlavaAI()
            loaded_image = ai_engine._load_image("http://example.com/image.jpg")

            mock_get.assert_called_once_with(
                "http://example.com/image.jpg",
                stream=True,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            assert isinstance(loaded_image, Image.Image)
            assert loaded_image.mode == "RGB"

    @patch("medvision_toolkit.radiology_helpers.requests.get")
    @patch("medvision_toolkit.radiology_helpers.AutoProcessor.from_pretrained")
    @patch(
        "medvision_toolkit.radiology_helpers.LlavaForConditionalGeneration.from_pretrained"
    )
    def test_load_image_resizes_large_inputs(
        self,
        mock_model_loader,
        mock_processor_loader,
        mock_requests_get,
    ):
        """Large remote images should be resized before inference."""
        mock_model_loader.return_value = MagicMock()
        mock_processor_loader.return_value = MagicMock()

        large_image = Image.new("RGB", (2048, 512), color="orange")
        buffer = BytesIO()
        large_image.save(buffer, format="PNG")
        buffer.seek(0)

        response = MagicMock()
        response.content = buffer.read()
        mock_requests_get.return_value = response

        ai_engine = LlavaAI()
        resized = ai_engine._load_image("http://example.com/large.png")

        assert max(resized.size) == DEFAULT_MAX_IMAGE_EDGE
        assert resized.size[0] == DEFAULT_MAX_IMAGE_EDGE
        assert resized.size[1] < DEFAULT_MAX_IMAGE_EDGE

    def test_init_rejects_invalid_max_image_edge(self):
        """Initialisation should fail for invalid resize limits."""
        with pytest.raises(ValueError):
            LlavaAI(max_image_edge=0)
