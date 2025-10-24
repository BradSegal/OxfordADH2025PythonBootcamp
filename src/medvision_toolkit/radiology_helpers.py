"""
Radiology AI Helper Module for Project MedVision.

This module provides two AI engines for analyzing medical images:
1. RadiologyAI: Production engine using Google's MedGemma-4B model
2. LlavaAI: Experimental engine using LLaVA-1.5-7B for benchmarking

Both classes abstract away the complexity of model loading, image preprocessing,
prompt templating, and inference, providing a simple `analyze()` interface.
"""

import base64
import logging
from io import BytesIO
from typing import Any, Dict, Optional, cast

import requests
import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    LlavaForConditionalGeneration,
)

logger = logging.getLogger(__name__)
_DEFAULT_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0"}


class RadiologyAI:
    """Primary AI engine for MedVision with selectable backends.

    Parameters
    ----------
    model_id : str, optional
        HF model identifier for the Transformers backend.
    backend : str, optional
        "gguf" to use the llama.cpp inference path, otherwise "transformers".
    gguf_filename : str, optional
        The GGUF file to load when `backend == "gguf"`.
    gguf_model_id : str, optional
        The repository containing the GGUF weights.
    gguf_mmproj : str, optional
        The multimodal projection file required by llama.cpp for vision inputs.
    """

    def __init__(
        self,
        model_id: str = "google/medgemma-4b-it",
        backend: str = "gguf",
        gguf_filename: str = "medgemma-4b-it-Q4_0.gguf",
        gguf_model_id: str = "unsloth/medgemma-4b-it-GGUF",
        gguf_mmproj: str = "mmproj-F16.gguf",
        gguf_stop: Optional[list[str]] = None,
    ) -> None:
        """
        Initialize the RadiologyAI engine by loading the MedGemma model.

        This method downloads and loads the model, which may take several minutes
        on the first run. The model is cached for subsequent uses.

        Parameters
        ----------
        model_id : str, optional
            The Hugging Face model identifier (default: "google/medgemma-4b-it").

        Raises
        ------
        Exception
            If the model fails to load due to network errors, insufficient memory,
            or missing dependencies. The exception will be re-raised to halt execution.
        """
        self.backend = backend.lower()
        self.chat_handler: Optional[Any] = None
        if self.backend == "gguf":
            self.device = "cpu"  # llama.cpp handles its own device usage
            stop_tokens = gguf_stop or ["<end_of_turn>"]
            self._init_gguf_backend(
                gguf_model_id, gguf_filename, gguf_mmproj, stop_tokens
            )
        else:
            self._init_transformers_backend(model_id)

    def _init_transformers_backend(self, model_id: str) -> None:
        logger.info(
            "Initializing RadiologyAI (Transformers) with model: %s...", model_id
        )
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("RadiologyAI (Transformers) using device: %s", self.device)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
            logger.info("RadiologyAI (Transformers) initialized successfully.")
        except Exception:
            logger.exception(
                "FATAL ERROR: Failed to initialize the Transformers backend."
            )
            raise

    def _init_gguf_backend(
        self, repo_id: str, filename: str, mmproj_filename: str, stop_tokens: list[str]
    ) -> None:
        logger.info(
            "Initializing RadiologyAI (GGUF) with repo %s and file %s...",
            repo_id,
            filename,
        )
        try:
            from llama_cpp import Llama, llama_chat_format  # type: ignore
            import llama_cpp._logger as llama_cpp_logger  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "llama-cpp-python is required for the GGUF backend. Please install it."
            ) from exc

        try:
            llama_cpp_logger.logger.setLevel(logging.CRITICAL + 10)
            gpu_available = False
            try:
                gpu_available = torch.cuda.is_available()
            except Exception:
                gpu_available = False

            self.gguf_gpu_layers = -1 if gpu_available else 0

            chat_handler = llama_chat_format.Llava15ChatHandler.from_pretrained(
                repo_id=repo_id,
                filename=mmproj_filename,
                verbose=False,
            )

            self.llm = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                verbose=False,
                n_gpu_layers=self.gguf_gpu_layers,
                chat_handler=chat_handler,
            )
            self.chat_handler = chat_handler
            self.processor = None
            self.gguf_stop_tokens = stop_tokens
            if gpu_available:
                logger.info("RadiologyAI (GGUF) initialized with GPU acceleration.")
            else:
                logger.info(
                    "RadiologyAI (GGUF) initialized on CPU; switch to a CUDA runtime for faster inference."
                )
        except Exception:
            logger.exception("FATAL ERROR: Failed to initialize the GGUF backend.")
            raise

    def analyze(
        self, image_path_or_url: str, prompt: str, persona: str = "radiologist"
    ) -> str:
        """
        Analyze a medical image and return a text-based report.

        This method takes an image (from URL or local path) and a text prompt,
        then generates a detailed analysis using the MedGemma model with the
        specified medical expert persona.

        Parameters
        ----------
        image_path_or_url : str
            The local file path or web URL of the medical image.
        prompt : str
            The question or instruction for the AI model.
        persona : str, optional
            The expert persona the AI should adopt (default: "radiologist").
            Examples: "radiologist", "cardiologist", "pathologist".

        Returns
        -------
        str
            A text analysis generated by the AI model.

        Examples
        --------
        >>> ai = RadiologyAI()
        >>> result = ai.analyze(
        ...     "chest_xray.jpg",
        ...     "Describe any abnormalities",
        ...     persona="radiologist"
        ... )
        Raises
        ------
        requests.exceptions.HTTPError
            If downloading an image from a URL fails.
        FileNotFoundError
            If the provided local image path does not exist.
        RuntimeError
            If model generation fails due to resource constraints or other runtime issues.
        """
        if self.backend == "gguf":
            return self._analyze_with_llama_cpp(image_path_or_url, prompt, persona)

        image = self._load_image(image_path_or_url)
        system_prompt = f"You are an expert {persona}."

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        prompt_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

        inputs = self.processor(text=prompt_text, images=image, return_tensors="pt").to(
            self.device, torch.bfloat16
        )

        output = self.model.generate(**inputs, max_new_tokens=300)

        decoded_text: str = self.processor.decode(output[0], skip_special_tokens=True)
        if "assistant\n" in decoded_text:
            return decoded_text.split("assistant\n", maxsplit=1)[-1].strip()
        return decoded_text.strip()

    def _analyze_with_llama_cpp(
        self, image_path_or_url: str, prompt: str, persona: str
    ) -> str:
        image = self._load_image(image_path_or_url)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        image_bytes = buffer.getvalue()
        data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode(
            "utf-8"
        )

        messages: list[Dict[str, Any]] = [
            {"role": "system", "content": f"You are an expert {persona}."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": data_url},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        response = cast(
            Dict[str, Any],
            self.llm.create_chat_completion(
                messages=cast(list[Any], messages),
                max_tokens=512,
                stop=self.gguf_stop_tokens,
            ),
        )

        message = response["choices"][0]["message"]
        content = (message.get("content") or "").strip()
        if content.startswith("assistant\n"):
            content = content.split("assistant\n", maxsplit=1)[-1].strip()
        return content

    def _load_image(self, image_path_or_url: str) -> Image.Image:
        """
        Load an image from either a local path or a URL.

        Parameters
        ----------
        image_path_or_url : str
            The local file path or web URL of the image.

        Returns
        -------
        PIL.Image.Image
            The loaded image in RGB format.

        Raises
        ------
        requests.exceptions.HTTPError
            If the URL cannot be fetched or returns a bad status code.
        FileNotFoundError
            If the local file path does not exist.
        """
        if image_path_or_url.startswith("http"):
            response = requests.get(
                image_path_or_url, stream=True, headers=_DEFAULT_HTTP_HEADERS
            )
            response.raise_for_status()  # Raise an exception for bad status codes
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_path_or_url).convert("RGB")
        return image


class LlavaAI:
    """
    An experimental AI engine for benchmarking, powered by the LLaVA model.

    This class provides an alternative vision-language model for comparison
    exercises. It uses the LLaVA-1.5-7B model, a general-purpose vision-language
    model trained on diverse image-text data.

    Attributes
    ----------
    device : str
        The compute device being used ('cuda' or 'cpu').
    model : LlavaForConditionalGeneration
        The loaded LLaVA vision-language model.
    processor : AutoProcessor
        The processor for handling text and image inputs.

    Examples
    --------
    >>> benchmark_engine = LlavaAI()
    >>> result = benchmark_engine.analyze(
    ...     "medical_image.jpg",
    ...     "What do you see in this image?"
    ... )
    """

    def __init__(self, model_id: str = "llava-hf/llava-1.5-7b-hf") -> None:
        """
        Initialize the LlavaAI engine by loading the LLaVA model.

        This method downloads and loads the model, which may take several minutes
        on the first run. The model is cached for subsequent uses.

        Parameters
        ----------
        model_id : str, optional
            The Hugging Face model identifier (default: "llava-hf/llava-1.5-7b-hf").

        Raises
        ------
        Exception
            If the model fails to load due to network errors, insufficient memory,
            or missing dependencies. The exception will be re-raised to halt execution.
        """
        logger.info("Initializing LlavaAI with model: %s...", model_id)
        logger.info(
            "LlavaAI initialization may take several minutes while the model downloads."
        )

        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("LlavaAI using device: %s", self.device)

            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
            logger.info("LlavaAI engine initialized successfully.")

        except Exception:
            logger.exception("FATAL ERROR: Failed to initialize LlavaAI.")
            logger.error(
                "This may be due to a lack of GPU memory. Ensure you are on a T4 GPU or better."
            )
            raise  # Re-raise the exception to halt execution

    def analyze(
        self, image_path_or_url: str, prompt: str, persona: str = "assistant"
    ) -> str:
        """
        Analyze an image and return a text-based description.

        This method takes an image (from URL or local path) and a text prompt,
        then generates an analysis using the LLaVA model.

        Parameters
        ----------
        image_path_or_url : str
            The local file path or web URL of the image.
        prompt : str
            The question or instruction for the AI model.
        persona : str, optional
            The role descriptor for the AI (default: "assistant").
            Note: LLaVA uses a simpler prompt format than MedGemma.

        Returns
        -------
        str
            A text analysis generated by the AI model.

        Examples
        --------
        >>> ai = LlavaAI()
        >>> result = ai.analyze("image.jpg", "What's in this image?")
        Raises
        ------
        requests.exceptions.HTTPError
            If downloading an image from a URL fails.
        FileNotFoundError
            If the provided local image path does not exist.
        RuntimeError
            If model generation fails due to resource constraints or other runtime issues.
        """
        image = self._load_image(image_path_or_url)

        # LLaVA uses a different prompt format: "USER: <image>\n{prompt}\nASSISTANT:"
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"You are an expert {persona}. {prompt}"},
                ],
            },
        ]

        # Process the inputs
        prompt_text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(images=image, text=prompt_text, return_tensors="pt").to(
            self.device, torch.bfloat16
        )

        # Generate the response
        output = self.model.generate(**inputs, max_new_tokens=300)

        # Decode and extract the assistant's response
        decoded_text: str = self.processor.decode(output[0], skip_special_tokens=True)

        # Extract only the assistant's response (after "ASSISTANT:")
        if "ASSISTANT:" in decoded_text:
            return decoded_text.split("ASSISTANT:", maxsplit=1)[-1].strip()
        return decoded_text.strip()

    def _load_image(self, image_path_or_url: str) -> Image.Image:
        """
        Load an image from either a local path or a URL.

        Parameters
        ----------
        image_path_or_url : str
            The local file path or web URL of the image.

        Returns
        -------
        PIL.Image.Image
            The loaded image in RGB format.

        Raises
        ------
        requests.exceptions.HTTPError
            If the URL cannot be fetched or returns a bad status code.
        FileNotFoundError
            If the local file path does not exist.
        """
        if image_path_or_url.startswith("http"):
            response = requests.get(
                image_path_or_url, stream=True, headers=_DEFAULT_HTTP_HEADERS
            )
            response.raise_for_status()  # Raise an exception for bad status codes
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_path_or_url).convert("RGB")
        return image
