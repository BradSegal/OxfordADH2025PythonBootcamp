"""
Radiology AI Helper Module for Project MedVision.

This module provides two AI engines for analyzing medical images:
1. RadiologyAI: Production engine using Google's MedGemma-4B model
2. LlavaAI: Experimental engine using LLaVA-1.5-7B for benchmarking

Both classes abstract away the complexity of model loading, image preprocessing,
prompt templating, and inference, providing a simple `analyze()` interface.
"""

import base64
import importlib.util
import logging
from io import BytesIO
from typing import Any, Callable, Dict, Iterable, Optional, Protocol, Sequence, cast

import requests
import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    LlavaForConditionalGeneration,
    StoppingCriteria,
    StoppingCriteriaList,
)

logger = logging.getLogger(__name__)
_DEFAULT_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0"}
DEFAULT_MAX_IMAGE_EDGE = 256
DEFAULT_STOP_MARKERS: tuple[str, ...] = ("<end_of_turn>", "\nUSER:", "USER:")
DEFAULT_CONTEXT_TOKENS = 8192
DEFAULT_MAX_GENERATED_TOKENS = 1024

_LANCZOS_FILTER: Any
try:
    _LANCZOS_FILTER = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - Pillow < 10 compatibility
    _LANCZOS_FILTER = Image.LANCZOS  # type: ignore[attr-defined]


def _enforce_max_image_edge(image: Image.Image, max_edge: int) -> Image.Image:
    """
    Constrain an image so its longest edge does not exceed ``max_edge`` pixels.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image in RGB mode.
    max_edge : int
        Maximum allowed size for the longest edge. Must be a positive integer.

    Returns
    -------
    PIL.Image.Image
        The original image if already within bounds, otherwise a resized copy.

    Raises
    ------
    ValueError
        If ``max_edge`` is not a positive integer.
    """
    if max_edge <= 0:
        raise ValueError("max_edge must be a positive integer.")

    width, height = image.size
    if max(width, height) <= max_edge:
        return image

    resized = image.copy()
    resized.thumbnail((max_edge, max_edge), _LANCZOS_FILTER)
    return resized


class _ChatProcessor(Protocol):
    """Protocol describing the processor interface required by RadiologyAI."""

    def apply_chat_template(
        self, conversation: Sequence[Dict[str, Any]], add_generation_prompt: bool = ...
    ) -> str: ...

    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]: ...

    def decode(
        self, token_ids: Sequence[int], skip_special_tokens: bool = ...
    ) -> str: ...

    def batch_decode(
        self, sequences: torch.Tensor, skip_special_tokens: bool = ...
    ) -> list[str]: ...


class _StopOnSubstrings(StoppingCriteria):
    """Custom stopping criteria that halts generation on defined substrings."""

    def __init__(self, stop_strings: Sequence[str], processor: _ChatProcessor) -> None:
        self.stop_strings = [marker for marker in stop_strings if marker]
        self.processor = processor

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:  # type: ignore[override]
        if not self.stop_strings:
            return False
        token_ids = input_ids[0].tolist()
        generated_text = self.processor.decode(token_ids, skip_special_tokens=False)
        return any(marker in generated_text for marker in self.stop_strings)


def llama_cpp_has_cuda_support() -> bool:
    """
    Determine whether the llama.cpp Python bindings expose CUDA kernels.

    Returns
    -------
    bool
        ``True`` when the CUDA extension module is present, otherwise ``False``.
    """

    try:
        return importlib.util.find_spec("llama_cpp.llama_cpp_cuda") is not None
    except (ImportError, AttributeError):  # pragma: no cover - defensive guard
        return False


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
    max_image_edge : int, optional
        Maximum size (in pixels) for the image's longest edge before inference.
    """

    def __init__(
        self,
        model_id: str = "unsloth/medgemma-4b-it",
        backend: str = "gguf",
        gguf_filename: str = "medgemma-4b-it-Q4_0.gguf",
        gguf_model_id: str = "unsloth/medgemma-4b-it-GGUF",
        gguf_mmproj: str = "mmproj-F16.gguf",
        gguf_stop: Optional[list[str]] = None,
        max_image_edge: int = DEFAULT_MAX_IMAGE_EDGE,
        context_tokens: Optional[int] = None,
        max_generated_tokens: Optional[int] = None,
        use_sampling: Optional[bool] = None,
        sampling_temperature: Optional[float] = None,
        sampling_top_p: Optional[float] = None,
        sampling_top_k: Optional[int] = None,
    ) -> None:
        """
        Initialize the RadiologyAI engine by loading the MedGemma model.

        This method downloads and loads the model, which may take several minutes
        on the first run. The model is cached for subsequent uses.

        Parameters
        ----------
        model_id : str, optional
            The Hugging Face model identifier (default: "google/medgemma-4b-it").

        max_image_edge : int, optional
            Maximum size for the image's longest edge (default: 256).
        context_tokens : int, optional
            Context window in tokens for text-image conversations. Shared across
            backends; defaults to ``DEFAULT_CONTEXT_TOKENS``.
        max_generated_tokens : int, optional
            Maximum number of tokens generated per response. Shared across
            backends; defaults to ``DEFAULT_MAX_GENERATED_TOKENS``.

        Raises
        ------
        Exception
            If the model fails to load due to network errors, insufficient memory,
            or missing dependencies. The exception will be re-raised to halt execution.
        ValueError
            If ``max_image_edge`` is not a positive integer.
        """
        if max_image_edge <= 0:
            raise ValueError("max_image_edge must be a positive integer.")
        self.context_tokens = context_tokens or DEFAULT_CONTEXT_TOKENS
        if self.context_tokens <= 0:
            raise ValueError("context_tokens must be a positive integer.")
        self.max_generated_tokens = max_generated_tokens or DEFAULT_MAX_GENERATED_TOKENS
        if self.max_generated_tokens <= 0:
            raise ValueError("max_generated_tokens must be a positive integer.")
        self.max_image_edge = max_image_edge
        self.backend = backend.lower()
        self.model_dtype: torch.dtype = torch.float32
        self.chat_handler: Optional[Any] = None
        self.processor: Optional[_ChatProcessor] = None
        # Model Properties
        self.model_id: str = model_id
        self.gguf_filename: str = gguf_filename
        self.gguf_model_id: str = gguf_model_id
        self.gguf_mmproj: str = gguf_mmproj
        self.gguf_stop: Optional[list[str]] = gguf_stop
        # Sampler settings
        self.use_sampling = True if use_sampling is None else bool(use_sampling)
        if sampling_temperature is not None and sampling_temperature <= 0:
            raise ValueError("sampling_temperature must be a positive float.")
        if sampling_top_p is not None and not (0 < sampling_top_p <= 1):
            raise ValueError("sampling_top_p must lie in the interval (0, 1].")
        if sampling_top_k is not None and sampling_top_k <= 0:
            raise ValueError("sampling_top_k must be a positive integer.")
        self.sampling_temperature = sampling_temperature or 0.7
        self.sampling_top_p = sampling_top_p or 0.9
        self.sampling_top_k = sampling_top_k
        base_stop_tokens: list[str] = []
        if gguf_stop is not None:
            for token in gguf_stop:
                if token and token not in base_stop_tokens:
                    base_stop_tokens.append(token)
        for marker in DEFAULT_STOP_MARKERS:
            if marker and marker not in base_stop_tokens:
                base_stop_tokens.append(marker)
        self.stop_tokens: list[str] = base_stop_tokens
        if self.backend == "gguf":
            self.device = "cpu"  # llama.cpp handles its own device usage
            self._init_gguf_backend(
                gguf_model_id,
                gguf_filename,
                gguf_mmproj,
                self.stop_tokens,
                self.context_tokens,
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
            if self.device == "cuda":
                bf16_supported = False
                try:
                    bf16_supported = torch.cuda.is_bf16_supported()
                except AttributeError:  # pragma: no cover - defensive for older torch
                    bf16_supported = False
                self.model_dtype = torch.bfloat16 if bf16_supported else torch.float16
            else:
                self.model_dtype = torch.float32
            logger.info(
                "RadiologyAI (Transformers) selecting dtype: %s", self.model_dtype
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                dtype=self.model_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            self.processor = cast(
                _ChatProcessor, AutoProcessor.from_pretrained(model_id, use_fast=True)
            )
            logger.info("RadiologyAI (Transformers) initialized successfully.")
        except Exception:
            logger.exception(
                "FATAL ERROR: Failed to initialize the Transformers backend."
            )
            raise

    def _init_gguf_backend(
        self,
        repo_id: str,
        filename: str,
        mmproj_filename: str,
        stop_tokens: list[str],
        context_tokens: int,
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
            torch_cuda_available = False
            try:
                torch_cuda_available = torch.cuda.is_available()
            except Exception:
                logger.warning(
                    "Unable to query torch CUDA availability; defaulting to CPU.",
                    exc_info=True,
                )
            llama_cuda_available = llama_cpp_has_cuda_support()
            if torch_cuda_available and not llama_cuda_available:
                logger.warning(
                    "CUDA GPU detected, but llama.cpp bindings lack CUDA support. "
                    "Reinstall llama-cpp-python with CUDA wheels to enable acceleration."
                )

            gpu_available = torch_cuda_available and llama_cuda_available

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
                n_ctx=context_tokens,
                n_gpu_layers=self.gguf_gpu_layers,
                chat_handler=chat_handler,
            )
            self.chat_handler = chat_handler
            self.processor = None
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
        self,
        image_path_or_url: str,
        prompt: str,
        persona: str = "radiologist",
        stream_callback: Optional[Callable[[str], None]] = None,
        *,
        use_sampling: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
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
        stream_callback : Callable[[str], None], optional
            Streaming callback invoked with incremental text chunks when the
            GGUF backend is used. When provided, generation is streamed while
            the final consolidated report is still returned.
        use_sampling : bool, optional
            Override the engine-level sampling flag for this request.
        temperature : float, optional
            Sampling temperature applied when ``use_sampling`` is enabled.
        top_p : float, optional
            Nucleus sampling probability (0 < top_p <= 1) when sampling.
        top_k : int, optional
            Limits sampling to the ``top_k`` most likely tokens when provided.

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
        if temperature is not None and temperature <= 0:
            raise ValueError("temperature must be a positive float.")
        if top_p is not None and not (0 < top_p <= 1):
            raise ValueError("top_p must lie in the interval (0, 1].")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        effective_use_sampling = (
            self.use_sampling if use_sampling is None else bool(use_sampling)
        )
        user_requested_sampling = any(
            value is not None for value in (temperature, top_p, top_k)
        )
        if user_requested_sampling and not effective_use_sampling:
            effective_use_sampling = True
        effective_temperature = (
            temperature if temperature is not None else self.sampling_temperature
        )
        effective_top_p = top_p if top_p is not None else self.sampling_top_p
        effective_top_k = top_k if top_k is not None else self.sampling_top_k

        if self.backend == "gguf":
            return self._analyze_with_llama_cpp(
                image_path_or_url,
                prompt,
                persona,
                stream_callback,
                effective_use_sampling,
                effective_temperature if effective_use_sampling else None,
                effective_top_p if effective_use_sampling else None,
                effective_top_k if effective_use_sampling else None,
            )

        image = self._load_image(image_path_or_url)
        if self.processor is None:
            raise RuntimeError("Transformers processor is not initialised.")
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
            cast(Sequence[Dict[str, Any]], messages), add_generation_prompt=True
        )

        encoded = self.processor(text=prompt_text, images=image, return_tensors="pt")
        input_token_length = (
            encoded["input_ids"].shape[-1] if "input_ids" in encoded else None
        )
        inputs: dict[str, Any] = {}
        for key, value in encoded.items():
            tensor = value.to(self.device) if hasattr(value, "to") else value
            inputs[key] = tensor

        pixel_key = "pixel_values"
        if pixel_key in inputs:
            inputs[pixel_key] = inputs[pixel_key].to(self.model_dtype)

        generate_kwargs: dict[str, Any] = {"max_new_tokens": self.max_generated_tokens}
        if self.backend != "transformers" and self.stop_tokens:
            stopping_list = StoppingCriteriaList(
                [_StopOnSubstrings(self.stop_tokens, self.processor)]
            )
            generate_kwargs["stopping_criteria"] = stopping_list

        generate_kwargs["do_sample"] = effective_use_sampling
        if effective_use_sampling:
            if effective_temperature is not None:
                generate_kwargs["temperature"] = effective_temperature
            if effective_top_p is not None:
                generate_kwargs["top_p"] = effective_top_p
            if effective_top_k is not None:
                generate_kwargs["top_k"] = effective_top_k

        output = self.model.generate(**inputs, **generate_kwargs)
        if input_token_length is not None:
            generated_tokens = output[:, input_token_length:]
        else:
            generated_tokens = output
        decoded_list = self.processor.batch_decode(
            generated_tokens, skip_special_tokens=True
        )
        decoded_text = decoded_list[0] if decoded_list else ""
        return self._sanitize_output(decoded_text)

    def _analyze_with_llama_cpp(
        self,
        image_path_or_url: str,
        prompt: str,
        persona: str,
        stream_callback: Optional[Callable[[str], None]],
        use_sampling: bool,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
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

        llama_kwargs: Dict[str, Any] = {
            "messages": cast(list[Any], messages),
            "max_tokens": self.max_generated_tokens,
            "stop": self.stop_tokens,
        }
        if use_sampling:
            if temperature is not None:
                llama_kwargs["temperature"] = temperature
            if top_p is not None:
                llama_kwargs["top_p"] = top_p
            if top_k is not None:
                llama_kwargs["top_k"] = top_k

        if stream_callback is not None:
            llama_kwargs_stream = dict(llama_kwargs)
            llama_kwargs_stream["stream"] = True
            stream = cast(
                Iterable[Dict[str, Any]],
                self.llm.create_chat_completion(**llama_kwargs_stream),
            )
            chunks: list[str] = []
            for update in stream:
                choices = update.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if isinstance(piece, str):
                    chunks.append(piece)
                    stream_callback(piece)
            raw_text = "".join(chunks)
        else:
            response = cast(
                Dict[str, Any],
                self.llm.create_chat_completion(**llama_kwargs),
            )
            message = response["choices"][0]["message"]
            raw_text = message.get("content") or ""

        return self._sanitize_output(raw_text)

    def _sanitize_output(self, generated_text: str) -> str:
        """Remove chat prefixes and GGUF stop tokens from model output."""

        cleaned = generated_text.strip()
        for marker in (
            "assistant\n",
            "assistant:",
            "model\n",
            "model:",
            "ASSISTANT:",
            "MODEL:",
        ):
            if cleaned.startswith(marker):
                cleaned = cleaned[len(marker) :].strip()

        user_marker = "\nUSER:"
        if user_marker in cleaned:
            cleaned = cleaned.split(user_marker, maxsplit=1)[0].rstrip()

        stop_markers = set(DEFAULT_STOP_MARKERS)
        stop_markers.update(self.stop_tokens)
        for token in stop_markers:
            if token:
                cleaned = cleaned.replace(token, "")

        cleaned = cleaned.strip()
        return cleaned

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
        return _enforce_max_image_edge(image, self.max_image_edge)


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

    def __init__(
        self,
        model_id: str = "llava-hf/llava-1.5-7b-hf",
        max_image_edge: int = DEFAULT_MAX_IMAGE_EDGE,
        max_generated_tokens: Optional[int] = None,
    ) -> None:
        """
        Initialize the LlavaAI engine by loading the LLaVA model.

        This method downloads and loads the model, which may take several minutes
        on the first run. The model is cached for subsequent uses.

        Parameters
        ----------
        model_id : str, optional
            The Hugging Face model identifier (default: "llava-hf/llava-1.5-7b-hf").
        max_image_edge : int, optional
            Maximum size for the image's longest edge (default: 256).

        Raises
        ------
        Exception
            If the model fails to load due to network errors, insufficient memory,
            or missing dependencies. The exception will be re-raised to halt execution.
        ValueError
            If ``max_image_edge`` is not a positive integer.
        """
        logger.info("Initializing LlavaAI with model: %s...", model_id)
        logger.info(
            "LlavaAI initialization may take several minutes while the model downloads."
        )

        try:
            if max_image_edge <= 0:
                raise ValueError("max_image_edge must be a positive integer.")
            self.max_image_edge = max_image_edge
            self.max_generated_tokens = (
                max_generated_tokens or DEFAULT_MAX_GENERATED_TOKENS
            )
            if self.max_generated_tokens <= 0:
                raise ValueError("max_generated_tokens must be a positive integer.")
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
        output = self.model.generate(**inputs, max_new_tokens=self.max_generated_tokens)

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
        return _enforce_max_image_edge(image, self.max_image_edge)
