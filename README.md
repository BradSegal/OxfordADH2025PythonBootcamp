# Project MedVision: Python for Applied Digital Health

Project MedVision powers the University of Oxford Applied Digital Health (ADH) MSc Python Bootcamp, delivered by the Computational Health Informatics (CHI) Lab. The bootcamp introduces medical and health-data trainees to Python and responsible clinical AI, using a pair of narrative-driven, ready-to-run Jupyter notebooks backed by the `medvision_toolkit` helper library.

## Why This Repository Exists
- **Audience:** ADH students with little or no prior programming experience.
- **Mission:** Deliver a polished, confidence-building learning journey that runs end-to-end inside a free Google Colab runtime.
- **Design:** Each notebook focuses on the student experience; all heavy lifting lives in the typed helper library so learners never face low-level infrastructure code.

## Notebooks at a Glance
Each notebook opens with a “Run Once” setup cell that clones the repository (when needed), installs requirements, and exposes the helper library. Execute cells in order inside Colab for the smoothest experience. Use the Colab badges below to launch the notebooks directly in Google Colab.

- **`00_scrub_in_systems_check.ipynb` (≈40 min):** A systems-readiness drill that walks students through verifying Python versions, confirming dependencies, and practising safe environment hygiene. Includes checklists and guided troubleshooting so learners build confidence before touching clinical data. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/BradSegal/OxfordADH2025PythonBootcamp/blob/master/notebooks/00_scrub_in_systems_check.ipynb)
- **`01_python_first_dose.ipynb` (≈60 min):** The core Python primer framed as morning rounds. Students manipulate typed patient profiles, iterate over vitals, and assemble a structured handover note while challenge cells enforce deliberate practice. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/BradSegal/OxfordADH2025PythonBootcamp/blob/master/notebooks/01_python_first_dose.ipynb)
- **`02_project_medvision_lab.ipynb` (≈90–120 min):** The capstone radiology AI lab. Learners ground imaging prompts in their handover data, compare MedGemma and LLaVA vision-language engines, document model behaviour, and reflect on safety workflows. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/BradSegal/OxfordADH2025PythonBootcamp/blob/master/notebooks/02_project_medvision_lab.ipynb)

## Getting Started in Google Colab
1. Open the notebook in Colab (File → Open Notebook → GitHub tab, paste the repository URL, or upload the `.ipynb` file).
2. Run the very first setup cell. It will clone/update the repo, install dependencies, and add `src/` to `sys.path`. Any failure stops immediately so you can investigate.
3. Continue running cells from top to bottom. Challenge cells intentionally raise `NotImplementedError` until learners supply an answer—this is by design.
4. Restart the runtime (`Runtime → Restart and run all`) if you need a clean slate; the setup cell handles re-provisioning each time.
5. On GPU runtimes, execute the GPU verification cell immediately after setup; it confirms that both PyTorch and llama.cpp loaded CUDA-enabled wheels and explains how to fix issues.

## GPU Runtime Requirements
- Launch Colab with a GPU runtime (`Runtime → Change runtime type → GPU`). The setup cell now raises an explicit error if CUDA is unavailable after the dependency install.
- The setup cell automatically installs the CUDA 12.1 PyTorch and llama.cpp wheels. If a network hiccup forces a CPU-only fallback, rerun the setup cell; the subsequent verification cell will flag any missing GPU support.
- For local development on GPU workstations, manually install a CUDA-enabled PyTorch build before `pip install -r requirements.txt`, then reinstall `llama-cpp-python` with CUDA support (`pip install --extra-index-url https://jllllll.github.io/llama-cpp-python/whl/cu121 llama-cpp-python==0.2.78`).

## Running Locally (Faculty & Contributors)
```bash
git clone https://github.com/BradSegal/OxfordADH2025PythonBootcamp.git
cd python-for-medicine-bootcamp
python -m venv .venv && source .venv/bin/activate  # or use your preferred env tool
pip install -r requirements.txt
# Optional: verify quality gates before committing changes
black .
ruff check . --fix
mypy .
pytest
```

Launch Jupyter Lab/Notebook and open any file in `notebooks/`. Because the setup cells handle installation, the notebooks work identically once the virtual environment is active.

## Core Library Overview (`src/medvision_toolkit`)
All complexity lives in the helper library so notebooks stay approachable. Key modules:

- **`learning.patient_profiles`**
  - Typed `TypedDict` structures for demographics, vitals, medications, and the composite patient profile.
  - Utilities such as `load_sample_patient()`, `compute_vital_statistics()`, `triage_pain_scores()`, and `generate_handover_note()` that enforce validation and raise explicit errors on invalid input.
  - Visual helpers like `plot_vital_trends()` for inline Matplotlib charts.
- **`learning.imaging_helpers`**
  - High-level functions to initialise the MedGemma (`initialize_medgemma_engine`) and LLaVA (`initialize_llava_engine`) backends without exposing model-loading intricacies.
  - Prompt-construction utilities (`build_radiology_prompt`) and display helpers (`load_and_display_image`, `render_ai_report`) that manage HTTP fetching, persona framing, and Markdown rendering.
  - Streaming-friendly display via `stream_ai_report`, which progressively updates notebook output when the GGUF backend is active.
- **`radiology_helpers`**
  - `RadiologyAI`: the production vision-language engine supporting both GGUF (llama.cpp) and Transformers backends, complete with strict input validation and fail-fast error handling.
  - `LlavaAI`: an experimental comparator that mirrors the same `analyze()` contract so students can benchmark outputs side by side.
  - Shared image-loading utilities that differentiate local paths and HTTPS sources while surfacing network/file errors immediately.
  - Unified token budgeting lives in `DEFAULT_CONTEXT_TOKENS` and `DEFAULT_MAX_GENERATED_TOKENS`; both backends read these constants. Adjust them (or pass `context_tokens` / `max_generated_tokens` to `RadiologyAI` and `LlavaAI`) when you need longer reports or tighter limits.

Because interfaces are strictly typed, extending the toolkit means honouring existing contracts—functions and methods should return explicit data structures, raise precise exceptions, and avoid silent fallbacks.

## Extending or Customising the Toolkit
- Keep new functionality inside `medvision_toolkit` so notebooks stay uncluttered. Export high-level helpers that the notebooks can call with minimal arguments.
- Preserve the fail-fast philosophy. If something goes wrong (missing weights, bad prompt, malformed data), raise a specific exception and document the remediation path.
- Add or update tests under `tests/` and validate with `pytest -m "not api"` for fast feedback. Use mocks (`unittest.mock.patch`) to isolate heavy transformer dependencies in unit tests.
- Run the quality gate trio (`black`, `ruff`, `mypy`) before opening a pull request to maintain a consistent learning experience for students.

## Need Support?
- **Students:** Flag issues to your bootcamp facilitators or teaching assistants—capture screenshots or error traces from the setup cell to speed triage.
- **Maintainers:** Use GitHub issues to report bugs or propose enhancements. Please include notebook context, reproduction steps, and any notebook output that demonstrates the problem.

Enjoy exploring clinical AI in Python, and welcome to the Oxford CHI Lab’s MedVision bootcamp!
