## Relevant Files

- `src/config.py` - Central configuration for `TOOL_IMAGE_MAX_DIMENSION` and `OLLAMA_VISION_MODEL`.
- `src/utils.py` - Core utilities for image file detection, resizing, Base64 encoding, and multimodal `HumanMessage` composition.
- `src/agent.py` - Agent loop orchestrator updated to accept and stream multimodal messages (`images` parameter) into LangGraph.
- `src/gaia_evaluator.py` - GAIA benchmark evaluator updated to directly inject resolved image files into the first inference turn.
- `main.py` - Rich CLI updated to detect dragged-and-dropped/specified image files and forward them to `LilyAgent`.
- `tests/test_utils.py` - Unit tests for image detection, image resizing, Base64 conversion, and token guards.
- `tests/test_agent.py` - Unit tests for multimodal message processing and streaming lifecycle in `LilyAgent`.
- `tests/test_gaia_evaluator.py` - Unit tests for image attachment detection and direct multimodal dispatch in `GAIABenchmarkAgent`.

### Notes

- Base64 images must be resized proportionally using `TOOL_IMAGE_MAX_DIMENSION` (default: 1120px) to prevent context token explosion and GPU VRAM exhaustion.
- Adhere to `langchain_ollama` multi-part content schema: `{"type": "text", "text": ...}` and `{"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}`.
- Maintain defensive fail-safe principles: invalid, corrupt, or unreadable image files must fail gracefully back to text-only mode with descriptive warnings without crashing the agent.
- Run tests using `.\.venv\Scripts\pytest.exe`.
- Run linter using `.\.venv\Scripts\ruff.exe check .`.

## Instructions for Completing Tasks

**IMPORTANT:** As you complete each task, you must check it off in this markdown file by changing `- [ ]` to `- [x]`. This helps track progress and ensures you don't skip any steps.

Example:
- `- [ ] 1.1 Read file` → `- [x] 1.1 Read file` (after completing)

Update the file after completing each sub-task, not just after completing an entire parent task.

## Tasks

- [x] 0.0 Create feature branch
  - [x] 0.1 Create and checkout a new branch for multimodal harness (`git checkout -b feature/multimodal-harness`)
- [x] 1.0 Image Processing & Multimodal Message Composition Utilities
  - [x] 1.1 Implement `is_image_file(file_path: str) -> bool` in `src/utils.py` detecting supported image formats (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.gif`)
  - [x] 1.2 Implement `encode_image_to_base64(image_source: str | bytes, max_dimension: int = TOOL_IMAGE_MAX_DIMENSION) -> str` with PIL auto-resize and EXIF orientation handling
  - [x] 1.3 Implement `build_multimodal_human_message(text: str, images: list[str | bytes] | None = None) -> HumanMessage` conforming to LangChain & Ollama specification
  - [x] 1.4 Add unit tests in `tests/test_utils.py` covering image validation, proportional downscaling, base64 integrity, and graceful handling of corrupt files
- [x] 2.0 Multimodal Capability in LilyAgent Engine
  - [x] 2.1 Update `LilyAgent.chat_stream()` in `src/agent.py` to accept optional `images: list[str | bytes] | None = None`
  - [x] 2.2 Construct multimodal `HumanMessage` dynamically when images are present, preserving system prompt and context trimming
  - [x] 2.3 Update `LilyAgent.chat()` in `src/agent.py` to forward `images` parameter to `chat_stream()`
  - [x] 2.4 Add unit tests in `tests/test_agent.py` verifying multimodal message ingestion and state execution
- [x] 3.0 Direct Vision Injection in GAIA Benchmark Harness
  - [x] 3.1 Update `GAIABenchmarkAgent.__call__()` in `src/gaia_evaluator.py` to inspect `resolved_file` with `is_image_file()`
  - [x] 3.2 Automatically encode and pass image files directly to `LilyAgent.chat_stream(prompt, images=[resolved_file])` on turn 1
  - [x] 3.3 Update `GAIA_SYSTEM_PROMPT` and prompt text to instruct the agent that visual input is directly present, avoiding blind exploratory PIL steps
  - [x] 3.4 Add unit tests in `tests/test_gaia_evaluator.py` verifying that attached image tasks invoke multimodal inference directly
- [ ] 4.0 Interactive Multimodal Support in Terminal CLI (Presentation Layer)
  - [ ] 4.1 Update input parser in `main.py` to detect image paths or drag-and-dropped image files from Windows console
  - [ ] 4.2 Render visual status badges (e.g. `🖼️ Image Attached: <filename>`) in Rich console when an image is loaded
  - [ ] 4.3 Forward detected image paths into `agent.chat_stream(user_input, images=[image_path])`
- [ ] 5.0 End-to-End Verification, Code Quality & Documentation
  - [ ] 5.1 Run full test suite (`pytest`) and Ruff linter (`ruff check .`), ensuring 100% green build
  - [ ] 5.2 Verify multimodal inference end-to-end with sample images (`.png` / `.jpg`)
  - [ ] 5.3 Commit all changes with clear conventional commit messages and push branch to origin
