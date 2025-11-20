# JOY Voice Assistant Blueprint

This repository provides a lightweight yet advanced voice assistant stack inspired by **Blade Runner 2049's JOY**. It stitches together speech recognition, small-footprint LLM reasoning, and speech synthesis while keeping GPU memory usage within ~12 GB via quantization and efficient loading options.

## Architecture

- **Speech-to-Text (STT):** Hugging Face `automatic-speech-recognition` pipeline (e.g., `distil-whisper/distil-small.en` for low latency and memory).
- **LLM Response:** Quantized causal language model (defaults to `microsoft/Phi-3-mini-4k-instruct`) loaded with 4-bit `bitsandbytes` to keep VRAM low.
- **Text-to-Speech (TTS):** Hugging Face `text-to-speech` pipeline (defaults to `facebook/mms-tts-eng`) for fast speech synthesis.
- **Coordinator:** `JOYAssistant` orchestrates STT → dialog → TTS with context tracking, safety filters, and streaming-friendly hooks.

## Getting Started

1. **Install dependencies** (Python 3.10+):
   ```bash
   pip install -e .
   ```

2. **Run the demo** with an input WAV file (16 kHz mono recommended):
   ```bash
   python -m joy.assistant --audio_path path/to/input.wav --output_path reply.wav
   ```

3. **Customize models** via CLI flags:
   ```bash
   python -m joy.assistant \
     --llm_model microsoft/Phi-3-mini-4k-instruct \
     --stt_model distil-whisper/distil-small.en \
     --tts_model facebook/mms-tts-eng
   ```

## Implementation Highlights

- **Quantization for VRAM efficiency:** LLMs are loaded with 4-bit NF4 quantization; STT uses float16 when CUDA is available.
- **Context-aware replies:** The assistant preserves prior turns and optionally injects a safety preamble to keep outputs grounded.
- **Audio utilities:** Normalization and WAV export are built-in for easy evaluation.
- **Extensible:** Swap models by ID, adjust generation parameters, or integrate tool-calling by extending `JOYAssistant`.

## Notes

- GPU is optional; the pipelines automatically fall back to CPU (with slower performance).
- Ensure FFmpeg is available for robust audio handling (used by `pydub`).
- The default models are sized to run on a 12 GB GPU; larger checkpoints can work with more aggressive offloading.
