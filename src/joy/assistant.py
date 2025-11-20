import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .config import AssistantConfig
from .pipelines import build_llm_pipeline, build_stt_pipeline, build_tts_pipeline, resolve_device
from .utils.audio import normalize_audio, save_wav


class JOYAssistant:
    """Coordinates STT → LLM → TTS for JOY."""

    def __init__(self, config: AssistantConfig | None = None):
        self.config = config or AssistantConfig()
        self.device = resolve_device(self.config.device)
        self.stt = build_stt_pipeline(self.config.models.stt_model, self.device)
        self.llm = build_llm_pipeline(self.config.models.llm_model, self.config.generation, self.device)
        self.tts = build_tts_pipeline(self.config.models.tts_model, self.device)
        self.history: List[Tuple[str, str]] = []

    def normalize_text(self, text: str) -> str:
        normalized = text
        if "strip" in self.config.text_normalizers:
            normalized = normalized.strip()
        if "collapse_spaces" in self.config.text_normalizers:
            normalized = re.sub(r"\s+", " ", normalized)
        if "lower_if_probably_command" in self.config.text_normalizers:
            if normalized and normalized[0].islower():
                normalized = normalized.lower()
        return normalized

    def transcribe(self, audio_path: str | Path) -> str:
        result = self.stt(str(audio_path))
        return self.normalize_text(result["text"])

    def _build_prompt(self, user_text: str) -> str:
        prompt_parts = [self.config.safety_preamble, "\n"] if self.config.safety_preamble else []
        recent_history = self.history[-self.config.memory_turns :]
        for user, assistant in recent_history:
            prompt_parts.append(f"User: {user}\nJOY: {assistant}\n")
        prompt_parts.append(f"User: {user_text}\nJOY:")
        return "".join(prompt_parts)

    def reply(self, user_text: str) -> str:
        prompt = self._build_prompt(user_text)
        outputs = self.llm(prompt)
        generated = outputs[0]["generated_text"]
        reply_text = generated[len(prompt) :].strip()
        self.history.append((user_text, reply_text))
        return reply_text

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        audio_dict: Dict[str, np.ndarray | int] = self.tts(text)
        audio = np.array(audio_dict["audio"]).astype(np.float32)
        sample_rate = int(audio_dict["sampling_rate"])
        return normalize_audio(audio), sample_rate

    def process_audio_request(
        self, audio_path: str | Path, output_path: str | Path | None = None
    ) -> Dict[str, str | Tuple[np.ndarray, int]]:
        transcript = self.transcribe(audio_path)
        answer = self.reply(transcript)
        audio, sr = self.synthesize(answer)
        if output_path:
            save_wav(output_path, audio, sr)
        return {"transcript": transcript, "response": answer, "audio": (audio, sr)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run JOY voice assistant on a WAV input file.")
    parser.add_argument("--audio_path", required=True, help="Path to input WAV file.")
    parser.add_argument("--output_path", default="joy_reply.wav", help="Where to store the synthesized reply.")
    parser.add_argument("--llm_model", default=None, help="Override LLM model id.")
    parser.add_argument("--stt_model", default=None, help="Override STT model id.")
    parser.add_argument("--tts_model", default=None, help="Override TTS model id.")
    parser.add_argument("--device", default=None, help="Force device selection (e.g., 'cuda', 'cpu').")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Max tokens for LLM generation.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p sampling parameter.")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Repetition penalty.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = AssistantConfig()
    if args.device:
        config.device = args.device
    if args.llm_model:
        config.models.llm_model = args.llm_model
    if args.stt_model:
        config.models.stt_model = args.stt_model
    if args.tts_model:
        config.models.tts_model = args.tts_model
    if args.max_new_tokens is not None:
        config.generation.max_new_tokens = args.max_new_tokens
    if args.temperature is not None:
        config.generation.temperature = args.temperature
    if args.top_p is not None:
        config.generation.top_p = args.top_p
    if args.repetition_penalty is not None:
        config.generation.repetition_penalty = args.repetition_penalty

    assistant = JOYAssistant(config)
    result = assistant.process_audio_request(args.audio_path, args.output_path)
    print("Transcript:", result["transcript"])
    print("Response:", result["response"])
    print(f"Audio saved to {args.output_path}")


if __name__ == "__main__":
    main()
