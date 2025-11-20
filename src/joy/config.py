from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model identifiers for each modality."""

    llm_model: str = "microsoft/Phi-3-mini-4k-instruct"
    stt_model: str = "distil-whisper/distil-small.en"
    tts_model: str = "facebook/mms-tts-eng"


@dataclass
class GenerationConfig:
    """Sampling parameters for LLM responses."""

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.05


@dataclass
class AssistantConfig:
    """End-to-end assistant configuration."""

    device: Optional[str] = None
    safety_preamble: str = (
        "You are JOY, a warm and insightful AI companion. Keep responses concise, "
        "helpful, and grounded to verifiable knowledge. Avoid roleplay beyond your scope."
    )
    memory_turns: int = 4
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    text_normalizers: List[str] = field(
        default_factory=lambda: ["strip", "collapse_spaces", "lower_if_probably_command"]
    )
