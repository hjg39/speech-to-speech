from dataclasses import dataclass, field
from typing import Optional

from speech_to_speech.backend_registry import LLM_BACKENDS, STT_BACKENDS, TTS_BACKENDS

_AUDIO_INPUT_LLM_BACKENDS = ", ".join(
    name for name, spec in LLM_BACKENDS.items() if spec.capabilities.supports_audio_input
)
_PROXY_LLM_BACKENDS = ", ".join(name for name, spec in LLM_BACKENDS.items() if spec.capabilities.supports_llm_proxy)


@dataclass
class ModuleArguments:
    device: Optional[str] = field(
        default=None,
        metadata={"help": "If specified, overrides the device for all handlers."},
    )
    mac_optimal_settings: bool = field(
        default=False,
        metadata={
            "help": "If specified, provides macOS defaults: Parakeet TDT for STT, MLX LM for the language "
            "model, Qwen3-TTS for TTS, and MPS for supported component devices. Explicit component, model, "
            "global-device, and component-device flags override these defaults. It does not select a command.",
        },
    )
    stt: Optional[str] = field(
        default="parakeet-tdt",
        metadata={
            "choices": tuple(STT_BACKENDS),
            "help": "The STT to use. Use 'none' to send VAD audio directly to an audio-input LLM. "
            f"Audio-input LLM backends: {_AUDIO_INPUT_LLM_BACKENDS}. Select an explicitly audio-capable "
            "model with --model_name. Default is 'parakeet-tdt'.",
        },
    )
    llm_backend: Optional[str] = field(
        default="responses-api",
        metadata={
            "choices": tuple(LLM_BACKENDS),
            "help": "The LLM backend to use. Default is 'responses-api'.",
        },
    )
    tts: Optional[str] = field(
        default="qwen3",
        metadata={
            "choices": tuple(TTS_BACKENDS),
            "help": "The TTS backend to use. Default is 'qwen3'.",
        },
    )
    log_level: str = field(
        default="info",
        metadata={"help": "Provide logging level. Example --log_level debug, default=info."},
    )
    log_transcripts: bool = field(
        default=False,
        metadata={
            "help": (
                "Write full user and assistant transcript text to the application log, for "
                "debugging STT, LLM, TTS and Realtime flows. Off by default: logs are often "
                "retained by service managers, containers and hosted log aggregators, so "
                "conversation content would outlive the conversation there. Default is False."
            )
        },
    )
    enable_live_transcription: bool = field(
        default=True,
        metadata={
            "help": "Enable live transcription display while user is speaking (works with parakeet-tdt). Default is true."
        },
    )
    enable_partial_prefill: bool = field(
        default=False,
        metadata={"help": "Warm the LLM with one private token after six stable transcript words. Default is false."},
    )
    live_transcription_update_interval: float = field(
        default=0.5,
        metadata={"help": "Update interval for live transcription in seconds (default: 0.5s = 500ms)"},
    )
    live_transcription_min_silence_ms: int = field(
        default=500,
        metadata={
            "help": "Minimum silence duration (ms) before ending speech when live transcription is enabled (default: 500ms)"
        },
    )
    enable_llm_proxy: bool = field(
        default=False,
        metadata={
            "help": f"Expose a proxy-capable LLM backend ({_PROXY_LLM_BACKENDS}) as an "
            "OpenAI-compatible HTTP endpoint on the realtime server. The server performs no authentication of "
            "its own: enable it only on a trusted network or behind a gateway that owns access control. Off by "
            "default."
        },
    )
    llm_proxy_connect_timeout_s: float = field(
        default=10.0,
        metadata={
            "help": "Connect timeout in seconds for LLM proxy requests to the upstream provider. Reads have no "
            "timeout (generation may take minutes). Default is 10.0."
        },
    )
    num_pipelines: int = field(
        default=1,
        metadata={
            "help": "Number of isolated pipeline instances in the pool. One uvicorn server listens on "
            "--port and routes each incoming client to the next free pipeline (each has its own "
            "VAD/STT/LM/TTS handlers and conversation state). Max concurrent websocket sessions equals "
            "num_pipelines; further connections are rejected. Default is 1."
        },
    )
    wake_word_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to an openWakeWord verifier ONNX model ('x' 1x16x96 in, sigmoid out) used "
            "by the WakeGate in front of VAD. When set, the robot is deaf until the wake word is "
            "heard; each detection re-arms listening for wake_word_arm_duration seconds. When unset, "
            "the WakeGate passes audio through unchanged (current behaviour)."
        },
    )
    wake_word_name: str = field(
        default="sciel",
        metadata={"help": "Label key of the wake model used with predict() (default: sciel)."},
    )
    wake_word_threshold: float = field(
        default=0.9,
        metadata={
            "help": "Verifier score threshold for the WakeGate (default: 0.9)."
        },
    )
    wake_word_patience: int = field(
        default=2,
        metadata={
            "help": "Consecutive 80 ms frames above the threshold before a wake alarm. "
            "openWakeWord 0.6.0's built-in patience is deadlocked, so the gate keeps its own "
            "counter (default: 2)."
        },
    )
    wake_word_arm_duration: float = field(
        default=20.0,
        metadata={
            "help": "Seconds the pipeline stays listening after a wake detection (default: 20)."
        },
    )
