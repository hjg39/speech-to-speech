"""WakeGate: server-side wake-word gate on the input audio stream.

Sits in the queue chain *ahead* of VADHandler:
    recv_audio_chunks_queue -> WakeGate -> wake_gate_queue -> VAD -> ...

Design (locked):
  * SLEEP state (not speaking, not in the armed window): the openWakeWord
    verifier runs on every 80 ms of mic audio. Audio is SWALLOWED — the robot
    is deaf.
  * On wake detection: set ``should_listen`` (arm the downstream VAD for a
    ~20 s window) and emit ``WakeDetectedEvent`` on the text_output_queue so
    the realtime client can play a fast expression cue. The "Sciel" word
    itself stays inside the verifier's rolling buffer and is swallowed, and a
    brief post-fire window of silence is also swallowed so the VAD (if the
    wake word ever reaches it) gets trailing silence to close its segment —
    the robot never transcribes or answers its own wake word.
  * SELF-TRIGGER guard: while the robot is speaking (response_playing) the
    verifier is not run and audio is swallowed, so its own TTS cannot
    re-wake it.
  * LISTEN state (in the armed window, not speaking): audio forwards
    byte-for-byte to VAD.
  * When the gate is NOT enabled (no model) it is a pure pass-through, so a
    deployment without wake-wording behaves exactly like the original chain.

The gate OWNS its wake state via ``_awake``. It does NOT read the shared
``should_listen`` event to decide whether it is awake: that flag is created
unset but is re-set by many actors (``_clean_unit`` on every unit release,
response-complete, the VAD's ``on_session_end``) and persists on pooled units,
so it is already True by the time a new session starts. If the gate keyed off
it, it would be perpetually awake and the verifier would never run. Instead
the gate decides awake/sleep from ``_awake`` and *actively* sets/clears
``should_listen`` as a downstream signal for the VAD.

Chunk contract: the gate forwards the ORIGINAL input item byte-for-byte (a
``(bytes, RuntimeConfig)`` tuple or plain ``bytes``), never re-chunked
frames. The Silero VAD iterator asserts a fixed 512-sample (16 kHz) per-call
contract (the service re-chunks the wire stream to 512 before the pipeline),
so the gate must pass the client's own frame sizes through untouched. The
1280-sample (80 ms) framing for ``predict()`` lives only in the gate's
internal buffer.

openWakeWord 0.6.0's ``predict(patience=...)`` is DEADLOCKED — the patience
check reads ``prediction_buffer[-N:]`` BEFORE the current frame is appended,
and the buffer is seeded with zeros (5-frame init), so a rising spike never
sees N consecutive >= thr among previous frames. Verified on the official
alexa model with a forced-constant 0.99 score: 0 emissions WITH patience,
20/25 WITHOUT. The gate therefore applies its OWN patience counter to the raw
``predict()`` score.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from time import perf_counter
from typing import Iterator

import numpy as np
import openwakeword

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.events import WakeDetectedEvent
from speech_to_speech.pipeline.handler_types import VADIn

logger = logging.getLogger(__name__)

FRAME_SAMPLES = 1280  # 80 ms at 16 kHz
BYTES_PER_FRAME = FRAME_SAMPLES * 2
REFRACTORY_S = 0.5      # 0.5 s after a fire: keep sleeping (same word can't re-trigger)
# Brief swallow AFTER the wake fires, sized to the wake-word TAIL that the
# verifier does not consume: the v4 model fires ~0.5 s into the word and the
# word (with its trailing silence) runs another ~1.1 s on the calibration
# clips, so the tail stays in the 1.28 s rolling window and keeps re-firing.
# Swallowing that window keeps the "Sciel" word from opening a phantom VAD
# segment the LLM would transcribe/answer. Set to 0.0 to let the tail through
# instead (noisy, but loses none of a fast "Sciel, ...").
POST_FIRE_SWALLOW_S = 1.2


@dataclass
class WakeGateConfig:
    """Wake-gate tuning (defaults are the calibrated v4 values)."""

    model_path: str
    name: str = "sciel"
    threshold: float = 0.9
    patience: int = 2            # consecutive 80 ms frames >= threshold
    arm_duration_s: float = 20.0


class WakeGate(BaseHandler[VADIn, VADIn]):
    """Gate the VAD stage on an audio wake word. Passes raw audio through."""

    def __init__(
        self,
        stop_event,
        queue_in,
        queue_out,
        setup_args: tuple = (),
        setup_kwargs: dict = {},
    ) -> None:
        # BaseHandler.__init__ runs setup(), and setup() is the ONLY owner of
        # the gate's runtime state (it loads the openWakeWord model). Do NOT
        # re-initialize any of that here: any assignment after super().__init__()
        # would wipe what setup() just built (this exact bug left the live gate
        # deaf — the v4 RNN was loaded by setup(), then `self._model = None`
        # nulled it 0 ms later, before the first audio frame; the offline smoke
        # test called setup() directly and so missed it). __init__ is a pure
        # pass-through to the base handler.
        super().__init__(stop_event, queue_in, queue_out, setup_args, setup_kwargs)
        self.cfg: WakeGateConfig
        self.should_listen
        self.response_playing
        self._lock: threading.Lock

    def setup(
        self,
        cfg: WakeGateConfig,
        should_listen,
        response_playing,
        *,
        text_output_queue=None,
    ) -> None:
        self.cfg = cfg
        self.should_listen = should_listen
        self.response_playing = response_playing
        self.text_output_queue = text_output_queue
        self._lock = threading.Lock()
        self._model = None
        self.enabled = bool(cfg.model_path)
        if self.enabled:
            self._model = openwakeword.Model(
                wakeword_models=[cfg.model_path], inference_framework="onnx"
            )
            if self._model is None:
                logger.error("WakeGate: model load returned None; running in pass-through mode")
                self.enabled = False
        else:
            logger.warning("WakeGate: no model path; running in pass-through mode")
        self._buf = bytearray()
        self._run = 0
        self._awake = False
        self._awake_logged = False
        self._armed_until = 0.0
        self._swallow_until = 0.0
        self._last_fire = 0.0
        self._hits_disabled = 0
        self._hits_speak = 0
        self._hits_awake = 0
        self._hits_refrac = 0
        self._hits_asleep = 0
        self._asleep_peak = 0.0
        self._watchdog = threading.Thread(
            target=self._watch_loop, daemon=True, name="wake-gate-watchdog"
        )
        self._watchdog.start()
        logger.info(
            "WakeGate: %s model %s thr %.2f patience %d arm %.0fs",
            "enabled," if self.enabled else "DISABLED,",
            cfg.model_path or "(none)", cfg.threshold, cfg.patience, cfg.arm_duration_s,
        )

    def _watch_loop(self) -> None:
        """Expiry watchdog: after the arm window elapses -> SLEEP.

        Independent of whether Harry spoke during the window (and of the _run
        counter, which freezes at the last frame's score) — otherwise a turn
        in flight at expiry would leave the robot listening forever. The gate
        is the sole authority on expiry. We never clear should_listen while
        the robot is speaking (response_playing): it would just be re-set at
        response-end, and the robot is expected to stay armed for Harry's
        reply during its own TTS.
        """
        self._awake = False
        self._awake_logged = False
        while not self.stop_event.is_set():
            try:
                if self.enabled and self._awake and perf_counter() >= self._armed_until:
                    self._awake = False
                    self._armed_until = 0.0
                    self._run = 0
                    self._buf.clear()
                    if not self.response_playing.is_set():
                        self.should_listen.clear()
                        if self.text_output_queue is not None:
                            self.text_output_queue.put(
                                WakeDetectedEvent(now=perf_counter(), arm_state="expired")
                            )
                        logger.info("WakeGate: arm window elapsed; back to SLEEP")
            except Exception as exc:  # pragma: no cover - watchdog never kills the gate
                logger.warning("WakeGate watchdog: %s", exc)
            self.stop_event.wait(0.5)

    @staticmethod
    def _is_audio(item) -> bool:
        if isinstance(item, (bytes, bytearray)):
            return True
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], (bytes, bytearray)):
            return True
        return False

    def process(self, input: VADIn) -> Iterator[VADIn]:
        rt_cfg = None
        if isinstance(input, tuple):
            input, rt_cfg = input
        if not self._is_audio(input):
            # Control messages / non-audio: pass through untouched.
            if rt_cfg is not None:
                yield (input, rt_cfg)
            else:
                yield input
            return
        if not self.enabled:
            # No model loaded: behave exactly like the original chain.
            self._hits_disabled += 1
            yield (input, rt_cfg) if rt_cfg is not None else input
            return

        out = (input, rt_cfg) if rt_cfg is not None else input
        now = perf_counter()
        speaking = self.response_playing.is_set()
        if speaking or now < self._swallow_until:
            # Robot TTS, or the brief post-fire window: swallow. The VAD would
            # drop it anyway (should_listen clear) and the self-trigger guard
            # keeps the verifier asleep.
            self._hits_speak += 1
            self._buf.clear()
            return

        if self._awake:
            # In the armed window: forward byte-for-byte. Do not accumulate —
            # the buffer would otherwise hold stale audio across the window.
            self._hits_awake += 1
            self._buf.clear()
            if not self._awake_logged:
                self._awake_logged = True
                logger.info(
                    "WakeGate: forward branch active (_awake=True, should_listen=%s)",
                    self.should_listen.is_set(),
                )
            yield out
            return

        # Asleep: accumulate into the verifier's 1280-sample frame buffer and
        # run it. Refractory: 0.5 s after a fire, keep sleeping so the same
        # "Sciel" word can't re-trigger. We never re-chunk the downstream
        # output (see module docstring: 512-sample VAD contract).
        if now - self._last_fire < REFRACTORY_S:
            self._hits_refrac += 1
            self._buf.clear()
            return
        self._hits_asleep += 1
        self._buf += input
        if len(self._buf) > 65536:
            self._buf = self._buf[-65536:]
        n_frames, peak = self._predict_chunk(now)
        self._asleep_peak = max(self._asleep_peak, peak)
        if self._hits_asleep % 128 == 0:
            # Periodic health line (~10 s of audio): proves the verifier is
            # running and shows the score floor. Silence should stay well
            # under the threshold; a sustained rise means the model, the
            # stream, or the threshold drifted.
            logger.info(
                "WakeGate: asleep, window peak score %.3f (thr %.2f)",
                self._asleep_peak, self.cfg.threshold,
            )
            self._asleep_peak = 0.0

    def _predict_chunk(self, now: float):
        """Drain the accumulator into 1280-sample frames and run the verifier.

        Returns (n_frames, peak_score). A fire is handled inside _predict
        (sets _awake, emits the realtime event); the "Sciel" word itself stays
        in the verifier's rolling buffer and is swallowed, so the VAD never
        transcribes or answers its own wake word.
        """
        if self._model is None:
            return (0, 0.0)
        n_frames = 0
        peak = 0.0
        while len(self._buf) >= BYTES_PER_FRAME:
            frame = bytes(self._buf[:BYTES_PER_FRAME])
            del self._buf[:BYTES_PER_FRAME]
            peak = max(peak, self._predict(frame, now))
            n_frames += 1
        return (n_frames, peak)

    def _predict(self, frame: bytes, now: float) -> float:
        """One raw 80 ms predict + own patience counter. Returns the score."""
        if self._model is None:
            return 0.0
        x = np.frombuffer(frame, dtype=np.int16)
        if len(x) != FRAME_SAMPLES:
            return 0.0
        with self._lock:
            score = float(self._model.predict(x).get(self.cfg.name, 0.0))
        self._run = self._run + 1 if score >= self.cfg.threshold else 0
        if self._run >= self.cfg.patience:
            self._run = 0
            self._awake = True
            self._armed_until = now + self.cfg.arm_duration_s
            self._last_fire = now
            self._swallow_until = now + POST_FIRE_SWALLOW_S
            self.should_listen.set()
            if self.text_output_queue is not None:
                self.text_output_queue.put(
                    WakeDetectedEvent(now=now, arm_state="armed")
                )
            logger.info(
                "WakeGate: wake detected (score %.3f); armed %.0f s",
                score, self.cfg.arm_duration_s,
            )
        return score

    # ── lifecycle ───────────────────────────────────────────────────────

    def on_session_end(self) -> None:
        was_awake = self._awake
        self._awake = False
        self._awake_logged = False
        self._armed_until = 0.0
        self._run = 0
        self._swallow_until = 0.0
        self._buf.clear()
        if self._model is not None:
            self._model.reset()
        # If we were in the armed window, tell the client to drop the cue and
        # re-sleep. Never clear should_listen while the robot is speaking —
        # it would just be re-set at response-end. (The VAD's own
        # on_session_end re-sets should_listen after this runs, but that is
        # harmless: the gate is now asleep and swallows all input, so the VAD
        # receives none until the next wake.)
        if was_awake and not self.response_playing.is_set():
            self.should_listen.clear()
            if self.text_output_queue is not None:
                self.text_output_queue.put(
                    WakeDetectedEvent(now=perf_counter(), arm_state="expired")
                )
        logger.info(
            "WakeGate: session end — branches: swallow=%d awake=%d refractory=%d asleep=%d; "
            "awake=%s should_listen=%s response_playing=%s",
            self._hits_speak, self._hits_awake, self._hits_refrac, self._hits_asleep,
            was_awake, self.should_listen.is_set(), self.response_playing.is_set(),
        )
        self._hits_disabled = 0
        self._hits_speak = 0
        self._hits_awake = 0
        self._hits_refrac = 0
        self._hits_asleep = 0
        self._asleep_peak = 0.0

    def cleanup(self) -> None:
        if self._model is not None:
            try:
                self._model.reset()
            except Exception:  # pragma: no cover - best-effort teardown
                pass
        self._model = None
        self._buf.clear()
        super().cleanup()
