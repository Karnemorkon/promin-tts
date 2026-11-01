#!/usr/bin/env python3
"""Export patriotyk/HolosTTS PyTorch weights to the Promin `holos-onnx` bundle.

The bundle is a directory of ONNX graphs plus a JSON manifest and a NumPy
voice archive.  Promin's Rust runtime (`crates/speech-inference`) runs the
stages in this order:

    encoder:  tokens[1,T], voice[1,256], speed[1]
              -> acoustic_tokens[1,512,T], prosodic_tokens[1,T,512], durations[1,T]

    prosody:  prosodic_tokens[1,T,512], frame_to_token[F], voice[1,256]
              -> f0[1,2F], energy[1,2F]

    decoder:  acoustic_tokens[1,512,T], frame_to_token[F],
              f0[1,2F], energy[1,2F], voice[1,256]
              -> audio[1,600F]

The optional streaming pair is:

    source:         f0[1,2F] -> source[1,600F]
    decoder_window: acoustic_tokens, frame_to_token, f0, energy, source, voice
                    -> audio[1,600F]

Everything is exported at the token level: the Rust side builds
`frame_to_token` from the predicted durations and the prosody/decoder graphs
gather frames internally.  This keeps the ONNX graphs independent of text
length and lets the server stream arbitrary windows.

Usage:
    python tools/export_holos_bundle.py \
        --repo /path/to/HolosTTS \
        --weights /path/to/holos.pt \
        --voices /path/to/voices.pt \
        --out /path/to/holos-onnx

The script needs `torch`, `numpy`, `onnx`, and (for validation)
`onnxruntime`.  Upstream HolosTTS imports `torchaudio`, `transformers`,
`librosa`, and `soundfile` even though the TTS model does not use them; those
imports are stubbed automatically when the packages are absent.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# Must be set before importing upstream `holos.modules`.
_FULL_LENGTH = 1_000_000
_SAMPLES_PER_FRAME = 600
_SAMPLE_RATE = 24_000


# ---------------------------------------------------------------------------
# Upstream import stubs
# ---------------------------------------------------------------------------


def _module_importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def install_import_stubs() -> None:
    """Stub optional upstream imports that the TTS model never executes."""
    if not _module_importable("torchaudio.models"):
        torchaudio = types.ModuleType("torchaudio")
        models = types.ModuleType("torchaudio.models")

        class Conformer:  # pragma: no cover - only used by VoiceEncoder
            pass

        models.Conformer = Conformer
        torchaudio.models = models
        sys.modules["torchaudio"] = torchaudio
        sys.modules["torchaudio.models"] = models

    if not _module_importable("transformers"):
        transformers = types.ModuleType("transformers")
        transformers.AlbertConfig = object
        transformers.AlbertModel = object
        sys.modules["transformers"] = transformers

    for name in ("librosa", "soundfile"):
        if not _module_importable(name):
            sys.modules[name] = types.ModuleType(name)

    if not _module_importable("huggingface_hub"):
        hub = types.ModuleType("huggingface_hub")

        def hf_hub_download(*_args: Any, **_kwargs: Any) -> str:
            raise RuntimeError(
                "huggingface_hub is required only for remote weights; "
                "pass local --weights/--voices paths"
            )

        hub.hf_hub_download = hf_hub_download
        sys.modules["huggingface_hub"] = hub


# ---------------------------------------------------------------------------
# Wrappers around the upstream modules
# ---------------------------------------------------------------------------


class EncoderExport(nn.Module):
    """Token-level text encoder + duration predictor."""

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.prosodic_text_encoder = model.prosodic_text_encoder
        self.duration_predictor = model.duration_predictor
        self.text_encoder = model.text_encoder

    def forward(
        self, tokens: torch.Tensor, voice: torch.Tensor, speed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths = torch.full(
            (tokens.shape[0],),
            tokens.shape[1],
            dtype=torch.long,
            device=tokens.device,
        )
        prosodic = self.prosodic_text_encoder(tokens, lengths)  # [1, 512, T]
        prosodic_tokens = prosodic.transpose(1, 2)  # [1, T, 512]
        logits = self.duration_predictor(
            prosodic_tokens, voice[:, 128:], lengths
        )  # [1, T, max_dur]
        duration = torch.sigmoid(logits).sum(dim=-1) / speed[:, None]
        durations = torch.round(duration).to(torch.int64)
        acoustic_tokens = self.text_encoder(tokens, lengths)  # [1, 512, T]
        return acoustic_tokens, prosodic_tokens, durations


class ProsodyExport(nn.Module):
    """Gather token features to frames, then predict f0/energy."""

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.predictor = model.prosodic_predictor

    def forward(
        self,
        prosodic_tokens: torch.Tensor,
        frame_to_token: torch.Tensor,
        voice: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        en = torch.index_select(prosodic_tokens, 1, frame_to_token).transpose(1, 2)
        # The upstream predictor uses lengths only for padding masks.  Every
        # frame in a bundle window is real, so a large constant makes all masks
        # all-ones and keeps the graph independent of the frame count.
        lengths = torch.full(
            (1,), _FULL_LENGTH, dtype=torch.long, device=en.device
        )
        f0, energy = self.predictor(en, voice[:, 128:], lengths)
        return f0, energy


def _harmonic_source(
    generator: nn.Module, f0: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """f0[1,2F] -> source[1,600F] and its STFT features [1,42,60F+1]."""
    f0_upsampled = generator.f0_upsamp(f0[:, None])  # [1,1,600F]
    source = generator.m_source(f0_upsampled.transpose(1, 2))
    source = source.transpose(1, 2)[:, 0, :]  # [1,600F]
    return source, generator.harmonic_features(source)


def _decode_frames(
    decoder: nn.Module,
    asr: torch.Tensor,
    f0: torch.Tensor,
    energy: torch.Tensor,
    style: torch.Tensor,
    harmonic: torch.Tensor,
) -> torch.Tensor:
    """Run the decoder body and the generator with precomputed harmonics.

    This mirrors `Decoder.forward(..., lengths=ones)` followed by
    `Generator.forward(..., lengths=ones)`, without the padding masks: every
    frame in a bundle window is real.  The final generator upsample adds a
    reflection pad, so the main branch is one frame longer than the harmonic
    source; `Generator.forward` crops to the shorter tensor.  Doing the same
    crop here keeps the output at exactly `600 * frames` samples.
    """
    generator = decoder.generator

    f0_conv = decoder.F0_conv(f0.unsqueeze(1))
    energy_conv = decoder.N_conv(energy.unsqueeze(1))
    x = torch.cat([asr, f0_conv, energy_conv], dim=1)
    x = decoder.encode(x, style)

    asr_res = decoder.asr_res(asr)
    residual = True
    for block in decoder.decode:
        if residual:
            x = torch.cat([x, asr_res, f0_conv, energy_conv], dim=1)
        x = block(x, style)
        if block.upsample_type != "none":
            residual = False

    for index in range(generator.num_upsamples):
        x = F.leaky_relu(x, 0.1)
        x = generator.ups[index](x)
        if index == generator.num_upsamples - 1:
            x = generator.reflection_pad(x)
        x_source = generator.noise_convs[index](harmonic)
        if index == generator.num_upsamples - 1:
            x = x[..., :-1]
        x_source = generator.noise_res[index](x_source, style)
        x = x + x_source

        summed = None
        for kernel in range(generator.num_kernels):
            block_out = generator.resblocks[index * generator.num_kernels + kernel](
                x, style
            )
            summed = block_out if summed is None else summed + block_out
        x = summed / generator.num_kernels

    x = F.leaky_relu(x)
    x = generator.conv_post(x)
    magnitude = torch.exp(x[:, : generator.post_n_fft // 2 + 1, :])
    phase = torch.sin(x[:, generator.post_n_fft // 2 + 1 :, :])
    return generator.stft.inverse(magnitude, phase).squeeze(1)


class DecoderExport(nn.Module):
    """Exact decoder for a complete utterance."""

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.decoder = model.decoder

    def forward(
        self,
        acoustic_tokens: torch.Tensor,
        frame_to_token: torch.Tensor,
        f0: torch.Tensor,
        energy: torch.Tensor,
        voice: torch.Tensor,
    ) -> torch.Tensor:
        asr = torch.index_select(acoustic_tokens, 2, frame_to_token)
        _, harmonic = _harmonic_source(self.decoder.generator, f0)
        return _decode_frames(
            self.decoder, asr, f0, energy, voice[:, :128], harmonic
        )


class SourceExport(nn.Module):
    """Harmonic excitation for the streaming decoder."""

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.generator = model.decoder.generator

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        source, _ = _harmonic_source(self.generator, f0)
        return source


class DecoderWindowExport(nn.Module):
    """Streaming decoder that consumes a precomputed source window."""

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.decoder = model.decoder

    def forward(
        self,
        acoustic_tokens: torch.Tensor,
        frame_to_token: torch.Tensor,
        f0: torch.Tensor,
        energy: torch.Tensor,
        source: torch.Tensor,
        voice: torch.Tensor,
    ) -> torch.Tensor:
        asr = torch.index_select(acoustic_tokens, 2, frame_to_token)
        harmonic = self.decoder.generator.harmonic_features(source)
        return _decode_frames(
            self.decoder, asr, f0, energy, voice[:, :128], harmonic
        )


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def inline_external_data(path: Path) -> None:
    """Fold a `.onnx.data` sidecar back into the model file."""
    import onnx

    data_file = Path(str(path) + ".data")
    if not data_file.exists():
        return
    model = onnx.load(str(path))
    onnx.save_model(model, str(path), save_as_external_data=False)
    data_file.unlink()
    print(f"[export] inlined external data from {data_file.name}")


def export_stage(
    wrapper: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    path: Path,
    opset: int,
) -> None:
    wrapper.eval()
    started = time.time()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example_inputs,
            str(path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    inline_external_data(path)
    size_mb = path.stat().st_size / 1e6
    print(f"[export] {path.name}: {size_mb:.1f} MB in {time.time() - started:.1f}s")


def write_voices(path: Path, voices: dict[str, torch.Tensor]) -> None:
    names = np.array(list(voices.keys()), dtype=np.str_)
    values = np.stack(
        [voice.detach().cpu().numpy().astype(np.float32).reshape(-1) for voice in voices.values()]
    )
    np.savez(path, names=names, values=values)
    print(f"[voices] {len(voices)} presets -> {path.name} {path.stat().st_size / 1024:.0f} KB")


def write_manifest(
    path: Path,
    *,
    vocab: list[str],
    max_frames: int,
    chunk_frames: int,
    context_frames: int,
    streaming: bool,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "format": "holos-onnx",
        "format_version": 2,
        "encoder": "encoder.onnx",
        "prosody": "prosody.onnx",
        "decoder": "decoder.onnx",
        "voices": "voices.npz",
        "vocab": vocab,
        "sample_rate": _SAMPLE_RATE,
        "samples_per_frame": _SAMPLES_PER_FRAME,
        "max_frames": max_frames,
        "decoder_mode": "exact",
        "alignment_format": "gather",
    }
    if streaming:
        manifest["streaming"] = {
            "source": "source.onnx",
            "decoder": "decoder_window.onnx",
            "chunk_frames": chunk_frames,
            "context_frames": context_frames,
        }
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[manifest] {path.name}: vocab={len(vocab)} max_frames={max_frames}")
    return manifest


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else float("nan")


def _tensor(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_bundle(
    out_dir: Path,
    model: Any,
    manifest: dict[str, Any],
    samples: list[str],
) -> dict[str, Any]:
    import onnxruntime as ort

    def session(name: str) -> ort.InferenceSession:
        return ort.InferenceSession(
            str(out_dir / name), providers=["CPUExecutionProvider"]
        )

    encoder = session("encoder.onnx")
    prosody = session("prosody.onnx")
    decoder = session("decoder.onnx")
    streaming = "streaming" in manifest
    source_session = session("source.onnx") if streaming else None
    window_session = session("decoder_window.onnx") if streaming else None

    report: dict[str, Any] = {"samples": [], "arbitrary_frames": [], "streaming": streaming}
    voice = model.voices["Speaker_0"].unsqueeze(0)
    voice_np = _tensor(voice)

    for text in samples:
        tokens = [0] + model.tokenizer(text) + [0]
        _check(all(0 <= token < len(manifest["vocab"]) for token in tokens),
               f"token outside manifest vocab for {text!r}: {tokens}")
        token_tensor = torch.tensor(tokens).unsqueeze(0)
        token_count = token_tensor.size(1)
        lengths = torch.tensor([token_count])

        acoustic_ref = model.text_encoder(token_tensor, lengths)
        prosodic_ref = model.prosodic_text_encoder(token_tensor, lengths).transpose(1, 2)
        encoder_ref = EncoderExport(model)
        duration_sums: dict[float, int] = {}

        for speed in (1.0, 0.7, 1.4):
            acoustic, prosodic_tokens, durations = encoder.run(
                ["acoustic_tokens", "prosodic_tokens", "durations"],
                {
                    "tokens": np.array([tokens], dtype=np.int64),
                    "voice": voice_np,
                    "speed": np.array([speed], dtype=np.float32),
                },
            )
            expected = encoder_ref(
                token_tensor, voice, torch.tensor([speed])
            )
            _check(acoustic.shape == (1, 512, token_count), f"acoustic shape {acoustic.shape}")
            _check(prosodic_tokens.shape == (1, token_count, 512), f"prosodic shape {prosodic_tokens.shape}")
            _check(durations.shape == (1, token_count), f"durations shape {durations.shape}")
            _check(durations[0].tolist() == expected[2][0].tolist(),
                   f"durations mismatch at speed {speed}: {durations} vs {expected[2]}")
            duration_sums[speed] = int(durations.sum())
            if speed == 1.0:
                _check(np.allclose(acoustic, _tensor(acoustic_ref), atol=3e-5),
                       f"acoustic parity failed for {text!r}")
                _check(np.allclose(prosodic_tokens, _tensor(prosodic_ref), atol=3e-5),
                       f"prosodic parity failed for {text!r}")

        _check(duration_sums[0.7] >= duration_sums[1.0] >= duration_sums[1.4],
               f"speed did not scale durations monotonically: {duration_sums}")

        durations = encoder.run(
            ["durations"],
            {
                "tokens": np.array([tokens], dtype=np.int64),
                "voice": voice_np,
                "speed": np.array([1.0], dtype=np.float32),
            },
        )[0][0].astype(np.int64)
        frames = int(durations.sum())
        frame_to_token = np.repeat(np.arange(token_count), durations)
        _check(frame_to_token.size == frames, "frame_to_token length mismatch")

        f0_ref, energy_ref = model.prosodic_predictor(
            torch.index_select(prosodic_ref, 1, torch.from_numpy(frame_to_token)).transpose(1, 2),
            voice[:, 128:],
            torch.tensor([2 * frames]),
        )
        f0, energy = prosody.run(
            ["f0", "energy"],
            {
                "prosodic_tokens": prosodic_tokens,
                "frame_to_token": frame_to_token,
                "voice": voice_np,
            },
        )
        _check(f0.shape == (1, 2 * frames) and energy.shape == (1, 2 * frames),
               f"prosody shapes {f0.shape}/{energy.shape}")
        _check(np.allclose(f0, _tensor(f0_ref), atol=3e-4), f"f0 parity failed for {text!r}")
        _check(np.allclose(energy, _tensor(energy_ref), atol=3e-4),
               f"energy parity failed for {text!r}")

        acoustic, prosodic_tokens, _ = encoder.run(
            ["acoustic_tokens", "prosodic_tokens", "durations"],
            {
                "tokens": np.array([tokens], dtype=np.int64),
                "voice": voice_np,
                "speed": np.array([1.0], dtype=np.float32),
            },
        )
        asr = torch.index_select(torch.from_numpy(acoustic), 2, torch.from_numpy(frame_to_token))

        audio = decoder.run(
            ["audio"],
            {
                "acoustic_tokens": acoustic,
                "frame_to_token": frame_to_token,
                "f0": f0,
                "energy": energy,
                "voice": voice_np,
            },
        )[0]
        expected_samples = frames * _SAMPLES_PER_FRAME
        _check(audio.shape == (1, expected_samples), f"decoder shape {audio.shape}, expected {expected_samples}")
        _check(np.isfinite(audio).all(), "decoder produced NaN/Inf")
        decoder_ref = model.decoder(
            asr,
            torch.from_numpy(f0),
            torch.from_numpy(energy),
            voice[:, :128],
            lengths=torch.tensor([2 * frames]),
        )
        decoder_ref_np = _tensor(decoder_ref).reshape(-1)
        audio_np = audio.reshape(-1)
        rms_ratio = float(
            np.sqrt(np.mean(audio_np ** 2)) / (np.sqrt(np.mean(decoder_ref_np ** 2)) + 1e-9)
        )
        decoder_cos = _cosine(audio_np, decoder_ref_np)
        _check(0.5 < rms_ratio < 2.0, f"decoder RMS ratio {rms_ratio}")
        _check(decoder_cos > 0.9, f"decoder cosine {decoder_cos}")

        window_result: dict[str, Any] | None = None
        if streaming:
            source = source_session.run(["source"], {"f0": f0})[0]
            _check(source.shape == (1, expected_samples), f"source shape {source.shape}")
            _check(np.isfinite(source).all(), "source produced NaN/Inf")
            _check(float(np.abs(source).max()) > 1e-3, "source is silent")

            window = window_session.run(
                ["audio"],
                {
                    "acoustic_tokens": acoustic,
                    "frame_to_token": frame_to_token,
                    "f0": f0,
                    "energy": energy,
                    "source": source,
                    "voice": voice_np,
                },
            )[0]
            _check(window.shape == (1, expected_samples), f"window shape {window.shape}")
            _check(np.isfinite(window).all(), "window produced NaN/Inf")
            with torch.no_grad():
                harmonic = model.decoder.generator.harmonic_features(torch.from_numpy(source))
                window_ref = _decode_frames(
                    model.decoder,
                    asr,
                    torch.from_numpy(f0),
                    torch.from_numpy(energy),
                    voice[:, :128],
                    harmonic,
                )
            window_diff = float(np.abs(window - _tensor(window_ref)).max())
            _check(window_diff < 2e-4, f"window parity max diff {window_diff}")
            window_result = {"max_abs_diff": window_diff}

        report["samples"].append(
            {
                "text": text,
                "tokens": token_count,
                "frames": frames,
                "durations": durations.tolist(),
                "decoder_cos": decoder_cos,
                "decoder_rms_ratio": rms_ratio,
                "window": window_result,
            }
        )
        print(
            f"[validate] {text!r}: T={token_count} F={frames} "
            f"decoder cos={decoder_cos:.4f} rms={rms_ratio:.3f}"
            + (f" window diff={window_result['max_abs_diff']:.2e}" if window_result else "")
        )

    # Independent dynamic-shape checks: the graphs must accept frame counts
    # that never occurred during tracing.
    generator = torch.Generator().manual_seed(0)
    token_count = 24
    acoustic = torch.randn(1, 512, token_count, generator=generator)
    voice_random = torch.randn(1, 256, generator=generator)
    for frames in (1, 2, 7, 37, 101):
        frame_to_token = torch.randint(0, token_count, (frames,), generator=generator)
        f0 = torch.rand(1, 2 * frames, generator=generator) * 300.0
        energy = torch.randn(1, 2 * frames, generator=generator)
        audio = decoder.run(
            ["audio"],
            {
                "acoustic_tokens": _tensor(acoustic),
                "frame_to_token": _tensor(frame_to_token),
                "f0": _tensor(f0),
                "energy": _tensor(energy),
                "voice": _tensor(voice_random),
            },
        )[0]
        _check(audio.shape == (1, frames * _SAMPLES_PER_FRAME),
               f"arbitrary decoder F={frames} shape {audio.shape}")
        _check(np.isfinite(audio).all(), f"arbitrary decoder F={frames} NaN/Inf")
        if streaming:
            source = source_session.run(["source"], {"f0": _tensor(f0)})[0]
            _check(source.shape == (1, frames * _SAMPLES_PER_FRAME),
                   f"arbitrary source F={frames} shape {source.shape}")
            window = window_session.run(
                ["audio"],
                {
                    "acoustic_tokens": _tensor(acoustic),
                    "frame_to_token": _tensor(frame_to_token),
                    "f0": _tensor(f0),
                    "energy": _tensor(energy),
                    "source": source,
                    "voice": _tensor(voice_random),
                },
            )[0]
            _check(window.shape == (1, frames * _SAMPLES_PER_FRAME),
                   f"arbitrary window F={frames} shape {window.shape}")
            _check(np.isfinite(window).all(), f"arbitrary window F={frames} NaN/Inf")
        report["arbitrary_frames"].append(frames)
        print(f"[validate] arbitrary F={frames}: decoder/window OK")

    return report


def validate_voices(path: Path, voices: dict[str, torch.Tensor]) -> None:
    with np.load(path) as archive:
        names = archive["names"]
        values = archive["values"]
    _check(names.dtype.kind == "U", f"names dtype {names.dtype}")
    _check(values.dtype == np.float32, f"values dtype {values.dtype}")
    _check(values.shape == (len(voices), 256), f"values shape {values.shape}")
    _check(names.tolist() == list(voices.keys()), "voice names changed order")
    for index, (name, voice) in enumerate(voices.items()):
        _check(np.allclose(values[index], _tensor(voice), atol=0.0),
               f"voice {name!r} was not preserved exactly")
    print(f"[validate] voices.npz: {len(names)} names, shape {values.shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export HolosTTS PyTorch weights to a Promin holos-onnx bundle."
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("HOLOS_REPO", ""),
        help="Path to the patriotyk/HolosTTS source checkout (must contain holos/).",
    )
    parser.add_argument(
        "--weights",
        default=os.environ.get("HOLOS_WEIGHTS", "holos.pt"),
        help="Path to holos.pt.",
    )
    parser.add_argument(
        "--voices",
        default=os.environ.get("HOLOS_VOICES", "voices.pt"),
        help="Path to voices.pt.",
    )
    parser.add_argument(
        "--out",
        default="holos-onnx",
        help="Output bundle directory.",
    )
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=20_000,
        help="Maximum frames the manifest allows (40 frames = 1 second).",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=10,
        help="Streaming chunk size in frames (10 frames = 250 ms).",
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=2,
        help="Streaming context on each side of a chunk, in frames.",
    )
    parser.add_argument(
        "--skip-streaming",
        action="store_true",
        help="Do not export source.onnx / decoder_window.onnx.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Export without loading the graphs in ONNX Runtime.",
    )
    parser.add_argument(
        "--report",
        default="",
        help="Optional path for a JSON validation report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve() if args.repo else None
    weights = Path(args.weights).expanduser().resolve()
    voices_path = Path(args.voices).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    if repo is None:
        raise SystemExit("--repo is required (path to the HolosTTS source checkout)")
    if not (repo / "holos" / "models.py").is_file():
        raise SystemExit(f"{repo} does not look like a HolosTTS checkout")
    if not weights.is_file():
        raise SystemExit(f"weights not found: {weights}")
    if not voices_path.is_file():
        raise SystemExit(f"voices not found: {voices_path}")
    if args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive")
    if args.chunk_frames <= 0 or args.context_frames < 0:
        raise SystemExit("streaming chunk/context values are invalid")

    install_import_stubs()
    sys.path.insert(0, str(repo))
    for name in list(sys.modules):
        if name == "holos" or name.startswith("holos."):
            del sys.modules[name]
    from holos.models import HolosTTS  # noqa: PLC0415

    print(f"[load] {weights}")
    model = HolosTTS(
        weights_path=str(weights), voices_path=str(voices_path), device="cpu"
    )
    model.eval()

    embedding_rows = int(model.text_encoder.embedding.weight.shape[0])
    if int(model.prosodic_text_encoder.embedding.weight.shape[0]) != embedding_rows:
        raise SystemExit("text encoders disagree on vocabulary size")
    if len(model.tokenizer.vocab) < embedding_rows:
        raise SystemExit(
            f"checkpoint vocab has {len(model.tokenizer.vocab)} entries, "
            f"but the embedding has {embedding_rows} rows"
        )
    vocab = list(model.tokenizer.vocab[:embedding_rows])
    if not vocab:
        raise SystemExit("checkpoint has an empty vocabulary")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Example tensors.  The actual graphs are dynamic in T/F; these values
    # only need to exercise every branch once.
    voice = model.voices["Speaker_0"].unsqueeze(0)
    tokens = torch.tensor([[0] + model.tokenizer("prɪˈwʲit ˈsʲwʲite") + [0]])
    speed = torch.tensor([1.0])
    with torch.no_grad():
        acoustic_tokens, prosodic_tokens, durations = EncoderExport(model)(
            tokens, voice, speed
        )
        frame_to_token = torch.repeat_interleave(
            torch.arange(tokens.size(1)), durations[0]
        )
        frames = int(frame_to_token.numel())
        f0, energy = ProsodyExport(model)(prosodic_tokens, frame_to_token, voice)
        source = SourceExport(model)(f0)
        _ = DecoderExport(model)(acoustic_tokens, frame_to_token, f0, energy, voice)
        if not args.skip_streaming:
            _ = DecoderWindowExport(model)(
                acoustic_tokens, frame_to_token, f0, energy, source, voice
            )
    print(f"[example] T={tokens.size(1)} F={frames}")

    export_stage(
        EncoderExport(model),
        (tokens, voice, speed),
        ["tokens", "voice", "speed"],
        ["acoustic_tokens", "prosodic_tokens", "durations"],
        {
            "tokens": {1: "tokens"},
            "acoustic_tokens": {2: "tokens"},
            "prosodic_tokens": {1: "tokens"},
            "durations": {1: "tokens"},
        },
        out_dir / "encoder.onnx",
        args.opset,
    )
    export_stage(
        ProsodyExport(model),
        (prosodic_tokens, frame_to_token, voice),
        ["prosodic_tokens", "frame_to_token", "voice"],
        ["f0", "energy"],
        {
            "prosodic_tokens": {1: "tokens"},
            "frame_to_token": {0: "frames"},
            "f0": {1: "frames2"},
            "energy": {1: "frames2"},
        },
        out_dir / "prosody.onnx",
        args.opset,
    )
    export_stage(
        DecoderExport(model),
        (acoustic_tokens, frame_to_token, f0, energy, voice),
        ["acoustic_tokens", "frame_to_token", "f0", "energy", "voice"],
        ["audio"],
        {
            "acoustic_tokens": {2: "tokens"},
            "frame_to_token": {0: "frames"},
            "f0": {1: "frames2"},
            "energy": {1: "frames2"},
            "audio": {1: "samples"},
        },
        out_dir / "decoder.onnx",
        args.opset,
    )

    if not args.skip_streaming:
        export_stage(
            SourceExport(model),
            (f0,),
            ["f0"],
            ["source"],
            {"f0": {1: "frames2"}, "source": {1: "samples"}},
            out_dir / "source.onnx",
            args.opset,
        )
        export_stage(
            DecoderWindowExport(model),
            (acoustic_tokens, frame_to_token, f0, energy, source, voice),
            ["acoustic_tokens", "frame_to_token", "f0", "energy", "source", "voice"],
            ["audio"],
            {
                "acoustic_tokens": {2: "tokens"},
                "frame_to_token": {0: "frames"},
                "f0": {1: "frames2"},
                "energy": {1: "frames2"},
                "source": {1: "samples"},
                "audio": {1: "samples"},
            },
            out_dir / "decoder_window.onnx",
            args.opset,
        )

    write_voices(out_dir / "voices.npz", model.voices)
    manifest = write_manifest(
        out_dir / "holos_onnx.json",
        vocab=vocab,
        max_frames=args.max_frames,
        chunk_frames=args.chunk_frames,
        context_frames=args.context_frames,
        streaming=not args.skip_streaming,
    )

    report: dict[str, Any] = {
        "bundle": str(out_dir),
        "weights": str(weights),
        "voices": str(voices_path),
        "vocab_size": len(vocab),
        "embedding_rows": embedding_rows,
        "samples_per_frame": _SAMPLES_PER_FRAME,
        "sample_rate": _SAMPLE_RATE,
        "max_frames": args.max_frames,
        "streaming": not args.skip_streaming,
        "files": {},
    }
    for path in sorted(out_dir.iterdir()):
        if path.is_file():
            report["files"][path.name] = path.stat().st_size

    validate_voices(out_dir / "voices.npz", model.voices)
    if not args.no_validate:
        samples = [
            "prɪˈwʲit ˈsʲwʲite",
            "t͡sɛ ˈdɔu̯ɦɪi̯ tɛst dʲlʲa pɛrɛˈwʲirke",
            "a",
        ]
        report["validation"] = validate_bundle(out_dir, model, manifest, samples)

    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[report] {report_path}")

    total = sum(report["files"].values())
    print(f"[done] {out_dir} ({total / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
