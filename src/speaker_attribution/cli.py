"""
EchoLocate — Speaker Attribution Command Line Interface (CLI)
=============================================================
Provides clean commands for enrolling speakers, identifying audio events,
running model benchmarks, and evaluating research ablations.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .adapters.audio_source_adapter import AudioSourceAdapter
from .encoders.factory import create_encoder
from .enrollment.database import SpeakerEnrollmentDB
from .experiments.ablation_runner import Experiment7AblationRunner
from .experiments.benchmark_runner import SpeakerModelBenchmarkRunner
from .identifier import SpeakerIdentifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger("EchoLocate.SpeakerCLI")


def cmd_enroll(args: argparse.Namespace) -> int:
    encoder = create_encoder(args.model)
    db = SpeakerEnrollmentDB(db_dir=args.db)
    audio, sr = AudioSourceAdapter.from_file(args.audio)

    prof = db.enroll_audio(speaker_id=args.speaker, audio=audio, encoder=encoder, sample_rate=sr)
    db.save(args.db)
    print(f"Successfully enrolled speaker '{prof.speaker_id}' (total samples: {prof.sample_count}, dim: {prof.embedding_dim})")
    return 0


def cmd_identify(args: argparse.Namespace) -> int:
    encoder = create_encoder(args.model)
    db = SpeakerEnrollmentDB(db_dir=args.db)
    ident = SpeakerIdentifier(
        encoder=encoder,
        enrollment_db=db,
        unknown_threshold=args.threshold,
        enable_temporal_smoothing=args.smooth,
    )

    audio, sr = AudioSourceAdapter.from_file(args.audio)
    res = ident.identify(audio, sample_rate=sr)
    print(json.dumps(res.to_dict(), indent=2))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    runner = SpeakerModelBenchmarkRunner(output_dir=args.output_dir)
    results = runner.run_benchmark(models=args.models, num_trials_per_condition=args.trials)
    print(f"Benchmark completed successfully! Results saved to {args.output_dir}")
    return 0


def cmd_ablation(args: argparse.Namespace) -> int:
    runner = Experiment7AblationRunner(output_dir=args.output_dir, model_name=args.model)
    results = runner.run_all_ablations(num_trials=args.trials)
    print(f"Ablation study completed! Results saved to {args.output_dir}/ablation_results.json")
    return 0


def main(cli_args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EchoLocate Speaker Attribution CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # enroll-speaker
    p_enroll = subparsers.add_parser("enroll-speaker", help="Enroll a speaker from WAV audio")
    p_enroll.add_argument("--speaker", type=str, required=True, help="Speaker identifier")
    p_enroll.add_argument("--audio", type=str, required=True, help="Path to WAV audio file")
    p_enroll.add_argument("--model", type=str, default="campp", help="Model: campp, eres2netv2, ecapa, mock")
    p_enroll.add_argument("--db", type=str, default="data/enrollment", help="Database directory")
    p_enroll.set_defaults(func=cmd_enroll)

    # identify-speaker
    p_ident = subparsers.add_parser("identify-speaker", help="Identify speaker from WAV audio")
    p_ident.add_argument("--audio", type=str, required=True, help="Path to WAV audio file")
    p_ident.add_argument("--model", type=str, default="campp", help="Model: campp, eres2netv2, ecapa, mock")
    p_ident.add_argument("--threshold", type=float, default=0.65, help="UNKNOWN decision threshold")
    p_ident.add_argument("--db", type=str, default="data/enrollment", help="Database directory")
    p_ident.add_argument("--smooth", action="store_true", help="Enable temporal smoothing")
    p_ident.set_defaults(func=cmd_identify)

    # benchmark-speaker-models
    p_bench = subparsers.add_parser("benchmark-speaker-models", help="Run benchmark across CAM++, ERes2NetV2, ECAPA")
    p_bench.add_argument("--models", nargs="+", default=["campp", "eres2netv2", "ecapa"], help="Models to benchmark")
    p_bench.add_argument("--trials", type=int, default=5, help="Trials per condition")
    p_bench.add_argument("--output-dir", type=str, default="results/speaker_attribution", help="Output directory")
    p_bench.set_defaults(func=cmd_benchmark)

    # run-ablations
    p_abl = subparsers.add_parser("run-ablations", help="Run Experiment 7 research ablations (E7.1 - E7.6)")
    p_abl.add_argument("--model", type=str, default="campp", help="Model architecture")
    p_abl.add_argument("--trials", type=int, default=5, help="Trials per condition")
    p_abl.add_argument("--output-dir", type=str, default="results/speaker_attribution", help="Output directory")
    p_abl.set_defaults(func=cmd_ablation)

    args = parser.parse_args(cli_args)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
