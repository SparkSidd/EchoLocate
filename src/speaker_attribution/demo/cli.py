"""
EchoLocate — Real Human Speaker Demonstration Interactive CLI
=============================================================
Interactive workflows for:
  - Live microphone enrollment (1/N ... N/N with quality checks)
  - Real-time speaker identification card display
  - UNKNOWN rejection testing with impostors
  - Blind evaluation session with truth blinding and final scoring
  - Enrolled speaker profile inspection
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from .quality_checker import EnrollmentQualityChecker
from .recorder import MicrophoneRecorder
from .session import DemonstrationSession

SAMPLE_SENTENCE_PROMPTS = [
    "My name is {name}, enrolling into the EchoLocate acoustic system.",
    "The quick brown fox jumps over the lazy dog.",
    "Acoustic sensor networks enable spatial microphone processing.",
    "Artificial intelligence processes real-time multichannel audio.",
    "We are validating human speaker verification with zero leakage.",
    "The weather outside is unpredictable today.",
    "EchoLocate estimates acoustic event arrival times.",
    "Clear speech production ensures high quality biometric centroids.",
]


def print_banner(title: str) -> None:
    print("\n" + "=" * 64)
    print(f"  {title.upper()}")
    print("=" * 64 + "\n")


def print_card(result: dict) -> None:
    spk = result["display_name"]
    status = result["decision"]
    conf = result["confidence"] * 100.0
    sim = result["similarity"]
    thresh = result["threshold"]
    margin = result["margin"]
    lat = result["latency_ms"]

    status_icon = "✅ KNOWN" if status == "KNOWN" else "⚠️ UNKNOWN SPEAKER"
    border = "─" * 58

    print(f"\n┌{border}┐")
    print(f"│  SPEAKER ATTRIBUTION RESULT                             │")
    print(f"├{border}┤")
    print(f"│  Decision:        {status_icon:<38}│")
    print(f"│  Speaker:         {spk:<38}│")
    print(f"│  Confidence:      {conf:>5.1f}% (Calibrated)                     │")
    print(f"│  Similarity:      {sim:>5.3f} (Threshold: {thresh:.3f})               │")
    print(f"│  Decision Margin: {margin:>+5.3f}                                 │")
    print(f"│  Inference Time:  {lat:>5.1f} ms                                │")
    print(f"├{border}┤")
    print(f"│  TOP CANDIDATES:                                        │")
    for alt in result["alternatives"][:3]:
        s_line = f"    #{alt['rank']} {alt['speaker_id']:<18} sim: {alt['similarity']:.3f}"
        print(f"│{s_line:<58}│")
    print(f"└{border}┘\n")


def enroll_flow(session: DemonstrationSession, recorder: MicrophoneRecorder, args: argparse.Namespace) -> None:
    print_banner("Live Human Speaker Enrollment")
    speaker_id = (args.speaker or input("Enter unique Speaker ID (e.g. rahul, siddharth): ")).strip()
    if not speaker_id:
        print("Error: Speaker ID cannot be empty.")
        return

    display_name = (args.name or input(f"Enter Display Name [{speaker_id}]: ")).strip() or speaker_id
    num_samples = int(args.num_samples or 5)

    print(f"\nEnrolling '{display_name}' ({speaker_id}) with {num_samples} independent utterances.")
    print("Each sample will undergo strict VAD, duration, RMS, and SNR quality checks.\n")

    accepted_count = 0
    while accepted_count < num_samples:
        step = accepted_count + 1
        prompt_sentence = SAMPLE_SENTENCE_PROMPTS[(step - 1) % len(SAMPLE_SENTENCE_PROMPTS)].format(name=display_name)
        print(f"--- Utterance {step}/{num_samples} ---")
        print(f"Suggested text: \"{prompt_sentence}\"")

        input("Press [ENTER] to start recording (3.0 seconds)... ")
        audio = recorder.record_live(duration_sec=3.0)

        ok, quality, profile = session.enroll_sample(
            speaker_id=speaker_id,
            display_name=display_name,
            audio=audio,
        )

        print(quality.summary_line())
        if ok and profile:
            accepted_count += 1
            print(f" Utterance {step}/{num_samples} ACCEPTED. Centroid updated (total samples: {profile.sample_count}).\n")
        else:
            print(f" REJECTED: {quality.status_reason}. Please speak clearly and try again.\n")

    print_banner(f"Enrollment Complete for {display_name}!")
    prof = session.db.get_profile(speaker_id)
    if prof:
        print(f"Profile ID:        {prof.speaker_id}")
        print(f"Display Name:      {display_name}")
        print(f"Model Identifier:  {prof.model_name}")
        print(f"Embedding Dim:     {prof.embedding_dim}")
        print(f"Total Centroids:   {prof.sample_count}")
        print(f"Saved to:          {session.db_dir.resolve()}\n")


def identify_flow(session: DemonstrationSession, recorder: MicrophoneRecorder, args: argparse.Namespace) -> None:
    print_banner("Real-Time Speaker Identification")
    if len(session.db) == 0:
        print("No speakers enrolled yet! Run enrollment first: python -m src.speaker_attribution.demo enroll")
        return

    print(f"Active Enrolled Speakers ({len(session.db)}): {', '.join(session.db.get_enrolled_speakers())}")
    print(f"Model: {session.model_name} | Threshold: {session.matcher.default_threshold:.3f}\n")

    while True:
        try:
            cmd = input("Press [ENTER] to record, or 'q' to quit: ").strip().lower()
            if cmd == "q":
                break

            audio = recorder.record_live(duration_sec=3.0)
            res = session.identify_sample(audio, use_temporal_tracking=True)
            print_card(res)
        except KeyboardInterrupt:
            break


def blind_test_flow(session: DemonstrationSession, recorder: MicrophoneRecorder, args: argparse.Namespace) -> None:
    print_banner("Blind Evaluation Testing Mode")
    if len(session.db) == 0:
        print("No speakers enrolled yet! Run enrollment first.")
        return

    enrolled = session.db.get_enrolled_speakers()
    print(f"Enrolled Known Speakers: {', '.join(enrolled)}")
    print("You can test ENROLLED speakers and UNSEEN/UNKNOWN impostor speakers.")
    print("Results are blinded during testing and summarized at the end.\n")

    num_tests = int(args.num_tests or 6)
    print(f"Running {num_tests} blind test trials.\n")

    for i in range(1, num_tests + 1):
        print(f"================ Trial #{i}/{num_tests} ================")
        true_label = input(f"Evaluator: enter true speaker ID (or 'unknown' for impostor): ").strip()
        if not true_label:
            true_label = "unknown"

        cond = input("Condition [normal / close / 1m / 2m / noise]: ").strip() or "normal"

        input("Speaker: press [ENTER] to record test utterance... ")
        audio = recorder.record_live(duration_sec=3.0)

        test_case = session.record_blind_test(true_speaker=true_label, audio=audio, condition=cond)
        # Note: Do not display whether it's correct! Keep it blinded.
        print(f"Trial #{i} recorded and processed. (Result stored in blind registry)\n")

    # Present final evaluation summary
    print_banner("Blind Evaluation Summary Results")
    summary = session.compute_summary()
    print(summary.format_report())

    # Display line-by-line comparison
    print("\nDetailed Test Log:")
    print(f"{'#':<4} {'True Speaker':<15} {'Predicted':<15} {'Status':<10} {'Sim':<7} {'Conf':<7} {'Verdict':<8}")
    print("-" * 72)
    for t in summary.test_cases:
        verdict = "CORRECT" if t["is_correct"] else "WRONG"
        print(
            f"{t['test_id']:<4} {t['true_speaker']:<15} {t['predicted_speaker']:<15} "
            f"{t['decision_status']:<10} {t['similarity']:<7.3f} {t['confidence']:<7.1f} {verdict:<8}"
        )

    # Save summary
    out_json = session.save_summary_json("results/speaker_attribution/real_human_demo_summary.json")
    print(f"\nFull machine-readable results saved to: {out_json.resolve()}\n")


def list_flow(session: DemonstrationSession) -> None:
    print_banner("Enrolled Speaker Profiles")
    speakers = session.db.get_enrolled_speakers()
    if not speakers:
        print("No speaker profiles found.")
        return

    print(f"{'Speaker ID':<18} {'Display Name':<20} {'Samples':<10} {'Model':<12} {'Dim':<6}")
    print("-" * 68)
    for s_id in speakers:
        prof = session.db.get_profile(s_id)
        if prof:
            d_name = prof.metadata.get("display_name", s_id) if prof.metadata else s_id
            print(f"{prof.speaker_id:<18} {d_name:<20} {prof.sample_count:<10} {prof.model_name:<12} {prof.embedding_dim:<6}")
    print()


def evaluate_profiles_flow(session: DemonstrationSession, args: argparse.Namespace) -> None:
    print_banner("Machine-Readable Speaker Verification Evaluation (Phase 11 & Phase 19)")
    split_n = getattr(args, "split", None)
    results = session.evaluate_speaker_profiles(split_first_n_enroll=split_n)
    if "error" in results:
        print(f"Error: {results['error']}")
        return

    print(f"Model:                 {results['model_name']}")
    print(f"Enrolled Speakers:     {', '.join(results['enrolled_speakers'])}")
    print(f"Genuine Trials:        {results['total_genuine_trials']} (Mean: {results['genuine_mean']:.4f} ± {results['genuine_std']:.4f})")
    print(f"Impostor Trials:       {results['total_impostor_trials']} (Mean: {results['impostor_mean']:.4f} ± {results['impostor_std']:.4f})")
    print(f"Separation Margin:     {results['separation_margin']:+.4f}")
    print(f"Score Overlap Ratio:   {results['score_overlap_ratio']:.4f}")
    print(f"Equal Error Rate(EER): {results['eer']:.4f}")
    print(f"Optimal Threshold:     {results['optimal_threshold']:.3f}")
    print(f"Active Threshold:      {results['active_threshold']:.3f} (FAR: {results['active_threshold_far']:.4f}, FRR: {results['active_threshold_frr']:.4f})")

    out_file = getattr(args, "output", None)
    if out_file:
        p = Path(out_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nMachine-readable JSON saved to: {p}")
    else:
        print("\n--- First 5 Trials ---")
        for t in results["trials"][:5]:
            print(f"  [#{t['trial_id']}] {t['speaker']} -> {t['enrolled_target']} ({'GENUINE' if t['is_genuine'] else 'IMPOSTOR'}): score={t['score']:.4f}, pred={t['predicted']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EchoLocate Real Human Speaker Demonstration")
    subparsers = parser.add_subparsers(dest="command", help="Demonstration command")

    # Enroll command
    p_enroll = subparsers.add_parser("enroll", help="Enroll a human speaker with live microphone")
    p_enroll.add_argument("--speaker", type=str, help="Speaker ID")
    p_enroll.add_argument("--name", type=str, help="Display Name")
    p_enroll.add_argument("--num-samples", type=int, default=5, help="Number of enrollment samples")
    p_enroll.add_argument("--model", type=str, default="campp", choices=["campp", "eres2netv2", "ecapa"])
    p_enroll.add_argument("--db-dir", type=str, default="demo_data/speakers")

    # Identify command
    p_ident = subparsers.add_parser("identify", help="Live speaker identification from microphone")
    p_ident.add_argument("--model", type=str, default="campp", choices=["campp", "eres2netv2", "ecapa"])
    p_ident.add_argument("--threshold", type=float, default=0.42, help="Decision threshold")
    p_ident.add_argument("--db-dir", type=str, default="demo_data/speakers")

    # Blind-test command
    p_blind = subparsers.add_parser("blind-test", help="Execute blinded evaluation session")
    p_blind.add_argument("--num-tests", type=int, default=6, help="Number of blind test trials")
    p_blind.add_argument("--model", type=str, default="campp")
    p_blind.add_argument("--threshold", type=float, default=0.42)
    p_blind.add_argument("--db-dir", type=str, default="demo_data/speakers")

    # List command
    p_list = subparsers.add_parser("list", help="List enrolled speaker profiles")
    p_list.add_argument("--db-dir", type=str, default="demo_data/speakers")

    # Evaluate command (Phase 11 & Phase 19)
    p_eval = subparsers.add_parser("evaluate-speaker-profiles", help="Run genuine vs impostor verification matrix")
    p_eval.add_argument("--model", type=str, default="campp", choices=["campp", "eres2netv2", "ecapa"])
    p_eval.add_argument("--threshold", type=float, default=0.42, help="Decision threshold")
    p_eval.add_argument("--split", type=int, default=None, help="Optional enrollment sample count split (e.g. 5)")
    p_eval.add_argument("--output", type=str, default="evaluation_results.json", help="Path to save machine-readable JSON")
    p_eval.add_argument("--db-dir", type=str, default="demo_data/speakers")

    # Web command
    p_web = subparsers.add_parser("web", help="Launch browser-based web demonstration server")
    p_web.add_argument("--port", type=int, default=8050, help="Port to bind web demo")
    p_web.add_argument("--model", type=str, default="campp")
    p_web.add_argument("--threshold", type=float, default=0.42)
    p_web.add_argument("--db-dir", type=str, default="demo_data/speakers")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "web":
        from .web_demo import launch_web_demo
        launch_web_demo(port=args.port, model_name=args.model, db_dir=args.db_dir)
        return

    db_dir = getattr(args, "db_dir", "demo_data/speakers")
    model_name = getattr(args, "model", "campp")
    threshold = getattr(args, "threshold", 0.42)

    session = DemonstrationSession(
        db_dir=db_dir,
        model_name=model_name,
        default_threshold=threshold,
    )
    recorder = MicrophoneRecorder()

    if args.command == "enroll":
        enroll_flow(session, recorder, args)
    elif args.command == "identify":
        identify_flow(session, recorder, args)
    elif args.command == "blind-test":
        blind_test_flow(session, recorder, args)
    elif args.command == "list":
        list_flow(session)
    elif args.command == "evaluate-speaker-profiles":
        evaluate_profiles_flow(session, args)


if __name__ == "__main__":
    main()
