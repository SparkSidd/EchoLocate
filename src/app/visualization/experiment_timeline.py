"""
EchoLocate — Complete Research Journey Timeline Visualizer
==========================================================
Encodes the scientific research narrative from Experiment 1 through 6.25,
illustrating the Problem -> Discovery -> Failure -> Next Solution progression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ResearchMilestone:
    """Detailed record of an individual experiment in the EchoLocate research arc."""
    experiment_id: str
    title: str
    phase: str
    problem: str
    discovery: str
    failure_mechanism: str
    next_solution: str
    badge_color: str
    key_metric: str


def get_complete_research_timeline() -> list[ResearchMilestone]:
    """Return chronological sequence of all 11 EchoLocate experiments."""
    return [
        ResearchMilestone(
            experiment_id="Exp 1",
            title="Classical Multilateration Baseline",
            phase="Foundations",
            problem="How to perform 2D acoustic localization from 3-4 microphones in an indoor room?",
            discovery="Pairwise GCC-PHAT + Non-Linear Least Squares achieves sub-centimeter accuracy in anechoic conditions.",
            failure_mechanism="Extreme vulnerability to early acoustic reflections (multipath).",
            next_solution="Experiment 2: Systematic reverberation and noise stress testing.",
            badge_color="#94a3b8",
            key_metric="Anechoic Error: 0.008 m",
        ),
        ResearchMilestone(
            experiment_id="Exp 2 / 2.5",
            title="Reverberation & Noise Stress Testing",
            phase="Stress Testing",
            problem="Under what exact acoustic boundary does pairwise GCC-PHAT + NLS fail?",
            discovery="Catastrophic phase transition discovered at RT60 = 0.38s (error jumps from 0.05m to >1.5m).",
            failure_mechanism="Pairwise peak hopping: strongest cross-correlation peak corresponds to room reflection rather than direct path.",
            next_solution="Experiment 3: Steered Response Power (SRP-PHAT) spatial aggregation.",
            badge_color="#64748b",
            key_metric="Critical Transition: RT60 = 0.38 s",
        ),
        ResearchMilestone(
            experiment_id="Exp 3",
            title="SRP-PHAT Spatial Likelihood Baseline",
            phase="Classical Spatial Likelihood",
            problem="Can global spatial likelihood accumulation bypass pairwise peak ambiguity?",
            discovery="SRP-PHAT integrates cross-correlations across all pairs simultaneously, achieving 94.8% success at transition edge.",
            failure_mechanism="High computational cost; coarse grid quantization limits precision.",
            next_solution="Experiment 3.5: Adversarial reality check and off-grid validation.",
            badge_color="#38bdf8",
            key_metric="Transition Accuracy: 94.8%",
        ),
        ResearchMilestone(
            experiment_id="Exp 3.5",
            title="Model-Mismatch & Leakage Validation Audit",
            phase="Scientific Audit",
            problem="Are Experiment 3 results inflated by simulation matching or grid alignment?",
            discovery="Audited zero data leakage; confirmed continuous off-grid sources degrade localization without quadratic sub-grid refinement.",
            failure_mechanism="Voiced speech pitch harmonics create persistent spatial pseudo-peaks.",
            next_solution="Experiment 4: Learned neural front-end (DeepGCC) for pseudo-peak suppression.",
            badge_color="#0ea5e9",
            key_metric="Audit Status: 100% Passed",
        ),
        ResearchMilestone(
            experiment_id="Exp 4",
            title="DeepGCC Learned Acoustic Enhancement",
            phase="Neural Enhancement",
            problem="Can a 1D ResNet suppress speech harmonic pseudo-peaks in GCC cross-correlations?",
            discovery="DeepGCC successfully suppressed harmonic pseudo-peaks in 78.4% of speech frames.",
            failure_mechanism="Probability smoothing caused spatial peak contrast collapse across room grids.",
            next_solution="Experiment 4.5: Calibrated likelihood sharpening and differentiable SRP.",
            badge_color="#f97316",
            key_metric="Harmonic Suppression: 78.4%",
        ),
        ResearchMilestone(
            experiment_id="Exp 4.5",
            title="Neural Calibration & Likelihood Sharpening",
            phase="Neural Calibration",
            problem="How to restore sharp spatial contrast while preserving harmonic suppression?",
            discovery="Temperature calibration (T=0.35) and multi-objective loss restored sharp spatial likelihood peaks.",
            failure_mechanism="Spatial peak contrast restored, but generalization to unseen acoustics unverified.",
            next_solution="Experiment 4.75: Cross-domain generalization and adversarial stress testing.",
            badge_color="#ea580c",
            key_metric="Spatial Contrast: +45.2%",
        ),
        ResearchMilestone(
            experiment_id="Exp 4.75",
            title="Adversarial Audit & Robustness Boundary",
            phase="Adversarial Validation",
            problem="Does calibrated DeepGCC maintain advantage under severe physical and acoustic mismatch?",
            discovery="Surprising scientific discovery: Raw SRP-PHAT remained more robust overall under severe physical array perturbations.",
            failure_mechanism="Neural models overfit to nominal geometry and synthetic training distributions.",
            next_solution="Experiment 5: Sim-to-real transfer and confidence-gated hybrid architecture.",
            badge_color="#c2410c",
            key_metric="Verdict: Classical Dominance",
        ),
        ResearchMilestone(
            experiment_id="Exp 5",
            title="Real-World Validation & Confidence Gating",
            phase="Sim-to-Real Transfer",
            problem="How does the pipeline perform on real recorded datasets (LOCATA & AV16.3)?",
            discovery="Method M12 (Confidence-Gated Hybrid) routes nominal speech to SRP and harmonic frames to DeepGCC, minimizing worst-case risk.",
            failure_mechanism="Hardware array misplacement and uncalibrated gain drift.",
            next_solution="Experiment 6: Real-time working interactive prototype.",
            badge_color="#22c55e",
            key_metric="LOCATA Error: 0.42 m",
        ),
        ResearchMilestone(
            experiment_id="Exp 6",
            title="Real-Time Interactive Working Prototype",
            phase="Deployment Validation",
            problem="Can the validated EchoLocate pipeline run in real time for live examiner demonstrations?",
            discovery="Achieved sub-real-time throughput (11.4 ms latency, RTF = 0.057x) with strictly causal temporal tracking.",
            failure_mechanism="Need for complete visual exploration of internal representations.",
            next_solution="Experiment 6.25: Comprehensive research visualization and analytics platform.",
            badge_color="#10b981",
            key_metric="Latency: 11.4 ms (RTF 0.057x)",
        ),
        ResearchMilestone(
            experiment_id="Exp 6.25",
            title="Complete Research Visualization Platform",
            phase="Examiner Presentation",
            problem="How to visually present the entire 6-experiment journey interactively without fake animations?",
            discovery="Unified 12-view interactive analytics platform with verified data provenance and mathematical fidelity.",
            failure_mechanism="Fully addressed via real-time diagnostics and examiner verification mode.",
            next_solution="Project Completed & Ready for Defense.",
            badge_color="#a855f7",
            key_metric="Visual Views: 12 Interactive Modes",
        ),
    ]
