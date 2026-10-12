"""
EchoLocate — Visualization Engine Package
"""

from src.app.visualization.data_integrity import (
    VisualizationIntegrityReport,
    audit_spatial_heatmap_payload,
    audit_waveform_payload,
)
from src.app.visualization.deepgcc_analysis import (
    DeepGCCAnalysisPayload,
    DeepGCCComparisonPair,
    analyze_deepgcc_comparisons,
)
from src.app.visualization.experiment_timeline import ResearchMilestone, get_complete_research_timeline
from src.app.visualization.gcc_analysis import GCCAnalysisPayload, GCCPairAnalysis, analyze_all_gcc_pairs
from src.app.visualization.latency import LatencyProfilePayload, PipelineStageLatency, compute_latency_profile
from src.app.visualization.method_comparison import (
    MethodPredictionVisual,
    MultiMethodComparisonPayload,
    compute_multi_method_comparison,
)
from src.app.visualization.provenance import VisualProvenanceMetadata, VisualProvenanceType
from src.app.visualization.spatial_heatmap import SpatialHeatmapPayload, compute_spatial_likelihood_heatmap
from src.app.visualization.spectrogram import SpectrogramPayload, compute_channel_spectrogram
from src.app.visualization.srp_explainer import (
    SRPExplainerPairSample,
    SRPExplainerStepResult,
    compute_srp_explainer_step,
)
from src.app.visualization.trajectory import (
    TrajectoryPointData,
    TrajectoryVisualizationPayload,
    format_trajectory_visualization,
)
from src.app.visualization.uncertainty import (
    RiskCoveragePoint,
    UncertaintyAnalysisPayload,
    UncertaintyEllipseData,
    compute_uncertainty_analysis,
)
from src.app.visualization.waveform import (
    ChannelWaveformData,
    MultichannelWaveformPayload,
    process_multichannel_waveform,
)

__all__ = [
    "VisualProvenanceType",
    "VisualProvenanceMetadata",
    "VisualizationIntegrityReport",
    "audit_waveform_payload",
    "audit_spatial_heatmap_payload",
    "ChannelWaveformData",
    "MultichannelWaveformPayload",
    "process_multichannel_waveform",
    "SpectrogramPayload",
    "compute_channel_spectrogram",
    "GCCPairAnalysis",
    "GCCAnalysisPayload",
    "analyze_all_gcc_pairs",
    "DeepGCCComparisonPair",
    "DeepGCCAnalysisPayload",
    "analyze_deepgcc_comparisons",
    "SpatialHeatmapPayload",
    "compute_spatial_likelihood_heatmap",
    "SRPExplainerPairSample",
    "SRPExplainerStepResult",
    "compute_srp_explainer_step",
    "TrajectoryPointData",
    "TrajectoryVisualizationPayload",
    "format_trajectory_visualization",
    "UncertaintyEllipseData",
    "RiskCoveragePoint",
    "UncertaintyAnalysisPayload",
    "compute_uncertainty_analysis",
    "PipelineStageLatency",
    "LatencyProfilePayload",
    "compute_latency_profile",
    "MethodPredictionVisual",
    "MultiMethodComparisonPayload",
    "compute_multi_method_comparison",
    "ResearchMilestone",
    "get_complete_research_timeline",
]
