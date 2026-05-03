# Fusion module
from .multimodal_fusion import (
    MultimodalFusion,
    MLPFusion,
    AttentionFusion,
    map_to_common_emotions,
    average_fusion,
    weighted_average_fusion,
    voting_fusion,
    max_fusion,
)

from .coherence_score import (
    coherence_kl,
    coherence_agreement,
    classify_coherence,
    fusion_with_coherence,
    coherence_statistics,
    kl_divergence,
    js_divergence,
)
