"""
Contextual Belief Updating
==========================

This package implements the core representation-learning problem described in the
NeurIPS submission "Learning Compact Market-State Representations from Prediction
Markets via Contextual Belief Updating".

Setup
-----
A *target* market A is observed at two times:
  - t - Δ  (stale time): the model receives A's *local* state here
  - t      (context time): A's own local state is withheld; the model receives the
           contemporaneous state of A's *family siblings* (and optionally external signals)

Two protocol variants are supported:
  - MVP outcome prediction: predict A's final binary outcome.
  - Main belief-update recovery: predict A's hidden current-state update at t.

Modules
-------
dataset : BeliefUpdatingDatasetBuilder — builds training examples from CanonicalDataset
model   : DeepSetsEncoder + BeliefUpdatingPredictor — the PyTorch set encoder + head
train   : BeliefUpdatingMVPExperiment — legacy 4-rung MVP outcome ladder
main    : BeliefUpdatingMainExperiment — main update-recovery ladder
"""

from polymarket_research.belief_updating.dataset import (
    AGG_CONTEXT_FEATURE_NAMES,
    BeliefUpdatingDatasetBuilder,
    BeliefUpdatingManifest,
    BeliefUpdatingSpec,
    CONTEXT_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    STALE_FEATURE_NAMES,
)
from polymarket_research.belief_updating.main import (
    MAIN_FLAT_FEATURE_NAMES,
    BeliefUpdatingMainConfig,
    BeliefUpdatingMainExperiment,
    BeliefUpdatingMainResults,
    BeliefUpdatingMainRungMetrics,
    MainExperimentArtifacts,
)
from polymarket_research.belief_updating.model import (
    BeliefUpdatingPredictor,
    BeliefUpdatingTorchDataset,
    DeepSetsEncoder,
)
from polymarket_research.belief_updating.train import (
    BeliefUpdatingMVPConfig,
    BeliefUpdatingMVPExperiment,
    BeliefUpdatingMVPResults,
    BeliefUpdatingMVPRungMetrics,
    MVPExperimentArtifacts,
)

__all__ = [
    "BeliefUpdatingSpec",
    "BeliefUpdatingManifest",
    "BeliefUpdatingDatasetBuilder",
    "STALE_FEATURE_NAMES",
    "CONTEXT_FEATURE_NAMES",
    "AGG_CONTEXT_FEATURE_NAMES",
    "GLOBAL_FEATURE_NAMES",
    "DeepSetsEncoder",
    "BeliefUpdatingPredictor",
    "BeliefUpdatingTorchDataset",
    "BeliefUpdatingMVPExperiment",
    "BeliefUpdatingMVPConfig",
    "BeliefUpdatingMVPResults",
    "BeliefUpdatingMVPRungMetrics",
    "MVPExperimentArtifacts",
    "BeliefUpdatingMainConfig",
    "BeliefUpdatingMainExperiment",
    "BeliefUpdatingMainResults",
    "BeliefUpdatingMainRungMetrics",
    "MainExperimentArtifacts",
    "MAIN_FLAT_FEATURE_NAMES",
]
