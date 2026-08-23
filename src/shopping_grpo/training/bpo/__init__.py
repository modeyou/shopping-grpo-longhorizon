"""Branching Policy Optimization for the Shopping agent."""

from .advantage import compute_bpo_advantage
from .branching import BranchCandidate, select_branch_candidate, shannon_entropy

__all__ = [
    "BranchCandidate", "compute_bpo_advantage",
    "select_branch_candidate", "shannon_entropy",
]
