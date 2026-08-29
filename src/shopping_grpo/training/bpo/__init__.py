"""Full Branching Policy Optimization support for Shopping Agent training."""

from shopping_grpo.training.bpo.advantage import compute_bpo_advantage
from shopping_grpo.training.bpo.branching import BranchCandidate, select_branch_candidate
from shopping_grpo.training.bpo.reward import completion_aligned_train_return

__all__ = [
    "BranchCandidate",
    "compute_bpo_advantage",
    "completion_aligned_train_return",
    "select_branch_candidate",
]
