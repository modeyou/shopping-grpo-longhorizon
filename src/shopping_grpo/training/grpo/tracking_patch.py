"""Deterministic veRL 0.8 tracking-lifecycle source transform."""

PATCH_MARKER = "SHOPPING_GRPO_TRACKING_FINISH_PATCH_V1"

ORIGINAL_CLOSE = '''    def __del__(self):
        if "wandb" in self.logger:
            self.logger["wandb"].finish(exit_code=0)
        if "swanlab" in self.logger:
            self.logger["swanlab"].finish()
        if "vemlp_wandb" in self.logger:
            self.logger["vemlp_wandb"].finish(exit_code=0)
        if "tensorboard" in self.logger:
            self.logger["tensorboard"].finish()
        if "clearml" in self.logger:
            self.logger["clearml"].finish()
        if "trackio" in self.logger:
            self.logger["trackio"].finish()
        if "file" in self.logger:
            self.logger["file"].finish()
'''

FINISH = '''    def finish(self):
        """Finish tracking once, from the trainer thread when possible."""
        if getattr(self, "_shopping_finished", False):
            return
        self._shopping_finished = True
        backends = getattr(self, "logger", {})
        try:
            if "wandb" in backends:
                backends["wandb"].finish(exit_code=0)
            if "swanlab" in backends:
                backends["swanlab"].finish()
            if "vemlp_wandb" in backends:
                backends["vemlp_wandb"].finish(exit_code=0)
            if "tensorboard" in backends:
                backends["tensorboard"].finish()
            if "clearml" in backends:
                backends["clearml"].finish()
            if "trackio" in backends:
                backends["trackio"].finish()
            if "file" in backends:
                backends["file"].finish()
        except Exception as exc:
            logger.warning("experiment tracker finish failed: %s", exc)

    def __del__(self):
        self.finish()
'''


def patch_source(source: str) -> str:
    """Add an idempotent explicit finish contract to veRL's Tracking."""
    if PATCH_MARKER in source:
        return source
    anchor = "        self.logger = {}\n"
    if source.count(anchor) != 1 or source.count(ORIGINAL_CLOSE) != 1:
        raise ValueError("pinned veRL Tracking source anchors mismatch")
    source = source.replace(
        anchor,
        anchor + f"        # {PATCH_MARKER}\n        self._shopping_finished = False\n",
        1,
    )
    return source.replace(ORIGINAL_CLOSE, FINISH, 1)
