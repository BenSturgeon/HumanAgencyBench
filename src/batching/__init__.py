# batching package

"""Runtime configuration for batching behaviour.

The main entry-point is ``install_batching_config`` which is called once after
the user’s YAML config is loaded.  It stores per-provider booleans (enabled or
disabled) on ``BatchEngine`` so that every call can respect them.
"""

from typing import Dict


def install_batching_config(cfg: Dict):
    """Injects `cfg['batching']` (if present) into the engine.

    Example structure::

        batching:
          anthropic: true
          openai:    false
          gemini:    true
    """

    from .engine import BatchEngine

    batching_section = cfg.get("batching", {}) if cfg else {}

    # Normalise keys to lowercase strings for predictable look-ups
    BatchEngine.allowed = {k.lower(): bool(v) for k, v in batching_section.items()} 