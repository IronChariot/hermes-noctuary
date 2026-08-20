"""Noctuary configuration.

Everything lives in ``$HERMES_HOME/noctuary.json`` with sane defaults (see
noctuary-requirements.md section 8). The dashboard's generic flat-json config
backend writes ``$HERMES_HOME/noctuary/config.json`` instead; both files are
read, with ``noctuary.json`` taking precedence, so either editing path works.

``get_config_schema()`` on the provider exposes only the keys a user is
likely to touch during ``hermes memory setup``; the long tail of tuning
knobs is documented here and in the README.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

CONFIG_BASENAME = "noctuary.json"
STORE_DIRNAME = "noctuary"

DEFAULTS: Dict[str, Any] = {
    # Section 8 keys
    "recallTokenBudget": 1000,
    "librarianModel": "",          # empty = the agent's main model
    "librarianProvider": "",       # empty = auto-detect from model/config
    "embeddingModel": "auto",      # auto | hash | <fastembed/sentence-transformers model name>
    "surfacePageLimit": 12,
    "consolidateSchedule": "nightly",  # informational; scheduling is external (hermes cron)
    "minUserTurns": 1,
    # Ingestion gate: platforms whose live turns are archived.
    "ingestPlatforms": ["discord"],
    # Recall tuning
    "maxRecallEntries": 6,
    "gistSimilarity": 0.50,
    "familiaritySimilarity": 0.32,
    "prefetchTimeoutSeconds": 30,
    # Decay / promotion (applied only during consolidation)
    "decayFactor": 0.95,
    "retrievalBoost": 0.15,
    "demotionFloor": 0.05,
    "massDemotionThreshold": 10,
    # Librarian LLM budget
    "llmTimeoutSeconds": 240,
    "librarianMaxTokens": 4096,
    "chunkChars": 60000,
    "relatedNodeLimit": 8,
}


@dataclass
class NoctuaryConfig:
    """Resolved configuration plus the paths derived from HERMES_HOME."""

    hermes_home: Path
    values: Dict[str, Any] = field(default_factory=dict)

    # -- paths ---------------------------------------------------------------

    @property
    def store_root(self) -> Path:
        return self.hermes_home / STORE_DIRNAME

    @property
    def config_path(self) -> Path:
        return self.hermes_home / CONFIG_BASENAME

    @property
    def soul_path(self) -> Path:
        return self.hermes_home / "SOUL.md"

    # -- typed accessors -----------------------------------------------------

    def get(self, key: str) -> Any:
        if key in self.values:
            return self.values[key]
        return DEFAULTS[key]

    def get_int(self, key: str) -> int:
        try:
            return int(self.get(key))
        except (TypeError, ValueError):
            return int(DEFAULTS[key])

    def get_float(self, key: str) -> float:
        try:
            return float(self.get(key))
        except (TypeError, ValueError):
            return float(DEFAULTS[key])

    def get_str(self, key: str) -> str:
        value = self.get(key)
        return str(value) if value is not None else ""

    def get_list(self, key: str) -> List[str]:
        value = self.get(key)
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, list):
            return [str(item) for item in value]
        return list(DEFAULTS[key])


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception as exc:
        logger.warning("noctuary: could not read config %s: %s", path, exc)
    return {}


def load_config(hermes_home: str | Path | None = None) -> NoctuaryConfig:
    """Load config for the given (or active) HERMES_HOME."""
    if hermes_home is None:
        from hermes_constants import get_hermes_home
        hermes_home = get_hermes_home()
    home = Path(hermes_home)

    values: Dict[str, Any] = {}
    # Dashboard flat-json location first, then noctuary.json on top.
    values.update(_read_json(home / STORE_DIRNAME / "config.json"))
    values.update(_read_json(home / CONFIG_BASENAME))
    return NoctuaryConfig(hermes_home=home, values=values)


def save_config_values(values: Dict[str, Any], hermes_home: str | Path) -> None:
    """Merge *values* into ``$HERMES_HOME/noctuary.json``."""
    home = Path(hermes_home)
    path = home / CONFIG_BASENAME
    existing = _read_json(path)
    existing.update(values)
    home.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
