"""Test bootstrap.

The plugin imports ``agent.memory_provider`` from hermes-agent. Point
``HERMES_AGENT_ROOT`` at a checkout, or keep the default sibling layout:

    <parent>/hermes-noctuary   (this repo)
    <parent>/hermes-agent
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HERMES_AGENT_ROOT = Path(
    os.environ.get("HERMES_AGENT_ROOT", REPO_ROOT.parent / "hermes-agent")
)

for entry in (str(REPO_ROOT), str(HERMES_AGENT_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

if not (HERMES_AGENT_ROOT / "agent" / "memory_provider.py").is_file():
    pytest.exit(
        f"hermes-agent checkout not found at {HERMES_AGENT_ROOT}; "
        "set HERMES_AGENT_ROOT",
        returncode=3,
    )


@pytest.fixture()
def hermes_home(tmp_path):
    home = tmp_path / "hermes-home"
    home.mkdir()
    return home


@pytest.fixture()
def cfg(hermes_home):
    from noctuary.config import NoctuaryConfig

    return NoctuaryConfig(
        hermes_home=hermes_home,
        values={
            "embeddingModel": "hash",
            # Hash-embedder similarities run lower than real models; relax
            # the recall floors so ranking (not absolute magnitude) is tested.
            "familiaritySimilarity": 0.05,
            "gistSimilarity": 0.25,
        },
    )


@pytest.fixture()
def store(cfg):
    from noctuary.store import NoctuaryStore

    s = NoctuaryStore(cfg.store_root)
    s.init()
    return s
