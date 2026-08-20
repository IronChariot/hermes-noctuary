"""Dashboard config schema for the Noctuary memory provider.

Loaded by path from the web server (never imported as part of the plugin
package), so it may only import from ``plugins.memory.config_schema``.

Storage note: the generic flat-json backend persists these fields to
``$HERMES_HOME/noctuary/config.json``. The provider reads that file and
``$HERMES_HOME/noctuary.json``, with the latter taking precedence.
"""

from plugins.memory.config_schema import (
    KIND_NUMBER,
    KIND_TEXT,
    ProviderConfigSchema,
    ProviderField,
)

CONFIG_SCHEMA = ProviderConfigSchema(
    name="noctuary",
    label="Noctuary",
    fields=(
        ProviderField(
            key="recallTokenBudget",
            label="Recall token budget",
            kind=KIND_NUMBER,
            default="1000",
            description="Token budget for the passive recall packet.",
            inline=True,
        ),
        ProviderField(
            key="embeddingModel",
            label="Embedding model",
            kind=KIND_TEXT,
            default="auto",
            description="auto, hash, or a fastembed/sentence-transformers model name.",
            inline=True,
        ),
        ProviderField(
            key="librarianModel",
            label="Librarian model",
            kind=KIND_TEXT,
            default="",
            description="Model for nightly consolidation. Empty = the agent's model.",
            inline=True,
        ),
        ProviderField(
            key="librarianProvider",
            label="Librarian provider",
            kind=KIND_TEXT,
            default="",
            description="Provider for the librarian model. Empty = auto.",
        ),
        ProviderField(
            key="surfacePageLimit",
            label="Surface page limit",
            kind=KIND_NUMBER,
            default="12",
            description="Target size of the surface (passive recall) layer.",
        ),
        ProviderField(
            key="minUserTurns",
            label="Activity gate",
            kind=KIND_NUMBER,
            default="1",
            description="Days with fewer user turns are skipped by the librarian.",
        ),
    ),
)
