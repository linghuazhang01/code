"""Local-first tokenizer loading for the unchanged G-OPD formatter."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any, Iterator
from unittest.mock import patch

LOGGER = logging.getLogger(__name__)
FORMATTER_MODEL = "Qwen/Qwen3-4B"
_FORMATTER_LOCK = RLock()


@lru_cache(maxsize=1)
def load_formatter_tokenizer() -> Any:
    """Resolve a cached snapshot first, avoiding remote metadata probes."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from transformers import AutoTokenizer

    try:
        config = hf_hub_download(
            FORMATTER_MODEL, "tokenizer_config.json", local_files_only=True,
        )
        snapshot = str(Path(config).parent)
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot, trust_remote_code=True, local_files_only=True,
        )
        LOGGER.info("LCB formatter tokenizer loaded from local snapshot=%s", snapshot)
        return tokenizer
    except (LocalEntryNotFoundError, OSError):
        LOGGER.warning("LCB tokenizer cache incomplete; fetching %s once", FORMATTER_MODEL)
        return AutoTokenizer.from_pretrained(FORMATTER_MODEL, trust_remote_code=True)


@contextmanager
def cached_formatter_tokenizer() -> Iterator[None]:
    """Reuse one tokenizer while the upstream formatter renders a prompt batch.

    Only the exact formatter model is intercepted. The patch is scoped to prompt
    construction and restored even on failure; unrelated model loads delegate.
    """
    from transformers import AutoTokenizer

    with _FORMATTER_LOCK:
        tokenizer = load_formatter_tokenizer()
        original = AutoTokenizer.from_pretrained

        def load(model: str, *args: Any, **kwargs: Any) -> Any:
            if model == FORMATTER_MODEL:
                return tokenizer
            return original(model, *args, **kwargs)

        with patch.object(AutoTokenizer, "from_pretrained", side_effect=load):
            yield
