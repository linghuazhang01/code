"""Local-first formatter loader tests without model downloads."""

import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

from eval.lcb_tokenizer import cached_formatter_tokenizer, load_formatter_tokenizer


class LocalCacheMiss(Exception):
    pass


class TokenizerCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        load_formatter_tokenizer.cache_clear()
        self.hub = ModuleType("huggingface_hub")
        self.hub.hf_hub_download = Mock(return_value="/cache/snapshots/rev/tokenizer_config.json")
        self.errors = ModuleType("huggingface_hub.errors")
        self.errors.LocalEntryNotFoundError = LocalCacheMiss
        self.transformers = ModuleType("transformers")
        self.transformers.AutoTokenizer = type("AutoTokenizer", (), {"from_pretrained": Mock()})
        self.loader = self.transformers.AutoTokenizer.from_pretrained
        self.modules = patch.dict(sys.modules, {
            "huggingface_hub": self.hub,
            "huggingface_hub.errors": self.errors,
            "transformers": self.transformers,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.addCleanup(load_formatter_tokenizer.cache_clear)

    def test_local_snapshot_and_memory_reuse(self) -> None:
        first = load_formatter_tokenizer()
        self.assertIs(first, load_formatter_tokenizer())
        self.hub.hf_hub_download.assert_called_once_with(
            "Qwen/Qwen3-4B", "tokenizer_config.json", local_files_only=True,
        )
        self.loader.assert_called_once_with(
            "/cache/snapshots/rev", trust_remote_code=True, local_files_only=True,
        )

    def test_cache_miss_falls_back_once(self) -> None:
        self.hub.hf_hub_download.side_effect = LocalCacheMiss()
        load_formatter_tokenizer()
        load_formatter_tokenizer()
        self.loader.assert_called_once_with("Qwen/Qwen3-4B", trust_remote_code=True)

    def test_incomplete_snapshot_falls_back(self) -> None:
        self.loader.side_effect = [OSError("missing vocab"), object()]
        load_formatter_tokenizer()
        self.assertEqual(self.loader.call_count, 2)

    def test_invalid_config_fails_without_network_fallback(self) -> None:
        self.loader.side_effect = ValueError("bad tokenizer config")
        with self.assertRaises(ValueError):
            load_formatter_tokenizer()
        self.assertEqual(self.loader.call_count, 1)

    def test_scoped_reuse_delegation_and_restoration(self) -> None:
        tokenizer = load_formatter_tokenizer()
        with self.assertRaises(RuntimeError):
            with cached_formatter_tokenizer():
                for _ in range(10):
                    self.assertIs(
                        self.transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"),
                        tokenizer,
                    )
                self.transformers.AutoTokenizer.from_pretrained("other")
                raise RuntimeError("formatter failed")
        self.assertIs(self.transformers.AutoTokenizer.from_pretrained, self.loader)
        self.assertEqual(self.loader.call_count, 2)
        self.loader.assert_called_with("other")


if __name__ == "__main__":
    unittest.main()
