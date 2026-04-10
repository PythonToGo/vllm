# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from collections.abc import Iterable

from vllm.v1.kv_offload.abstract import OffloadKey
from vllm.v1.kv_offload.cpu.policies.abstract import BlockStatus, CachePolicy


class LFUCachePolicy(CachePolicy):
    """LFU cache policy with recency tie-breaking within each frequency."""

    def __init__(self, cache_capacity: int):
        self.cache_capacity = cache_capacity
        self.blocks: dict[OffloadKey, BlockStatus] = {}
        self.freqs: dict[OffloadKey, int] = {}
        self.freq_buckets: dict[int, OrderedDict[OffloadKey, None]] = {}
        self.min_freq = 0

    def get(self, key: OffloadKey) -> BlockStatus | None:
        return self.blocks.get(key)

    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        self.blocks[key] = block
        self.freqs[key] = 1
        self.freq_buckets.setdefault(1, OrderedDict())[key] = None
        self.min_freq = 1

    def remove(self, key: OffloadKey) -> None:
        freq = self.freqs.pop(key)
        del self.blocks[key]
        bucket = self.freq_buckets[freq]
        del bucket[key]
        if not bucket:
            del self.freq_buckets[freq]
            if self.min_freq == freq:
                self._update_min_freq()

    def touch(self, keys: Iterable[OffloadKey]) -> None:
        for key in reversed(list(keys)):
            if key not in self.blocks:
                continue

            old_freq = self.freqs[key]
            new_freq = old_freq + 1

            old_bucket = self.freq_buckets[old_freq]
            del old_bucket[key]
            if not old_bucket:
                del self.freq_buckets[old_freq]
                if self.min_freq == old_freq:
                    self.min_freq = new_freq

            self.freqs[key] = new_freq
            self.freq_buckets.setdefault(new_freq, OrderedDict())[key] = None

    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        if n == 0:
            return []

        candidates: list[tuple[OffloadKey, BlockStatus]] = []
        already_selected: set[OffloadKey] = set()

        for freq in sorted(self.freq_buckets):
            for key in self.freq_buckets[freq]:
                block = self.blocks[key]
                if (
                    block.ref_cnt == 0
                    and key not in protected
                    and key not in already_selected
                ):
                    candidates.append((key, block))
                    already_selected.add(key)
                    if len(candidates) == n:
                        break
            if len(candidates) == n:
                break

        if len(candidates) < n:
            return None

        for key, _ in candidates:
            self.remove(key)

        return candidates

    def _update_min_freq(self) -> None:
        self.min_freq = min(self.freq_buckets, default=0)
