import torch
from torch import Tensor

from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized

from torch.utils.data.sampler import Sampler

T_co = TypeVar('T_co', covariant=True)

class ModifiedSubsetRandomSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement and stores output indices.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], generator=None) -> None:
        self.indices = indices
        self.generator = generator
        self.samplerIndex = []

    def __iter__(self) -> Iterator[int]:
        self.samplerIndex = []
        for i in torch.randperm(len(self.indices), generator=self.generator):
            #print("          Index '%i' will be used." % (self.indices[i]))
            self.samplerIndex.append(self.indices[i])
            yield self.indices[i]

    def __len__(self) -> int:
        return len(self.indices)
