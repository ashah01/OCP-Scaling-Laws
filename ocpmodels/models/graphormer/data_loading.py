# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import lru_cache
from pathlib import Path
from typing import Dict, Sequence, Union

import numpy as np
import torch
from torch import Tensor


class PBCDataset:
    def __init__(self):
        self.cell_offsets = torch.tensor(
            [
                [-1, -1, 0],
                [-1, 0, 0],
                [-1, 1, 0],
                [0, -1, 0],
                [0, 1, 0],
                [1, -1, 0],
                [1, 0, 0],
                [1, 1, 0],
            ],
        ).float()
        self.n_cells = self.cell_offsets.size(0)
        self.cutoff = 8
        self.filter_by_tag = True

    def preprocess(self, data):

        pos = data["pos"]
        cell = data["cell"]
        atoms = data["atomic_numbers"]
        tags = data["tags"]
        batch_size = data["cell"].size(0)

        indices = [0, *map(lambda x: x.item(), data.natoms)]
        for i in range(batch_size):
            pos_ = pos[indices[i] : indices[i + 1] + indices[i]]
            tags_ = tags[indices[i] : indices[i + 1] + indices[i]]
            offsets = torch.matmul(self.cell_offsets, cell[i]).view(
                self.n_cells, 1, 3
            )
            expand_pos = (
                pos_.unsqueeze(0).expand(self.n_cells, -1, -1) + offsets
            ).view(-1, 3)
            src_pos = pos_[tags_ > 1] if self.filter_by_tag else pos_

            dist: Tensor = (
                src_pos.unsqueeze(1) - expand_pos.unsqueeze(0)
            ).norm(dim=-1)
            used_mask = (dist < self.cutoff).any(dim=0) & tags_.ne(2).repeat(
                self.n_cells
            )  # not copy ads
            used_expand_pos = expand_pos[used_mask]

            used_expand_tags = tags_.repeat(self.n_cells)[
                used_mask
            ]  # original implementation use zeros, need to test
        # TODO: Integrate pad1d collation into batched preprocessing
        return dict(
            pos=torch.cat([pos, used_expand_pos], dim=0),
            atoms=torch.cat([atoms, atoms.repeat(self.n_cells)[used_mask]]),
            tags=torch.cat([tags, used_expand_tags]),
            real_mask=torch.cat(
                [
                    torch.ones_like(tags, dtype=torch.bool),
                    torch.zeros_like(used_expand_tags, dtype=torch.bool),
                ]
            ),
        )


def pad_1d(samples: Sequence[Tensor], fill=0, multiplier=8):
    max_len = max(x.size(0) for x in samples)
    max_len = (max_len + multiplier - 1) // multiplier * multiplier
    n_samples = len(samples)
    out = torch.full(
        (n_samples, max_len, *samples[0].shape[1:]),
        fill,
        dtype=samples[0].dtype,
    )
    for i in range(n_samples):
        x_len = samples[i].size(0)
        out[i][:x_len] = samples[i]
    return out


class AtomDataset:
    def __init__(self, dataset, keyword):
        super().__init__()
        self.dataset = dataset
        self.keyword = keyword
        self.atom_list = [
            1,
            5,
            6,
            7,
            8,
            11,
            13,
            14,
            15,
            16,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            55,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
        ]
        # fill others as unk
        unk_idx = len(self.atom_list) + 1
        self.atom_mapper = torch.full((128,), unk_idx)
        for idx, atom in enumerate(self.atom_list):
            self.atom_mapper[atom] = idx + 1  # reserve 0 for paddin

    def get(self):
        atoms: Tensor = self.dataset[self.keyword]
        return self.atom_mapper[atoms]

    def collater(self, samples):
        return pad_1d(samples)


class KeywordDataset:
    def __init__(self, dataset, keyword, is_scalar=False, pad_fill=0):
        super().__init__()
        self.dataset = dataset
        self.keyword = keyword
        self.is_scalar = is_scalar
        self.pad_fill = pad_fill

    def get(self):
        return self.dataset[self.keyword]

    def collater(self, samples):
        if self.is_scalar:
            return torch.tensor(samples)
        return pad_1d(samples, fill=self.pad_fill)


def load_dataset(data):
    data.atomic_numbers = data.atomic_numbers.long()

    pbc_dataset = PBCDataset()
    result = pbc_dataset.preprocess(data)

    atoms = AtomDataset(result, "atoms").get()
    tags = KeywordDataset(result, "tags").get()
    real_mask = KeywordDataset(result, "real_mask").get()

    pos = KeywordDataset(result, "pos").get()

    return atoms, tags, pos, real_mask
