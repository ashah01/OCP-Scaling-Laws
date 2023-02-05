import torch

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)
from ocpmodels.models.base import BaseModel


@registry.register_model("graphormer")
class Graphormer(BaseModel):
    def __init__(
        self, num_atoms, bond_feat_dim, num_targets, regress_forces=True
    ):
        self.regress_forces = regress_forces
        super().__init__()

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        # pos = data.pos
        # batch = data.batch
        (
            edge_index,
            dist,
            _,
            cell_offsets,
            offsets,
            neighbors,
        ) = self.generate_graph(data)
        return data

    def forward(self, data):
        if self.regress_forces:
            data.pos.requires_grad_(True)
        energy = self._forward(data)

        if self.regress_forces:
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    data.pos,
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True,
                )[0]
            )
            return energy, forces
        else:
            return energy

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
