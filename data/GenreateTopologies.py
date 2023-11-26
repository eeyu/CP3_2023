import numpy as np

import paths
import utils_public as up
import matplotlib.pyplot as plt
from enum import Enum

class Constraint(Enum):
    H_BC = 1
    V_BC = 2
    H_LOAD = 3
    V_LOAD = 4
    VOL_FRAC = 5

class TopologySet:
    def __init__(self, topology, constraints):
        # Topology in (64, 64)
        # Constraints 5 x (64, 64)
        self.topology = topology
        self.constraints = {
            Constraint.H_BC: constraints[0].toarray() if constraints[0] is not None else None,
            Constraint.V_BC:  constraints[1].toarray() if constraints[1] is not None else None,
            Constraint.H_LOAD:  constraints[2].toarray() if constraints[2] is not None else None,
            Constraint.V_LOAD:  constraints[3].toarray() if constraints[3] is not None else None,
            Constraint.VOL_FRAC:  constraints[4],
        }
        self._constraints_raw = constraints


    def sample_masked_topology(self, mask_topology=False, mask_constraints=False):
        # For topology
        topology = self.topology
        if mask_topology:
            mask = up.random_n_masks(np.array((64, 64)), 4, 7).astype(bool)  # From the utils file - feel free to check it out
            topology = self.topology * (1 - mask) + 0.5 * (mask)

        # For constraint
        constraints = self._constraints_raw.copy()
        if mask_constraints:
            mask = np.random.choice(range(5), 2, replace=False)
            for j in mask:
                constraints[j] = None

        return TopologySet(topology=topology, constraints=constraints)

class Topologies:
    def __init__(self):
        self.topologies = np.load(paths.HOME_PATH + "topologies_train.npy") # 12k
        self.constraints_sparse = np.load(paths.HOME_PATH + "constraints_train.npy", allow_pickle=True)
        self.length = len(self.topologies)

        self.topology_sets = []
        for i in range(len(self.topologies)):
            self.topology_sets.append(TopologySet(self.topologies[i], self.constraints_sparse[i]))
        # Constraints:
        # - H BC
        # - V BC
        # - H Loads
        # - V Loads
        # - Volume Fraction
    def get_unmasked_data(self, i):
        return self.topology_sets[i]

    def get_masked_data(self, i, mask_topology=False, mask_constraints=False):
        return self.topology_sets[i].sample_masked_topology(mask_topology=mask_topology, mask_constraints=mask_constraints)

    def get_topology_tensor(self, indices, mask=False):
        tensor = []
        for i in indices:
            tensor.append(self.topology_sets[i].sample_masked_topology(mask_topology=mask))
        return np.array(tensor)

    def get_constraint_tensor(self, indices, constraint_type: Constraint, mask=False):
        tensor = []
        for i in indices:
            constraints = self.topology_sets[i].sample_masked_topology(mask_constraints=mask)
            tensor.append(constraints[constraint_type])
        return np.array(tensor)

if __name__ == "__main__":
    topologies = Topologies()
    topology_set_i = topologies.get_unmasked_data(104)
    masked_topology = topologies.get_masked_data(104, mask_topology=True, mask_constraints=False)

    constraints_0 = topology_set_i.constraints[Constraint.V_BC]
    up.plot_n_topologies([topology_set_i.topology, masked_topology.topology])
    # if constraints_0 is None:
    #     print("None")
    # else:
    #     plt.imshow(constraints_0, cmap="gray")
    #     plt.show()