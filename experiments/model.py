import gurobipy as gp
from gurobipy import GRB
import itertools
from collections.abc import Mapping, Sequence
import numpy as np

from instance import Instance, Arc, Node

class Model:
    def __init__(self, instance: Instance):
        # TODO: clean this up yuck
        self.instance = instance
        model = self.model = gp.Model("asymmetric")
        model.setParam("OutputFlag", 0)

        world = instance.world
        scenarios = instance.scenarios
        demands = instance.demands
        capacities = instance.capacities
        paths_per_od = instance.paths_per_od
        active_arcs = instance.active_arcs
        arc_to_idx = instance.arc_to_idx
        tau = instance.tau
        phi = instance.phi

        num_arcs = len(active_arcs)
        arc_indices = self.arc_indices = np.arange(num_arcs)

        T = self.T = {od: i for i, od in enumerate(demands.keys())}
        od_list = list(demands.keys())

        profiles_per_od = self.profiles_per_od = {
            od: list(itertools.combinations_with_replacement(paths_per_od[od], n_k))
            for od, n_k in demands.items()
        }

        joint_action_profiles = self.joint_action_profiles = list(itertools.product(*profiles_per_od.values()))
        num_joint = len(joint_action_profiles)

        path_arc_indices = self.path_arc_indices = {
            (od, i): [arc_to_idx[a] for a in path]
            for od, paths in paths_per_od.items()
            for i, path in enumerate(paths)
        }

        # m := len(profiles_per_od[?])
        # m = (n + k - 1) choose (k - 1)
        od_path_flows: Mapping[Node, np.ndarray] = {} # (m x num_arcs), possible path flows induced by od
        od_arc_flows: Mapping[Node, np.ndarray] = {} # (m x num_arcs), possible arc flows induced by od
        for od, paths in paths_per_od.items():
            # (m x k)
            counts = od_path_flows[od] = np.array([
                [profile.count(p) for p in paths]
                for profile in profiles_per_od[od]
            ])

            # map paths to their arcs
            path_to_arcs = np.zeros((len(paths), num_arcs))
            for i, path in enumerate(paths):
                path_to_arcs[i, path_arc_indices[od, i]] = 1

            # (m x num_arcs)
            od_arc_flows[od] = counts @ path_to_arcs


        # TODO: optimize combining od flows
        # suppose matrices A, B, C:
        # A = [a0
        #      a1]
        # B = [b0
        #      b1]
        # C = [c0
        #      c1]
        # add all combinations of rows:
        # X = [c0 + a0 + b0
        #      c0 + a0 + b1
        #      c0 + a1 + b0
        #      c0 + a1 + b1
        #      c1 + a0 + b0
        #      c1 + a0 + b1
        #      c1 + a1 + b0
        #      c1 + a1 + b1]
        # i.e. repeat the left, tile the right


        # combine od flows
        # (num_joint x num_arcs)
        flow_matrix = self.flow_matrix = np.array([
            sum(combo)
            for combo in itertools.product(
                *(od_arc_flows[od] for od in od_list)
            )
        ], dtype=int)

        # TODO: can be similarly optimized to above
        # (num_joint,)
        path_flows = self.path_flows = {
            (od, i): np.array([
                joint_profile[T[od]].count(path)
                for joint_profile in joint_action_profiles
            ])
            for od, paths in paths_per_od.items()
            for i, path in enumerate(paths)
        }

        path_as_set = {
            path: set(path)
            for paths in paths_per_od.values()
            for path in paths
        }

        # partition the arcs in p1 into two groups: (p1 & p0, p1 - p0)
        partitioned_paths = self.partitioned_paths = {}
        for od, paths in paths_per_od.items():
            for p0_idx, p0 in enumerate(paths):
                p0_set = path_as_set[p0]
                for p1_idx, p1 in enumerate(paths):
                    if p0_idx != p1_idx:
                        shared = [arc_to_idx[a] for a in p1 if a in p0_set]
                        dev = [arc_to_idx[a] for a in p1 if a not in p0_set]
                        partitioned_paths[od, p0_idx, p1_idx] = (shared, dev)

        # decision: sigma[omega, a] := \prob[a \mid \theta]
        sigma = self.sigma = {
            scenario_name: [
                model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0, ub=1,
                    name=f"sigma_{scenario_name}_{action_profile_idx}"
                )
                for action_profile_idx in range(len(joint_action_profiles))
            ]
            for scenario_name in scenarios
        }

        # constraint: \sum_{rho \in \mathcal{A}} sigma[\omega, rho] = 1
        for scenario_name in scenarios:
            model.addConstr(
                gp.quicksum(sigma[scenario_name]) == 1,
                name=f"sigma_{scenario_name}"
            )

        pot_coeffs = []
        cost_coeffs = []
        self.all_sigma_vars = list(
            itertools.chain.from_iterable(
                sigma_vars
                for sigma_vars in sigma.values()
            )
        )

        for scenario_name, mu in scenarios.items():
            tai_matrix = tau[scenario_name]
            phi_matrix = phi[scenario_name]

            # per joint profile
            potentials = phi_matrix[arc_indices, flow_matrix].sum(axis=1)
            costs = (flow_matrix * tai_matrix[arc_indices, flow_matrix]).sum(axis=1)

            pot_coeffs.extend((mu * potentials).tolist())
            cost_coeffs.extend((mu * costs).tolist())

        self.pot_coeffs = pot_coeffs
        self.cost_coeffs = cost_coeffs

    def gen_expected_potential(self):
        return gp.LinExpr(self.pot_coeffs, self.all_sigma_vars)

    def gen_expected_social_cost(self):
        return gp.LinExpr(self.cost_coeffs, self.all_sigma_vars)

    def debug(self):
        model = self.model
        scenarios = self.instance.scenarios
        sigma = self.sigma
        joint_action_profiles = self.joint_action_profiles
        profiles_per_od = self.profiles_per_od
        paths_per_od = self.instance.paths_per_od

        model.optimize()
        assert model.Status == GRB.OPTIMAL, f"Optimization failed: {model.Status}"

        out = []
        out.append(f"{model.ObjVal=:.2f}")
        out.append(f"{model.Runtime=:.3f}s")

        for scenario_name in scenarios:
            sigma_vars = sigma[scenario_name]
            out.append(f"\nState: {scenario_name}")

            for action_profile_idx, joint_profile in enumerate(joint_action_profiles):
                prob = sigma_vars[action_profile_idx].X
                if prob > 1e-3:
                    out.append(f"\tJoint Profile {action_profile_idx}: ({prob=:.3f})")
                    for od, od_profile in zip(profiles_per_od.keys(), joint_profile):
                        path_flows = tuple(od_profile.count(path) for path in paths_per_od[od])
                        out.append(f"\t\t{od}: {path_flows}")

        return out, model.ObjVal
