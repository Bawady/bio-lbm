from src.Serializable import Serializable
from typing import Callable
import numpy as np


class MassActionKinetics:

    def init(self, reactants, products, rate, params):
        self.rate = rate
        self.params = params
        self.reactants = [r[0] for r in reactants if r[1] != 0]
        self.externals = [r[0] for r in reactants if r[1] == 0]
        self.stoich_coeffs = [r[1] for r in reactants if r[1] != 0]
        self.weights = [-r[1] for r in reactants if r[1] != 0]
        self.products = [p[0] for p in products]
        self.species = self.reactants.copy()
        for p in products:
            try:
                i = self.species.index(p[0])
                self.weights[i] += p[1]
            except ValueError:
                self.species.append(p[0])
                self.weights.append(p[1])

    def react(self, cap_at_zero=False, flavor=0):
        tmp = self.rate(self.reactants, self.externals, self.products, **self.params)
        for i in range(len(self.reactants)):
            if cap_at_zero:
                bac = self.reactants[i].density
                capped = np.where(bac <= 0, 0, bac)
                tmp *= capped**self.stoich_coeffs[i]
            else:
                tmp *= self.reactants[i].density**self.stoich_coeffs[i]
        for i in range(len(self.species)):
            if self.weights[i] == 0:
                continue
            reac_term = self.weights[i] * tmp
            spec = self.species[i]
            for j in range(self.species[i].lattice.Q):
                if flavor == 1:
                    # Hilpert's original term, our results are with reac_term only
                    self.species[i].f_coll[j] += reac_term #* spec.f[j] / spec.density
                elif flavor == 2:
                    # Reaction term for QO94
                    spec.f_coll[j] += spec.lattice.dt * reac_term / spec.lattice.Q
                else:
                    spec.f_coll[j] += spec.lattice.weights[j] * spec.lattice.dt * reac_term


class CRN:
    def __init__(self):
        self.reactions = []

    def add_reaction(self, reactants: dict, products: dict, rate: Callable, **params: dict) -> None:
        reaction = MassActionKinetics()
        reaction.init(reactants, products, rate, params)
        self.reactions.append(reaction)

    def react(self, **react_params) -> None:
        for r in self.reactions:
            r.react(**react_params)
