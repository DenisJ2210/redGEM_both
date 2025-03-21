"""
.. module:: pytfa
   :platform: Unix, Windows
   :synopsis: Thermodynamics-based Flux Analysis

.. moduleauthor:: pyTFA team

Sampling wrappers for pytfa models
"""

import numpy as np
from sympy.core.singleton import S
from time import time
from cobra.sampling import OptGPSampler, shared_np_array
from vendor.cobra.sampling.hr_sampler import HRSampler
from vendor.cobra.sampling.achr import ACHRSampler
from optlang.interface import OPTIMAL

class GeneralizedHRSampler(HRSampler):

    def __init__(self, model, thinning, nproj=None, seed=None):
        """
        Adapted from cobra.flux_analysis.sampling.py
        _________________________________________

        Initialize a new sampler object.
        """
        print("Initializing GeneralizedHRSampler...")
        # This currently has to be done to reset the solver basis which is
        # required to get deterministic warmup point generation
        # (in turn required for a working `seed` arg)
        HRSampler.__init__(self, model, thinning, seed=seed)
        print("HRSampler initialized.")

        if model.solver.is_integer:
            raise TypeError("Sampling does not work with integer problems :(")

        self.model = model.copy()
        self.thinning = thinning

        if nproj is None:
            self.nproj = int(min(len(self.model.variables) ** 3, 1e6))
            print(f"nproj set to default value: {self.nproj}")
        else:
            self.nproj = nproj
            print(f"nproj set to: {self.nproj}")

        self.n_samples = 0
        self.retries = 0

        print("Building problem matrix...")
        self.problem = self._HRSampler__build_problem()
        print("Problem matrix built.")

        # Set up a map from reaction -> forward/reverse variable
        print("Setting up variable index mapping...")
        var_idx = {v: idx for idx, v in enumerate(self.model.variables)}
        self.var_idx = np.array(
            [var_idx[v] for v in self.model.variables]
        )
        print("Variable index mapping set up.")

        self.warmup = None
        if seed is None:
            self._seed = int(time())
        else:
            self._seed = seed

        # Avoid overflow
        self._seed = self._seed % np.iinfo(np.int32).max
        print(f"GeneralizedHRSampler initialized with seed: {self._seed}")

    def generate_fva_warmup(self):
        """
        Adapted from cobra.flux_analysis.sampling.py
        __________________________________________

        Generate the warmup points for the sampler.
        """
        print("Generating FVA warmup points...")
        self.n_warmup = 0
        idx = np.hstack([self.var_idx])
        self.warmup = np.zeros((len(idx), len(self.model.variables)))
        self.model.objective = S.Zero
        self.model.objective.direction = "max"
        variables = self.model.variables

        for i in idx:
            # Omit fixed reactions
            if self.problem.variable_fixed[i]:
                self.model.logger.info(f"Skipping fixed variable {variables[i].name}")
                continue

            print(f"Maximizing variable {variables[i].name}...")
            self.model.objective.set_linear_coefficients({variables[i]: 1})
            self.model.slim_optimize()

            if not self.model.solver.status == OPTIMAL:
                self.model.logger.info(f"Cannot maximize variable {variables[i].name}, skipping it.")
                continue

            primals = self.model.solver.primal_values
            sol = [primals[v.name] for v in self.model.variables]
            self.warmup[self.n_warmup,] = sol
            self.n_warmup += 1
            print(f"Stored warmup point {self.n_warmup}.")

            # Revert objective
            self.model.objective.set_linear_coefficients({variables[i]: 0})

        # Shrink warmup points to measure
        self.warmup = shared_np_array((self.n_warmup, len(variables)), self.warmup[0:self.n_warmup,])
        print(f"FVA warmup points generated. Total warmup points: {self.n_warmup}")


# Next, we redefine the analysis class as both inheriting from the
# GeneralizedHRSampler, and then the original classes. This will overwrite the
# inherited methods from HRSampler, while keeping the lower level ones from the
# samplers

class GeneralizedACHRSampler(GeneralizedHRSampler, ACHRSampler):
    def __init__(self, model, thinning=100, seed=None):
        """
        Adapted from cobra.flux_analysis.analysis
        __________________________________________
        Initialize a new ACHRSampler."""
        print("Initializing GeneralizedACHRSampler...")
        GeneralizedHRSampler.__init__(self, model, thinning, seed=seed)
        self.generate_fva_warmup()
        self.prev = self.center = self.warmup.mean(axis=0)
        np.random.seed(self._seed)
        print("GeneralizedACHRSampler initialized.")

class GeneralizedOptGPSampler(GeneralizedHRSampler, OptGPSampler):
    def __init__(self, model, processes, thinning=100, seed=None):
        """
        Adapted from cobra.flux_analysis.sampling.py
        __________________________________________
        Initialize a new OptGPSampler."""
        print("Initializing GeneralizedOptGPSampler...")
        GeneralizedHRSampler.__init__(self, model, thinning, seed=seed)
        self.generate_fva_warmup()
        self.processes = processes

        # This maps our saved center into shared memory,
        # meaning they are synchronized across processes
        self.center = shared_np_array((len(self.model.variables),), self.warmup.mean(axis=0))
        print("GeneralizedOptGPSampler initialized.")

def sample(model, n, method="optgp", thinning=100, processes=1, seed=None):
    """
    Sample valid flux distributions from a thermo cobra_model.

    Function adapted from cobra.flux_analysis.sample to display all solver
    variables
    """
    print(f"Starting sampling with method '{method}'...")
    
    if method == "optgp":
        print("Using GeneralizedOptGPSampler...")
        sampler = GeneralizedOptGPSampler(model, processes, thinning=thinning, seed=seed)
    elif method == "achr":
        print("Using GeneralizedACHRSampler...")
        sampler = GeneralizedACHRSampler(model, thinning=thinning, seed=seed)
    else:
        raise ValueError("Method must be 'optgp' or 'achr'!")

    print(f"Generating {n} samples...")
    result = sampler.sample(n, fluxes=False)
    print("Sampling completed.")
    return result
