# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from redgempy.gem_processing import identify_biomass_reaction


def find_max_biomass_growth_rate(model):
    """
    Identify the biomass reaction in a genome-scale model (GEM) and calculate
    the maximum biomass growth rate by setting it as the objective function.

    Args:
        model (cobra.Model): A genome-scale metabolic model.

    Returns:
        tuple:
            cobra.Model: The original model restored to its initial state.
            float: The maximum biomass growth rate.

    Raises:
        ValueError: If the biomass reaction cannot be identified.
    """
    # Step 1: Identify the biomass reaction
    if not hasattr(model, "biomass_reaction") or model.biomass_reaction is None:
        biomass_rxn, _, model = identify_biomass_reaction(model)
        if not biomass_rxn:
            raise ValueError("Unable to identify a biomass reaction in the model.")
        model.biomass_reaction = (
            biomass_rxn  # Optionally store it in the model for reuse
        )
    else:
        biomass_rxn = model.biomass_reaction

    # Step 2: Save the original model objective for resetting later
    original_objective = model.objective
    original_objective_direction = model.objective_direction

    # Step 3: Modify the model objective to maximize biomass
    model.objective = biomass_rxn
    model.objective_direction = "max"

    # Step 4: Solve the optimization problem
    solution = model.optimize()
    max_growth_rate = solution.objective_value if solution.status == "optimal" else None

    # Step 5: Reset the model to its original state
    model.objective = original_objective
    model.objective_direction = original_objective_direction

    # Log an error if no optimal solution was found
    if max_growth_rate is None:
        print("Warning: No optimal solution was found for biomass maximization.")

    return model, max_growth_rate
