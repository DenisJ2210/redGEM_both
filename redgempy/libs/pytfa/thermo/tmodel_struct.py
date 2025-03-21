# -*- coding: utf-8 -*-
"""
.. module:: pytfa
   :platform: Unix, Windows
   :synopsis: Thermodynamics-based Flux Analysis

.. moduleauthor:: pyTFA team

Thermodynamic cobra_model class and methods definition

"copied from pytfa but idea is to create a model with tfa structure without deltag constraints
'should contain FU BU integers and fu+bu<=1

"""

import re
import pandas as pd
from cobra import Model

from libs.pytfa.core.model import LCSBModel
from libs.pytfa.thermo import std
from libs.pytfa.optim.constraints import (
    SimultaneousUse,
    BackwardDirectionCoupling,
    ForwardDirectionCoupling,
)
from libs.pytfa.optim.variables import (
    ForwardUseVariable,
    BackwardUseVariable,
)
from libs.pytfa.utils import numerics
from libs.pytfa.utils.logger import get_bistream_logger

from libs.pytfa.io.dict import (
    model_from_dict,
    model_to_dict,
)
from libs.pytfa.optim.utils import copy_solver_configuration

BIGM = numerics.BIGM
BIGM_THERMO = numerics.BIGM_THERMO
BIGM_DG = numerics.BIGM_DG
BIGM_P = numerics.BIGM_P
EPSILON = numerics.EPSILON
MAX_STOICH = 10


class ThermoModelStructure(LCSBModel, Model):
    """
    A class representing a cobra_model with thermodynamics information

    """

    def __init__(
        self,
        thermo_data=None,
        model=Model(),
        name=None,
        temperature=std.TEMPERATURE_0,
        min_ph=std.MIN_PH,
        max_ph=std.MAX_PH,
    ):
        """

        :param float temperature: the temperature (K) at which to perform the calculations
        :param dict thermo_data: The thermodynamic database
        :type temperature: float
        """

        LCSBModel.__init__(self, model, name)

        self.logger = get_bistream_logger("ME model" + str(self.name))

        self.TEMPERATURE = temperature
        self.thermo_data = thermo_data
        self.parent = model

        # CONSTANTS
        self.MAX_pH = max_ph
        self.MIN_pH = min_ph

    def prepare(self):
        """Prepares a COBRA toolbox cobra_model for TFBA analysis by doing the following:

        1. checks if a reaction is a transport reaction
        2. checks the ReactionDB for Gibbs energies of formation of metabolites
        3. computes the Gibbs energies of reactions

        """

        self.logger.info("# Model preparation starting...")

        # Number of reactions
        num_rxns = len(self.reactions)

        # Number of metabolites
        num_mets = len(self.metabolites)

        for i in range(num_mets):
            met = self.metabolites[i]
            # self._prepare_metabolite(met)

        # And now, reactions !

        self.logger.debug("computing reaction thermodynamic data")

        # Look for the proton enzyme...
        # proton = {}
        # for i in range(num_mets):
        #     if (self.metabolites[i].formula == 'H'
        #             or ('seed_id' in self.metabolites[i].annotation
        #                 and self.metabolites[i].annotation[
        #                     'seed_id'] == 'cpd00067')):
        #         proton[self.metabolites[i].compartment] = self.metabolites[i]
        #
        # if len(proton) == 0:
        #     raise Exception("Cannot find proton")
        # else:
        #     self._proton_of = proton
        #
        # # Iterate over each reaction
        # for i in range(num_rxns):
        #     reaction = self.reactions[i]
        #     self._prepare_reaction(reaction)

        self.logger.info("# Model preparation done.")

    def _convert_metabolite(self, met, add_potentials, verbose):
        """
        Given a enzyme, proceeds to create the necessary variables and
        constraints for thermodynamics-based modeling

        :param met:
        :return:
        """

        if add_potentials:
            P_lb = -BIGM_P  # kcal/mol
            P_ub = BIGM_P  # kcal/mol

        # exclude protons and water and those without deltaGF
        # P_met: P_met - RT*LC_met = DGF_met
        # metformula = met.formula
        # metDeltaGF = met.thermo.deltaGf_tr
        # metDEltaGFerr = met.thermo.deltaGf_err
        # metComp = met.compartment
        # metLConc_lb = log(self.compartments[metComp]['c_min'])
        # metLConc_ub = log(self.compartments[metComp]['c_max'])
        # Comp_pH = self.compartments[metComp]['pH']
        # LC = None

        # if metformula == 'H2O':
        #     LC = self.add_variable(LogConcentration, met, lb=0, ub=0)
        #
        # elif metformula == 'H':
        #     LC = self.add_variable(
        #         LogConcentration,
        #         met,
        #         lb=log(10 ** -Comp_pH),
        #         ub=log(10 ** -Comp_pH))
        #
        # elif ('seed_id' in met.annotation
        #       and met.annotation['seed_id'] == 'cpd11416'):
        #     # we do not create the thermo variables for biomass enzyme
        #     pass
        #
        # elif metDeltaGF < 10 ** 6:
        #     if verbose:
        #         self.logger.debug('generating thermo variables for {}'.format(met.id))
        #     LC = self.add_variable(LogConcentration,
        #                            met,
        #                            lb=metLConc_lb,
        #                            ub=metLConc_ub)
        #
        #     if add_potentials:
        #         DGOF = self.add_variable(DeltaGFormstd,
        #                                  met,
        #                                  lb=-BIGM,
        #                                  ub=BIGM)
        #
        #         self.DGoF_vars[met] = DGOF
        #
        #         P = self.add_variable(ThermoPotential,
        #                               met,
        #                               lb=P_lb,
        #                               ub=P_ub)
        #
        #         self.P_vars[met] = P
        #         # Formulate the constraint
        #         expr = self.RT * LC + DGOF - P
        #         self.add_constraint(
        #             PotentialConstraint,
        #             met,
        #             expr,
        #             lb=0, ub=0)
        #
        # else:
        #     self.logger.debug('NOT generating thermo variables for {}'.format(met.id))
        #
        # if LC != None:
        #     # Register the variable to find it more easily
        #     self.LC_vars[met] = LC

    def _convert_reaction(self, rxn, add_potentials, add_displacement, verbose):
        """

        :param rxn:
        :param add_potentials:
        :param add_displacement:
        :param verbose:
        :return:
        """

        #  RT = self.RT

        DGR_lb = -BIGM_THERMO  # kcal/mol
        DGR_ub = BIGM_THERMO  # kcal/mol

        epsilon = self.solver.configuration.tolerances.feasibility

        # Is it a water transport reaction ?
        H2OtRxns = False
        # if rxn.thermo['isTrans'] and len(rxn.reactants) == 1:
        #     if rxn.reactants[0].annotation['seed_id'] == 'cpd00001':
        #         H2OtRxns = True

        # Is it a drain reaction ?
        NotDrain = len(rxn.metabolites) > 1

        # if the reaction is flagged with rxnThermo, and it's not a H2O
        # transport, we will add thermodynamic constraints
        # if rxn.thermo['computed'] and not H2OtRxns:
        #     if verbose:
        #         self.logger.debug('generating thermo constraint for {}'.format(rxn.id))
        #
        #     # add the delta G as a variable
        #     DGR = self.add_variable(DeltaG, rxn, lb=DGR_lb, ub=DGR_ub)
        #
        #     # add the delta G naught as a variable
        #     RxnDGerror = rxn.thermo['deltaGRerr']
        #     DGoR = self.add_variable(DeltaGstd,
        #                              rxn,
        #                              lb=rxn.thermo['deltaGR'] - RxnDGerror,
        #                              ub=rxn.thermo['deltaGR'] + RxnDGerror)
        #
        #     # Initialization of indices and coefficients for all possible
        #     # scenaria:
        #     LC_TransMet = 0
        #     LC_ChemMet = 0
        #     P_expr = 0
        #     DGoF_expr = 0
        #
        #     if rxn.thermo['isTrans']:
        #         # calculate the DG component associated to transport of the
        #         # enzyme. This will be added to the constraint on the Right
        #         # Hand Side (RHS)
        #
        #         transportedMets = find_transported_mets(rxn)
        #
        #         # Chemical coefficient, it is the enzyme's coefficient...
        #         # + transport coeff for reactants
        #         # - transport coeff for products
        #         chem_stoich = rxn.metabolites.copy()
        #
        #         # Adding the terms for the transport part
        #         for seed_id, trans in transportedMets.items():
        #             for type_ in ['reactant', 'product']:
        #                 if trans[type_].formula != 'H':
        #                     LC_TransMet += (self.LC_vars[trans[type_]]
        #                                     * RT
        #                                     * trans['coeff']
        #                                     * (
        #                                         -1 if type_ == 'reactant' else 1))
        #                     if add_potentials:
        #                         DGoF_expr += (self.DGoF_vars[trans[type_]] * trans['coeff']
        #                                       * (-1 if type_ == 'reactant' else 1))
        #
        #                 chem_stoich[trans[type_]] += (trans['coeff']
        #                                               * (
        #                                                   1 if type_ == "reactant" else -1))
        #
        #         # Also add the chemical reaction part
        #         chem_stoich = {met: val for met, val in chem_stoich.items()
        #                        if val != 0}
        #
        #         for met in chem_stoich:
        #             metFormula = met.formula
        #             if metFormula not in ['H', 'H2O']:
        #                 LC_ChemMet += self.LC_vars[met] * RT * chem_stoich[met]
        #
        #                 if add_potentials:
        #                     DGoF_expr += self.DGoF_vars[met] * rxn.metabolites[met]
        #
        #     else:
        #         # if it is just a regular chemical reaction
        #         for met in rxn.metabolites:
        #             metformula = met.formula
        #             if metformula not in ['H', 'H2O']:
        #                 # we use the LC here as we already accounted for the
        #                 # changes in deltaGFs in the RHS term
        #                 LC_ChemMet += (self.LC_vars[met]
        #                                * RT
        #                                * rxn.metabolites[met])
        #
        #                 if add_potentials:
        #                     DGoF_expr += self.DGoF_vars[met] * rxn.metabolites[met]
        #
        #     # G: - DGR_rxn + DGoRerr_Rxn
        #     #   + RT * StoichCoefProd1 * LC_prod1
        #     #   + RT * StoichCoefProd2 * LC_prod2
        #     #   + RT * StoichCoefSub1 * LC_subs1
        #     #   + RT * StoichCoefSub2 * LC_subs2
        #     #   - ...
        #     #   = 0
        #
        #     # Formulate the constraint
        #     CLHS = DGoR - DGR + LC_TransMet + LC_ChemMet
        #     self.add_constraint(NegativeDeltaG, rxn, CLHS, lb=0, ub=0)
        #
        #     if add_displacement:
        #         lngamma = self.add_variable(ThermoDisplacement,
        #                                     rxn,
        #                                     lb=-BIGM_P,
        #                                     ub=BIGM_P)
        #
        #         # ln(Gamma) = +DGR/RT (DGR < 0 , rxn is forward, ln(Gamma) < 0d
        #         expr = lngamma - 1 / RT * DGR
        #         self.add_constraint(DisplacementCoupling,
        #                             rxn,
        #                             expr,
        #                             lb=0,
        #                             ub=0)
        #
        #     # TODO: we need an exeption for transport reactions that also add the
        #     # Tansport potetnial
        #     if add_potentials:
        #         expr = DGoR - DGoF_expr
        #
        #         if rxn.thermo['isTrans']:
        #             RT_sum_H_LC_tpt = rxn.thermo['breakdown']['RT_sum_H_LC_tpt']
        #             sum_F_memP_charge = rxn.thermo['breakdown']['sum_F_memP_charge']
        #             sum_stoich_NH = rxn.thermo['breakdown']['sum_stoich_NH']
        #
        #             RHS = sum_F_memP_charge + RT_sum_H_LC_tpt + sum_stoich_NH
        #
        #             self.add_constraint(PotentialCoupling,
        #                                 rxn,
        #                                 expr,
        #                                 lb=RHS,
        #                                 ub=RHS, )
        #
        #         else:
        #             self.add_constraint(PotentialCoupling,
        #                                 rxn,
        #                                 expr,
        #                                 lb=0,
        #                                 ub=0)
        #
        #     # Create the use variables constraints and connect them to the
        #     # deltaG if the reaction has thermo constraints
        #
        #     # Note DW: implementing suggestions of maria
        #     # FU_rxn: 1000 FU_rxn + DGR_rxn < 1000 - epsilon
        #     FU_rxn = self.add_variable(ForwardUseVariable, rxn)
        #
        #     CLHS = DGR + FU_rxn * BIGM_THERMO
        #
        #     self.add_constraint(ForwardDeltaGCoupling,
        #                         rxn,
        #                         CLHS,
        #                         ub=BIGM_THERMO - epsilon * 1e3)
        #
        #     # BU_rxn: 1000 BU_rxn - DGR_rxn < 1000 - epsilon
        #     BU_rxn = self.add_variable(BackwardUseVariable, rxn)
        #
        #     CLHS = BU_rxn * BIGM_THERMO - DGR
        #
        #     self.add_constraint(BackwardDeltaGCoupling,
        #                         rxn,
        #                         CLHS,
        #                         ub=BIGM_THERMO - epsilon * 1e3)
        #
        # else:
        #     if not NotDrain:
        #         self.logger.debug('generating only use constraints for drain reaction'
        #                           + rxn.id)
        #     else:
        #         self.logger.debug(
        #             'generating only use constraints for reaction' + rxn.id)
        #
        #     FU_rxn = self.add_variable(ForwardUseVariable, rxn)
        #     BU_rxn = self.add_variable(BackwardUseVariable, rxn)

        # create the prevent simultaneous use constraints
        # SU_rxn: FU_rxn + BU_rxn <= 1

        FU_rxn = self.add_variable(ForwardUseVariable, rxn)
        BU_rxn = self.add_variable(BackwardUseVariable, rxn)
        CLHS = FU_rxn + BU_rxn
        self.add_constraint(SimultaneousUse, rxn, CLHS, ub=1)

        # create constraints that control fluxes with their use variables
        # UF_rxn: F_rxn - M FU_rxn < 0
        F_rxn = rxn.forward_variable
        CLHS = F_rxn - FU_rxn * BIGM
        self.add_constraint(ForwardDirectionCoupling, rxn, CLHS, ub=0)

        # UR_rxn: R_rxn - M RU_rxn < 0
        R_rxn = rxn.reverse_variable
        CLHS = R_rxn - BU_rxn * BIGM
        self.add_constraint(BackwardDirectionCoupling, rxn, CLHS, ub=0)

    def convert(self, add_potentials=False, add_displacement=False, verbose=True):
        """Converts a cobra_model into a tFBA ready cobra_model by adding the
        thermodynamic constraints required

        .. warning::
            This function requires you to have already called
            :func:`~.pytfa.ThermoModel.prepare`, otherwise it will raise an Exception !

        """

        self.logger.info("# Model conversion starting...")

        ###########################################
        # CONSTANTS & PARAMETERS for tFBA problem #
        ###########################################

        # value for the bigM in big M constraints such as:
        # UF_rxn: F_rxn - M*FU_rxn < 0
        bigM = BIGM
        # Check each reactions' bounds
        for reaction in self.reactions:
            if (
                reaction.lower_bound < -bigM - EPSILON
                or reaction.upper_bound > bigM + EPSILON
            ):
                raise Exception("flux bounds too wide or big M not big enough")
            if reaction.lower_bound < -bigM:
                reaction.lower_bound = -bigM
            if reaction.upper_bound > bigM:
                reaction.upper_bound = bigM

        ###################
        # INPUTS & CHECKS #
        ###################

        # check if cobra_model reactions has been checked if they are transport reactions
        # try:
        #     for reaction in self.reactions:
        #         if not 'isTrans' in reaction.thermo:
        #             reaction.thermo['isTrans'] = check_transport_reaction(
        #                 reaction)
        # except:
        #     raise Exception('Reaction thermo data missing. '
        #                     + 'Please run ThermoModel.prepare()')

        # FIXME Use generalized rule (ext to the function)
        # formatting the enzyme and reaction names to remove brackets
        replacements = {"_": re.compile(r"[\[\(]"), "": re.compile(r"[\]\)]")}
        for items in [self.metabolites, self.reactions]:
            for item in items:
                for rep in replacements:
                    item.name = re.sub(replacements[rep], rep, item.name)

        self.LC_vars = {}
        self.P_vars = {}
        self.DGoF_vars = {}

        for met in self.metabolites:
            self._convert_metabolite(met, add_potentials, verbose)

        ## For each reaction...
        for rxn in self.reactions:
            self._convert_reaction(rxn, add_potentials, add_displacement, verbose)

        # CONSISTENCY CHECKS

        # Creating the objective
        if len(self.objective.variables) == 0:
            self.logger.warning("Objective not found")

        self.logger.info("# Model conversion done.")
        self.logger.info("# Updating cobra_model variables...")
        self.repair()
        self.logger.info("# cobra_model variables are up-to-date")

    def print_info(self, specific=False):
        """
        Print information and counts for the cobra_model
        :return:
        """
        if not specific:
            LCSBModel.print_info(self)

        n_metabolites = len(self.metabolites)
        n_reactions = len(self.reactions)
        n_metabolites_thermo = len(
            [x for x in self.metabolites if hasattr(x, "thermo") and x.thermo["id"]]
        )
        n_reactions_thermo = len(
            [
                x
                for x in self.reactions
                if x.id is not None and hasattr(x, "thermo") and x.thermo["computed"]
            ]
        )

        info = pd.DataFrame(columns=["value"])
        info.loc["num metabolites(thermo)"] = n_metabolites_thermo
        info.loc["num reactions(thermo)"] = n_reactions_thermo
        info.loc["pct metabolites(thermo)"] = n_metabolites_thermo / n_metabolites * 100
        info.loc["pct reactions(thermo)"] = n_reactions_thermo / n_reactions * 100
        info.index.name = "key"

        print(info)

    def __deepcopy__(self, memo):
        """

        :param memo:
        :return:
        """

        return self.copy()

    def copy(self):

        "previous"

        # dictmodel = model_to_dict(self)
        # new = model_from_dict(dictmodel)
        dictmodel = model_to_dict(self)
        new = model_from_dict(dictmodel)

        copy_solver_configuration(self, new)

        return new
