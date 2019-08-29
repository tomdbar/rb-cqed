from rb_cqed.globals import i, d, R2args, Singleton
from rb_cqed.runner_objs import Cavity, CavityBiref
import numpy as np
np.set_printoptions(threshold=np.inf)

from abc import ABC, abstractmethod
from itertools import product

import qutip as qt

'''
This is just some notes on the below.  Essentially I want to minimise re-computation of the operators I track through
the simulations (practically these are lists of matrices at each time step).  To do this I define a class that takes
the base experimental set-up (an Atom instance, a Cavity instance, and the ketbra dictionary already computed
to set up the origional simulation) and return the list of operator matrices at every time-step:

    _EmissionOperators: returns the operators for the total photon emission from the cavity.
    _NumberOperators: returns the operators for the total photon number inside the cavity.

These classes keep a record of every set of operators they calculate (uniquely defined for a given experimental setup by
the time series, t_series, and the basis in which we are looking, given by R_ZL) and returns the pre-computed operators
if an equivilent set already exists. 

To keep track of different possible experimental setups (i.e. different cavity/atom instances), we generate the 
_xxxOperators instances through a (singleton) factory.  This will return the _xxxOperators instance that already exists
if a suitable one is found, otherwise it creates a new _xxxOperators instance and adds it to its list. 
'''

class EmissionOperatorsFactory(metaclass=Singleton):

    emission_operators = []

    @classmethod
    def get(cls, atom, cavity, ketbras, verbose):
        for em_op in cls.emission_operators:
            if em_op._is_compatible(atom, cavity):
                if verbose: print("\n\tFound suitable _EmissionOperators obj for setup.", end='')
                return em_op
        else:
            if type(cavity)==Cavity:
                em_op = cls._EmissionOperatorsCavitySingle(atom,cavity,ketbras,verbose)
            elif type(cavity)==CavityBiref:
                em_op = cls._EmissionOperatorsCavityBiref(atom,cavity,ketbras,verbose)
            else:
                raise Exception('Unrecognised cavity type:', type(cavity))
            cls.emission_operators.append(em_op)
            return em_op

    class _EmissionOperators(ABC):

        def __init__(self, atom, cavity, ketbras, verbose):
            if verbose: print("\n\tCreating new _EmissionOperators obj for setup.", end='')

            self.atom = atom
            self.cavity = cavity
            self.ketbras = ketbras
            self.verbose=verbose

        @abstractmethod
        def get(self):
            raise NotImplementedError()

        def _is_compatible(self, atom, cavity):
            if all([self.atom==atom,self.cavity==cavity]):
                return True
            else:
                return False

    class _EmissionOperatorsCavitySingle(_EmissionOperators):

        def __init__(self, *args):
            super().__init__(*args)

            self.a = qt.tensor(qt.qeye(self.atom.M), qt.destroy(self.cavity.N))
            self.an = self.a.dag() * self.a
            self.em = 2*self.cavity.kappa*self.an

        def get(self, t_series=None):
            return self.em if not t_series else [self.em]*len(t_series)

    class _EmissionOperatorsCavityBiref(_EmissionOperators):

        def __init__(self, *args):
            super().__init__(*args)

            self.operator_series = []

            def kb(a, b):
                return self.ketbras[str([a, b])]

            all_atom_states = list(self.atom.configured_states)

            self.em_fast_1 = sum(map(lambda s: kb([s, 1, 0], [s, 1, 0]) + kb([s, 1, 1], [s, 1, 1]), all_atom_states))
            self.em_fast_2 = sum(map(lambda s: kb([s, 0, 1], [s, 0, 1]) + kb([s, 1, 1], [s, 1, 1]), all_atom_states))
            self.em_fast_3 = sum(map(lambda s: kb([s, 0, 1], [s, 0, 1]) - kb([s, 1, 0], [s, 1, 0]), all_atom_states))
            self.em_fast_4 = sum(map(lambda s: kb([s, 0, 1], [s, 1, 0]), all_atom_states))
            self.em_fast_5 = sum(map(lambda s: kb([s, 1, 0], [s, 0, 1]), all_atom_states))

        def get(self, t_series, R_ZL):
            for t, R, kappa1, kappa2, deltaP, op_series in self.operator_series:
                if all([np.array_equal(t, t_series), np.array_equal(R, R_ZL)]):
                    if self.verbose: print("\n\tFound suitable pre-computed emission operator series.", end='')
                    return op_series
            return self.__generate(t_series, R_ZL)

        def __generate(self, t_series, R_ZL):
            if self.verbose: print("\n\tCreating new number operator series.", end='')
            R_ZM = self.cavity.R_ML.getH() * R_ZL

            alpha_ZM, beta_ZM, phi1_ZM, phi2_ZM = R2args(R_ZM)
            R_MC = self.cavity.R_CL.getH() * self.cavity.R_ML
            alpha_MC, beta_MC, phi1_MC, phi2_MC = R2args(R_MC)

            kappa1, kappa2, deltaP = self.cavity.kappa1, self.cavity.kappa2, self.cavity.deltaP

            emArot1 = 2 * (alpha_MC ** 2 * alpha_ZM ** 2 * kappa1 +
                           beta_MC ** 2 * beta_ZM ** 2 * kappa2) * self.em_fast_1 + \
                      2 * (beta_MC ** 2 * alpha_ZM ** 2 * kappa1 +
                           alpha_MC ** 2 * beta_ZM ** 2 * kappa2) * self.em_fast_2 + \
                      4 * (alpha_MC * alpha_ZM * beta_MC * beta_ZM * kappa1 ** 0.5 * kappa2 ** 0.5) * \
                      np.cos(phi1_MC + phi2_MC + phi1_ZM - phi2_ZM) * self.em_fast_3

            emArot2 = 2 * (alpha_MC ** 2 * beta_ZM ** 2 * kappa1 +
                           beta_MC ** 2 * alpha_ZM ** 2 * kappa2) * self.em_fast_1 + \
                      2 * (beta_MC ** 2 * beta_ZM ** 2 * kappa1 +
                           alpha_MC ** 2 * alpha_ZM ** 2 * kappa2) * self.em_fast_2 - \
                      4 * (alpha_MC * alpha_ZM * beta_MC * beta_ZM * kappa1 ** 0.5 * kappa2 ** 0.5) * \
                      np.cos(phi1_MC + phi2_MC + phi1_ZM - phi2_ZM) * self.em_fast_3

            emBsrot1 = [
                2 * np.exp(-i * deltaP * t) * np.exp(-i * (2 * phi1_MC + phi1_ZM + phi2_ZM)) * \
                (
                        np.exp(2 * i * phi2_ZM) * alpha_MC ** 2 * alpha_ZM * beta_ZM * kappa1 ** 0.5 * kappa2 ** 0.5 -
                        np.exp(2 * i * (
                                    phi1_MC + phi2_MC + phi1_ZM)) * alpha_ZM * beta_MC ** 2 * beta_ZM * kappa1 ** 0.5 * kappa2 ** 0.5 +
                        np.exp(i * (phi1_MC + phi2_MC + phi1_ZM + phi2_ZM)) * alpha_MC * beta_MC * (
                                    alpha_ZM ** 2 * kappa1 -
                                    beta_ZM ** 2 * kappa2)
                ) * self.em_fast_4
                for t in t_series]

            emBsrot2 = [
                -2 * np.exp(-i * deltaP * t) * np.exp(-i * (2 * phi1_MC + phi1_ZM + phi2_ZM)) * \
                (
                        np.exp(2 * i * phi2_ZM) * alpha_MC ** 2 * alpha_ZM * beta_ZM * kappa1 ** 0.5 * kappa2 ** 0.5 -
                        np.exp(2 * i * (
                                    phi1_MC + phi2_MC + phi1_ZM)) * alpha_ZM * beta_MC ** 2 * beta_ZM * kappa1 ** 0.5 * kappa2 ** 0.5 -
                        np.exp(i * (phi1_MC + phi2_MC + phi1_ZM + phi2_ZM)) * alpha_MC * beta_MC * (
                                    beta_ZM ** 2 * kappa1 -
                                    alpha_ZM ** 2 * kappa2)
                ) * self.em_fast_4
                for t in t_series]

            emCsrot1 = [
                -2 * np.exp(i * deltaP * t) * np.exp(-i * (2 * phi2_MC + phi1_ZM + phi2_ZM)) * \
                (
                        -np.exp(2 * i * (
                                    phi1_MC + phi2_MC + phi1_ZM)) * alpha_MC ** 2 * alpha_ZM * beta_ZM * kappa1 ** 0.5 * kappa2 ** 0.5 +
                        np.exp(2 * i * phi2_ZM) * beta_MC ** 2 * alpha_ZM * beta_ZM * kappa1 ** 0.5 * kappa2 ** 0.5 -
                        np.exp(i * (phi1_MC + phi2_MC + phi1_ZM + phi2_ZM)) * alpha_MC * beta_MC * (
                                    alpha_ZM ** 2 * kappa1 -
                                    beta_ZM ** 2 * kappa2)
                ) * self.em_fast_5
                for t in t_series]

            emCsrot2 = [
                2 * np.exp(i * deltaP * t) * np.exp(-i * (2 * phi2_MC + phi1_ZM + phi2_ZM)) * \
                (
                        -np.exp(2 * i * (
                                    phi1_MC + phi2_MC + phi1_ZM)) * alpha_MC ** 2 * alpha_ZM * beta_ZM * kappa1 ** 0.5 * kappa2 ** 0.5 +
                        np.exp(2 * i * phi2_ZM) * beta_MC ** 2 * alpha_ZM * beta_ZM * kappa1 ** 0.5 * kappa2 ** 0.5 +
                        np.exp(i * (phi1_MC + phi2_MC + phi1_ZM + phi2_ZM)) * alpha_MC * beta_MC * (
                                    beta_ZM ** 2 * kappa1 -
                                    alpha_ZM ** 2 * kappa2)
                ) * self.em_fast_5
                for t in t_series]

            emRot1s = [emArot1 + emBrot1 + emCrot1 for emBrot1, emCrot1 in zip(emBsrot1, emCsrot1)]
            emRot2s = [emArot2 + emBrot2 + emCrot2 for emBrot2, emCrot2 in zip(emBsrot2, emCsrot2)]

            self.operator_series.append( (t_series, R_ZL, kappa1, kappa2, deltaP, (emRot1s, emRot2s)) )

            return emRot1s, emRot2s

class NumberOperatorsFactory(metaclass=Singleton):

    number_operators = []

    @classmethod
    def get(cls, atom, cavity, ketbras, verbose):
        for an_op in cls.number_operators:
            if an_op._is_compatible(atom, cavity):
                if verbose: print("\n\tFound suitable _NumberOperators obj for setup.", end='')
                return an_op
        else:
            if type(cavity)==Cavity:
                an_op = cls._NumberOperatorsCavitySingle(atom,cavity,ketbras,verbose)
            elif type(cavity)==CavityBiref:
                an_op = cls._NumberOperatorsCavityBiref(atom,cavity,ketbras,verbose)
            else:
                raise Exception('Unrecognised cavity type:', type(cavity))
            cls.number_operators.append(an_op)
            return an_op

    class _NumberOperators(ABC):

        def __init__(self, atom, cavity, ketbras, verbose):

            if verbose: print("\n\tCreating new _NumberOperators obj for setup.", end='')

            self.atom = atom
            self.cavity = cavity
            self.ketbras = ketbras
            self.verbose = verbose

        @abstractmethod
        def get(self):
            raise NotImplementedError()

        def _is_compatible(self, atom, cavity):
            if all([self.atom == atom, self.cavity == cavity]):
                return True
            else:
                return False

    class _NumberOperatorsCavitySingle(_NumberOperators):

        def __init__(self, *args):
            super().__init__(*args)

            self.a = qt.tensor(qt.qeye(self.atom.M), qt.destroy(self.cavity.N))
            self.an = self.a.dag() * self.a

        def get(self, t_series=None):
            return self.an if not t_series else [self.an]*len(t_series)

    class _NumberOperatorsCavityBiref(_NumberOperators):

        def __init__(self, *args):
            super().__init__(*args)

            self.operator_series = []

            def kb(a, b):
                return self.ketbras[str([a, b])]

            all_atom_states = list(self.atom.configured_states)

            self.an_fast_1 = sum(map(lambda s: kb([s, 1, 0], [s, 1, 0]) + kb([s, 1, 1], [s, 1, 1]), all_atom_states))
            self.an_fast_2 = sum(map(lambda s: kb([s, 0, 1], [s, 0, 1]) + kb([s, 1, 1], [s, 1, 1]), all_atom_states))
            self.an_fast_3 = sum(map(lambda s: kb([s, 0, 1], [s, 1, 0]), all_atom_states))
            self.an_fast_4 = sum(map(lambda s: kb([s, 1, 0], [s, 0, 1]), all_atom_states))

        def get(self, t_series, R_ZL):
            for t, R, deltaP, op_series in self.operator_series:
                if all([np.array_equal(t,t_series),
                        np.array_equal(R, R_ZL),
                        deltaP==self.cavity.deltaP]):
                    if self.verbose: print("\n\tFound suitable pre-computed number operator series.", end='')
                    return op_series
            return self.__generate(t_series, R_ZL)

        def __generate(self, t_series, R_ZL):
            if self.verbose: print("\n\tCreating new number operator series.", end='')
            R_ZC = self.cavity.R_CL.getH() * R_ZL

            alpha_ZC, beta_ZC, phi1_ZC, phi2_ZC = R2args(R_ZC)
            delta_phi = phi2_ZC - phi1_ZC
            deltaP = self.cavity.deltaP

            an0P = (alpha_ZC ** 2 * self.an_fast_1 + beta_ZC ** 2 * self.an_fast_2)
            an0M = (alpha_ZC ** 2 * self.an_fast_2 + beta_ZC ** 2 * self.an_fast_1)

            an1s = [alpha_ZC * beta_ZC * (
                    np.exp(-i * deltaP * t) * np.exp(i * delta_phi) * self.an_fast_3 + \
                    np.exp(i * deltaP * t) * np.exp(-i * delta_phi) * self.an_fast_4
            )
                    for t in t_series]

            anRots = [[an0P + an1, an0M - an1] for an1 in an1s]
            anRots = [list(i) for i in zip(*anRots)]

            self.operator_series.append( (t_series, R_ZL, deltaP, anRots) )

            return anRots

class AtomicOperatorsFactory(metaclass=Singleton):

    atomic_operators = []

    @classmethod
    def get(cls, atom, cavity, ketbras, verbose):
        for at_op in cls.atomic_operators:
            if at_op._is_compatible(atom):
                if (type(cavity)==Cavity and type(at_op)==cls._AtomicOperatorsCavitySingle) or \
                   (type(cavity)==CavityBiref and type(at_op)==cls._AtomicOperatorsCavityBiref):
                    if verbose: print("\n\tFound suitable _AtomicOperators obj for setup.", end='')
                    return at_op
        else:
            if type(cavity)==Cavity:
                at_op = cls._AtomicOperatorsCavitySingle(atom, ketbras, verbose)
            elif type(cavity)==CavityBiref:
                at_op = cls._AtomicOperatorsCavityBiref(atom, ketbras, verbose)
            else:
                raise Exception('Unrecognised cavity type:', type(cavity))
            cls.atomic_operators.append(at_op)
            return at_op

    class _AtomicOperators():

        def __init__(self, atom, ketbras, verbose):
            if verbose: print("\n\tCreating new _AtomicOperators obj for setup.", end='')

            self.atom = atom
            self.ketbras = ketbras
            self.verbose = verbose

        def get_at_op(self, states=[]):
            if type(states) != list:
                states = [states]
            if not states:
                return list(self.at_ops.values())
            else:
                try:
                    return [self.at_ops[s] for s in states]
                except KeyError:
                    raise KeyError('Invalid atomic state entered.  Valid options are ', list(self.at_ops))

        def get_sp_op(self):
            return self.sp_op

        def _is_compatible(self, atom):
            if self.atom == atom:
                return True
            else:
                return False

    class _AtomicOperatorsCavitySingle(_AtomicOperators):

        def __init__(self, *args):
            super().__init__(*args)

            def kb(a, b):
                return self.ketbras[str([a, b])]

            self.at_ops = {}
            for s in self.atom.configured_states:
                self.at_ops[s]= kb([s,0], [s,0]) + kb([s,1], [s,1])

            spont_decay_ops = []

            # for g,x,branching_ratio in self.atom.get_spontaneous_emission_channels():
            for g,x,r in self.atom.get_spontaneous_emission_channels():
                try:
                    # spont_decay_ops.append(branching_ratio * np.sqrt(2 * self.atom.gamma) *
                    spont_decay_ops.append(np.sqrt(r * 2 * self.atom.gamma) *
                                           qt.tensor(
                                             qt.basis(self.atom.M, self.atom.get_state_id(g)) * qt.basis(self.atom.M, self.atom.get_state_id(x)).dag(),
                                             qt.qeye(Cavity.N)))
                except KeyError:
                    pass

            self.sp_op = sum([x.dag() * x for x in spont_decay_ops])

    class _AtomicOperatorsCavityBiref(_AtomicOperators):

        def __init__(self, *args):
            super().__init__(*args)

            self.operator_series = []

            def kb(a, b):
                return self.ketbras[str([a, b])]

            self.at_ops = {}
            for s in self.atom.configured_states:
                self.at_ops[s]= kb([s,0,0],[s,0,0]) + kb([s,1,0],[s,1,0]) + kb([s,0,1],[s,0,1]) + kb([s,1,1],[s,1,1])

            spont_decay_ops = []

            # for g,x,branching_ratio in self.atom.get_spontaneous_emission_channels():
            for g,x,r in self.atom.get_spontaneous_emission_channels():
                try:
                    # spont_decay_ops.append(branching_ratio * np.sqrt(2 * self.atom.gamma) *
                    spont_decay_ops.append(np.sqrt(r * 2 * self.atom.gamma) *
                                           qt.tensor(
                                             qt.basis(self.atom.M, self.atom.get_state_id(g)) * qt.basis(self.atom.M, self.atom.get_state_id(x)).dag(),
                                             qt.qeye(Cavity.N),
                                             qt.qeye(Cavity.N)))
                except KeyError:
                    pass

            self.sp_op = sum([x.dag() * x for x in spont_decay_ops])

class StatesFactory(metaclass=Singleton):

    states = []
    #todo account for reconfigurable decays in atom==atom, cavity==cavity
    @classmethod
    def get(cls, atom, cavity, verbose=False):
        for s in cls.states:
            if s._is_compatible(atom, cavity):
                if verbose: print("\n\tFound suitable _States obj for setup.", end='')
                return s
        else:
            if type(cavity)==Cavity:
                s = cls._StatesCavitySingle(atom, cavity)
            elif type(cavity)==CavityBiref:
                s = cls._StatesCavityBiref(atom, cavity)
            else:
                raise Exception('Unrecognised cavity type:', type(cavity))
            cls.states.append(s)
            return s

    class _States(ABC):

        def __init__(self, atom, cavity):
            self.atom = atom
            self.cavity = cavity

            self.kets = {}
            self.bras = {}
            self.ketbras = {}

            states = self._get_states_list()

            for state in states:
                self.kets[str(state)] = self.ket(*state)
                self.bras[str(state)] = self.bra(*state)

            for x in list(map(list, list(product(*[states, states])))):
                self.ketbras[str(x)] = self.ket(*x[0]) * self.bra(*x[1])

        @abstractmethod
        def ket(self, *args):
            raise NotImplementedError()

        @abstractmethod
        def bra(self, *args):
            raise NotImplementedError()

        @abstractmethod
        def _get_states_list(self):
            raise NotImplementedError

        def _is_compatible(self, atom, cavity):
            if (self.atom == atom) and (self.cavity == cavity):
                return True
            else:
                return False

    class _StatesCavitySingle(_States):

        def ket(self, atom_state, cav):
            try:
                ket = self.kets[str([atom_state, cav])]
            except KeyError:
                ket = qt.tensor(qt.basis(self.atom.M, self.atom.get_state_id(atom_state)),
                             qt.basis(self.cavity.N, cav))
                self.kets[str([atom_state, cav])] = ket

            return ket

        def bra(self, atom_state, cav):
            try:
                bra = self.bras[str([atom_state, cav])]
            except KeyError:
                bra = qt.tensor(qt.basis(self.atom.M, self.atom.get_state_id(atom_state)),
                             qt.basis(self.cavity.N, cav)).dag()
                self.bras[str([atom_state, cav])] = bra

            return bra

        def _get_states_list(self):
            s = [list(self.atom.configured_states), self.cavity.cavity_states]
            return list(map(list, list(product(*s))))

    class _StatesCavityBiref(_States):

        def __init__(self, *args):
            super().__init__(*args)

        def ket(self, atom_state, cav_X, cav_Y):
            try:
                ket = self.kets[str([atom_state, cav_X, cav_Y])]
            except KeyError:
                ket = qt.tensor(qt.basis(self.atom.M, self.atom.get_state_id(atom_state)),
                             qt.basis(self.cavity.N, cav_X),
                             qt.basis(self.cavity.N, cav_Y))
                self.kets[str([atom_state, cav_X, cav_Y])] = ket

            return ket

        def bra(self, atom_state, cav_X, cav_Y):
            try:
                bra = self.bras[str([atom_state, cav_X, cav_Y])]
            except KeyError:
                bra = qt.tensor(qt.basis(self.atom.M, self.atom.get_state_id(atom_state)),
                             qt.basis(self.cavity.N, cav_X),
                             qt.basis(self.cavity.N, cav_Y)).dag()
                self.bras[str([atom_state, cav_X, cav_Y])] = bra

            return bra

        def _get_states_list(self):
            s = [list(self.atom.configured_states), self.cavity.cavity_states, self.cavity.cavity_states]
            return list(map(list, list(product(*s))))