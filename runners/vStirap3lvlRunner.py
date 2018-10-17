import numpy as np
import matplotlib.pyplot as plt
import copy
from shutil import copyfile
import fileinput
import time
from itertools import chain, product
from dataclasses import dataclass, field, Field, asdict
from typing import Callable
from qutip import *

import os
import sys

import io
from contextlib import redirect_stdout, redirect_stderr

np.set_printoptions(threshold=np.inf)

# Global parameters
d = 3.584*10**(-29)
i = np.complex(0,1)
compiled_hamiltonians = []

def R2args(R):
    alpha = np.clip(np.abs(R[0, 0]), 0, 1)
    phi1, phi2 = np.angle(R[0, 0]), np.angle(R[1, 0])
    beta = np.sqrt(1 - alpha ** 2)
    return alpha, beta, phi1, phi2

@dataclass(eq=False)
class RunnerDataClass:

    def _eq_ignore_fields(self):
        return []

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            def remove_ignore_fields(d):
                for k in self._eq_ignore_fields():
                    d.pop(k, None)
                return d

            d1, d2 = map(lambda x: remove_ignore_fields(asdict(x)), [self, other])

            def eq(x,y):
                if all(map(lambda z: type(z) in [np.matrix, np.array], [x,y])):
                    return np.array_equal(x,y)
                else:
                    return x==y

            return all([eq(x,y) for x,y in zip(d1.values(), d2.values())])

        else:
            return False

@dataclass(eq=False)
class Atom(RunnerDataClass):
    atom_states: dict = field(default_factory=dict)
    M: int = 4
    gamma: float = 3 * 2. * np.pi
    branching_ratios: list = field(default_factory=list)
    R_AL: np.matrix = np.sqrt(1 / 2) * np.matrix([[1, i],
                                                  [i, 1]])
    def __post_init__(self):
        if self.atom_states == {}:
            self.atom_states = {"x0": 0, "gM": 1, "gP": 2, "d": 3}
        self.M = len(self.atom_states)
        if self.branching_ratios == []:
            self.branching_ratios = [0.,0.,1.]

    def _eq_ignore_fields(self):
        return []

    @classmethod
    def get_couplings_sigma_plus(cls, delta) -> list:
        return [
            ('gM', 'x0', delta, 1)
        ]

    @classmethod
    def get_couplings_sigma_minus(cls, delta) -> list:
        return [
            ('gP', 'x0', delta, -1)
        ]

    def get_spontaneous_emission_channels(self):
        return  [
            # |F',mF'> --> |F=1,mF=-1>
            ('gM', 'x0', self.branching_ratios[0]),
            ('gP', 'x0', self.branching_ratios[1]),
            ('d', 'x0', self.branching_ratios[2])
        ]

    def check_coupling(self,g,x):
        if not g in self.atom_states or not x in self.atom_states:
            raise Exception("Invalid atom state entered.\nConfigured states are",
                            print(self.atom_states.values()))

@dataclass(eq=False)
class Cavity(RunnerDataClass):
    N: int = 2
    cavity_states: list = field(default_factory=list)
    g: float = 3 * 2. * np.pi
    kappa1: float = 3 * 2. * np.pi
    kappa2: float = 3 * 2. * np.pi
    deltaP: float = 0 * 2. * np.pi
    R_CL: np.matrix = np.matrix([[1, 0],
                                 [0, 1]])
    R_ML: np.matrix = np.sqrt(1 / 2) * np.matrix([[1, i],
                                                  [i, 1]])

    def __post_init__(self):
        self.cavity_states += [0,1]
        self.N = len(self.cavity_states)

    def _eq_ignore_fields(self):
        return ['g']

@dataclass(eq=False)
class LaserCoupling(RunnerDataClass):
    omega0: float
    g: str
    x: str
    deltaL: float
    args_ham: dict
    pulse_shape: str = 'np.piecewise(t, [t<length_pulse], [np.sin((np.pi/length_pulse)*t)**2,0])'
    setup_pyx: list = field(default_factory=list)
    add_pyx: list = field(default_factory=list)

    def _eq_ignore_fields(self):
        return ['omega0', 'deltaL', 'args_ham']

@dataclass(eq=False)
class CavityCoupling(RunnerDataClass):
    g0: float
    g: str
    x: str
    deltaC: float
    deltaM: int

    def _eq_ignore_fields(self):
        return ['g0','deltaC']

@dataclass
class CompiledHamiltonian:
    name: str
    hams: list
    atom: Atom
    cavity: Cavity
    laser_couplings: list
    cavity_couplings: list
    c_ops: list
    tdname: str
    tdfunc: Callable

    def can_use(self, atom, cavity, laser_couplings, cavity_couplings):
        can_use = True
        if self.atom != atom:
            can_use = False
        if self.cavity != cavity:
            can_use = False
        for x,y in list(zip(self.laser_couplings, laser_couplings)) + \
                   list(zip(self.cavity_couplings, cavity_couplings)):
            if x != y:
                can_use = False
        return can_use

    def can_modify(self, atom, cavity, laser_couplings, cavity_couplings):
        can_modify = True
        if self.atom != atom:
            can_modify = False
        if self.cavity != cavity:
            can_modify = False
        for x, y in list(zip(self.cavity_couplings, cavity_couplings)):
            if x != y:
                can_modify = False
        for x, y in list(zip(self.laser_couplings, laser_couplings)):
            cx,cy = copy.copy(x),  copy.copy(y)
            cx.pulse_shape = ''; cy.pulse_shape = ''
            if cx != cy:
                # If the only difference between the laser couplings is the pulse shape, we can
                # modify the compiled .pyx file to fix this, without re-compiling it.
                can_modify = False
        return can_modify

class ExperimentalRunner():

    def __init__(self,
                 atom=Atom(),
                 cavity=Cavity(),
                 laser_couplings = [],
                 cavity_couplings = [],
                 verbose = False):
        self.atom = atom
        self.cavity = cavity
        self.laser_couplings = laser_couplings if type(laser_couplings)==list else [laser_couplings]
        self.cavity_couplings = cavity_couplings if type(cavity_couplings)==list else [cavity_couplings]
        self.verbose = verbose

        self.hams = []
        self.c_op_list = []
        self.args_hams = dict([('i', i)])

        self.ketbras = self.__configure_ketbras()

        self.compiled_hamiltonian = self.__generate_hamiltonian()

    def __ket(self, atom_state, cav_X, cav_Y):
        return tensor(basis(self.atom.M, self.atom.atom_states[atom_state]),
                      basis(self.cavity.N, cav_X),
                      basis(self.cavity.N, cav_Y))

    def __bra(self, atom_state, cav_X, cav_Y):
        return self.__ket(atom_state, cav_X, cav_Y).dag()

    def __configure_ketbras(self):

        kets, bras = {}, {}
        ketbras = {}

        s = [list(self.atom.atom_states), self.cavity.cavity_states, self.cavity.cavity_states]
        states = list(map(list, list(product(*s))))
        for state in states:
            kets[str(state)] = self.__ket(*state)
            bras[str(state)] = self.__bra(*state)

        for x in list(map(list, list(product(*[states, states])))):
            ketbras[str(x)] = self.__ket(*x[0]) * self.__bra(*x[1])

        return ketbras

    def __generate_hamiltonian(self):

        for ham in compiled_hamiltonians:
            if ham.can_use(self.atom, self.cavity, self.laser_couplings, self.cavity_couplings):
                if self.verbose:
                    print("Pre-compiled Hamiltonian, {0}.pyx, is suitable to run this experiment.".format(ham.name))
                self.__configure_c_ops(args_only=True)
                self.__configure_laser_couplings(args_only=True)
                self.__configure_cavity_couplings(args_only=True)
                ham.atom=self.atom
                ham.cavity=self.cavity
                ham.laser_couplings = self.laser_couplings
                ham.cavity_couplings = self.cavity_couplings
                self.hams = ham.hams
                self.c_op_list = ham.c_ops
                return ham

        # for ham in compiled_hamiltonians:
        #     if ham.can_modify(self.atom, self.cavity, self.laser_couplings, self.cavity_couplings):
        #         if self.verbose:
        #             print("Pre-compiled Hamiltonian, {0}.pyx, can be modified to run this experiment.".
        #                   format(ham.name))
        #         ham_new = self.__copy_and_modify_hamiltonian(ham)
        #         self.hams = ham_new.hams
        #         self.c_op_list = ham_new.c_ops
        #         self.__configure_c_ops(args_only=True)
        #         self.__configure_laser_couplings(args_only=True)
        #         self.__configure_cavity_couplings(args_only=True)
        #         compiled_hamiltonians.append(ham_new)
        #         if self.verbose:
        #             print("\t{0}.pyx is a modified copy ready to run this experiment.".format(ham_new.name))
        #         return ham_new

        if self.verbose:
            print("No suitable pre-compiled Hamiltonian found.  Generating Cython file...", end='')
            t_start = time.time()

        self.__configure_c_ops()
        self.__configure_laser_couplings()
        self.__configure_cavity_couplings()

        name = 'ExperimentalRunner_Hamiltonian_{0}'.format(len(compiled_hamiltonians))

        self.hams = list(chain(*self.hams))

        try:
            with io.StringIO() as buf, redirect_stderr(buf):
                rhs_generate(self.hams, self.c_op_list, args=self.args_hams, name=name, cleanup=False)
                # print('buf:\n', buf.getvalue())
        except:
            if self.verbose:
                print("\n\tException in rhs comp...adding additional setups...", end='')
            for laser_couping in self.laser_couplings:
                if laser_couping.setup_pyx != [] or laser_couping.add_pyx != []:
                    with fileinput.FileInput(name + '.pyx', inplace=True) as file:
                        toWrite_setup = True
                        toWrite_add = True
                        for line in file:
                            if '#' not in line and toWrite_setup:
                                for input in laser_couping.setup_pyx:
                                    print(input, end='\n')
                                toWrite_setup = False
                            if '@cython.cdivision(True)' in line and toWrite_add:
                                for input in laser_couping.add_pyx:
                                    print(input, end='\n')
                                toWrite_add = False
                            print(line, end='')
                        fileinput.close()
            if self.verbose:
                print("and trying rhs generate again...", end='')
            code = compile('from ' + name + ' import cy_td_ode_rhs', '<string>', 'exec')
            exec(code, globals())
            solver.config.tdfunc = cy_td_ode_rhs

        compiled_hamiltonian = CompiledHamiltonian(name=name,
                                                   hams=self.hams,
                                                   atom=self.atom,
                                                   cavity=self.cavity,
                                                   laser_couplings=self.laser_couplings,
                                                   cavity_couplings=self.cavity_couplings,
                                                   c_ops=self.c_op_list,
                                                   tdname=solver.config.tdname,
                                                   tdfunc=solver.config.tdfunc)

        compiled_hamiltonians.append(compiled_hamiltonian)

        if self.verbose:
            print("done.\n\tNew file is {0}.pyx.  Generated in {1} seconds.".format(name,
                                                                                    np.round(time.time()-t_start,3)))

        return compiled_hamiltonian

    def __configure_c_ops(self, args_only=False):

        self.args_hams.update({"deltaP": self.cavity.deltaP})

        if not args_only:
            # Define collapse operators
            c_op_list = []

            R_MC = self.cavity.R_CL.getH() * self.cavity.R_ML
            alpha_MC, beta_MC, phi1_MC, phi2_MC = R2args(R_MC)

            aX = tensor(qeye(self.atom.M), destroy(self.cavity.N), qeye(self.cavity.N))
            aY = tensor(qeye(self.atom.M), qeye(self.cavity.N), destroy(self.cavity.N))

            aM1X = np.conj(np.exp(i * phi1_MC) * alpha_MC) * aX
            aM1Y = np.conj(np.exp(i * phi2_MC) * beta_MC) * aY
            aM2X = np.conj(-np.exp(-i * phi2_MC) * beta_MC) * aX
            aM2Y = np.conj(np.exp(-i * phi1_MC) * alpha_MC) * aY

            '''
            Deprecated TDB 16-08-18: Group collapse terms into fewest operators for speed.
            '''
            # for kappa, aMX, aMY in zip([self.cavity.kappa1, self.cavity.kappa2], [aM1X, aM2X], [aM1Y, aM2Y]):
            #     c_op_list.append(2 * kappa * lindblad_dissipator(aMX))
            #     c_op_list.append(2 * kappa * lindblad_dissipator(aMY))
            #     c_op_list.append([2 * kappa * (sprepost(aMY, aMX.dag())
            #                                    - 0.5 * spost(aMX.dag() * aMY)
            #                                    - 0.5 * spre(aMX.dag() * aMY)),
            #                       'exp(i*deltaP*t)'])
            #     c_op_list.append([2 * kappa * (sprepost(aMX, aMY.dag())
            #                                    - 0.5 * spost(aMY.dag() * aMX)
            #                                    - 0.5 * spre(aMY.dag() * aMX)),
            #                       'exp(-i*deltaP*t)'])

            c_op_list.append(2 * self.cavity.kappa1 * lindblad_dissipator(aM1X) +
                             2 * self.cavity.kappa1 * lindblad_dissipator(aM1Y) +
                             2 * self.cavity.kappa2 * lindblad_dissipator(aM2X) +
                             2 * self.cavity.kappa2 * lindblad_dissipator(aM2Y))
            c_op_list.append([2 * self.cavity.kappa1 * (sprepost(aM1Y, aM1X.dag())
                                                        - 0.5 * spost(aM1X.dag() * aM1Y)
                                                        - 0.5 * spre(aM1X.dag() * aM1Y)) +
                              2 * self.cavity.kappa2 * (sprepost(aM2Y, aM2X.dag())
                                                        - 0.5 * spost(aM2X.dag() * aM2Y)
                                                        - 0.5 * spre(aM2X.dag() * aM2Y)),
                              'exp(i*deltaP*t)'])
            c_op_list.append([2 * self.cavity.kappa1 * (sprepost(aM1X, aM1Y.dag())
                                                        - 0.5 * spost(aM1Y.dag() * aM1X)
                                                        - 0.5 * spre(aM1Y.dag() * aM1X)) +
                              2 * self.cavity.kappa2 * (sprepost(aM2X, aM2Y.dag())
                                                        - 0.5 * spost(aM2Y.dag() * aM2X)
                                                        - 0.5 * spre(aM2Y.dag() * aM2X)),
                              'exp(-i*deltaP*t)'])

            spontEmmChannels = self.atom.get_spontaneous_emission_channels()

            spontDecayOps = []

            for x in spontEmmChannels:
                try:
                    spontDecayOps.append(x[2] * np.sqrt(2 * self.atom.gamma) *
                                         tensor(
                                             basis(self.atom.M, self.atom.atom_states[x[0]]) *
                                                basis(self.atom.M, self.atom.atom_states[x[1]]).dag(),
                                             qeye(self.cavity.N),
                                             qeye(self.cavity.N)))
                except KeyError:
                    pass

            c_op_list += spontDecayOps

            self.c_op_list = c_op_list

    def __configure_laser_couplings(self, args_only=False):

        for laser_coupling in self.laser_couplings:
            g, x = laser_coupling.g, laser_coupling.x
            self.atom.check_coupling(g, x)

            Omega = laser_coupling.omega0
            Omega_lab = 'Omega_{0}{1}'.format(g, x)
            omegaL = laser_coupling.deltaL
            omegaL_lab = 'omegaL_{0}{1}'.format(g, x)

            self.args_hams.update({Omega_lab: Omega,
                                   omegaL_lab: omegaL})
            self.args_hams.update(laser_coupling.args_ham)

            if not args_only:
                pulse_shape = laser_coupling.pulse_shape

                def kb(a, b):
                    return self.ketbras[str([a, b])]

                self.hams.append([
                    [-(1 / 2) * (
                            (kb([g, 0, 0], [x, 0, 0]) + kb([g, 0, 1], [x, 0, 1]) +
                             kb([g, 1, 0], [x, 1, 0]) + kb([g, 1, 1], [x, 1, 1])) +
                            (kb([x, 0, 0], [g, 0, 0]) + kb([x, 0, 1], [g, 0, 1]) +
                             kb([x, 1, 0], [g, 1, 0]) + kb([x, 1, 1], [g, 1, 1]))
                    ), '{0} * {1} * cos({2}*t)'.format(Omega_lab, pulse_shape, omegaL_lab)],
                    [i * (1 / 2) * (
                            (kb([x, 0, 0], [g, 0, 0]) + kb([x, 0, 1], [g, 0, 1]) +
                             kb([x, 1, 0], [g, 1, 0]) + kb([x, 1, 1], [g, 1, 1])) -
                            (kb([g, 0, 0], [x, 0, 0]) - kb([g, 0, 1], [x, 0, 1]) -
                             kb([g, 1, 0], [x, 1, 0]) - kb([g, 1, 1], [x, 1, 1]))
                    ), '{0} * {1} * sin({2}*t)'.format(Omega_lab, pulse_shape, omegaL_lab)]
                ])

    def __configure_cavity_couplings(self, args_only=False):

        # Rotation from Atom -> Cavity: R_LA: {|+>,|->} -> {|X>,|Y>}
        R_AC = self.cavity.R_CL.getH() * self.atom.R_AL
        alpha_AC, beta_AC, phi1_AC, phi2_AC = R2args(R_AC)

        self.args_hams.update({"alpha_AC": alpha_AC,
                               "beta_AC": beta_AC,
                               "phi1_AC": phi1_AC,
                               "phi2_AC": phi2_AC})

        def kb(a, b):
            return self.ketbras[str([a, b])]

        for cavity_coupling in self.cavity_couplings:
            g, x = cavity_coupling.g, cavity_coupling.x
            self.atom.check_coupling(g, x)

            g0 = cavity_coupling.g0
            g0_lab = 'g0_{0}{1}'.format(g, x)

            if g0 != self.cavity.g:
                print('\n\tWARNING: CavityCoupling does not have the same ' + \
                      'atom-cavity coupling rate (g0/2pi={0}MHz) as the ' + \
                      'configured cavity (g0/2pi={1}MHz) .  I hope you ' + \
                      'know what you are doing...'.format(
                    *[np.round(x/(2*np.pi)) for x in [g0,self.cavity.g]]
                ))

            omegaC = cavity_coupling.deltaC
            omegaC_X = omegaC + self.cavity.deltaP / 2
            omegaC_Y = omegaC - self.cavity.deltaP / 2
            omegaC_X_lab = 'omegaC_X_{0}{1}'.format(g, x)
            omegaC_Y_lab = 'omegaC_Y_{0}{1}'.format(g, x)
            self.args_hams.update({g0_lab: g0,
                                   omegaC_X_lab: omegaC_X,
                                   omegaC_Y_lab: omegaC_Y})

            if not args_only:
                deltaM = cavity_coupling.deltaM

                if deltaM == 1:
                    H_coupling = [
                        [-1 * alpha_AC * (
                                kb([g, 1, 0], [x, 0, 0]) + kb([g, 1, 1], [x, 0, 1]) +
                                kb([x, 0, 0], [g, 1, 0]) + kb([x, 0, 1], [g, 1, 1])
                        ), '{0} * cos({1}*t + phi1_AC)'.format(g0_lab, omegaC_X_lab)],

                        [-i * 1 * alpha_AC * (
                                kb([g, 1, 0], [x, 0, 0]) + kb([g, 1, 1], [x, 0, 1]) -
                                kb([x, 0, 0], [g, 1, 0]) - kb([x, 0, 1], [g, 1, 1])
                        ), '{0} * sin({1}*t + phi1_AC)'.format(g0_lab ,omegaC_X_lab)],

                        [-1 * beta_AC * (
                                kb([g, 0, 1], [x, 0, 0]) + kb([g, 1, 1], [x, 1, 0]) +
                                kb([x, 0, 0], [g, 0, 1]) + kb([x, 1, 0], [g, 1, 1])
                        ), '{0} * cos({1}*t + phi2_AC)'.format(g0_lab ,omegaC_Y_lab)],

                        [-i * 1 * beta_AC * (
                                kb([g, 0, 1], [x, 0, 0]) + kb([g, 1, 1], [x, 1, 0]) -
                                kb([x, 0, 0], [g, 0, 1]) - kb([x, 1, 0], [g, 1, 1])
                        ), '{0} * sin({1}*t + phi2_AC)'.format(g0_lab, omegaC_Y_lab)]
                    ]

                elif deltaM == -1:
                    H_coupling = [
                        [-1 *  alpha_AC * (
                                kb([g, 0, 1], [x, 0, 0]) + kb([g, 1, 1], [x, 1, 0]) +
                                kb([x, 0, 0], [g, 0, 1]) + kb([x, 1, 0], [g, 1, 1])
                        ), '{0} * cos({1}*t - phi1_AC)'.format(g0_lab, omegaC_Y_lab)],

                        [-i * 1 * alpha_AC * (
                                kb([g, 0, 1], [x, 0, 0]) + kb([g, 1, 1], [x, 1, 0]) -
                                kb([x, 0, 0], [g, 0, 1]) - kb([x, 1, 0], [g, 1, 1])
                        ), '{0} * sin({1}*t - phi1_AC)'.format(g0_lab, omegaC_Y_lab)],

                        [1 *  beta_AC * (
                                kb([g, 1, 0], [x, 0, 0]) + kb([g, 1, 1], [x, 0, 1]) +
                                kb([x, 0, 0], [g, 1, 0]) + kb([x, 0, 1], [g, 1, 1])
                        ), '{0} * cos({1}*t - phi2_AC)'.format(g0_lab, omegaC_X_lab)],

                        [i * 1 * beta_AC * (
                                kb([g, 1, 0], [x, 0, 0]) + kb([g, 1, 1], [x, 0, 1]) -
                                kb([x, 0, 0], [g, 1, 0]) - kb([x, 0, 1], [g, 1, 1])
                        ), '{0} * sin({1}*t - phi2_AC)'.format(g0_lab, omegaC_X_lab)]
                    ]

                else:
                    raise Exception("deltaM must be +/-1")

                self.hams.append(H_coupling)

    def __copy_and_modify_hamiltonian(self, ham):
        '''
        Copies and modifies a pre-compiled Hamiltonian to suit the current experimental set-up
        :param hamiltonian:
        :return:
        '''
        name = 'ExperimentalRunner_Hamiltonian_{0}'.format(len(compiled_hamiltonians))

        copyfile(ham.name + '.pyx', name +  '.pyx')

        old_pulses = [x.pulse_shape for x in ham.laser_couplings]
        new_pulses = [x.pulse_shape for x in self.laser_couplings]

        with fileinput.FileInput(name +  '.pyx', inplace=True) as file:
            for line in file:
                if 'spmvpy' in line:
                    for old, new in zip(old_pulses,new_pulses):
                        print(line.replace(old, new), end='')
                else:
                    print(line, end='')
        fileinput.close()

        hams_new = ham.hams
        for old_coupling, new_coupling in zip(ham.laser_couplings, self.laser_couplings):
            id = '{0}{1}'.format(new_coupling.g, new_coupling.x)
            old = old_coupling.pulse_shape
            new = new_coupling.pulse_shape
            hams_new = [[H, coeff.replace(old, new) if id and old in coeff else coeff]
                        for H, coeff in hams_new]

        if self.verbose:
            t_start = time.time()
            print("\tCompiling new rhs function.",end='...')
        code = compile('from ' + name + ' import cy_td_ode_rhs', '<string>', 'exec')
        exec(code, globals())
        solver.config.tdfunc = cy_td_ode_rhs
        if self.verbose:
            print("done in {0} seconds".format(np.round(time.time() - t_start, 3)))

        return CompiledHamiltonian(name=name,
                                  hams=hams_new,
                                  atom=ham.atom,
                                  cavity=ham.cavity,
                                  laser_couplings=self.laser_couplings,
                                  cavity_couplings=ham.cavity_couplings,
                                  c_ops=ham.c_ops,
                                  tdname=name,
                                  tdfunc=cy_td_ode_rhs)

    def run(self, psi0=['gM',0,0], t_length=1.2, n_steps=201):
        t, t_step = np.linspace(0, t_length, n_steps, retstep=True)

        # Clears the rhs memory, so that when we set rhs_reuse to true, it has nothing
        # and so uses our compiled hamiltonian.  We do this as setting rhs_reuse=True
        # prevents the .pyx files from being deleted after the first run.
        rhs_clear()
        opts = Options(rhs_reuse=True, rhs_filename=self.compiled_hamiltonian.name)

        psi0 = self.__ket(*psi0)

        if self.verbose:
            t_start = time.time()
            print("Running simulation with {0} timesteps".format(n_steps), end="...")
        solver.config.tdfunc = self.compiled_hamiltonian.tdfunc
        solver.config.tdname = self.compiled_hamiltonian.tdname
        output = mesolve(self.hams,
                         psi0,
                         t,
                         self.c_op_list,
                         [],
                         args=self.args_hams,
                         options=opts)
        if self.verbose:
            print("finished in {0} seconds".format(np.round(time.time()-t_start,3)))

        return ExperimentalResults(output,
                                   self.compiled_hamiltonian,
                                   self.args_hams,
                                   self.ketbras,
                                   self.verbose)

'''
This is just some notes on the below.  Essentially I want to minimise re-computiation of the operators I track through
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
class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class EmissionOperatorsFactory(metaclass=Singleton):

    emission_operators = []

    @classmethod
    def get(cls, atom, cavity, ketbras, verbose):
        for em_op in cls.emission_operators:
            if em_op._is_compatible(atom, cavity):
                if verbose: print("\tFound suitable _EmissionOperators obj for setup.")
                return em_op
        else:
            em_op = cls._EmissionOperators(atom,cavity,ketbras,verbose)
            cls.emission_operators.append(em_op)
            return em_op

    class _EmissionOperators():

        def __init__(self, atom, cavity, ketbras, verbose):
            self.atom = atom
            self.cavity = cavity
            self.ketbras = ketbras
            self.verbose=verbose

            if verbose: print("\tCreating new _EmissionOperators obj for setup.")

            def kb(a, b):
                return self.ketbras[str([a, b])]

            all_atom_states = list(self.atom.atom_states)

            self.em_fast_1 = sum(map(lambda s: kb([s, 1, 0], [s, 1, 0]) + kb([s, 1, 1], [s, 1, 1]), all_atom_states))
            self.em_fast_2 = sum(map(lambda s: kb([s, 0, 1], [s, 0, 1]) + kb([s, 1, 1], [s, 1, 1]), all_atom_states))
            self.em_fast_3 = sum(map(lambda s: kb([s, 0, 1], [s, 0, 1]) - kb([s, 1, 0], [s, 1, 0]), all_atom_states))
            self.em_fast_4 = sum(map(lambda s: kb([s, 0, 1], [s, 1, 0]), all_atom_states))
            self.em_fast_5 = sum(map(lambda s: kb([s, 1, 0], [s, 0, 1]), all_atom_states))

            self.operator_series = []

        def get(self, t_series, R_ZL):
            for t, R, kappa1, kappa2, deltaP, op_series in self.operator_series:
                if all([np.array_equal(t,t_series),np.array_equal(R, R_ZL)]):
                    if self.verbose: print("\tFound suitable pre-computed emission operator series.")
                    return op_series
            return self.__generate(t_series, R_ZL)

        def __generate(self, t_series, R_ZL):
            if self.verbose: print("\tCreating new number operator series.")
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

        def _is_compatible(self, atom, cavity):
            if all([self.atom==atom,self.cavity==cavity]):
                return True
            else:
                return False

class NumberOperatorsFactory(metaclass=Singleton):

    number_operators = []

    @classmethod
    def get(cls, atom, cavity, ketbras, verbose):
        for an_op in cls.number_operators:
            if an_op._is_compatible(atom, cavity):
                if verbose: print("\tFound suitable _NumberOperators obj for setup.")
                return an_op
        else:
            an_op = cls._NumberOperators(atom, cavity, ketbras,verbose)
            cls.number_operators.append(an_op)
            return an_op

    class _NumberOperators():

        def __init__(self, atom, cavity, ketbras, verbose):
            self.atom = atom
            self.cavity = cavity
            self.ketbras = ketbras
            self.verbose = verbose

            if verbose: print("\tCreating new _NumberOperators obj for setup.")

            def kb(a, b):
                return self.ketbras[str([a, b])]

            all_atom_states = list(self.atom.atom_states)

            self.an_fast_1 = sum(map(lambda s: kb([s, 1, 0], [s, 1, 0]) + kb([s, 1, 1], [s, 1, 1]), all_atom_states))
            self.an_fast_2 = sum(map(lambda s: kb([s, 0, 1], [s, 0, 1]) + kb([s, 1, 1], [s, 1, 1]), all_atom_states))
            self.an_fast_3 = sum(map(lambda s: kb([s, 0, 1], [s, 1, 0]), all_atom_states))
            self.an_fast_4 = sum(map(lambda s: kb([s, 1, 0], [s, 0, 1]), all_atom_states))

            self.operator_series = []

        def get(self, t_series, R_ZL):
            for t, R, deltaP, op_series in self.operator_series:
                if all([np.array_equal(t,t_series),
                        np.array_equal(R, R_ZL),
                        deltaP==self.cavity.deltaP]):
                    if self.verbose: print("\tFound suitable pre-computed number operator series.")
                    return op_series
            return self.__generate(t_series, R_ZL)

        def __generate(self, t_series, R_ZL):
            if self.verbose: print("\tCreating new number operator series.")
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

        def _is_compatible(self, atom, cavity):
            if all([self.atom == atom, self.cavity == cavity]):
                return True
            else:
                return False

class AtomicOperatorsFactory(metaclass=Singleton):

    atomic_operators = []

    @classmethod
    def get(cls, atom, ketbras, verbose):
        for at_op in cls.atomic_operators:
            if at_op._is_compatible(atom):
                if verbose: print("\tFound suitable _AtomicOperators obj for setup.")
                return at_op
        else:
            at_op = cls._AtomicOperators(atom, ketbras, verbose)
            cls.atomic_operators.append(at_op)
            return at_op

    class _AtomicOperators():

        def __init__(self, atom, ketbras, verbose):
            self.atom = atom
            self.ketbras = ketbras
            self.verbose = verbose

            if verbose: print("\tCreating new _AtomicOperators obj for setup.")

            def kb(a, b):
                return self.ketbras[str([a, b])]

            atom_states=self.atom.atom_states
            self.at_ops = {}
            for s in atom_states:
                self.at_ops[s]= kb([s,0,0],[s,0,0]) + kb([s,1,0],[s,1,0]) + kb([s,0,1],[s,0,1]) + kb([s,1,1],[s,1,1])

            spont_decay_ops = []

            for g,x,branching_ratio in self.atom.get_spontaneous_emission_channels():
                try:
                    spont_decay_ops.append(branching_ratio * np.sqrt(2 * self.atom.gamma) *
                                           tensor(
                                             basis(self.atom.M, atom_states[g]) * basis(self.atom.M, atom_states[x]).dag(),
                                             qeye(Cavity.N),
                                             qeye(Cavity.N)))
                except KeyError:
                    pass

            self.sp_op = sum([x.dag() * x for x in spont_decay_ops])

        def get_at_op(self, states=[]):
            if type(states)!= list:
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

class ExperimentalResults():

    def __init__(self, output, hamiltonian, args, ketbras, verbose=False):
        self.output = output
        self.hamiltonian = hamiltonian
        self.args = args
        self.ketbras = ketbras
        self.verbose = verbose

        self.emission_operators = EmissionOperatorsFactory.get(hamiltonian.atom,
                                                               hamiltonian.cavity,
                                                               self.ketbras,
                                                               self.verbose)
        self.number_operators = NumberOperatorsFactory.get(hamiltonian.atom,
                                                           hamiltonian.cavity,
                                                           self.ketbras,
                                                           self.verbose)
        self.atomic_operators = AtomicOperatorsFactory.get(hamiltonian.atom,
                                                           self.ketbras,
                                                           self.verbose)

    def get_cavity_emission(self, R_ZL, i_output=[]):
        if type(R_ZL)!=np.matrix:
            raise ValueError("R_ZL must be a numpy matrix.")
        emP_t, emM_t = self.emission_operators.get(self.output.times, R_ZL)

        emP, emM = np.abs(np.array(
            [expect(list(an_list), state) for an_list, state in zip(zip(emP_t, emM_t), self.__get_output_states(i_output))]
        )).T

        return emP, emM

        return list(zip(*[iter(ems)] * 2))

    def get_cavity_number(self, R_ZL, i_output=[]):
        if type(R_ZL)!=np.matrix:
            raise ValueError("R_ZL must be a numpy matrix.")
        anP_t, anM_t = self.number_operators.get(self.output.times, R_ZL)

        anP, anM = np.abs(np.array(
            [expect(list(an_list), state) for an_list, state in zip(zip(anP_t, anM_t), self.__get_output_states(i_output))]
        )).T

        return anP, anM

    def get_atomic_population(self, states=[], i_output=[]):
        at_ops = self.atomic_operators.get_at_op(states)
        return np.abs(expect(at_ops, self.__get_output_states(i_output)))

    def get_total_spontaneous_emission(self, i_output=[]):
        sp_op = self.atomic_operators.get_sp_op()
        return expect(sp_op, self.__get_output_states(i_output))

    def __get_output_states(self,i_output):
        if not i_output:
            out_states = self.output.states
        elif type(i_output)==int:
            out_states = self.output.states[i_output]
        elif len(i_output)==2:
            out_states = self.output.states[i_output[0]:i_output[1]]
        else:
            raise TypeError('i_output must be [], an integer, or a list/tuple of length 2')
        return out_states