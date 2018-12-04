import numpy as np
import matplotlib.pyplot as plt
import copy
import fileinput
import textwrap
import csv
import time
import io
import os
from dataclasses import dataclass, field, asdict, InitVar
from typing import Any
from abc import ABC, abstractmethod
from contextlib import redirect_stdout, redirect_stderr
from itertools import chain, product
from qutip import *

try:
    import seaborn as sns
    plt.style.use('seaborn')
except ImportError:
    pass
plt.rcParams['text.usetex'] = True

np.set_printoptions(threshold=np.inf)

##########################################
# Globals                                #
##########################################
d = 3.584*10**(-29)
i = np.complex(0,1)

def R2args(R):
    alpha = np.clip(np.abs(R[0, 0]), 0, 1)
    phi1, phi2 = np.angle(R[0, 0]), np.angle(R[1, 0])
    beta = np.sqrt(1 - alpha ** 2)
    return alpha, beta, phi1, phi2

##########################################
# Data Classes                           #
##########################################
#TODO: turn RunnerDataClass into ABC and drop dataclass decorators?
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
class Atom87Rb(RunnerDataClass):
    g_states: dict = field(default_factory=dict)
    x_states: dict = field(default_factory=dict)
    configured_states: list = field(default_factory=list)
    transition_strengths: dict = field(default_factory=dict)
    g_detunings: dict = field(default_factory=dict)
    x_detunings: dict = field(default_factory=dict)
    M: int = 4
    gamma: float = 3 * 2. * np.pi
    '''
    Rotation from Atom -> Lab: R_AL: {|+>,|->} -> {|H>,|V>}

    R_AL.|+>_A = |+>_L --> R_AL.(1 0)^tr = (1  i)^tr / sqrt(2)
    R_AL.|->_A = |->_L --> R_AL.(0 1)^tr = (1 -i)^tr / sqrt(2)
    '''
    R_AL: np.matrix = np.sqrt(1 / 2) * np.matrix([[1, i],
                                                  [i, 1]])
    params_file: InitVar[str] = './params/exp_params_0MHz.csv'
    x_zero_energy_state: InitVar[str] = 'x2'

    def __post_init__(self, params_file, x_zero_energy_state):
        self.__configure_states()
        params_dict = self.__load_params(params_file)
        self.__configure_transition_strengths(params_dict)
        self.__configure_detunings(params_dict, x_zero_energy_state)

    def __configure_states(self):
        # Default levels for the 87Rb D2 line.
        g_lvls = [
            'g1M', 'g1', 'g1P',  # F=1,mF=-1,0,+1 respectively
            'g2MM', 'g2M', 'g2', 'g2P', 'g2PP',  # F=2,mF=-2,-1,0,+1,+2 respectively
            ]

        x_lvls = [
            'x0', # F'=1,mF'=0
            'x1M', 'x1', 'x1P', # F'=1,mF'=-1,0,+1 respectively
            'x2MM', 'x2M', 'x2', 'x2P', 'x2PP', # F'=2,mF'=-2,-1,0,+1,+2 respectively
            'x3MMM', 'x3MM','x3M','x3','x3P','x3PP', 'x3PPP' # F'=3,mF'=-3,-2-1,0,+1,+2,+3 respectively
        ]

        # If self.configured_states was not passed through, infer the configured states from the state_dicts if they
        # were passed through.  If not presume all states are to be configured.
        if self.configured_states == []:
            for states, lvls in zip([self.g_states, self.x_states],[g_lvls, x_lvls]):
                if states == {}:
                    self.configured_states += lvls
                else:
                    self.configured_states += list(states.keys())

        # If self.configured_states was passed through, then check it makes sense!
        else:
            # If a configured state list and configured state dictionaries are passed in, warn that this is
            # defining the same thing twice and so one will be ignored...
            if self.g_states != {} or self.x_states != {}:
                if self.x_states == []:
                    s = 'Atom87Rb.g_states'
                elif self.g_states != []:
                    s = 'Atom87Rb.x_states'
                else:
                    s = 'Atom87Rb.g_states/Atom87Rb.x_states'
                raise Warning(textwrap.dedent('''\
                    {0} and Atom87Rb.configured_states both passed to constructor.
                    Please only configure your atomic states in one way! {0} will be ignored.\
                    '''.format(s))
                )
            # If configured_states contains unrecognised states then throw an error because I don't know what to do!
            if set(self.configured_states).difference(g_lvls + x_lvls) !=[]:
                raise Exception(textwrap.dedent('''\
                    Atom87Rb.configured_states contains states not recognised for 87Rb.
                    \tUnrecognised states: {0}
                    \tAllowed states: {1}\
                    '''.format(self.configured_states.difference(g_lvls + x_lvls), g_lvls + x_lvls))
                )

        # Now we know the configured states are sensible and have warned about any assumptions we are making.
        for states, lvls in zip([self.g_states, self.x_states], [g_lvls, x_lvls]):
            # configured_lvls = set(self.configured_states).intersection(lvls)
            for k, v in zip(lvls, range(len(lvls))):
                states[k] = v
        self.M = len(self.configured_states)

    def __load_params(self, params_file):
        params_dict = {}
        with open(params_file) as file:
            reader = csv.reader(file)
            for row in reader:
                params_dict[str(row[0])] = float(row[1])
        return params_dict

    def __configure_transition_strengths(self, params_dict):
        if self.transition_strengths != {}:
            raise Warning(textwrap.dedent('''\
                Atom87Rb.transition_strengths are derived from the params file.  Explicitly passed values will be ignored.\
                '''))
        for x in self.x_states.keys():
            self.transition_strengths[x] = {}
            for g in self.g_states.keys():
                try:
                    self.transition_strengths[x][g] = params_dict['CG{0}{1}'.format(g,x)]
                except KeyError:
                    # If transition strength doesn't exist in param file, set to 0 in case anyone ever asks for the
                    # coupling of a disallowed transition.
                    self.transition_strengths[x][g] = 0
            if np.abs(sum([x**2 for x  in self.transition_strengths[x].values()]) - 0.5) > 1e-3:
                raise Warning(textwrap.dedent('''\
                    The sum of the transition strengths from each excited level of the 87Rb D2 line should be 1/2.
                    For configured state {0} it is {1}.  Check the parameter file or carry on if you know what
                    you are doing.'''.format(x,sum(self.transition_strengths[x].values()))))

    def __configure_detunings(self, params_dict, x_zero_energy_state):
        if self.g_detunings != {} or self.x_detunings != {}:
            raise Warning(textwrap.dedent('''\
                Atom87Rb.*_detunings are derived from the params file.  Explicitly passed values will be ignored.\
                '''))
        self.g_detunings, self.x_detunings = {}, {}

        deltaEx0, deltaEx1, deltaEx3 = [params_dict['deltaE{0}'.format(x)] for x in ['x0','x1','x3']]

        # Note F=2 level is defined as zero energy by convection in my parameter files.
        def get_level_offset(x):
            if 'x0' in x: return deltaEx0
            elif 'x1' in x: return deltaEx1
            elif 'x3' in x: return deltaEx3
            else: return 0

        for x in self.x_states.keys():
            self.x_detunings[x] = params_dict['deltaZ{0}'.format(x)] + get_level_offset(x)

        if x_zero_energy_state not in self.x_states.keys():
            raise Exception(textwrap.dedent('''\
                The value passed as the zero-energy excited state in the R.W.A. ({0}) is not a valid state.\
                '''.format(x_zero_energy_state)))

        if x_zero_energy_state != 'x2':
            zero_offset = self.x_detunings[x_zero_energy_state]
            self.x_detunings = {k:v-zero_offset for k,v in self.x_detunings.items()}

        deltaZ = params_dict['deltaZ']

        def get_ground_splitting(g):
            return deltaZ * (g.count('P') - g.count('M'))

        for g in self.g_states.keys():
            self.g_detunings[g] = get_ground_splitting(g)

    def get_couplings(self, deltaL, deltaM) -> list:
        couplings = []
        for g,x in product(self.g_states.keys(), self.x_states.keys()):
            # deltaM = mFx - mFg
            if type(deltaM) != list:
                deltaM = [deltaM]
            if self.get_deltaM(g,x) in deltaM:
                couplings.append((g,
                                  x,
                                 self.transition_strengths[x][g],
                                 self.get_detuning(g,x) - deltaL,
                                 self.get_deltaM(g, x)))
        return couplings

    def get_deltaM(self, g, x) -> int:
        return (x.count('P') - x.count('M')) - (g.count('P') - g.count('M'))

    def get_detuning(self, g, x):
        '''
        Get the detuning (in the r.w.a. with configured zero energy states) of the g <-> x transition.

        :param g: Ground state identifier
        :param x: Excited state identifier
        :return: Float of detuning.
        '''
        return self.x_detunings[x] - self.g_detunings[g]

    def get_couplings_sigma_plus(self, deltaL) -> list:
        return self.get_couplings(deltaL, 1)

    def get_couplings_sigma_minus(self, deltaL) -> list:
        return self.get_couplings(deltaL, -1)

    def get_couplings_pi(self, deltaL) -> list:
        return self.get_couplings(deltaL, 0)

    def check_coupling(self,g,x):
        if not g in self.g_states or not x in self.x_states:
            raise Exception("Invalid atom states (g={0}, x={1}) entered.\nConfigured states are {2}".format(
                        g, x,
                        [list(self.g_states.keys()), list(self.x_states.keys())]))

    def get_spontaneous_emission_channels(self):
        # Get all the spontaneous emission channels and normalise the decay rates to sum to 1, as is required
        # by Qutip.mesolve.
        gs, xs, CGs = [], [], []

        for x in self.x_states:
            for g, CG in self.transition_strengths[x].items():
                gs.append(g)
                xs.append(x)
                CGs.append(CG)

        norm_CG = sum(CGs)
        sp_emm_channels = zip(gs,xs,[CG/norm_CG for CG in CGs])

        return sp_emm_channels

    def get_state_id(self, state_name):
        try:
            return self.g_states[state_name]
        except KeyError:
            return self.x_states[state_name]
        except KeyError:
            raise KeyError('Invalid atom state name.')

    def _eq_ignore_fields(self) -> list:
        return []

@dataclass(eq=False)
class Atom4lvl(RunnerDataClass):
    g_states: dict = field(default_factory=dict)
    x_states: dict = field(default_factory=dict)
    configured_states: list = field(default_factory=list)
    transition_strengths: dict = field(default_factory=dict)
    g_detunings: dict = field(default_factory=dict)
    x_detunings: dict = field(default_factory=dict)
    M: int = 4
    gamma: float = 3 * 2. * np.pi
    '''
    Rotation from Atom -> Lab: R_AL: {|+>,|->} -> {|H>,|V>}

    R_AL.|+>_A = |+>_L --> R_AL.(1 0)^tr = (1  i)^tr / sqrt(2)
    R_AL.|->_A = |->_L --> R_AL.(0 1)^tr = (1 -i)^tr / sqrt(2)
    '''
    R_AL: np.matrix = np.sqrt(1 / 2) * np.matrix([[1, i],
                                                  [i, 1]])

    def __post_init__(self):
        # if self.configured_states == []:
        self.__configure_states()
        if self.transition_strengths == {}:
            self.__configure_transition_strengths()
        self.__configure_detunings()

    def __configure_states(self):
        # Default levels for the 87Rb D2 line.
        g_lvls = ['gM', 'g', 'gP']
        x_lvls = ['x']

        # If self.configured_states was not passed through, infer the configured states from the state_dicts if they
        # were passed through.  If not presume all states are to be configured.
        if self.configured_states == []:
            for states, lvls in zip([self.g_states, self.x_states],[g_lvls, x_lvls]):
                if states == {}:
                    self.configured_states += lvls
                else:
                    self.configured_states += list(states.keys())

        # If self.configured_states was passed through, then check it makes sense!
        else:
            # If a configured state list and configured state dictionaries are passed in, warn that this is
            # defining the same thing twice and so one will be ignored...
            if self.g_states != {} or self.x_states != {}:
                if self.x_states == []:
                    s = 'Atom4lvl.g_states'
                elif self.g_states != []:
                    s = 'Atom4lvl.x_states'
                else:
                    s = 'Atom4lvl.g_states/Atom4lvl.x_states'
                raise Warning(textwrap.dedent('''\
                    {0} and Atom4lvl.configured_states both passed to constructor.
                    Please only configure your atomic states in one way! {0} will be ignored.\
                    '''.format(s))
                )
            # If configured_states contains unrecognised states then throw an error because I don't know what to do!
            if set(self.configured_states).difference(g_lvls + x_lvls) !=[]:
                raise Exception(textwrap.dedent('''\
                    Atom4lvl.configured_states contains states not recognised for 87Rb.
                    \tUnrecognised states: {0}
                    \tAllowed states: {1}\
                    '''.format(self.configured_states.difference(g_lvls + x_lvls), g_lvls + x_lvls))
                )

        # Now we know the configured states are sensible and have warned about any assumptions we are making.
        for states, lvls in zip([self.g_states, self.x_states], [g_lvls, x_lvls]):
            # configured_lvls = set(self.configured_states).intersection(lvls)
            for k, v in zip(lvls, range(len(lvls))):
                states[k] = v
        self.M = len(self.configured_states)

        # if self.g_states == {}:
        #     self.g_states = {"gM": 0, "g": 1, "gP": 2}
        # if self.x_states == {}:
        #     self.x_states = {"x":3}
        # for states in [self.g_states, self.x_states]:
        #     self.configured_states += list(states.keys())
        # self.M = len(self.configured_states)

    def __configure_transition_strengths(self):
        for x in self.x_states.keys():
            self.transition_strengths[x] = {}
            for g in self.g_states.keys():
                self.transition_strengths[x][g] = 1

    def __configure_detunings(self):
        for g in self.g_states.keys():
            self.g_detunings[g] = 0
        for x in self.x_states.keys():
            self.x_detunings[x] = 0

    def get_couplings(self, deltaL, deltaM) -> list:
        couplings = []
        for g,x in product(self.g_states.keys(), self.x_states.keys()):
            # deltaM = mFx - mFg
            if type(deltaM) != list:
                deltaM = [deltaM]
            if self.get_deltaM(g,x) in deltaM:
                couplings.append((g,
                                  x,
                                 self.transition_strengths[x][g],
                                 self.get_detuning(g,x) - deltaL,
                                 self.get_deltaM(g, x)))
        return couplings

    def get_deltaM(self, g, x) -> int:
        return (x.count('P') - x.count('M')) - (g.count('P') - g.count('M'))

    def get_detuning(self, g, x):
        '''
        Get the detuning (in the r.w.a. with configured zero energy states) of the g <-> x transition.

        :param g: Ground state identifier
        :param x: Excited state identifier
        :return: Float of detuning.
        '''
        return self.x_detunings[x] - self.g_detunings[g]

    def get_couplings_sigma_plus(self, deltaL) -> list:
        return self.get_couplings(deltaL, 1)

    def get_couplings_sigma_minus(self, deltaL) -> list:
        return self.get_couplings(deltaL, -1)

    def get_couplings_pi(self, deltaL) -> list:
        return self.get_couplings(deltaL, 0)

    def check_coupling(self,g,x):
        if not g in self.g_states or not x in self.x_states:
            raise Exception("Invalid atom states (g={0}, x={1}) entered.\nConfigured states are {2}".format(
                        g, x,
                        [list(self.g_states.keys()), list(self.x_states.keys())]))

    def get_spontaneous_emission_channels(self):
        # Get all the spontaneous emission channels and normalise the decay rates to sum to 1, as is required
        # by Qutip.mesolve.
        gs, xs, CGs = [], [], []

        for x in self.x_states:
            for g, CG in self.transition_strengths[x].items():
                gs.append(g)
                xs.append(x)
                CGs.append(CG)

        norm_CG = sum(CGs)
        sp_emm_channels = zip(gs,xs,[CG/norm_CG for CG in CGs])

        return sp_emm_channels

    def get_state_id(self, state_name):
        try:
            return self.g_states[state_name]
        except KeyError:
            return self.x_states[state_name]
        except KeyError:
            raise KeyError('Invalid atom state name.')

    def _eq_ignore_fields(self) -> list:
        return []

@dataclass(eq=False)
class Cavity(RunnerDataClass):
    N: int = 2
    cavity_states: list = field(default_factory=list)
    g: float = 3 * 2. * np.pi
    kappa: float = 3 * 2. * np.pi

    def __post_init__(self):
        self.cavity_states += [0,1]
        self.N = len(self.cavity_states)

    def _eq_ignore_fields(self):
        return ['g']

@dataclass(eq=False)
class CavityBiref(RunnerDataClass):
    N: int = 2
    cavity_states: list = field(default_factory=list)
    g: float = 3 * 2. * np.pi
    kappa1: float = 3 * 2. * np.pi
    kappa2: float = 3 * 2. * np.pi
    deltaP: float = 0 * 2. * np.pi
    # Rotation from Lab -> Cavity: R_CL:  {|X>,|Y>} -> {|H>,|V>}
    #   Default is {|X>,|Y>} = {|H>,|V>}
    R_CL: np.matrix = np.matrix([[1, 0],
                                 [0, 1]])
    # Rotation from Lab -> Cavity: R_ML:  {|M1>,|M2>} -> {|H>,|V>}
    #   Default is M1 = (|H> + i|V>)/sqrt(2), M2 = (|H> - i|V>)/sqrt(2). i.e. circularly polarised decay rates.
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
    deltaM: Any
    args_ham: dict
    pulse_shape: str = 'np.piecewise(t, [t<length_pulse], [np.sin((np.pi/length_pulse)*t)**2,0])'
    couple_off_resonance: bool = False
    g_coupled: list = field(default_factory=list)
    x_coupled: list = field(default_factory=list)
    setup_pyx: list = field(default_factory=list)
    add_pyx: list = field(default_factory=list)
    is_user_configured: bool = False

    def __post_init__(self):
        # If autofilling off resonance couplings is True, the manually passed lists of states to consider will
        # be ignored.  Warn the user of this.
        if self.couple_off_resonance and (self.g_coupled != [] or self.x_coupled != []):
            print(textwrap.dedent('''\
                    Warning: If LaserCoupling.autofill_off_resonance is True, the following manually passed states to\
                    autofill couplings to will be ignored:\n\
                        \tLaserCoupling.g_coupled = {0}\n\
                        \tLaserCoupling.x_coupled = {1}\
                    '''.format(self.g_coupled, self.x_coupled)))

        elif not self.couple_off_resonance:
            if self.g_coupled == []:
                self.g_coupled = [self.g]
            if self.x_coupled == []:
                self.x_coupled = [self.x]

        if type(self.deltaM) != list:
            self.deltaM = [self.deltaM]
        if not all([int(x) in [0,-1,1] for x in self.deltaM]):
            raise Exception(textwrap.dedent('''\
                deltaM must be {0,1,-1} or some combination thereof.\
            '''))

    def _eq_ignore_fields(self):
        return ['omega0', 'deltaL', 'args_ham']

@dataclass(eq=False)
class CavityCoupling(RunnerDataClass):
    g0: float
    g: str
    x: str
    deltaC: float
    deltaM: Any
    couple_off_resonance: bool = False
    g_coupled: list = field(default_factory=list)
    x_coupled: list = field(default_factory=list)
    is_user_configured: bool = False

    def __post_init__(self):
        # If autofilling off resonance couplings is True, the manually passed lists of states to consider will
        # be ignored.  Warn the user of this.
        if self.couple_off_resonance and (self.g_coupled != [] or self.x_coupled != []):
            print(textwrap.dedent('''\
                    Warning: If LaserCoupling.autofill_off_resonance is True, the following manually passed states to\
                    autofill couplings to will be ignored:\n\
                        \tLaserCoupling.g_coupled = {0}\n\
                        \tLaserCoupling.x_coupled = {1}\
                    '''.format(self.g_coupled, self.x_coupled)))

        elif not self.couple_off_resonance:
            if self.g_coupled == []:
                self.g_coupled = [self.g]
            if self.x_coupled == []:
                self.x_coupled = [self.x]

        if type(self.deltaM) != list:
            self.deltaM = [self.deltaM]
        if not all([int(x) in [0,-1,1] for x in self.deltaM]):
            raise Exception(textwrap.dedent('''\
                deltaM must be {0,1,-1} or some combination thereof.\
            '''))

    def _eq_ignore_fields(self):
        return ['g0','deltaC']

##########################################
# Runner and Results                     #
##########################################
class ExperimentalRunner():

    def __init__(self,
                 atom,
                 cavity,
                 laser_couplings,
                 cavity_couplings,
                 verbose = False):
        self.atom = atom
        self.cavity = cavity
        self.laser_couplings = laser_couplings if type(laser_couplings)==list else [laser_couplings]
        self.cavity_couplings = cavity_couplings if type(cavity_couplings)==list else [cavity_couplings]
        self.verbose = verbose

        # Before additional off-resonance couplings are inferred in CompiledHamiltonianFactory.get(...), flag the
        # couplings explicitly set by the user.  This will be used to decide how to plot the results in
        # ExperimentalResults.plot(...).
        for coupling in self.laser_couplings + self.cavity_couplings:
            coupling.is_user_configured = True

        self.compiled_hamiltonian = CompiledHamiltonianFactory.get(self.atom,
                                                                   self.cavity,
                                                                   self.laser_couplings,
                                                                   self.cavity_couplings,
                                                                   self.verbose)

    def run(self, psi0, t_length=1.2, n_steps=201):

        t, t_step = np.linspace(0, t_length, n_steps, retstep=True)

        # If the initial state is not a Obj, convert it to a ket using the systems states factory.
        if not isinstance(psi0, qobj.Qobj):
            psi0 = self.compiled_hamiltonian.states.ket(*psi0)

        # Clears the rhs memory, so that when we set rhs_reuse to true, it has nothing
        # and so uses our compiled hamiltonian.  We do this as setting rhs_reuse=True
        # prevents the .pyx files from being deleted after the first run.
        rhs_clear()
        opts = Options(rhs_reuse=True, rhs_filename=self.compiled_hamiltonian.name)

        if self.verbose:
            t_start = time.time()
            print("Running simulation with {0} timesteps".format(n_steps), end="...")
        solver.config.tdfunc = self.compiled_hamiltonian.tdfunc
        solver.config.tdname = self.compiled_hamiltonian.name

        output = mesolve(self.compiled_hamiltonian.hams,
                         psi0,
                         t,
                         self.compiled_hamiltonian.c_op_list,
                         [],
                         args=self.compiled_hamiltonian.args_hams,
                         options=opts)

        if self.verbose:
            print("finished in {0} seconds".format(np.round(time.time()-t_start,3)))

        return ExperimentalResultsFactory.get(output, self.compiled_hamiltonian, self.verbose)

    def ket(self, *args):
        return self.compiled_hamiltonian.states.ket(*args)

    def bra(self, *args):
        return self.compiled_hamiltonian.states.bra(*args)

##########################################
# Factories                              #
##########################################
class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class CompiledHamiltonianFactory(metaclass=Singleton):

    __compiled_hamiltonians = []

    @classmethod
    def get(cls, atom, cavity, laser_couplings, cavity_couplings, verbose=True):

        ham = None

        for c_ham in cls.__compiled_hamiltonians:
            if c_ham._is_compatible(atom, cavity, laser_couplings, cavity_couplings):
                if verbose:
                    print("Pre-compiled Hamiltonian, {0}.pyx, is suitable to run this experiment.".format(c_ham.name))

                ham = copy.deepcopy(c_ham)

                ham.atom = copy.deepcopy(atom)
                ham.cavity = copy.deepcopy(cavity)
                ham.laser_couplings = copy.deepcopy(laser_couplings)
                ham.cavity_couplings = copy.deepcopy(cavity_couplings)
                ham._configure_c_ops(args_only=True)
                ham._configure_laser_couplings(args_only=True)
                ham._configure_cavity_couplings(args_only=True)

        if not ham:
            if verbose:
                print("No suitable pre-compiled Hamiltonian found.  Generating Cython file...", end='')
                t_start = time.time()

            if type(cavity)==Cavity:
                com_ham_cls = cls._CompiledHamiltonianCavitySingle
            elif type(cavity)==CavityBiref:
                com_ham_cls = cls._CompiledHamiltonianCavityBiref
            else:
                raise Exception('Unrecognised cavity type:', type(cavity))

            ham = com_ham_cls(atom, cavity, laser_couplings, cavity_couplings,
                              'ExperimentalRunner_Hamiltonian_{0}_{1}'.format(
                              len(cls.__compiled_hamiltonians),
                              os.getpid()))
            if verbose:
                print("done.\n\tNew file is {0}.pyx.  Generated in {1} seconds.".format(ham.name,
                                                                                        np.round(time.time() - t_start, 3)))

            cls.__compiled_hamiltonians.append(ham)

        return ham

    @classmethod
    def get_all(cls):
        return cls.__compiled_hamiltonians

    @classmethod
    def clear(cls):
        cls.__compiled_hamiltonians = []

    class _CompiledHamiltonian(ABC):

        def __init__(self, atom, cavity, laser_couplings, cavity_couplings, name, verbose=False):

            # These are deep copies, so that if the atom, cavity, coupling objects are edited by the user after
            # compilation, the in memory versions of these with which the Hamiltonian was compiled is left unchanged.
            self.atom = copy.deepcopy(atom)
            self.cavity = copy.deepcopy(cavity)
            self.laser_couplings = copy.deepcopy(laser_couplings)
            self.cavity_couplings = copy.deepcopy(cavity_couplings)
            self.name = name

            self.states = StatesFactory.get(self.atom, self.cavity, verbose)

            # Prepare args_dict and the lists for the Hamiltonians and collapse operators.
            self.args_hams = dict([('i', i)])
            self.hams = []
            self.c_op_list = []

            self._configure_c_ops()
            self._configure_laser_couplings()
            self._configure_cavity_couplings()

            self.name = name

            self.tdfunc = self._compile(verbose)

        @abstractmethod
        def _configure_c_ops(self, args_only=False):
            raise NotImplementedError()

        @abstractmethod
        def _configure_laser_couplings(self, args_only=False):
            raise NotImplementedError()

        @abstractmethod
        def _configure_cavity_couplings(self, args_only=False):
            raise NotImplementedError()

        def _compile(self, verbose=False):
            '''
            Compile the .pyx file into an executable Cython function.
            :param verbose: Whether to print compilation details.
            :return: None
            '''
            self.hams = list(chain(*self.hams))

            # Define the function we will overwrite in the global namespace to run the simuluation.
            global cy_td_ode_rhs
            def cy_td_ode_rhs():
                raise NotImplementedError

            try:
                if verbose:
                    rhs_generate(self.hams, self.c_op_list, args=self.args_hams, name=self.name, cleanup=False)
                else:
                    with io.StringIO() as buf, redirect_stderr(buf):
                        rhs_generate(self.hams, self.c_op_list, args=self.args_hams, name=self.name, cleanup=False)
            except:
                if verbose:
                    print("\n\tException in rhs comp...adding additional setups...", end='')
                    # print('buf:\n', buf.getvalue())
                for laser_couping in self.laser_couplings:
                    if laser_couping.setup_pyx != [] or laser_couping.add_pyx != []:
                        with fileinput.FileInput(self.name + '.pyx', inplace=True) as file:
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
                if verbose:
                    print("and trying rhs generate again...", end='')
                code = compile('from ' + self.name + ' import cy_td_ode_rhs', '<string>', 'exec')
                exec(code, globals())

                return cy_td_ode_rhs

        def _is_compatible(self, atom, cavity, laser_couplings, cavity_couplings):
            '''
            Check whether the Hamiltonian can be used to simulate the given system without
            recompiling the .pyx file.
            :param atom:
            :param cavity:
            :param laser_couplings:
            :param cavity_couplings:
            :return: Boolean
            '''
            can_use = True
            if self.atom != atom:
                can_use = False
            if self.cavity != cavity:
                can_use = False
            if  ( len(self.laser_couplings) != len(laser_couplings) ) or \
                ( len(self.cavity_couplings) != len(cavity_couplings)):
                can_use = False
            else:
                for x,y in list(zip(self.laser_couplings, laser_couplings)) + \
                           list(zip(self.cavity_couplings, cavity_couplings)):
                    if x != y:
                        can_use = False
            return can_use

    class _CompiledHamiltonianCavitySingle(_CompiledHamiltonian):

        def _configure_c_ops(self, args_only=False):
            '''
            Internal function to populate the list of collapse operators for the configured
            atom and cavity.

            :param args_only: Change only the configured arguments for the simulated Hamiltonians,
                              not the Hamiltonians themselves.
            :return: None
            '''
            if not args_only:
                self.c_op_list = []

                # Cavity decay
                self.c_op_list.append(np.sqrt(2 * self.cavity.kappa) * tensor(qeye(self.atom.M), destroy(self.cavity.N)))

                # Spontaneous decay
                spont_decay_ops = []

                for g, x, r in self.atom.get_spontaneous_emission_channels():
                    try:
                        spont_decay_ops.append(r * np.sqrt(2 * self.atom.gamma) *
                                             tensor(
                                                 basis(self.atom.M, self.atom.get_state_id(g)) *
                                                 basis(self.atom.M, self.atom.get_state_id(x)).dag(),
                                                 qeye(self.cavity.N)))
                    except KeyError:
                        pass

                self.c_op_list += spont_decay_ops

        def _configure_laser_couplings(self, args_only=False):
            '''
            Internal function to configure the laser couplings by adding the required terms
            to the list of Hamiltonians.

            :param args_only: Change only the configured arguments for the simulated Hamiltonians,
                              not the Hamiltonians themselves.
            :return: None
            '''
            # Define the shorthand function for ketbras first, so we don't re-define it every time in the nested loops
            # below.
            def kb(a, b):
                return self.states.ketbras[str([a, b])]

            for laser_coupling in self.laser_couplings:
                g, x = laser_coupling.g, laser_coupling.x
                self.atom.check_coupling(g, x)

                if self.atom.get_deltaM(g, x) not in laser_coupling.deltaM:
                    print(textwrap.dedent('''\
                        Laser coupling {0}-{1} ignored as transition does not have the specified polarisation(s): {2}.\
                    '''.format(g,x,laser_coupling.deltaM)))
                else:
                    # Get the detuning of the desired transtion (in the r.w.a of the atom) and correct this for
                    # the specified laser detuning.  This is the detuning of the laser in the r.w.a of the atom.
                    detuning = self.atom.get_detuning(g, x) - laser_coupling.deltaL

                    # Get all couplings, in the form (g, x, transition strength, detuning, deltaM).
                    couplings = self.atom.get_couplings(detuning, laser_coupling.deltaM)

                    # If we are not automatically adding all off-resonance couplings, select only the couplings in the
                    # S_off_resonance lists.
                    if not laser_coupling.couple_off_resonance:
                        couplings = [c for c in couplings if (c[0] in laser_coupling.g_coupled and
                                                              c[1] in laser_coupling.x_coupled)]

                    # Create the values and labels for the coupling strength and detunings.  The label will be passed
                    # into the cython compiled Hamiltonain, the value will be associated with this label in the
                    # arguments dictionary of the Hamiltonian.
                    self.args_hams.update(laser_coupling.args_ham)

                    for g, x, transition_strength, detuning, deltaM in couplings:
                        Omega = transition_strength * laser_coupling.omega0
                        Omega_lab = 'Omega_{0}{1}'.format(g, x)
                        omegaL = detuning
                        omegaL_lab = 'omegaL_{0}{1}'.format(g, x)
                        self.args_hams.update({Omega_lab: Omega,
                                               omegaL_lab: omegaL})

                        # If we are not just updating the arguments dictionary (i.e. we are preparing to compile a fresh
                        # Hamiltonian), append the coupling to the list of Hamiltonian terms.
                        if not args_only:
                            pulse_shape = laser_coupling.pulse_shape

                            self.hams.append([
                                [-(1/2) * (
                                        (kb([g, 0], [x, 0]) + kb([g, 1], [x, 1])) +
                                        (kb([x, 0], [g, 0]) + kb([x, 1], [g, 1]))
                                ), '{0} * {1} * cos({2}*t)'.format(Omega_lab, pulse_shape, omegaL_lab)],
                                [i * (1/2) * (
                                        (kb([x, 0], [g, 0]) + kb([x, 1], [g, 1])) -
                                        (kb([g, 0], [x, 0]) + kb([g, 1], [x, 1]))
                                ), '{0} * {1} * sin({2}*t)'.format(Omega_lab, pulse_shape, omegaL_lab)]
                            ])

        def _configure_cavity_couplings(self, args_only=False):
            '''
            Internal function to configure the cavity couplings by adding the required terms
            to the list of Hamiltonians.

            :param args_only: Change only the configured arguments for the simulated Hamiltonians,
                              not the Hamiltonians themselves.
            :return: None
            '''
            def kb(a, b):
                return self.states.ketbras[str([a, b])]

            for cavity_coupling in self.cavity_couplings:
                g, x = cavity_coupling.g, cavity_coupling.x
                self.atom.check_coupling(g, x)

                if self.atom.get_deltaM(g, x) not in cavity_coupling.deltaM:
                    print(textwrap.dedent('''\
                        Laser coupling {0}-{1} ignored as transition does not have the specified polarisation(s): {2}.\
                    '''.format(g,x,cavity_coupling.deltaM)))
                else:
                    # Get the detuning of the desired transtion (in the r.w.a of the atom) and correct this for
                    # the specified laser detuning.  This is the detuning of the laser in the r.w.a of the atom.
                    detuning = self.atom.get_detuning(g, x) - cavity_coupling.deltaC

                    # Get all couplings, in the form (g, x, transition strength, detuning, deltaM).
                    couplings = self.atom.get_couplings(detuning, cavity_coupling.deltaM)

                    # If we are not automatically adding all off-resonance couplings, select only the couplings in the
                    # S_off_resonance lists.
                    if not cavity_coupling.couple_off_resonance:
                        couplings = [c for c in couplings if (c[0] in cavity_coupling.g_coupled and
                                                              c[1] in cavity_coupling.x_coupled)]


                    for g, x, transition_strength, detuning, deltaM in couplings:
                        g0 = transition_strength * cavity_coupling.g0
                        g0_lab = 'g0_{0}{1}'.format(g, x)
                        omegaC = detuning
                        omegaC_lab = 'omegaC_{0}{1}'.format(g, x)
                        self.args_hams.update({g0_lab: g0,
                                               omegaC_lab: omegaC})

                        # If we are not just updating the arguments dictionary (i.e. we are preparing to compile a fresh
                        # Hamiltonian), append the coupling to the list of Hamiltonian terms.
                        if not args_only:

                            self.hams.append([
                                [-1 * (
                                        kb([g, 1], [x, 0]) + kb([x, 0], [g, 1])
                                ), '{0} * cos({1}*t)'.format(g0_lab, omegaC_lab)],
                                [-i * (
                                        kb([g, 1], [x, 0]) - kb([x, 0], [g, 1])
                                ), '{0} * sin({1}*t)'.format(g0_lab, omegaC_lab)]
                            ])

    class _CompiledHamiltonianCavityBiref(_CompiledHamiltonian):

        def _configure_c_ops(self, args_only=False):
            '''
            Internal function to populate the list of collapse operators for the configured
            atom and cavity.

            :param args_only: Change only the configured arguments for the simulated Hamiltonians,
                              not the Hamiltonians themselves.
            :return: None
            '''
            self.args_hams.update({"deltaP": self.cavity.deltaP})

            if not args_only:
                # Define collapse operators
                R_MC = self.cavity.R_CL.getH() * self.cavity.R_ML
                alpha_MC, beta_MC, phi1_MC, phi2_MC = R2args(R_MC)

                aX = tensor(qeye(self.atom.M), destroy(self.cavity.N), qeye(self.cavity.N))
                aY = tensor(qeye(self.atom.M), qeye(self.cavity.N), destroy(self.cavity.N))

                aM1X = np.conj(np.exp(i * phi1_MC) * alpha_MC) * aX
                aM1Y = np.conj(np.exp(i * phi2_MC) * beta_MC) * aY
                aM2X = np.conj(-np.exp(-i * phi2_MC) * beta_MC) * aX
                aM2Y = np.conj(np.exp(-i * phi1_MC) * alpha_MC) * aY

                # Group collapse terms into fewest operators for speed.
                self.c_op_list.append(2 * self.cavity.kappa1 * lindblad_dissipator(aM1X) +
                                      2 * self.cavity.kappa1 * lindblad_dissipator(aM1Y) +
                                      2 * self.cavity.kappa2 * lindblad_dissipator(aM2X) +
                                      2 * self.cavity.kappa2 * lindblad_dissipator(aM2Y))
                self.c_op_list.append([2 * self.cavity.kappa1 * (sprepost(aM1Y, aM1X.dag())
                                                            - 0.5 * spost(aM1X.dag() * aM1Y)
                                                            - 0.5 * spre(aM1X.dag() * aM1Y)) +
                                       2 * self.cavity.kappa2 * (sprepost(aM2Y, aM2X.dag())
                                                            - 0.5 * spost(aM2X.dag() * aM2Y)
                                                            - 0.5 * spre(aM2X.dag() * aM2Y)),
                                       'exp(i*deltaP*t)'])
                self.c_op_list.append([2 * self.cavity.kappa1 * (sprepost(aM1X, aM1Y.dag())
                                                            - 0.5 * spost(aM1Y.dag() * aM1X)
                                                            - 0.5 * spre(aM1Y.dag() * aM1X)) +
                                       2 * self.cavity.kappa2 * (sprepost(aM2X, aM2Y.dag())
                                                            - 0.5 * spost(aM2Y.dag() * aM2X)
                                                            - 0.5 * spre(aM2Y.dag() * aM2X)),
                                       'exp(-i*deltaP*t)'])

                spont_decay_ops = []

                for g, x, r in self.atom.get_spontaneous_emission_channels():
                    try:
                        spont_decay_ops.append(r * np.sqrt(2 * self.atom.gamma) *
                                             tensor(
                                                 basis(self.atom.M, self.atom.get_state_id(g)) *
                                                 basis(self.atom.M, self.atom.get_state_id(x)).dag(),
                                                 qeye(self.cavity.N),
                                                 qeye(self.cavity.N)))
                    except KeyError:
                        pass

                self.c_op_list += spont_decay_ops

        def _configure_laser_couplings(self, args_only=False):
            '''
            Internal function to configure the laser couplings by adding the required terms
            to the list of Hamiltonians.

            :param args_only: Change only the configured arguments for the simulated Hamiltonians,
                              not the Hamiltonians themselves.
            :return: None
            '''
            # Define the shorthand function for ketbras first, so we don't re-define it every time in the nested loops
            # below.
            def kb(a, b):
                return self.states.ketbras[str([a, b])]

            for laser_coupling in self.laser_couplings:
                g, x = laser_coupling.g, laser_coupling.x
                self.atom.check_coupling(g, x)

                if self.atom.get_deltaM(g, x) not in laser_coupling.deltaM:
                    print(textwrap.dedent('''\
                        Laser coupling {0}-{1} ignored as transition does not have the specified polarisation(s): {2}.\
                    '''.format(g,x,laser_coupling.deltaM)))
                else:
                    # Get the detuning of the desired transtion (in the r.w.a of the atom) and correct this for
                    # the specified laser detuning.  This is the detuning of the laser in the r.w.a of the atom.
                    detuning = self.atom.get_detuning(g, x) - laser_coupling.deltaL

                    # Get all couplings, in the form (g, x, transition strength, detuning, deltaM).
                    couplings = self.atom.get_couplings(detuning, laser_coupling.deltaM)

                    # If we are not automatically adding all off-resonance couplings, select only the couplings in the
                    # S_off_resonance lists.
                    if not laser_coupling.couple_off_resonance:
                        couplings = [c for c in couplings if (c[0] in laser_coupling.g_coupled and
                                                              c[1] in laser_coupling.x_coupled)]

                    # Create the values and labels for the coupling strength and detunings.  The label will be passed
                    # into the cython compiled Hamiltonain, the value will be associated with this label in the
                    # arguments dictionary of the Hamiltonian.
                    self.args_hams.update(laser_coupling.args_ham)

                    for g, x, transition_strength, detuning, deltaM in couplings:
                        Omega = transition_strength * laser_coupling.omega0
                        Omega_lab = 'Omega_{0}{1}'.format(g, x)
                        omegaL = detuning
                        omegaL_lab = 'omegaL_{0}{1}'.format(g, x)
                        self.args_hams.update({Omega_lab: Omega,
                                               omegaL_lab: omegaL})

                        # If we are not just updating the arguments dictionary (i.e. we are preparing to compile a fresh
                        # Hamiltonian), append the coupling to the list of Hamiltonian terms.
                        if not args_only:
                            pulse_shape = laser_coupling.pulse_shape

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

        def _configure_cavity_couplings(self, args_only=False):
            '''
            Internal function to configure the cavity couplings by adding the required terms
            to the list of Hamiltonians.

            :param args_only: Change only the configured arguments for the simulated Hamiltonians,
                              not the Hamiltonians themselves.
            :return: None
            '''
            # Rotation from Atom -> Cavity: R_LA: {|+>,|->} -> {|X>,|Y>}
            R_AC = self.cavity.R_CL.getH() * self.atom.R_AL
            alpha_AC, beta_AC, phi1_AC, phi2_AC = R2args(R_AC)

            self.args_hams.update({"alpha_AC": alpha_AC,
                                   "beta_AC": beta_AC,
                                   "phi1_AC": phi1_AC,
                                   "phi2_AC": phi2_AC})

            def kb(a, b):
                return self.states.ketbras[str([a, b])]

            for cavity_coupling in self.cavity_couplings:
                g, x = cavity_coupling.g, cavity_coupling.x
                self.atom.check_coupling(g, x)

                if self.atom.get_deltaM(g, x) not in cavity_coupling.deltaM:
                    print(textwrap.dedent('''\
                        Laser coupling {0}-{1} ignored as transition does not have the specified polarisation(s): {2}.\
                    '''.format(g,x,cavity_coupling.deltaM)))
                else:
                    # Get the detuning of the desired transtion (in the r.w.a of the atom) and correct this for
                    # the specified laser detuning.  This is the detuning of the laser in the r.w.a of the atom.
                    detuning = self.atom.get_detuning(g, x) - cavity_coupling.deltaC

                    # Get all couplings, in the form (g, x, transition strength, detuning, deltaM).
                    couplings = self.atom.get_couplings(detuning, cavity_coupling.deltaM)

                    # If we are not automatically adding all off-resonance couplings, select only the couplings in the
                    # S_off_resonance lists.
                    if not cavity_coupling.couple_off_resonance:
                        couplings = [c for c in couplings if (c[0] in cavity_coupling.g_coupled and
                                                              c[1] in cavity_coupling.x_coupled)]


                    for g, x, transition_strength, detuning, deltaM in couplings:
                        g0 = transition_strength * cavity_coupling.g0
                        g0_lab = 'g0_{0}{1}'.format(g, x)

                        omegaC = cavity_coupling.deltaC
                        omegaC_X = omegaC + self.cavity.deltaP / 2
                        omegaC_Y = omegaC - self.cavity.deltaP / 2
                        omegaC_X_lab = 'omegaC_X_{0}{1}'.format(g, x)
                        omegaC_Y_lab = 'omegaC_Y_{0}{1}'.format(g, x)
                        self.args_hams.update({g0_lab: g0,
                                               omegaC_X_lab: omegaC_X,
                                               omegaC_Y_lab: omegaC_Y})

                        if not args_only:
                            deltaM = self.atom.get_deltaM(g, x)

                            if deltaM == 1:
                                H_coupling = [
                                    [-1 * alpha_AC * (
                                            kb([g, 1, 0], [x, 0, 0]) + kb([g, 1, 1], [x, 0, 1]) +
                                            kb([x, 0, 0], [g, 1, 0]) + kb([x, 0, 1], [g, 1, 1])
                                    ), '{0} * cos({1}*t + phi1_AC)'.format(g0_lab, omegaC_X_lab)],

                                    [-i * 1 * alpha_AC * (
                                            kb([g, 1, 0], [x, 0, 0]) + kb([g, 1, 1], [x, 0, 1]) -
                                            kb([x, 0, 0], [g, 1, 0]) - kb([x, 0, 1], [g, 1, 1])
                                    ), '{0} * sin({1}*t + phi1_AC)'.format(g0_lab, omegaC_X_lab)],

                                    [-1 * beta_AC * (
                                            kb([g, 0, 1], [x, 0, 0]) + kb([g, 1, 1], [x, 1, 0]) +
                                            kb([x, 0, 0], [g, 0, 1]) + kb([x, 1, 0], [g, 1, 1])
                                    ), '{0} * cos({1}*t + phi2_AC)'.format(g0_lab, omegaC_Y_lab)],

                                    [-i * 1 * beta_AC * (
                                            kb([g, 0, 1], [x, 0, 0]) + kb([g, 1, 1], [x, 1, 0]) -
                                            kb([x, 0, 0], [g, 0, 1]) - kb([x, 1, 0], [g, 1, 1])
                                    ), '{0} * sin({1}*t + phi2_AC)'.format(g0_lab, omegaC_Y_lab)]
                                ]

                            elif deltaM == -1:
                                H_coupling = [
                                    [-1 * alpha_AC * (
                                            kb([g, 0, 1], [x, 0, 0]) + kb([g, 1, 1], [x, 1, 0]) +
                                            kb([x, 0, 0], [g, 0, 1]) + kb([x, 1, 0], [g, 1, 1])
                                    ), '{0} * cos({1}*t - phi1_AC)'.format(g0_lab, omegaC_Y_lab)],

                                    [-i * 1 * alpha_AC * (
                                            kb([g, 0, 1], [x, 0, 0]) + kb([g, 1, 1], [x, 1, 0]) -
                                            kb([x, 0, 0], [g, 0, 1]) - kb([x, 1, 0], [g, 1, 1])
                                    ), '{0} * sin({1}*t - phi1_AC)'.format(g0_lab, omegaC_Y_lab)],

                                    [1 * beta_AC * (
                                            kb([g, 1, 0], [x, 0, 0]) + kb([g, 1, 1], [x, 0, 1]) +
                                            kb([x, 0, 0], [g, 1, 0]) + kb([x, 0, 1], [g, 1, 1])
                                    ), '{0} * cos({1}*t - phi2_AC)'.format(g0_lab, omegaC_X_lab)],

                                    [i * 1 * beta_AC * (
                                            kb([g, 1, 0], [x, 0, 0]) + kb([g, 1, 1], [x, 0, 1]) -
                                            kb([x, 0, 0], [g, 1, 0]) - kb([x, 0, 1], [g, 1, 1])
                                    ), '{0} * sin({1}*t - phi2_AC)'.format(g0_lab, omegaC_X_lab)]
                                ]

                            else:
                                raise Exception("deltaM must be +/-1.  Transition {0} -> {1} has {2}.".format(g, x, deltaM))

                            self.hams.append(H_coupling)

class ExperimentalResultsFactory():

    @classmethod
    def get(cls, output, compiled_hamiltonian, verbose=False):
        if type(compiled_hamiltonian.cavity) == Cavity:
            exp_res = cls._ExperimentalResultsSingle(output, compiled_hamiltonian, verbose=False)
        elif type(compiled_hamiltonian.cavity) == CavityBiref:
            exp_res = cls._ExperimentalResultsBiref(output, compiled_hamiltonian, verbose=False)
        else:
            raise Exception('Unrecognised cavity type:', type(compiled_hamiltonian.cavity))

        return exp_res

    class _ExperimentalResults(ABC):

        def __init__(self, output, compiled_hamiltonian, verbose=False):
            self.output = output
            self.compiled_hamiltonian = compiled_hamiltonian
            self.args = self.compiled_hamiltonian.args_hams
            self.ketbras = self.compiled_hamiltonian.states.ketbras
            self.verbose = verbose

            self.emission_operators = EmissionOperatorsFactory.get(self.compiled_hamiltonian.atom,
                                                                   self.compiled_hamiltonian.cavity,
                                                                   self.ketbras,
                                                                   self.verbose)
            self.number_operators = NumberOperatorsFactory.get(self.compiled_hamiltonian.atom,
                                                               self.compiled_hamiltonian.cavity,
                                                               self.ketbras,
                                                               self.verbose)
            self.atomic_operators = AtomicOperatorsFactory.get(self.compiled_hamiltonian.atom,
                                                               self.compiled_hamiltonian.cavity,
                                                               self.ketbras,
                                                               self.verbose)


        def _get_output_states(self, i_output):
            if not i_output:
                out_states = self.output.states
            elif type(i_output) == int:
                out_states = self.output.states[i_output]
            elif len(i_output) == 2:
                out_states = self.output.states[i_output[0]:i_output[1]]
            else:
                raise TypeError('i_output must be [], an integer, or a list/tuple of length 2')
            return out_states

        @abstractmethod
        def get_cavity_emission(self, *args):
            raise NotImplementedError()

        @abstractmethod
        def get_cavity_number(self, *args):
            raise NotImplementedError()

        @abstractmethod
        def get_atomic_population(self, *args):
            raise NotImplementedError()

        @abstractmethod
        def get_total_spontaneous_emission(self, *args):
            raise NotImplementedError()

        @abstractmethod
        def plot(self, *args):
            raise NotImplementedError()

        def _plot_atomic_populations(self, atom_states):
            t = self.output.times

            # Get the atom and it's states for configuring the plot of atomic populations.
            atom = self.compiled_hamiltonian.atom
            atom_g_states = list(atom.g_states.keys())
            atom_x_states = list(atom.x_states.keys())

            if atom_states != None:
                # If atom_states are configured, plot these, sorting them into ground and excited states.
                # Note: unrecognised states will be silently ignored.
                g_levels_plt_list = [g for g in atom_g_states if g in atom_states]
                x_levels_plt_list = [x for x in atom_x_states if x in atom_states]
            elif type(atom) == Atom4lvl:
                # For a 3-level atom we will plot all states by default.
                g_levels_plt_list = atom_g_states
                x_levels_plt_list = atom_x_states
            else:
                # Atom is Atom87Rb by process of elimination.  By default we only plot states explicitly coupled
                # by the user configured laser and cavity couplings.
                g_levels_plt_list = []
                x_levels_plt_list = []

                for coupling in self.compiled_hamiltonian.laser_couplings + self.compiled_hamiltonian.cavity_couplings:
                    if coupling.is_user_configured:
                        g_levels_plt_list.append(coupling.g)
                        x_levels_plt_list.append(coupling.x)
                        # Sort lists (for plotting order) by the order the states are listed on the atom.  It's nice
                        # to have consistent state ordering across multiple plots.
                        g_levels_plt_list = sorted(g_levels_plt_list, key=lambda s: atom_g_states.index(s))
                        x_levels_plt_list = sorted(x_levels_plt_list, key=lambda s: atom_x_states.index(s))

            # Remove duplicates (which either result from multiple couplings to one level, or the user passing
            # duplicate state keys).
            g_levels_plt_list = list(set(g_levels_plt_list))
            x_levels_plt_list = list(set(x_levels_plt_list))

            exp_at_ground = zip(g_levels_plt_list, self.get_atomic_population(g_levels_plt_list))
            exp_at_excited = zip(x_levels_plt_list, self.get_atomic_population(x_levels_plt_list))

            f, (a) = plt.subplots(1, 1, sharex=True, figsize=(12, 11 / 4))

            a.set_title('\\textbf{Atomic state}', fontsize=16)

            for lab, exp_at in exp_at_ground:
                a.plot(t, exp_at, label='${}$'.format(lab))
            for lab, exp_at in exp_at_excited:
                a.plot(t, exp_at, '--', label='${}$'.format(lab))

            a.set_xlabel('Time, $\mu s$')
            a.set_ylabel('Population')
            a.legend(loc=1)

            return f

    class _ExperimentalResultsSingle(_ExperimentalResults):

        def get_cavity_emission(self, i_output=[]):
            return np.abs(expect(self.emission_operators.get(), self._get_output_states(i_output)))
            em = self.emission_operators.get()
            # return np.array([(x * em).tr() for x in self._get_output_states(i_output)])

        def get_cavity_number(self, i_output=[]):
            return np.abs(expect(self.number_operators.get(), self._get_output_states(i_output)))
            # an = self.number_operators.get()
            # return np.array([(x * an).tr() for x in self._get_output_states(i_output)])

        def get_atomic_population(self, states=[], i_output=[]):
            at_ops = self.atomic_operators.get_at_op(states)
            return np.abs(expect(at_ops, self._get_output_states(i_output)))

        def get_total_spontaneous_emission(self, i_output=[]):
            sp_op = self.atomic_operators.get_sp_op()
            return expect(sp_op, self._get_output_states(i_output))

        def plot(self, atom_states=None):
            exp_an = self.get_cavity_number()
            exp_em = self.get_cavity_emission()

            exp_sp = self.get_total_spontaneous_emission()

            t = self.output.times
            tStep = np.mean(np.ediff1d(t))

            n_ph = np.trapz(exp_em, dx=tStep)
            n_sp = np.trapz(exp_sp, dx=tStep)

            print('Photon emission:', np.round(n_ph, 3))
            print('Spontaneous emission:', np.round(n_sp, 3))

            # Plot the results
            f1, ((a1a, a1b)) = plt.subplots(1, 2, sharex=True, figsize=(12, 11. / 4))

            a1a.set_title('\\textbf{Cavity mode population}', fontsize=16)
            a1b.set_title('\\textbf{Cavity emission rate} ($1/\mu s$)', fontsize=16, fontweight='bold')

            ###
            axA, axB = a1a, a1b

            axA.plot(t, exp_an)
            axA.set_ylabel('Cavity mode population')

            axB.plot(t, exp_em)
            axB.set_ylabel('Cavity emission rate, $1/\mu s$')

            ###

            f2 = self._plot_atomic_populations(atom_states)

            return f1, f2

    # TODO: allow nested lists to be passed to atom-_states for producing multiple plots.
    class _ExperimentalResultsBiref(_ExperimentalResults):

        def get_cavity_emission(self, R_ZL, i_output=[]):
            if type(R_ZL) != np.matrix:
                raise ValueError("R_ZL must be a numpy matrix.")
            emP_t, emM_t = self.emission_operators.get(self.output.times, R_ZL)

            emP, emM = np.abs(np.array(
                [expect(list(an_list), state) for an_list, state in
                 zip(zip(emP_t, emM_t), self._get_output_states(i_output))]
            )).T

            return emP, emM

            return list(zip(*[iter(ems)] * 2))

        def get_cavity_number(self, R_ZL, i_output=[]):
            if type(R_ZL) != np.matrix:
                raise ValueError("R_ZL must be a numpy matrix.")
            anP_t, anM_t = self.number_operators.get(self.output.times, R_ZL)

            anP, anM = np.abs(np.array(
                [expect(list(an_list), state) for an_list, state in
                 zip(zip(anP_t, anM_t), self._get_output_states(i_output))]
            )).T

            return anP, anM

        def get_atomic_population(self, states=[], i_output=[]):
            at_ops = self.atomic_operators.get_at_op(states)
            return np.abs(expect(at_ops, self._get_output_states(i_output)))

        def get_total_spontaneous_emission(self, i_output=[]):
            sp_op = self.atomic_operators.get_sp_op()
            return expect(sp_op, self._get_output_states(i_output))

        def plot(self, atom_states=None):
            '''
            Plot a summary of the simulation results.
            :param atom_states: The displayed atomic populations.  Default behaviour is automatically configured
            to choose sensible states.
            :return:
            '''
            #TODO: relabel plots in terms of the cavity basis, mirror basis, atomic basis etc.
            exp_anX, exp_anY = self.get_cavity_number(self.compiled_hamiltonian.cavity.R_CL)
            exp_emX, exp_emY = self.get_cavity_emission(self.compiled_hamiltonian.cavity.R_CL)
            exp_anP, exp_anM = self.get_cavity_number(self.compiled_hamiltonian.atom.R_AL)
            exp_emP, exp_emM = self.get_cavity_emission(self.compiled_hamiltonian.atom.R_AL)

            exp_sp = self.get_total_spontaneous_emission()

            t = self.output.times
            tStep = np.mean(np.ediff1d(t))

            n_X = np.trapz(exp_emX, dx=tStep)
            n_Y = np.trapz(exp_emY, dx=tStep)
            n_ph = n_X + n_Y
            n_sp = np.trapz(exp_sp, dx=tStep)

            plt.rcParams['text.usetex'] = True

            print('Photon emission:', np.round(n_ph, 3))
            print('Photon emission in |X>, |Y>:', np.round(n_X, 3), np.round(n_Y, 3))
            print('Spontaneous emission:', np.round(n_sp, 3))

            # Plot the results
            f1, ((a1a, a1b),
                 (a2a, a2b)) = plt.subplots(2, 2, sharex=True, figsize=(12, 11. / 2))

            a1a.set_title('\\textbf{Cavity mode population}', fontsize=16)
            a1b.set_title('\\textbf{Cavity emission rate} ($1/\mu s$)', fontsize=16, fontweight='bold')

            ###
            axA, axB = a1a, a1b

            axA.plot(t, exp_anP, label='$+ (\sigma^{+})$')
            axA.plot(t, exp_anM, label='$- (\sigma^{-})$')
            axA.set_ylabel('Cavity mode population')
            axA.legend(loc=1)

            axB.plot(t, exp_emP, label='$+ (\sigma^{+})$')
            axB.plot(t, exp_emM, label='$- (\sigma^{-})$')
            axB.set_ylabel('Cavity emission rate, $1/\mu s$')
            axB.legend(loc=1)

            ###
            axA, axB = a2a, a2b

            axA.plot(t, exp_anX, label='$X$')
            axA.plot(t, exp_anY, label='$Y$')
            axA.set_ylabel('Cavity mode population')
            axA.legend(loc=1)

            axB.plot(t, exp_emX, label='$X$')
            axB.plot(t, exp_emY, label='$Y$')
            axB.set_ylabel('Cavity emission rate, $1/\mu s$')
            axB.legend(loc=1)

            f2 = self._plot_atomic_populations(atom_states)

            return f1, f2

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
                if verbose: print("\tFound suitable _EmissionOperators obj for setup.")
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
            if verbose: print("\tCreating new _EmissionOperators obj for setup.")

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

            self.a = tensor(qeye(self.atom.M), destroy(self.cavity.N))
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

class NumberOperatorsFactory(metaclass=Singleton):

    number_operators = []

    @classmethod
    def get(cls, atom, cavity, ketbras, verbose):
        for an_op in cls.number_operators:
            if an_op._is_compatible(atom, cavity):
                if verbose: print("\tFound suitable _NumberOperators obj for setup.")
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

            if verbose: print("\tCreating new _NumberOperators obj for setup.")

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

            self.a = tensor(qeye(self.atom.M), destroy(self.cavity.N))
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

class AtomicOperatorsFactory(metaclass=Singleton):

    atomic_operators = []

    @classmethod
    def get(cls, atom, cavity, ketbras, verbose):
        for at_op in cls.atomic_operators:
            if at_op._is_compatible(atom):
                if (type(cavity)==Cavity and type(at_op)==cls._AtomicOperatorsCavitySingle) or \
                   (type(cavity)==CavityBiref and type(at_op)==cls._AtomicOperatorsCavityBiref):
                    if verbose: print("\tFound suitable _AtomicOperators obj for setup.")
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
            if verbose: print("\tCreating new _AtomicOperators obj for setup.")

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

            for g,x,branching_ratio in self.atom.get_spontaneous_emission_channels():
                try:
                    spont_decay_ops.append(branching_ratio * np.sqrt(2 * self.atom.gamma) *
                                           tensor(
                                             basis(self.atom.M, self.atom.get_state_id(g)) * basis(self.atom.M, self.atom.get_state_id(x)).dag(),
                                             qeye(Cavity.N)))
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

            for g,x,branching_ratio in self.atom.get_spontaneous_emission_channels():
                try:
                    spont_decay_ops.append(branching_ratio * np.sqrt(2 * self.atom.gamma) *
                                           tensor(
                                             basis(self.atom.M, self.atom.get_state_id(g)) * basis(self.atom.M, self.atom.get_state_id(x)).dag(),
                                             qeye(Cavity.N),
                                             qeye(Cavity.N)))
                except KeyError:
                    pass

            self.sp_op = sum([x.dag() * x for x in spont_decay_ops])

class StatesFactory(metaclass=Singleton):

    states = []

    @classmethod
    def get(cls, atom, cavity, verbose=False):
        for s in cls.states:
            if s._is_compatible(atom, cavity):
                if verbose: print("\tFound suitable _States obj for setup.")
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
                ket = tensor(basis(self.atom.M, self.atom.get_state_id(atom_state)),
                             basis(self.cavity.N, cav))
                self.kets[str([atom_state, cav])] = ket

            return ket

        def bra(self, atom_state, cav):
            try:
                bra = self.bras[str([atom_state, cav])]
            except KeyError:
                bra = tensor(basis(self.atom.M, self.atom.get_state_id(atom_state)),
                             basis(self.cavity.N, cav)).dag()
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
                ket = tensor(basis(self.atom.M, self.atom.get_state_id(atom_state)),
                             basis(self.cavity.N, cav_X),
                             basis(self.cavity.N, cav_Y))
                self.kets[str([atom_state, cav_X, cav_Y])] = ket

            return ket

        def bra(self, atom_state, cav_X, cav_Y):
            try:
                bra = self.bras[str([atom_state, cav_X, cav_Y])]
            except KeyError:
                bra = tensor(basis(self.atom.M, self.atom.get_state_id(atom_state)),
                             basis(self.cavity.N, cav_X),
                             basis(self.cavity.N, cav_Y)).dag()
                self.bras[str([atom_state, cav_X, cav_Y])] = bra

            return bra

        def _get_states_list(self):
            s = [list(self.atom.configured_states), self.cavity.cavity_states, self.cavity.cavity_states]
            return list(map(list, list(product(*s))))