__all__ = ['Atom4lvl', 'Cavity', 'CavityBiref', 'LaserCoupling', 'CavityCoupling']
#todo: Add Atom87Rb when tested.

from rb_cqed.globals import i

import numpy as np
import textwrap
import csv
from dataclasses import dataclass, field, asdict, InitVar
from typing import Any
from itertools import product

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
class Atom4lvl(RunnerDataClass):
    g_states: dict = field(default_factory=dict)
    x_states: dict = field(default_factory=dict)
    configured_states: list = field(default_factory=list)
    transition_strengths: dict = field(default_factory=dict)
    sink_state: str = None
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
        self.__configure_transition_strengths()
        self.__configure_detunings()
        self.__configure_rotation_matrix()

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
            if list(set(self.configured_states).difference(g_lvls + x_lvls)) !=[]:
                raise Exception(textwrap.dedent('''\
                    Atom4lvl.configured_states contains states not recognised for 87Rb.
                    \tUnrecognised states: {0}
                    \tAllowed states: {1}\
                    '''.format(list(set(self.configured_states).difference(g_lvls + x_lvls)), g_lvls + x_lvls))
                )

        # Now we know the configured states are sensible and have warned about any assumptions we are making.
        lvl_idx = 0
        for states, lvls in zip([self.g_states, self.x_states], [g_lvls, x_lvls]):
            configured_lvls = [lvl for lvl in lvls if lvl in self.configured_states]
            for lvl in configured_lvls:
                states[lvl] = lvl_idx
                lvl_idx+=1
        self.M = len(self.configured_states)

    def __configure_transition_strengths(self):
        if self.sink_state != None:
            if not self.sink_state in self.g_states:
                raise Exception(textwrap.dedent('''\
                    Atom4lvl.sink_state must be a configured groud state.\
                    \tUnrecognised state: {0}\
                    \tAllowed states: {1}\
                    '''.format(self.sink_state, list(self.g_states.keys()))))
            else:
                if self.transition_strengths != {}:
                    print(textwrap.dedent('''Warning: \
                    Atom4lvl.sink_state will overwrite the couplings passed in Atom4lvl.transition_strengths to ensure \
                    that their no coupling (other than through spontaneous decay) between the sink_state and other \
                    levels.
                    '''))

        if self.transition_strengths == {}:
            for x in self.x_states.keys():
                self.transition_strengths[x] = {}
                for g in self.g_states.keys():
                    self.transition_strengths[x][g] = 1

        if self.sink_state != None:
            for x in self.x_states.keys():
                self.transition_strengths[x][self.sink_state] = 0

    def __configure_detunings(self):
        for g in self.g_states.keys():
            self.g_detunings[g] = 0
        for x in self.x_states.keys():
            self.x_detunings[x] = 0

    def __configure_rotation_matrix(self):
        if type(self.R_AL) is list:
            self.R_AL = np.matrix(self.R_AL)

        # Check the rotation matrices provided are unitary!
        if self.R_AL.shape != (2,2):
            raise Exception('Invalid value passed for R_AL.  Rotation matrices must have dimension (2,2).')
        if not np.allclose(np.eye(self.R_AL.shape[0]), self.R_AL.H * self.R_AL):
            raise Exception('Invalid value passed for R_AL.  Rotation matrices must be unitary.')

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

        if self.sink_state == None:
            for x in self.x_states:
                for g, CG in self.transition_strengths[x].items():
                    gs.append(g)
                    xs.append(x)
                    CGs.append(CG)
        else:
            for x in self.x_states:
                for g, CG in self.transition_strengths[x].items():
                    gs.append(g)
                    xs.append(x)
                    if g != self.sink_state:
                        CGs.append(0)
                    else:
                        CGs.append(1)

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
    params_file: InitVar[str] = './atom87rb_params/exp_params_0MHz.csv'
    x_zero_energy_state: InitVar[str] = 'x2'

    def __post_init__(self, params_file, x_zero_energy_state):
        self.__configure_states()
        params_dict = self.__load_params(params_file)
        self.__configure_transition_strengths(params_dict)
        self.__configure_detunings(params_dict, x_zero_energy_state)
        self.__configure_rotation_matrix()

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
            if list(set(self.configured_states).difference(g_lvls + x_lvls)) !=[]:
                raise Exception(textwrap.dedent('''\
                    Atom87Rb.configured_states contains states not recognised for 87Rb.
                    \tUnrecognised states: {0}
                    \tAllowed states: {1}\
                    '''.format(list(self.configured_states.difference(g_lvls + x_lvls)), g_lvls + x_lvls))
                )

        # Now we know the configured states are sensible and have warned about any assumptions we are making.
        lvl_idx = 0
        for states, lvls in zip([self.g_states, self.x_states], [g_lvls, x_lvls]):
            configured_lvls = [lvl for lvl in lvls if lvl in self.configured_states]
            for lvl in configured_lvls:
                states[lvl] = lvl_idx
                lvl_idx += 1
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
                Atom87Rb.transition_strengths are derived from the atom87rb_params file.  Explicitly passed values will be ignored.\
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
                Atom87Rb.*_detunings are derived from the atom87rb_params file.  Explicitly passed values will be ignored.\
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

    def __configure_rotation_matrix(self):
        if type(self.R_AL) is list:
            self.R_AL = np.matrix(self.R_AL)

        # Check the rotation matrices provided are unitary!
        if self.R_AL.shape != (2,2):
            raise Exception('Invalid value passed for R_AL.  Rotation matrices must have dimension (2,2).')
        if not np.allclose(np.eye(self.R_AL.shape[0]), self.R_AL.H * self.R_AL):
            raise Exception('Invalid value passed for R_AL.  Rotation matrices must be unitary.')

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

        if type(self.R_CL) is list:
            self.R_CL = np.matrix(self.R_CL)
        if type(self.R_ML) is list:
            self.R_ML = np.matrix(self.R_ML)

        # Check the rotation matrices provided are unitary!
        for R, lab in zip([self.R_CL, self.R_ML],['R_CL', 'R_ML']):
            if R.shape != (2,2):
                raise Exception('Invalid value passed for {0}.  Rotation matrices must have dimension (2,2).'.format(lab))
            if not np.allclose(np.eye(R.shape[0]), R.H * R):
                raise Exception('Invalid value passed for {0}.  Rotation matrices must be unitary.'.format(lab))

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
