__all__ = ['ExperimentalRunner']

from rb_cqed.globals import i, R2args, Singleton
from rb_cqed.runner_objs import Atom4lvl, Atom87Rb, Cavity, CavityBiref, LaserCoupling, CavityCoupling
from rb_cqed.qu_factories import EmissionOperatorsFactory, AtomicOperatorsFactory, NumberOperatorsFactory, StatesFactory

import numpy as np
np.set_printoptions(threshold=np.inf)

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

import copy
import fileinput
import textwrap
import time
import os
from abc import ABC, abstractmethod
from itertools import chain, product

import qutip as qt

# Monkey patch the qutip rhs_generate function with our customised version that prepends the constant part of the
# lagrangian to any time-dependent terms, rather than appending it as is done in qutip version 4.3.1.  Once this
# bug is fixed in the qutip distribution this monkey patch can be removed.
from rb_cqed.qutip_patches.rhs_generate import rhs_generate as rhs_generate_patch
qt.rhs_generate = rhs_generate_patch

# For certain applications we want to further edit the auto-generated .pyx files prior to their compilation into
# Cython function, so here we import the two steps ran within rhs_generate to allow us this option.
from rb_cqed.qutip_patches.rhs_generate import rhs_prepare, rhs_compile

try:
    import seaborn as sns
    plt.style.use('seaborn')
except ImportError:
    pass

##########################################
# Runner                                 #
##########################################
class ExperimentalRunner():

    def __init__(self,
                 atom,
                 cavity,
                 laser_couplings,
                 cavity_couplings,
                 verbose=False,
                 reconfigurable_decay_rates=False,
                 ham_pyx_dir=None,
                 force_compile=False):
        self.atom = atom
        self.cavity = cavity
        self.laser_couplings = laser_couplings if type(laser_couplings) == list else [laser_couplings]
        self.cavity_couplings = cavity_couplings if type(cavity_couplings) == list else [cavity_couplings]
        self.verbose = verbose
        self.reconfigurable_decay_rates = reconfigurable_decay_rates
        self.ham_pyx_dir = ham_pyx_dir
        self.force_compile=force_compile

        # Before additional off-resonance couplings are inferred in CompiledHamiltonianFactory.get(...), flag the
        # couplings explicitly set by the user.  This will be used to decide how to plot the results in
        # ExperimentalResults.plot(...).
        for coupling in self.laser_couplings + self.cavity_couplings:
            coupling.is_user_configured = True

        self.compiled_hamiltonian = CompiledHamiltonianFactory.get(self.atom,
                                                                   self.cavity,
                                                                   self.laser_couplings,
                                                                   self.cavity_couplings,
                                                                   self.verbose,
                                                                   self.reconfigurable_decay_rates,
                                                                   self.ham_pyx_dir,
                                                                   self.force_compile)

    def run(self, psi0, t_length=1.2, n_steps=201):

        t, t_step = np.linspace(0, t_length, n_steps, retstep=True)

        # If the initial state is not a Obj, convert it to a ket using the systems states factory.
        if not isinstance(psi0, qt.qobj.Qobj):
            psi0 = self.compiled_hamiltonian.states.ket(*psi0)

        # Clears the rhs memory, so that when we set rhs_reuse to true, it has nothing
        # and so uses our compiled hamiltonian.  We do this as setting rhs_reuse=True
        # prevents the .pyx files from being deleted after the first run.
        qt.rhs_clear()
        # opts = Options(rhs_reuse=True, rhs_filename=self.compiled_hamiltonian.name)
        opts = qt.Options(rhs_filename=self.compiled_hamiltonian.name)

        if self.verbose:
            t_start = time.time()
            print("Running simulation with {0} timesteps".format(n_steps), end="...")
        qt.solver.config.tdfunc = self.compiled_hamiltonian.tdfunc
        qt.solver.config.tdname = self.compiled_hamiltonian.name

        output = qt.mesolve(self.compiled_hamiltonian.hams,
                            psi0,
                            t,
                            self.compiled_hamiltonian.c_op_list,
                            [],
                            args=self.compiled_hamiltonian.args_hams,
                            options=opts)

        if self.verbose:
            print("finished in {0} seconds".format(np.round(time.time() - t_start, 3)))

        return ExperimentalResultsFactory.get(output, self.compiled_hamiltonian, self.verbose)

    def ket(self, *args):
        return self.compiled_hamiltonian.states.ket(*args)

    def bra(self, *args):
        return self.compiled_hamiltonian.states.bra(*args)


##########################################
# Factories                              #
##########################################
class CompiledHamiltonianFactory(metaclass=Singleton):
    __compiled_hamiltonians = []

    @classmethod
    def get(cls, atom, cavity, laser_couplings, cavity_couplings, verbose=True, reconfigurable_decay_rates=False,
            ham_pyx_dir=None, force_compile=False):

        ham = None

        if not force_compile:
            for c_ham in cls.__compiled_hamiltonians:
                if c_ham._is_compatible(atom, cavity, laser_couplings, cavity_couplings, reconfigurable_decay_rates):
                    if verbose:
                        if c_ham.ham_pyx_dir != None:
                            print(
                                "Pre-compiled Hamiltonian, {0}.pyx, is suitable to run this experiment.".format(c_ham.name))
                        else:
                            print("A pre-compiled Hamiltonian is suitable to run this experiment.")

                    ham = copy.deepcopy(c_ham)

                    ham.atom = copy.deepcopy(atom)
                    ham.cavity = copy.deepcopy(cavity)
                    ham.laser_couplings = copy.deepcopy(laser_couplings)
                    ham.cavity_couplings = copy.deepcopy(cavity_couplings)
                    ham._configure_c_ops(args_only=True)
                    ham._configure_laser_couplings(args_only=True)
                    ham._configure_cavity_couplings(args_only=True)

        if ham is None:
            if verbose:
                print("No suitable pre-compiled Hamiltonian found.  Generating and compiling Cython file...", end='')
                t_start = time.time()

            if type(cavity) == Cavity:
                com_ham_cls = cls._CompiledHamiltonianCavitySingle
            elif type(cavity) == CavityBiref:
                com_ham_cls = cls._CompiledHamiltonianCavityBiref
            else:
                raise Exception('Unrecognised cavity type:', type(cavity))

            ham = com_ham_cls(atom, cavity, laser_couplings, cavity_couplings,
                              'ExperimentalRunner_Hamiltonian_{0}_{1}'.format(
                                  len(cls.__compiled_hamiltonians),
                                  os.getpid()),
                              verbose,
                              reconfigurable_decay_rates,
                              ham_pyx_dir)
            if verbose:
                if ham.ham_pyx_dir != None:
                    print("done.\n\tNew file is {0}.pyx.  Complete in {1} seconds.".format(
                        ham.name, np.round(time.time() - t_start, 3)))
                else:
                    print("done.\n\tThe pyx file was deleted after compilation.  Complete in {0} seconds.".format(
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

        def __init__(self, atom, cavity, laser_couplings, cavity_couplings, name, verbose=False,
                     reconfigurable_decay_rates=False, ham_pyx_dir=None):

            # These are deep copies, so that if the atom, cavity, coupling objects are edited by the user after
            # compilation, the in memory versions of these with which the Hamiltonian was compiled is left unchanged.
            self.atom = copy.deepcopy(atom)
            self.cavity = copy.deepcopy(cavity)
            self.laser_couplings = copy.deepcopy(laser_couplings)
            self.cavity_couplings = copy.deepcopy(cavity_couplings)
            self.name = name
            self.verbose = verbose
            self.reconfigurable_decay_rates = reconfigurable_decay_rates
            self.ham_pyx_dir = ham_pyx_dir

            self.states = StatesFactory.get(self.atom, self.cavity, verbose)

            # Prepare args_dict and the lists for the Hamiltonians and collapse operators.
            self.args_hams = dict([('i', i)])
            self.hams = []
            self.c_op_list = []

            self._configure_c_ops()
            self._configure_laser_couplings()
            self._configure_cavity_couplings()

            self._clean_args_hams()

            # If no laser or cavity couplings were configured, add a 'no-action' coupling.  This is simply a workaround
            # required as Qutip.mesolve gets upset if it is passed empty Hamiltonian lists.
            if self.hams == []:
                self.hams.append([self._get_dummy_coupling()])

            self.name = name

            self.tdfunc = self._compile(self.verbose)

        @abstractmethod
        def _configure_c_ops(self, args_only=False):
            raise NotImplementedError()

        @abstractmethod
        def _configure_laser_couplings(self, args_only=False):
            raise NotImplementedError()

        @abstractmethod
        def _configure_cavity_couplings(self, args_only=False):
            raise NotImplementedError()

        @abstractmethod
        def _get_dummy_coupling(self):
            raise NotImplementedError()

        def _clean_args_hams(self):
            '''
            Cleans self.args_hams such that it has consistent typing, e.g. all integers are oncverted to floats.

            This is important as the compiled C function wants consistent typing.
            :return: None
            '''
            self.args_hams = {k:v if type(v)!=int else float(v) for k, v in self.args_hams.items()}

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

            if self.ham_pyx_dir == None:
                cleanup = True
            else:
                cleanup = False

            customise_pyx = False
            for laser_couping in self.laser_couplings:
                if laser_couping.setup_pyx != [] or laser_couping.add_pyx != []:
                    customise_pyx = True
                    break

            if not customise_pyx:
                if verbose:
                    print("\n\tcompiling Cython function with rhs_generate(...)", end='...')
                qt.rhs_generate(self.hams, self.c_op_list, args=self.args_hams, name=self.name, cleanup=cleanup)

            else:
                if verbose:
                    print("\n\tadditional setup required:\n\t\tpreparing .pyx file with rhs_prepare(...)", end='...')
                rhs_prepare(self.hams, self.c_op_list, args=self.args_hams, name=self.name)

                if verbose:
                    print("\n\t\tcustomising .pyx file", end='...')
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
                    print("\n\t\tcompiling Cython function with rhs_compile", end='...')
                rhs_compile(cleanup)

            if cleanup == True:
                try:
                    os.remove(self.name + '.pyx')
                except:
                    pass

            else:
                if not os.path.isdir(self.ham_pyx_dir):
                    try:
                        if self.verbose:
                            print('creating directory for Hamiltonian pyx file at {}'.format(self.ham_pyx_dir),
                                  end='...')
                        os.mkdir(self.ham_pyx_dir)
                        if self.verbose:
                            print('moving pyx file', end='...')
                        os.rename(self.name + '.pyx', os.path.join(self.ham_pyx_dir, self.name + '.pyx'))

                    except Exception as e:
                        if self.verbose:
                            print('failed with {}.'.format(e.str()), end='...')

                return cy_td_ode_rhs

        def _is_compatible(self, atom, cavity, laser_couplings, cavity_couplings, reconfigurable_decay_rates):
            '''
            Check whether the Hamiltonian can be used to simulate the given system without
            recompiling the .pyx file.
            :param atom:
            :param cavity:
            :param laser_couplings:
            :param cavity_couplings:
            :param reconfigurable_decay_rates
            :return: Boolean
            '''
            can_use = True

            if (type(atom) != type(self.atom)) or (type(cavity) != type(self.cavity)):
                can_use = False
            else:
                # If decay rates are reconfigurable, allow them to be different.
                if self.reconfigurable_decay_rates:
                    # Clone the items for comparison so we don't reset the decay rates on the atom/cavity we are actually
                    # going to use.
                    atom = copy.copy(atom)
                    atom.gamma = self.atom.gamma

                    cavity = copy.copy(cavity)

                    if type(cavity) == Cavity:
                        cavity.kappa = self.cavity.kappa
                    else:
                        cavity.kappa1 = self.cavity.kappa1
                        cavity.kappa2 = self.cavity.kappa2

                if self.atom != atom:
                    can_use = False
                if self.cavity != cavity:
                    can_use = False
                if (len(self.laser_couplings) != len(laser_couplings)) or \
                        (len(self.cavity_couplings) != len(cavity_couplings)):
                    can_use = False
                else:
                    for x, y in list(zip(self.laser_couplings, laser_couplings)) + \
                                list(zip(self.cavity_couplings, cavity_couplings)):
                        if x != y:
                            can_use = False
                if self.reconfigurable_decay_rates != reconfigurable_decay_rates:
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
            if self.reconfigurable_decay_rates:
                self.args_hams.update({"sqrt_gamma": np.sqrt(self.atom.gamma),
                                       "sqrt_kappa": np.sqrt(self.cavity.kappa)})

            if not args_only:
                self.c_op_list = []

                if not self.reconfigurable_decay_rates:
                    # Cavity decay
                    self.c_op_list.append(
                        np.sqrt(2 * self.cavity.kappa) * qt.tensor(qt.qeye(self.atom.M), qt.destroy(self.cavity.N)))
                else:
                    self.c_op_list.append(
                        [np.sqrt(2) * qt.tensor(qt.qeye(self.atom.M), qt.destroy(self.cavity.N)), "sqrt_kappa"])

                # Spontaneous decay
                spont_decay_ops = []

                if not self.reconfigurable_decay_rates:
                    for g, x, r in self.atom.get_spontaneous_emission_channels():
                        try:
                            spont_decay_ops.append(np.sqrt(r * 2 * self.atom.gamma) *
                                                   qt.tensor(
                                                       qt.basis(self.atom.M, self.atom.get_state_id(g)) *
                                                       qt.basis(self.atom.M, self.atom.get_state_id(x)).dag(),
                                                       qt.qeye(self.cavity.N)))
                        except KeyError:
                            pass

                else:
                    for g, x, r in self.atom.get_spontaneous_emission_channels():
                        try:
                            spont_decay_ops.append(np.sqrt(r * 2) *
                                                   qt.tensor(
                                                       qt.basis(self.atom.M, self.atom.get_state_id(g)) *
                                                       qt.basis(self.atom.M, self.atom.get_state_id(x)).dag(),
                                                       qt.qeye(self.cavity.N)))
                        except KeyError:
                            pass
                    spont_decay_ops = [[sum(spont_decay_ops), 'sqrt_gamma']]

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
                    '''.format(g, x, laser_coupling.deltaM)))
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
                                        (kb([g, 0], [x, 0]) + kb([g, 1], [x, 1])) +
                                        (kb([x, 0], [g, 0]) + kb([x, 1], [g, 1]))
                                ), '{0} * {1} * cos({2}*t)'.format(Omega_lab, pulse_shape, omegaL_lab)],
                                [i * (1 / 2) * (
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
                    '''.format(g, x, cavity_coupling.deltaM)))
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

        def _get_dummy_coupling(self):
            M, N = self.atom.M, self.cavity.N
            return qt.tensor(qt.qobj.Qobj(np.zeros((M, M))), qt.qobj.Qobj(np.zeros((N, N))))

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
            if self.reconfigurable_decay_rates:
                self.args_hams.update({"sqrt_gamma": np.sqrt(self.atom.gamma),
                                       "kappa1": self.cavity.kappa1,
                                       "kappa2": self.cavity.kappa2})

            if not args_only:

                # Define collapse operators
                R_MC = self.cavity.R_CL.getH() * self.cavity.R_ML
                alpha_MC, beta_MC, phi1_MC, phi2_MC = R2args(R_MC)

                aX = qt.tensor(qt.qeye(self.atom.M), qt.destroy(self.cavity.N), qt.qeye(self.cavity.N))
                aY = qt.tensor(qt.qeye(self.atom.M), qt.qeye(self.cavity.N), qt.destroy(self.cavity.N))

                aM1X = np.conj(np.exp(i * phi1_MC) * alpha_MC) * aX
                aM1Y = np.conj(np.exp(i * phi2_MC) * beta_MC) * aY
                aM2X = np.conj(-np.exp(-i * phi2_MC) * beta_MC) * aX
                aM2Y = np.conj(np.exp(-i * phi1_MC) * alpha_MC) * aY

                # Group collapse terms into fewest operators for speed.
                if not self.reconfigurable_decay_rates:
                    self.c_op_list.append(2 * self.cavity.kappa1 * qt.lindblad_dissipator(aM1X) +
                                          2 * self.cavity.kappa1 * qt.lindblad_dissipator(aM1Y) +
                                          2 * self.cavity.kappa2 * qt.lindblad_dissipator(aM2X) +
                                          2 * self.cavity.kappa2 * qt.lindblad_dissipator(aM2Y))
                    self.c_op_list.append([2 * self.cavity.kappa1 * (qt.sprepost(aM1Y, aM1X.dag())
                                                                     - 0.5 * qt.spost(aM1X.dag() * aM1Y)
                                                                     - 0.5 * qt.spre(aM1X.dag() * aM1Y)) +
                                           2 * self.cavity.kappa2 * (qt.sprepost(aM2Y, aM2X.dag())
                                                                     - 0.5 * qt.spost(aM2X.dag() * aM2Y)
                                                                     - 0.5 * qt.spre(aM2X.dag() * aM2Y)),
                                           'exp(i*deltaP*t)'])
                    self.c_op_list.append([2 * self.cavity.kappa1 * (qt.sprepost(aM1X, aM1Y.dag())
                                                                     - 0.5 * qt.spost(aM1Y.dag() * aM1X)
                                                                     - 0.5 * qt.spre(aM1Y.dag() * aM1X)) +
                                           2 * self.cavity.kappa2 * (qt.sprepost(aM2X, aM2Y.dag())
                                                                     - 0.5 * qt.spost(aM2Y.dag() * aM2X)
                                                                     - 0.5 * qt.spre(aM2Y.dag() * aM2X)),
                                           'exp(-i*deltaP*t)'])
                else:
                    self.c_op_list += \
                        [[2 * qt.lindblad_dissipator(aM1X) + 2 * qt.lindblad_dissipator(aM1Y),
                          'kappa1'],
                         [2 * qt.lindblad_dissipator(aM2X) + 2 * qt.lindblad_dissipator(aM2Y),
                          'kappa2'],
                         [2 * (qt.sprepost(aM1Y, aM1X.dag()) - 0.5 * qt.spost(aM1X.dag() * aM1Y) - 0.5 * qt.spre(
                             aM1X.dag() * aM1Y)),
                          'kappa1 * exp(i*deltaP*t)'],
                         [2 * (qt.sprepost(aM2Y, aM2X.dag()) - 0.5 * qt.spost(aM2X.dag() * aM2Y) - 0.5 * qt.spre(
                             aM2X.dag() * aM2Y)),
                          'kappa2 * exp(i*deltaP*t)'],
                         [2 * (qt.sprepost(aM1X, aM1Y.dag()) - 0.5 * qt.spost(aM1Y.dag() * aM1X) - 0.5 * qt.spre(
                             aM1Y.dag() * aM1X)),
                          'kappa1 * exp(-i*deltaP*t)'],
                         [2 * (qt.sprepost(aM2X, aM2Y.dag()) - 0.5 * qt.spost(aM2Y.dag() * aM2X) - 0.5 * qt.spre(
                             aM2Y.dag() * aM2X)),
                          'kappa2 * exp(-i*deltaP*t)']]

                # Spontaneous decay
                spont_decay_ops = []

                if not self.reconfigurable_decay_rates:
                    for g, x, r in self.atom.get_spontaneous_emission_channels():
                        try:
                            # r * spont_decay_ops.append(np.sqrt(2 * self.atom.gamma) *
                            spont_decay_ops.append(np.sqrt(r * 2 * self.atom.gamma) *
                                                   qt.tensor(
                                                       qt.basis(self.atom.M, self.atom.get_state_id(g)) *
                                                       qt.basis(self.atom.M, self.atom.get_state_id(x)).dag(),
                                                       qt.qeye(self.cavity.N),
                                                       qt.qeye(self.cavity.N)))
                        except KeyError:
                            pass

                else:
                    for g, x, r in self.atom.get_spontaneous_emission_channels():
                        try:
                            spont_decay_ops.append(np.sqrt(r * 2) *
                                                   qt.tensor(
                                                       qt.basis(self.atom.M, self.atom.get_state_id(g)) *
                                                       qt.basis(self.atom.M, self.atom.get_state_id(x)).dag(),
                                                       qt.qeye(self.cavity.N),
                                                       qt.qeye(self.cavity.N)))
                        except KeyError:
                            pass
                    spont_decay_ops = [[sum(spont_decay_ops), 'sqrt_gamma']]

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
                    '''.format(g, x, laser_coupling.deltaM)))
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
                    '''.format(g, x, cavity_coupling.deltaM)))
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
                                raise Exception(textwrap.dedent('''\
                                deltaM must be +/-1 for a cavity-coupled transition.\
                                Transition {0} -> {1} has {2}.'''.format(g, x, deltaM)))

                            self.hams.append(H_coupling)

        def _get_dummy_coupling(self):
            M, N = self.atom.M, self.cavity.N
            return qt.tensor(qt.qobj.Qobj(np.zeros((M, M))), qt.qobj.Qobj(np.zeros((N, N))),
                             qt.qobj.Qobj(np.zeros((N, N))))

# todo: make color ordering the same for all results
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

    # TODO: have plot function take argument to give the output plot an overall title
    class _ExperimentalResults(ABC):

        def __init__(self, output, compiled_hamiltonian, verbose=False):
            self.output = output
            self.compiled_hamiltonian = compiled_hamiltonian
            self.args = self.compiled_hamiltonian.args_hams
            self.ketbras = self.compiled_hamiltonian.states.ketbras
            self.verbose = verbose

            self.tStep = np.mean(np.ediff1d(self.output.times))

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
        def get_total_cavity_emission(self, *args):
            raise NotImplementedError()

        @abstractmethod
        def get_cavity_number(self, *args):
            raise NotImplementedError()

        @abstractmethod
        def get_atomic_population(self, *args):
            raise NotImplementedError()

        # @abstractmethod
        # def get_spontaneous_emission(self, *args):
        #     raise NotImplementedError()
        #
        # @abstractmethod
        # def get_total_spontaneous_emission(self, *args):
        #     raise NotImplementedError()

        @abstractmethod
        def plot(self, *args):
            raise NotImplementedError()

        @abstractmethod
        def _plot_cavity_summary(self, *args):
            raise NotImplementedError()

        def get_spontaneous_emission(self, i_output=[]):
            sp_op = self.atomic_operators.get_sp_op()
            return qt.expect(sp_op, self._get_output_states(i_output))

        def get_total_spontaneous_emission(self):
            exp_sp = self.get_spontaneous_emission()
            n_sp = np.trapz(exp_sp, dx=self.tStep)
            return n_sp

        def _plot_atomic_populations(self, atom_states):
            # If no atom_states were asked for explicitly, return as we don't want this plot.
            if atom_states == []:
                return None

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

            f, (ax) = plt.subplots(1, 1, sharex=True, figsize=(12, 11 / 4))

            ax.set_title('\\textbf{Atomic state}', fontsize=16)

            for lab, exp_at in exp_at_ground:
                ax.plot(t, exp_at, label='${}$'.format(lab))
            for lab, exp_at in exp_at_excited:
                ax.plot(t, exp_at, '--', label='${}$'.format(lab))

            ax.set_xlabel('Time')
            ax.set_ylabel('Population')
            ax.legend(loc=1)

            return f

        def _chop_plot_array(self, arr, tol=1e-10):
            '''
            Chop small values in the array to zero.
            :param arr: The array (or list of arrays) to be plotted.
            :param tol: The absolute tolerance before cropping to zero.
            :return: Value-chopped array (or list of arrays).
            '''

            def chop_arr(a):
                a.real[abs(a.real) < tol] = 0.0
                # a.imag[abs(a.imag) < tol] = 0.0
                return a

            if type(arr) in [list, tuple]:
                return [chop_arr(a) for a in arr]
            else:
                return chop_arr(arr)

            return arr

    class _ExperimentalResultsSingle(_ExperimentalResults):

        def get_cavity_emission(self, i_output=[]):
            return np.abs(qt.expect(self.emission_operators.get(), self._get_output_states(i_output)))

        def get_total_cavity_emission(self):
            return np.trapz(self.get_cavity_emission(), dx=self.tStep)

        def get_cavity_number(self, i_output=[]):
            return np.abs(qt.expect(self.number_operators.get(), self._get_output_states(i_output)))

        def get_atomic_population(self, states=[], i_output=[]):
            at_ops = self.atomic_operators.get_at_op(states)
            return np.abs(qt.expect(at_ops, self._get_output_states(i_output)))

        def _plot_cavity_summary(self, abs_tol=1e-10):
            exp_an = self._chop_plot_array(self.get_cavity_number())
            exp_em = self._chop_plot_array(self.get_cavity_emission())

            t = self.output.times

            # Plot the results
            f, ((ax1, ax2)) = plt.subplots(1, 2, sharex=True, figsize=(12, 11. / 4))

            ax1.set_title('\\textbf{Cavity population}', fontsize=16)
            ax2.set_title('\\textbf{Cavity emission}', fontsize=16, fontweight='bold')

            ax1.plot(t, exp_an)
            ax1.set_ylabel('Cavity mode population')

            ax2.plot(t, exp_em)
            ax2.set_ylabel('Cavity emission rate, $1/\mu s$')

            return f, [[exp_an], [exp_em]]

        def plot(self, atom_states=None, abs_tol=1e-10):
            f1, [[exp_an], [exp_em]] = self._plot_cavity_summary(abs_tol)
            f2 = self._plot_atomic_populations(atom_states)

            n_ph = np.trapz(exp_em, dx=self.tStep)
            n_sp = self.get_total_spontaneous_emission()

            ph_em_str = '\\textbf{' + 'Photon emission: {}'.format(np.round(n_ph, 3)) + '}'
            sp_em_str = '\\textbf{' + 'Spontaneous emission: {}'.format(np.round(n_sp, 3)) + '}'

            summary_str = "\n".join([ph_em_str, sp_em_str])

            a1x0 = f1.axes[0].get_position().x0

            f1.text(a1x0, 1.02, summary_str, wrap=True,
                    fontsize=14, style='oblique',
                    verticalalignment='bottom', horizontalalignment='left', multialignment='left',
                    bbox={'facecolor': '#EAEAF2',  # sns darkgrid default background color
                          'edgecolor': 'black',
                          'capstyle': 'round'})

            return f1, f2

    # TODO: allow nested lists to be passed to atom-states for producing multiple plots.
    class _ExperimentalResultsBiref(_ExperimentalResults):

        def get_cavity_emission(self, R_ZL, i_output=[]):
            if type(R_ZL) != np.matrix:
                raise ValueError("R_ZL must be a numpy matrix.")
            emP_t, emM_t = self.emission_operators.get(self.output.times, R_ZL)

            emP, emM = np.abs(np.array(
                [qt.expect(list(an_list), state) for an_list, state in
                 zip(zip(emP_t, emM_t), self._get_output_states(i_output))]
            )).T

            return emP, emM
            # return list(zip(*[iter(ems)] * 2))

        def get_total_cavity_emission(self, R_ZL):
            emP, emM = self.get_cavity_emission(R_ZL)
            return np.trapz(emP, dx=self.tStep), np.trapz(emM, dx=self.tStep)

        def get_cavity_number(self, R_ZL, i_output=[]):
            if type(R_ZL) != np.matrix:
                raise ValueError("R_ZL must be a numpy matrix.")
            anP_t, anM_t = self.number_operators.get(self.output.times, R_ZL)

            anP, anM = np.abs(np.array(
                [qt.expect(list(an_list), state) for an_list, state in
                 zip(zip(anP_t, anM_t), self._get_output_states(i_output))]
            )).T

            return anP, anM

        def get_atomic_population(self, states=[], i_output=[]):
            at_ops = self.atomic_operators.get_at_op(states)
            return np.abs(qt.expect(at_ops, self._get_output_states(i_output)))

        def get_spontaneous_emission(self, i_output=[]):
            sp_op = self.atomic_operators.get_sp_op()
            return qt.expect(sp_op, self._get_output_states(i_output))

        def get_total_spontaneous_emission(self):
            exp_sp = self.get_spontaneous_emission()
            n_sp = np.trapz(exp_sp, dx=self.tStep)
            return n_sp

        def _plot_cavity_summary(self, R_ZL, basis_name, basis_labels, abs_tol=1e-10):
            exp_an1, exp_an2 = self._chop_plot_array(self.get_cavity_number(R_ZL))
            exp_em1, exp_em2 = self._chop_plot_array(self.get_cavity_emission(R_ZL))

            lab1, lab2 = basis_labels

            t = self.output.times

            # Plot the results
            f, ((ax1, ax2)) = plt.subplots(1, 2, sharex=True, figsize=(12, 11. / 4))

            ax1.set_title(basis_name, loc='left', fontweight='bold', fontsize=14)

            ax1.plot(t, exp_an1, label=lab1)
            ax1.plot(t, exp_an2, label=lab2)
            ax1.set_ylabel('Cavity mode population')
            ax1.legend(loc=1)

            ax2.plot(t, exp_em1, label=lab1)
            ax2.plot(t, exp_em2, label=lab2)
            ax2.set_ylabel('Cavity emission rate, $1/\mu s$')
            ax2.legend(loc=1)

            return f, [[exp_an1, exp_an2], [exp_em1, exp_em2]]

        def plot(self, atom_states=None, pol_bases=['atomic', 'cavity'], abs_tol=1e-10):
            '''
            Plot a summary of the simulation results.
            :param atom_states: The displayed atomic populations.  Default behaviour is automatically configured
            to choose sensible states.
            :return:
            '''
            configured_bases_aliases = {'cavity': ['cavity', 'cav', 'c'],
                                        'atomic': ['atomic', 'atom', 'a'],
                                        'mirror': ['mirror', 'mir', 'm'],
                                        'lab': ['lab', 'linear', 'l'],
                                        'circ': ['circ', 'circular']}

            def __get_pol_basis_info(basis):
                # Make basis string lowercase.
                basis = basis.lower()
                if basis in configured_bases_aliases['cavity']:
                    return self.compiled_hamiltonian.cavity.R_CL, 'Cavity basis', ['X', 'Y']
                elif basis in configured_bases_aliases['atomic']:
                    return self.compiled_hamiltonian.atom.R_AL, 'Atomic basis', ['$+$', '$-$']
                elif basis in configured_bases_aliases['mirror']:
                    return self.compiled_hamiltonian.cavity.R_ML, 'Mirror basis', ['$M_1$', '$M_2$']
                elif basis in configured_bases_aliases['lab']:
                    return np.matrix([[1, 0], [0, 1]]), 'Lab basis', ['H', 'V']
                elif basis in configured_bases_aliases['circ']:
                    return np.sqrt(1 / 2) * np.matrix([[1, i], [i, 1]]), 'Circularly polarised basis', ['$\sigma^{+}$',
                                                                                                        '$\sigma^{-}$']
                else:
                    raise KeyError(textwrap.dedent('''\
                        Invalid polarisation bases keyword entered: {0}.\
                        Valid values are {1}.'''.format(basis, list(configured_bases_aliases.values()))))

            pol_bases_info = []
            for basis in pol_bases:
                if type(basis) is str:
                    pol_bases_info.append(__get_pol_basis_info(basis))
                elif type(basis) in [list, tuple]:
                    pol_bases_info.append(basis)
                else:
                    raise Exception(textwrap.dedent('''\
                        Unrecognised pol_bases option {0}.  Should be either:\
                        \t- A recognised polarisation basis keyword: {1}\
                        \t- A list of the form [[2x2 Rotation matrix from basis to lab]\
                                                Basis name
                                                [Basis state label 1, Basis state label 2]]'''.format(basis,
                                                                                                      list(
                                                                                                          configured_bases_aliases.values()))))

            f_list = []
            emm_summary_str_list = []
            n_ph = None

            # Helper function to format basis labels into the string format to show as ket's in the plots.
            def ketstr(s, cap="$"):
                if s[0] == '$':
                    s = s[1:]
                if s[-1] == '$':
                    s = s[:-1]
                return cap + "|" + s + "\\rangle" + cap

            for R_ZL, basis_name, basis_labels in pol_bases_info:

                basis_labels = [ketstr(x) for x in basis_labels]

                f, [[exp_an1, exp_an2], [exp_em1, exp_em2]] = self._plot_cavity_summary(R_ZL, basis_name, basis_labels,
                                                                                        abs_tol)
                f_list.append(f)
                n_1 = np.trapz(exp_em1, dx=self.tStep)
                n_2 = np.trapz(exp_em2, dx=self.tStep)
                if n_ph == None:
                    n_ph = n_1 + n_2

                emm_summary_str_list.append('$\\rightarrow$ Photon emission in {0}, {1}: {2}, {3}'.format(
                    basis_labels[0], basis_labels[1], np.round(n_1, 3), np.round(n_2, 3)))

            # Plot the atomic states summary.
            f2 = self._plot_atomic_populations(atom_states)

            # If a cavity summary plot exists:
            #   1. add the column titles to the first figure,
            #   2. add the summary text of the photon emission.
            if f_list != []:
                ax1, ax2 = f_list[0].axes
                ttl1 = ax1.set_title('\\textbf{Cavity population}', fontsize=16, fontweight='bold')
                ttl2 = ax2.set_title('\\textbf{Cavity emission}', fontsize=16, fontweight='bold')
                for ttl in [ttl1, ttl2]:
                    ttl.set_position([.5, 1.08])

                n_sp = self.get_total_spontaneous_emission()

                ph_em_str = '\\textbf{' + 'Total photon emission: {}'.format(np.round(n_ph, 3)) + '}'
                sp_em_str = '\\textbf{' + 'Total spontaneous emission: {}'.format(np.round(n_sp, 3)) + '}'

                # Create summary string out of flatten list of summaries
                summary_str = "\n".join(
                    [item for sublist in [[ph_em_str], emm_summary_str_list, [sp_em_str]] for item in sublist])

                if f_list != []:
                    f_top = f_list[0]
                else:
                    f_top = f2

                x0 = f_top.axes[0].get_position().x0

                f_top.text(x0, 1.09, summary_str, wrap=True,
                           fontsize=14, style='oblique', usetex=True,
                           verticalalignment='bottom', horizontalalignment='left', multialignment='left',
                           bbox={'facecolor': '#EAEAF2',  # sns darkgrid default background color
                                 'edgecolor': 'black',
                                 'capstyle': 'round'})

            return f_list + [f2]
