from unittest import TestCase, SkipTest

import numpy as np
import copy
from rb_cqed.rb_cqed import ExperimentalRunner, Atom4lvl, Cavity, LaserCoupling, CavityCoupling

# Set the absolute tolerance of the tests.
abs_tol = 1e-4

class TestVacuumRabiOscillations(TestCase):
    '''
    TestVacuumRabiOscillations
    '''

    @classmethod
    def setUpClass(cls):
        cls.atom4lvl_default = Atom4lvl(gamma=0)

        cls.cav_default = Cavity(g=5 * 2. * np.pi,
                                 kappa=0. * 2. * np.pi)

        cls.cavity_coupling_default = CavityCoupling(g0=cls.cav_default.g,
                                                     g='gP', x='x',
                                                     deltaC=0 * 2 * np.pi,
                                                     deltaM=[1, -1],
                                                     couple_off_resonance=False)

        cls.psi0 = ['x', 0]
        cls.t_length = 1
        cls.n_steps = 501

    def setUp(self):
        self.atom4lvl = copy.copy(self.atom4lvl_default)
        self.cav = copy.copy(self.cav_default)
        self.cavity_coupling = copy.copy(self.cavity_coupling_default)

    def _run_experiment(self):
        return ExperimentalRunner(
            atom=self.atom4lvl,
            cavity=self.cav,
            laser_couplings=[],
            cavity_couplings=self.cavity_coupling,
            verbose=True
        ).run(self.psi0, self.t_length, self.n_steps)

    def test_vacuum_rabi_oscillations(self):
        results = self._run_experiment()
        pop_gP, pop_x = results.get_atomic_population(['gP', 'x'])[:,
                        [0, 5, 10, 15, 20, 25, 50, 75, 100, 200, 300, 400, 500]]

        exp_pop_gP = np.array([0., 0.09549, 0.34549, 0.65451, 0.90451, 1.,
                               0., 1., 0., 0., 0, 0, 0])

        for meas, exp in zip([pop_gP, pop_x], [exp_pop_gP, 1 - exp_pop_gP]):
            for a, b in zip(meas, exp):
                self.assertAlmostEqual(a, b, delta=abs_tol)

    def test_detuned_cavity_vacuum_rabi_oscillations(self):
        self.cavity_coupling.deltaC = 10 * 2 * np.pi
        results = self._run_experiment()

        pop_gP, pop_x = results.get_atomic_population(['gP', 'x'])[:,
                        [0, 5, 10, 15, 20, 25, 50, 75, 100, 200, 300, 400, 500]]

        exp_pop_gP = np.array([0., 0.09237, 0.30122, 0.47223, 0.47901, 0.31656,
                               0.46455, 0.06919, 0.13173, 0.3881, 0.49893, 0.34742,
                               0.09325])

        for meas, exp in zip([pop_gP, pop_x], [exp_pop_gP, 1 - exp_pop_gP]):
            for a, b in zip(meas, exp):
                self.assertAlmostEqual(a, b, delta=abs_tol)

    def test_damped_vacauum_rabi_oscillations(self):
        self.atom4lvl.gamma = 0.3 * 2 * np.pi
        self.cav.kappa = 0.3 * 2 * np.pi

        results = self._run_experiment()

        pop_gP, pop_g, pop_gM, pop_x = results.get_atomic_population(['gP', 'g', 'gM', 'x'])[:,
                                       [0, 5, 10, 15, 20, 25, 50, 75, 100, 200, 300, 400, 500]]

        exp_pop_gP = np.array([0., 0.10508, 0.35039, 0.63736, 0.85911, 0.94055,
                               0.20901, 0.85415, 0.35238, 0.51816, 0.59617, 0.63286,
                               0.65013])
        exp_pop_g = np.array([0., 0.01194, 0.02132, 0.02705, 0.02938, 0.02973,
                              0.05253, 0.07292, 0.08857, 0.13024, 0.14984, 0.15907,
                              0.16341])
        exp_pop_gM = np.array([0., 0.01194, 0.02132, 0.02705, 0.02938, 0.02973,
                               0.05253, 0.07292, 0.08857, 0.13024, 0.14984, 0.15907,
                               0.16341])
        exp_pop_x = np.array([1., 0.87104, 0.60697, 0.30854, 0.08212, 0.,
                              0.68592, 0., 0.47049, 0.22136, 0.10415, 0.049,
                              0.02305])

        for meas, exp in zip([pop_gP, pop_g, pop_gM, pop_x], [exp_pop_gP, exp_pop_g, exp_pop_gM, exp_pop_x]):
            for a, b in zip(meas, exp):
                self.assertAlmostEqual(a, b, delta=abs_tol)

class TestRabiOscillations(TestCase):
    '''
    TestRabiOscillations
    '''

    @classmethod
    def setUpClass(cls):
        cls.atom4lvl_default = Atom4lvl(gamma=0)

        cls.cav_default = Cavity(g=5 * 2. * np.pi,
                                 kappa=0. * 2. * np.pi)


        # Our pulse will be on for 0.75us and then switch off.
        # Note that couple_off_resonance=False means we only consider the 'gM' <-> 'x' transition coupled.
        length_pulse = 0.75
        cls.laser_coupling_default = LaserCoupling(omega0=5 * 2 * np.pi,
                                                   g='gM', x='x',
                                                   deltaL=0 * 2 * np.pi,
                                                   deltaM=[1, -1],
                                                   pulse_shape='np.piecewise(t, [t<length_pulse], [1,0])',
                                                   args_ham={"length_pulse": length_pulse},
                                                   couple_off_resonance=False)

        cls.psi0 = ['gM', 0]
        cls.t_length = 1
        cls.n_steps = 501

    def setUp(self):
        self.atom4lvl = copy.copy(self.atom4lvl_default)
        self.cav = copy.copy(self.cav_default)
        self.laser_coupling = copy.copy(self.laser_coupling_default)

    def _run_experiment(self):
        return ExperimentalRunner(
            atom=self.atom4lvl,
            cavity=self.cav,
            laser_couplings=self.laser_coupling,
            cavity_couplings=[],
            verbose=True
        ).run(self.psi0, self.t_length, self.n_steps)

    def test_rabi_oscillations(self):
        results = self._run_experiment()
        pop_gM, pop_x = results.get_atomic_population(['gM', 'x'])[:,
                        [0, 5, 10, 15, 20, 25, 50, 75, 100, 200, 300, 400, 500]]

        exp_pop_gM = np.array([1., 0.97553, 0.90451, 0.79389, 0.65451, 0.5,
                               0., 0.5, 1., 0.99999, 0.99998, 0.49998,
                               0.49998])

        for meas, exp in zip([pop_gM, pop_x], [exp_pop_gM, 1 - exp_pop_gM]):
            for a, b in zip(meas, exp):
                self.assertAlmostEqual(a, b, delta=abs_tol)

    def test_rabi_oscillations_with_off_resonance_couplings(self):
        self.laser_coupling.couple_off_resonance=True

        results = self._run_experiment()

        pop_gP, pop_g, pop_gM, pop_x = results.get_atomic_population(['gP', 'g', 'gM', 'x'])[:,
                                       [0, 5, 10, 15, 20, 25, 50, 75, 100, 200, 300, 400, 500]]

        exp_pop_gP = np.array([0., 0.00015, 0.00236, 0.01145, 0.03413, 0.07728,
                               0.64457, 0.98198, 0.40085, 0.86324, 0.01915, 0.6236, 0.6236])
        exp_pop_g = np.array([0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0.])
        exp_pop_gM = np.array([1., 0.97558, 0.90527, 0.79748, 0.66465, 0.5213,
                               0.03887, 0.00008, 0.1346, 0.00502, 0.74242, 0.04424,
                               0.04424])
        exp_pop_x = np.array([0., 0.02427, 0.09237, 0.19108, 0.30122, 0.40142,
                              0.31657, 0.01794, 0.46455, 0.13174, 0.23843, 0.33216,
                              0.33216])

        for meas, exp in zip([pop_gP, pop_g, pop_gM, pop_x], [exp_pop_gP, exp_pop_g, exp_pop_gM, exp_pop_x]):
            for a, b in zip(meas, exp):
                self.assertAlmostEqual(a, b, delta=abs_tol)

class TestVStirap(TestCase):
    '''
    TestRabiOscillations
    '''

    @classmethod
    def setUpClass(cls):
        cls.atom4lvl_default = Atom4lvl(gamma=0)
        cls.cav_default = Cavity(g=5 * 2. * np.pi,
                                 kappa=5. * 2. * np.pi)

        cls.cavity_coupling_default = CavityCoupling(g0=cls.cav_default.g,
                                                     g='gP', x='x',
                                                     deltaC=0 * 2 * np.pi,
                                                     deltaM=[1, -1],
                                                     couple_off_resonance=False)

        cls.psi0 = ['gM', 0]
        cls.t_length = 1
        cls.n_steps = 501

    def setUp(self):
        self.atom4lvl = copy.copy(self.atom4lvl_default)
        self.cav = copy.copy(self.cav_default)
        self.cavity_coupling = copy.copy(self.cavity_coupling_default)

    def _run_experiment(self):
        return ExperimentalRunner(
            atom=self.atom4lvl,
            cavity=self.cav,
            laser_couplings=self.laser_coupling,
            cavity_couplings=self.cavity_coupling,
            verbose=True
        ).run(self.psi0, self.t_length, self.n_steps)

    def test_py_pulse(self):
        length_pulse = 1
        self.laser_coupling = LaserCoupling(omega0=5 * 2 * np.pi,
                                            g='gM', x='x',
                                            deltaL=0 * 2 * np.pi,
                                            deltaM=[1, -1],
                                            pulse_shape='np.piecewise(t, [t<length_pulse], [np.sin((np.pi/length_pulse)*t)**2,0])',
                                            args_ham={"length_pulse": length_pulse},
                                            couple_off_resonance=False
                                            )

        results = self._run_experiment()
        self.check_results(results)

    def test_c_pulse(self):
        pulse_c_str = \
'''
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double complex pulse_c(float t, float t_start, float t_end, float omega):
    if t_start<=t<=t_end: return sin(omega*(t-t_start))**2
    else: return 0
'''

        length_pulse = 1
        w_pulse = np.pi / length_pulse

        self.laser_coupling = LaserCoupling(omega0=5 * 2 * np.pi,
                                            g='gM', x='x',
                                            deltaL=0 * 2 * np.pi,
                                            deltaM=[1, -1],
                                            pulse_shape='pulse_c(t, 0, length_pulse, w_pulse)',
                                            args_ham={"length_pulse": length_pulse,
                                                      "w_pulse": w_pulse},
                                            setup_pyx=[''],
                                            add_pyx=[pulse_c_str],
                                            couple_off_resonance=False)

        results = self._run_experiment()
        self.check_results(results)

    def check_results(self, results):

        emm = results.get_total_cavity_emission()
        exp_emm = 0.99682
        self.assertAlmostEqual(emm, exp_emm, delta=abs_tol)

        pop_gM, pop_x = results.get_atomic_population(['gM', 'x'])[:,
                        [0, 50, 100, 150, 200, 250, 300, 400, 500]]

        exp_pop_gM = np.array([1., 0.99799, 0.92714, 0.61125, 0.21337, 0.04808, 0.01239, 0.00343,
                               0.00318])
        exp_pop_x = np.array([0., 0.00137, 0.02682, 0.07246, 0.04931, 0.01091, 0.00211, 0.00009,
                              0.])

        for meas, exp in zip([pop_gM, pop_x], [exp_pop_gM, exp_pop_x]):
            for a, b in zip(meas, exp):
                self.assertAlmostEqual(a, b, delta=abs_tol)