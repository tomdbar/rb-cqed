from unittest import TestCase

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
