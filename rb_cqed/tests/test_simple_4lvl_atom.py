from unittest import TestCase

import numpy as np
from rb_cqed.rb_cqed import ExperimentalRunner, Atom4lvl, Cavity, LaserCoupling, CavityCoupling

# Set the absolute tolerance of the tests.
abs_tol = 1e-4

class TestVacuumRabiOscillations(TestCase):
    '''
    TestVacuumRabiOscillations
    '''

    @classmethod
    def setUpClass(cls):
        cls.atom4lvl = Atom4lvl(gamma=0)
        cls.cav = Cavity(g=5 * 2. * np.pi,
                         kappa=0. * 2. * np.pi)

        cls.cavity_coupling = CavityCoupling(g0=cls.cav.g,
                                             g='gP', x='x',
                                             deltaC=0 * 2 * np.pi,
                                             deltaM=[1, -1],
                                             couple_off_resonance=False)

        cls.psi0 = ['x', 0]
        cls.t_length = 1
        cls.n_steps = 501

    def test_vacuum_rabi_oscillations(self):

        runner = ExperimentalRunner(atom=self.atom4lvl,
                                    cavity=self.cav,
                                    laser_couplings=[],
                                    cavity_couplings=self.cavity_coupling,
                                    verbose=True)

        results = runner.run(self.psi0, self.t_length, self.n_steps)

        pop_gP, pop_x = results.get_atomic_population(['gP', 'x'])[:,
                        [0, 5, 10, 15, 20, 25, 50, 75, 100, 200, 300, 400, 500]]

        exp_pop_gP = np.array([0., 0.09549, 0.34549, 0.65451, 0.90451, 1.,
                               0., 1., 0., 0., 0, 0, 0])

        for meas, exp in zip([pop_gP, pop_x], [exp_pop_gP, 1 - exp_pop_gP]):
            for a, b in zip(meas, exp):
                self.assertAlmostEqual(a, b, delta=abs_tol)

    def test_detuned_cavity_vacuum_rabi_oscillations(self):
        pass

    def test_damped_vacauum_rabi_oscillations(self):
        pass