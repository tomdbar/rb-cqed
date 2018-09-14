import numpy as np
import scipy.constants as const
import os
import datetime
import time
import csv
import _pickle as pickle
from itertools import chain, product

from qutip import *
np.set_printoptions(threshold=np.inf)

# Global parameters
d = 3.584*10**(-29)
i = np.complex(0,1)

data_folder = './data'

class vStirap_3lvl_runner():

    def __init__(self,
                 kappa=1 * 2.*np.pi,
                 gamma = 1 * 2.*np.pi,
                 branching_ratios = [1/3,1/3,1/3],
                 alpha_CL = 1,
                 phi1_CL = 0 * np.pi / 180,
                 phi2_CL= 0 * np.pi / 180,
                 deltaP = 0,
                 deltaL = 0,
                 deltaC = 0,
                 atom_states = {"x0":0,"gM":1,"gP":2,"d":3},
                 cavity_states = [0,1]):

        self.hams = []
        self.args_hams = dict([('i',i)])

        self.kappa = kappa
        self.gamma = gamma

        self.branching_ratios=branching_ratios

        self.alpha_CL = alpha_CL
        self.beta_CL = np.sqrt(1 - self.alpha_CL ** 2)
        self.phi1_CL = phi1_CL
        self.phi2_CL = phi2_CL

        self.__configure_birefringence(self.alpha_CL, self.beta_CL, self.phi1_CL, self.phi2_CL)

        self.deltaP = deltaP
        self.deltaL = deltaL
        self.deltaC = deltaC

        self.atom_states = atom_states
        self.cavity_states = cavity_states

        self.M = len(self.atom_states)
        self.N = len(self.cavity_states)

        self.ketbras = self.__configure_ketbras()

        self.c_op_list = self.__get_c_ops(self.branching_ratios)

    def run(self, amp_drive, len_drive=1, n_steps=201, psi0 = ['u',0,0]):

        self.psi0 = psi0

        A = amp_drive
        w = np.pi / len_drive

        t, self.tStep = np.linspace(0, len_drive, n_steps, retstep=True)

        self.H, args_ham = self.__generate_Hamiltonian(A, w)
        self.args = {**args_ham, **dict([['A',A],['w',w]])}

        def ket(atom, cavH, cavV):
            return tensor(basis(self.M, self.atom_states[atom]), basis(self.N, cavH), basis(self.N, cavV))

        psi0 = ket(*self.psi0)
        output = mesolve(self.H, psi0, t, self.c_op_list, [], args=self.args)

        return output

    def __configure_ketbras(self):

        def ket(atom, cavX, cavY):
            return tensor(basis(self.M, self.atom_states[atom]), basis(self.N, cavX), basis(self.N, cavY))

        def bra(atom, cavX, cavY):
            return ket(atom, cavX, cavY).dag()

        kets, bras = {}, {}
        ketbras = {}

        s = [list(self.atom_states), self.cavity_states, self.cavity_states]
        states = list(map(list, list(product(*s))))
        for state in states:
            kets[str(state)] = ket(*state)
            bras[str(state)] = bra(*state)

        for x in list(map(list, list(product(*[states, states])))):
            ketbras[str(x)] = ket(*x[0]) * bra(*x[1])

        return ketbras

    def __configure_birefringence(self, alpha_CL, beta_CL, phi1_CL, phi2_CL):
        self.R_AL = np.sqrt(1 / 2) * np.matrix([[1, i],
                                                [i, 1]])

        self.R_CL = np.matrix([[alpha_CL * np.exp(i * phi1_CL), -beta_CL * np.exp(-i * phi2_CL)],
                          [beta_CL * np.exp(i * phi2_CL), alpha_CL * np.exp(-i * phi1_CL)]])
        self.R_LC = self.R_CL.getH()

        '''
        Rotation from Atom -> Cavity: R_LA: {|+>,|->} -> {|X>,|Y>}
        '''
        self.R_AC = self.R_LC * self.R_AL
        self.R_CA = self.R_AC.getH()

        '''
        alpha, phi for writing the atomic basis in terms of the cavity basis
        '''
        self.alpha_AC = np.abs(self.R_AC[0, 0])
        self.beta_AC = np.sqrt(1 - self.alpha_AC ** 2)
        self.phi1_AC, self.phi2_AC = np.angle(self.R_AC[0, 0]), np.angle(self.R_AC[1, 0])

        self.args_hams.update({"alpha":self.alpha_AC,
                               "beta":self.beta_AC,
                               "phi1":self.phi1_AC,
                               "phi2":self.phi2_AC})

    def __get_c_ops(self, branching_ratios):
        # Define collapse operators
        c_op_list = []

        aX = tensor(qeye(self.M), destroy(self.N), qeye(self.N))
        aY = tensor(qeye(self.M), qeye(self.N), destroy(self.N))

        # Cavity decay rate
        c_op_list.append(np.sqrt(2 * self.kappa) * aX)
        c_op_list.append(np.sqrt(2 * self.kappa) * aY)

        spontEmmChannels = [
            # |F',mF'> --> |F=1,mF=-1>
            ('gM', 'x0', branching_ratios[0]),
            ('gP', 'x0', branching_ratios[1]),
            ('d', 'x0', branching_ratios[2])
        ]

        spontDecayOps = []

        for x in spontEmmChannels:
            try:
                spontDecayOps.append(x[2] * np.sqrt(2 * self.gamma) *
                                     tensor(
                                         basis(self.M, self.atom_states[x[0]]) * basis(self.M, self.atom_states[x[1]]).dag(), qeye(self.N),
                                         qeye(self.N)))
            except KeyError:
                pass

        c_op_list += spontDecayOps

        self.sigma_spontDecayOp = sum([x.dag() * x for x in spontDecayOps])

        return c_op_list

    def __check_coupling(self,g,x):
        if not g in self.atom_states or not x in self.atom_states:
            raise Exception("Invalid atom state entered.\nConfigured states are",
                            print(self.atom_states.values()))

    '''
    Create a laser coupling.

    Parameters:
        Omega - The peak rabi frequency of the pump pulse.
        g - The ground atomic atomic level.
        x - The excited atomic level.
        omegaL - The detuning of the pump laser.
        deltaM - The angular momentum change from g --> x.  This is ignored but included
                 for consistancy with the cavityCoupling function.
        args_list - A dictionary of arguments for the qutip simulation.
        pulseShape - The shape of the pump pulse.

    Returns:
        (List of cython-ready Hamiltonian terms,
         args_list with relevant parameters added)
    '''
    def add_laser_coupling(self, Omega=1*2.*np.pi, g='gM', x='x0', omegaL=None, pulse_length=1):

        self.__check_coupling(g,x)

        omegaL = self.deltaL if omegaL is None else omegaL

        omegaL_lab = 'omegaL_{0}{1}'.format(g, x)
        pulse_shape = 'np.piecewise(t, [t<pulse_length_{0}{1}], [np.sin(w_stirap_{0}{1}*t)**2,0])'.format(g, x)

        self.args_hams[omegaL_lab] = omegaL
        self.args_hams.update([('w_stirap_{0}{1}'.format(g, x), np.pi/pulse_length),
                               ('pulse_length_{0}{1}'.format(g, x), pulse_length)])

        def kb(a, b):
            return self.ketbras[str([a, b])]

        self.hams.append([
                        [-(Omega / 2) * (
                                (kb([g, 0, 0], [x, 0, 0]) + kb([g, 0, 1], [x, 0, 1]) +
                                 kb([g, 1, 0], [x, 1, 0]) + kb([g, 1, 1], [x, 1, 1])) +
                                (kb([x, 0, 0], [g, 0, 0]) + kb([x, 0, 1], [g, 0, 1]) +
                                 kb([x, 1, 0], [g, 1, 0]) + kb([x, 1, 1], [g, 1, 1]))
                        ), '{0} * np.cos({1}*t)'.format(pulse_shape, omegaL_lab)],
                        [i * (Omega / 2) * (
                                (kb([x, 0, 0], [g, 0, 0]) + kb([x, 0, 1], [g, 0, 1]) +
                                 kb([x, 1, 0], [g, 1, 0]) + kb([x, 1, 1], [g, 1, 1])) -
                                (kb([g, 0, 0], [x, 0, 0]) - kb([g, 0, 1], [x, 0, 1]) -
                                 kb([g, 1, 0], [x, 1, 0]) - kb([g, 1, 1], [x, 1, 1]))
                        ), '{0} * np.sin({1}*t)'.format(pulse_shape, omegaL_lab)]
                         ])

    '''
    Create a cavity coupling.

    Parameters:
        g0 - The atom-cavity coupling rate.
        g - The ground atomic atomic level.
        x - The excited atomic level.
        omegaC - The detuning of the cavity resonance.
        deltaM - The angular momentum change from g --> x.
        args_list - A dictionary of arguments for the qutip simulation.

    Returns:
        (List of cython-ready Hamiltonian terms,
         args_list with relevant parameters added)
    '''
    def add_cavity_coupling(self, g0=1*2.*np.pi, g='gP', x='x0', omegaC=None, deltaM=1):
        self.__check_coupling(g, x)

        omegaC = self.deltaC if omegaC is None else omegaC

        omegaC_X = omegaC + self.deltaP / 2
        omegaC_Y = omegaC - self.deltaP / 2
        omegaC_X_lab = 'omegaC_X_{0}{1}'.format(g, x)
        omegaC_Y_lab = 'omegaC_Y_{0}{1}'.format(g, x)

        self.args_hams[omegaC_X_lab] = omegaC_X
        self.args_hams[omegaC_Y_lab] = omegaC_Y

        def kb(a, b):
            return self.ketbras[str([a, b])]

        if deltaM == 1:
            H_coupling = [
                        [-g0 * self.alpha_AC * (
                                kb([g, 1, 0], [x, 0, 0]) + kb([g, 1, 1], [x, 0, 1])
                              + kb([x, 0, 0], [g, 1, 0]) + kb([x, 0, 1], [g, 1, 1])
                        ), 'np.cos({0}*t + phi1)'.format(omegaC_X_lab)],

                        [-i * g0 * self.alpha_AC * (
                                kb([g, 1, 0], [x, 0, 0]) + kb([g, 1, 1], [x, 0, 1])
                              - kb([x, 0, 0], [g, 1, 0]) - kb([x, 0, 1], [g, 1, 1])
                        ), 'np.sin({0}*t + phi1)'.format(omegaC_X_lab)],

                        [g0 * self.beta_AC * (
                                kb([g, 0, 1], [x, 0, 0]) + kb([g, 1, 1], [x, 1, 0])
                              + kb([x, 0, 0], [g, 0, 1]) + kb([x, 1, 0], [g, 1, 1])
                        ), 'np.cos({0}*t + phi2)'.format(omegaC_Y_lab)],

                        [i * g0 * self.beta_AC * (
                                kb([g, 0, 1], [x, 0, 0]) + kb([g, 1, 1], [x, 1, 0])
                              - kb([x, 0, 0], [g, 0, 1]) - kb([x, 1, 0], [g, 1, 1])
                        ), 'np.sin({0}*t + phi2)'.format(omegaC_Y_lab)]
                        ]

        elif deltaM == -1:
            H_coupling = [
                        [-g0 * self.alpha_AC * (
                                kb([g, 0, 1], [x, 0, 0]) + kb([g, 1, 1], [x, 1, 0])
                              + kb([x, 0, 0], [g, 0, 1]) + kb([x, 1, 0], [g, 1, 1])
                        ), 'np.cos({0}*t - phi1)'.format(omegaC_X_lab)],

                        [-i * g0 * self.alpha_AC * (
                                kb([g, 0, 1], [x, 0, 0]) + kb([g, 1, 1], [x, 1, 0])
                              - kb([x, 0, 0], [g, 0, 1]) - kb([x, 1, 0], [g, 1, 1])
                        ), 'np.sin({0}*t - phi1)'.format(omegaC_X_lab)],

                        [-g0 * self.beta_AC * (
                                kb([g, 1, 0], [x, 0, 0]) + kb([g, 1, 1], [x, 0, 1])
                              + kb([x, 0, 0], [g, 1, 0]) + kb([x, 0, 1], [g, 1, 1])
                        ), 'np.cos({0}*t - phi2)'.format(omegaC_Y_lab)],

                        [-i * g0 * self.beta_AC * (
                                kb([g, 1, 0], [x, 0, 0]) + kb([g, 1, 1], [x, 0, 1])
                              - kb([x, 0, 0], [g, 1, 0]) - kb([x, 0, 1], [g, 1, 1])
                        ), 'np.sin({0}*t - phi2)'.format(omegaC_Y_lab)]
                        ]

        else:
            raise Exception("deltaM must be +/-1")

        self.hams.append(H_coupling)

    def __generate_hamiltonian(self, ham_name='H_Stirap'):
        rhs_generate(self.hams, self.c_op_list, args=self.args_hams, name=ham_name, cleanup=False)
        self.compiled_ham_name = ham_name

    def run(self, psi0=['gM',0,0], t_length=1.2, n_steps=201, ham_name='H_Stirap'):

        t, t_step = np.linspace(0, t_length, n_steps, retstep=True)

        self.hams = list(chain(*self.hams))

        if ham_name!=None:
            self.__generate_hamiltonian(ham_name)
            opts = Options(rhs_reuse=False, rhs_filename=ham_name)
        else:
            opts = Options()

        def ket(atom, cavX, cavY):
            return tensor(basis(self.M, self.atom_states[atom]), basis(self.N, cavX), basis(self.N, cavY))

        psi0 = ket(*psi0)

        return mesolve(self.hams, psi0, t, self.c_op_list, [], args=self.args_hams, options=opts)

    def cleanup(self):
        try:
            os.remove(self.compiled_ham_name + ".pyx")
            print("Removed ", self.compiled_ham_name + ".pyx")
        except:
            pass


if __name__=='__main__':

    exp_folder_name = os.path.join(data_folder, datetime.date.today().__str__())

    save_loc = os.path.join(exp_folder_name, 'test')

    print('Running exp...', end=" ")

    runner = vStirap_3lvl_runner()

    runner.add_cavity_coupling()
    runner.add_laser_coupling()

    output = runner.run()

    print('done.\nSaving...',end=" ")

    if not os.path.exists(save_loc):

        os.makedirs(save_loc)

        qsave(output, os.path.join(save_loc,'output'))

        with open(os.path.join(save_loc,'vStirap_3lvl_runner__dict__.pkl'), 'wb') as out:
            pickle.dump(runner.__dict__, out, -1)
        print('done.')

 # if __name__=='__main__':
#
#     '''
#     Single offset/power scan
#     '''
#     exp_folder_name = os.path.join('../data/nlz/17-10-23/offset scans_no nlz/500us Photons_0.7g/A14_Z14/raw', 'u2g')
#
#     params = p = load_params(os.path.join('./params new/no nlz', 'exp_params_14MHz.csv'))
#     #params = p = load_params(os.path.join('./params new', 'exp_params_14MHz.csv'))
#
#     offset = 60.9;
#
#     # for offset in range(-40,130,5):
#     for offset in [-60,-55,-50,-45]:
#     #for A in range(1,62,3):
#     # for offset in [120]:
#
#         save_loc = os.path.join(exp_folder_name, '{0}MHz'.format(offset))
#
#         print('Running exp for {0}MHz...'.format(offset), end=" ")
#
#         A = 14
#         # offset = 93
#
#         exp = Stirap_Experimental_Runner(params=params,
#                                          kappa= 3.75 * 2. * np.pi /2,
#                                          deltaL = 2*p['deltaZ']* 2*np.pi + offset*2*np.pi, # |u> -> |g>
#                                          #deltaL=-2 * p['deltaZ'] * 2 * np.pi + offset * 2 * np.pi,  # |g> -> |u>
#                                          deltaC = offset*2*np.pi)
#         #output = exp.run_g2u(amp_drive=(A*2*np.pi)/ np.sqrt((1./6)), len_drive=0.33, psi0=['g',0,0])
#         output = exp.run(amp_drive=A*2* np.pi, len_drive=0.5, psi0=['u', 0, 0])
#
#         print('done.')
#
#         print('Saving...', end=" ")
#         if not os.path.exists(save_loc):
#             os.makedirs(save_loc)
#
#         qsave(output, os.path.join(save_loc,'output'))
#
#         with open(os.path.join(save_loc,'Stirap_Experimental_Runner__dict__.pkl'), 'wb') as out:
#             pickle.dump(exp.__dict__, out, -1)
#         print('done.')
#
#     print ('Experiments finished.')

