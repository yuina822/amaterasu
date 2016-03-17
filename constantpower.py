#!/usr/bin/env python3

import math
import sys
import configparser

import numpy as np
import toresupra as ts

from toresupra import ToreSupra
from tqdm import tqdm
from scipy.integrate import ode


class ConstantPower(ToreSupra):

    def ion_cyclotron_resonant_heating_power(self):
        return 1.5e+6

    def lower_hybrid_power(self):
        return 2.6e+6

    def parallel_refrection_index(self):
        return 1.75

    def ohmic_voltage(self, Ip, Ioh):
        return 0

    def simulation(self, t, s, r):
        I, Wth, psi = s[0:2], s[2], s[3:]

        dI = self.total_plasma_current(I, Wth, psi, r)
        dWth = self.plasma_thermal_energy(Wth, I[0])
        dpsi = self.poloidal_flux(psi, I, Wth, r)

        return np.r_[dI, dWth, dpsi]


def main():
    x = ConstantPower()

    conf = 'constantpower.conf'
    config = configparser.ConfigParser()
    config.read(conf)

    # Setting initial values
    tend = float(config['Simulation']['SimulationTime'])
    q0c = float(config['Simulation']['InitialSafetyFactorCenter'])
    q0e = float(config['Simulation']['InitialSafetyFactorEdge'])
    psi0e = float(config['Simulation']['InitialPoloidalFluxEdge'])
    Ip0 = float(config['Simulation']['InitialPlasmaCurrent'])
    N = int(config['Simulation']['Dimension'])
    tstep = float(config['Simulation']['SamplingTime'])

    t = np.linspace(0, tend, num=tend/tstep)
    r = np.linspace(0, 1, num=N)

    I0 = [Ip0, 0]
    Ii0 = 0
    Wth0 = x.initial_plasma_thermal_energy(Ip0)
    psi0 = x.initial_poloidal_flux(psi0e, q0c, q0e, r)
    tt = 0

    y0 = y = np.r_[I0, Wth0, psi0]
    sim = ode(x.simulation)
    sim.set_initial_value(y0, 0)
    sim.set_f_params(r)
    sim.set_integrator('vode', method='bdf', nsteps=1.0e+6)
    for i in tqdm(t[1:]):
        if not sim.successful():
            break
        sim.integrate(i)
        y = np.vstack((y, sim.y))
        tt = np.append(tt, sim.t)

    I, Wth, psi = y[:, 0:2], y[:, 2], y[:, 3:]
 
    np.savetxt('I.txt', I)
    np.savetxt('Wth.txt', Wth)
    np.savetxt('psi.txt', psi)
    np.savetxt('t.txt', tt)

if __name__ == '__main__':
    main()
