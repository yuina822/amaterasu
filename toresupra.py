import math

import numpy as np
import numpy.linalg as nplin
import scipy.integrate as spi
import scipy.constants as sc

"""
References
----------
..[1] E. Witrant, E. Joffrin, S. Brémond, G. Giruzzi, D. Mazon, O. Barana and P. Moreau,
"A control-oriented model of the current profile in tokamak plasma",
Plasma Physics and Controlled Fusion, vol. 49, no. 7, pp. 1075--1105, 2007.

..[2] F. B. Argomedo, E. Witrant, C. Prieur, S. Brémond, R. Nouailletas, and J. Artaud,
"Lyapunov-Based Distributed Control of the Safety Factor Profile in a Tokamak Plasma",
Nuclear Fusion, vol. 53, no. 3, pp. 5--33, 2013.

..[3] F. Kazarian-Vibert, X. Litaudon, D. Moreau, R. Arslanbekov, G. T. Hoang, and Y. Peysson,
“Full steady-state operation in tore supra,”
Plasma Physics and Controlled Fusion, vol. 38, no. 12, pp. 2113--2131, 1996.
"""

# Major radius [m]
R0 = 2.34

# Minor radius [m]
a = 0.78


class ToreSupra(object):
    def __init__(self):
        # Toroida magnetic field at center [T]
        self.Bphi0 = 3.69

        # Plasma inductance [H]
        self.Lp = 20.3e-6

        # Plasma resistance [Ω]
        self.Rp = 5.0e-6

        # Ohmic inductance [H]
        self.Loh = 0.58

        # Ohmic resistance [Ω]
        self.Roh = 29.0e-3

        # Mutual inductance [H]
        self.M = 2.8e-3

        # Exponential peaking coefficient of the electron density
        self.gamman = 1.5

        # Exponential peaking coefficient of the q-profile
        self.gammaq = 3.7

    def ion_cyclotron_resonant_heating_power(self):
        """
        Ion cyclotron resonance heating power

        """
        pass

    def lower_hybrid_power(self):
        """
        Lower hybrid power

        """
        pass

    def parallel_refrection_index(self):
        """
        Parallel refrection index

        """
        pass

    def ohmic_voltage(self, Ip, Ioh):
        """
        Ohmic Voltage

        """
        pass

    def confinement_efficiency(self, Ip, Wth):
        """
        Confinement efficiency

        Parameters
        ----------
        Ip : float
            Total plasma current
        Wth : float
            Plasma thermal energy
    
        Returns
        -------
        betatheta : float
            Confinement efficiency

        """
        mu_0 = sc.mu_0

        betatheta = 8.0*Wth / (3.0*mu_0*R0*Ip**2)

        return betatheta

    def normalized_internal_inductance(self, Ip, dpsidr, r):
        """
        Normalized internal inductance

        Parameters
        ----------
        Ip : float
            Total plasma current
        dpsidr : ndarray
            Magnetic flux of the poloidal field
            differentiated by normalized radius
        r : ndarray
            Normaized radius

        Returns
        -------
        li : float
            Normalized internal inductance

        """
        mu_0 = sc.mu_0

        def f(i): return r[i] * dpsidr[i]**2

        y = f(np.arange(len(r)))
        li = 8*math.pi**2 * spi.simps(y, x=r) / (mu_0**2 * R0**2 * Ip**2)

        return li

    def initial_safety_factor_profile(self, q0c, q0e, r):
        """
        Initial safety factor profile

        Parameters
        ----------
        Ip0 : float
            Initial total plasma current
        q0c : float
            Initial q-value at the center
        r : ndarray
            Normalized radius

        Returns
        -------
        q0 : ndarray
            The initial safety factor

        """
        gammaq = self.gammaq
        Bphi0 = self.Bphi0
        mu_0 = sc.mu_0

        q0 = (q0c - q0e)*(1 - r**gammaq) + q0e

        return q0

    def initial_poloidal_flux(self, psi0e, q0c, q0e, r):
        """
        Initial magnetic flux of the poloidal field

        Parameters
        ----------
        Ip0 : float
            Initial total plasma current
        psi0e : float
            Initial magnetic flux of the poloidal field at the edge
        q0c : float
            Initial q-value at the center
        r : ndarray
            Normalized radius

        Returns
        -------
        psi0 : ndarray
            The initial magnetic flux profile of the poloidal field

        """
        Bphi0 = self.Bphi0

        def f(x):
            return x / self.initial_safety_factor_profile(q0c, q0e, x)

        y = np.array([spi.quad(f, r[i], 1)[0] for i in np.arange(len(r))])
        psi0 = a**2 * Bphi0 * y + psi0e

        return psi0

    def plasma_effective_charge(self):
        """
        Plasma effective charge

        Reterns
        -------
        Zeff : float
            plasma effective charge

        """
        Zeff = 1.8

        return Zeff

    def electron_line_average_density(self):
        """
        Electron line average density

        Returns
        -------
        elad : float
            electron line average density

        """
        elad = 1.45e+19

        return elad

    def total_input_power(self):
        """
        Total input power

        Returns
        -------
        Ptot : float
            Total input power

        """
        Picrh = self.ion_cyclotron_resonant_heating_power()
        Plh = self.lower_hybrid_power()

        Ptot = Picrh + Plh

        return Ptot

    def alphalh(self, Ip):
        """
        Amplitude parameter of electron temperature profile
        if lower hybrid current drive power is not 0

        Parameters
        ----------
        Ip : float
            Total plasma current

        Returns
        -------
        alh : float
            Amplitude paramter electron temperature profile

        """
        Bphi0 = self.Bphi0
        e = sc.e

        Nll = self.parallel_refrection_index()
        Picrh = self.ion_cyclotron_resonant_heating_power()
        Ptot = self.total_input_power()

        alh = (e*1.0e+19)**-0.87 * (Ip * 1.0e-6)**-0.43 * \
                Bphi0**0.63 * Nll**0.25 * (1.0 + Picrh/Ptot)**0.15

        return alh

    def betalh(self, Ip):
        """
        Dilation parameter of electron temperature profile
        if lower hybrid current drive power is not 0

        Parameters
        ----------
        Ip : float
            Total plasma current

        Returns
        -------
        blh : float
            Dilation paramter of electron temperature profile

        """
        Bphi0 = self.Bphi0
        e = sc.e

        Nll = self.parallel_refrection_index()
        elad = self.electron_line_average_density()

        blh = - (e*1.0e+19)**3.88 * (Ip * 1.0e-6)**0.31 * \
            Bphi0**-0.86 * (elad * 1.0e-19)**-0.39 * Nll**-1.15

        return blh

    def gammalh(self, Ip):
        """
        Translation parameter of electron temperature profile
        if lower hybrid current drive power is not 0

        Parameters
        ----------
        Ip : float
            Total plasma current

        Returns
        -------
        glh : float
            Translation parameter

        """
        Bphi0 = self.Bphi0
        e = sc.e

        Nll = self.parallel_refrection_index()
        Picrh = self.ion_cyclotron_resonant_heating_power()
        Ptot = self.total_input_power()

        glh = (e*1.0e+19)**1.77 * (Ip*1.0e-6)**1.4 * Bphi0**-1.76 * \
                Nll**-0.45 * (1.0 + Picrh/Ptot)**-0.54

        return glh

    def density_ratio(self):
        """
        The density ratio

        Returns
        -------
        ani : float
            The density ratio

        """
        Zeff = self.plasma_effective_charge()

        ani = (7.0 - Zeff)/6.0

        if not 0.0 < ani < 1.0:
            raise ValueError('The density ratio must be from 0.0 to 1.0')

        return ani

    def electron_density_profile(self, r):
        """
        Electron density profile

        Parameters
        ----------
        r : ndarray
            Normalized radius

        Returns
        -------
        ne: ndarray
            Electron density profile

        """
        gamman = self.gamman
        elad = self.electron_line_average_density()

        ne = (gamman + 1.0) * (1.0 - r**gamman) * elad / gamman

        return ne

    def ion_density_profile(self, r):
        """
        Ion density profile

        Parameters
        ----------
        r : ndarray
            Normalized radius

        Returns
        -------
        ni : ndarray
            Ion density profile

        """
        alphani = self.density_ratio()
        ne = self.electron_density_profile(r)

        ni = alphani * ne

        return ni

    def ion_ratio_to_electron_temperature(self, Ip):
        """
        The ratio of ion to electron temperature

        Parameters
        ----------
        Ip : float
            Total plasma current

        Returns
        -------
        alphaTi : float
            The ratio of ion to electron temperature

        """
        Bphi0 = self.Bphi0

        elad = self.electron_line_average_density()
        Picrh = self.ion_cyclotron_resonant_heating_power()
        Ptot = self.total_input_power()
        Plh = self.lower_hybrid_power()

        alphaTi = 1.0 - 0.31 * (((Ip * 1.0e-6)/Bphi0)**-0.38 * \
                (elad * 1.0e-19)**-0.90 * (1.0 + Picrh/Ptot)**-1.62 * \
                (1.0 + Plh/Ptot)**1.36)

        if alphaTi < 0:
            raise ValueError('The ratio of ion to electron temperature \
                                must be positive or zero')

        return alphaTi

    def normalized_directivity(self):
        """
        Normalized directivity

        Returns
        -------
        Dn : float
            Normalized directivity

        """
        Nll = self.parallel_refrection_index()

        Dn = 2.03 - 0.63*Nll

        if Dn < 0:
            raise ValueError('Normalized directivity must be positive or zero')

        return Dn

    def lower_hybrid_current_drive_effiency(self, Ip):
        """
        Lower hybrid current drive effiency

        Paramters
        ---------
        Ip : float
            Total plasma current

        Returns
        -------
        etalh : float
            Lower hybrid current effiency

        """
        Dn = self.normalized_directivity()
        Zeff = self.plasma_effective_charge()
        tauth = self.thermal_energy_confinement_time(Ip)

        # etalh = 3.39 * Dn**0.26 * tauth**0.46 * Zeff**-0.13 * 1.0e+19
        etalh = 1.18 * Dn**0.55 * (Ip*1.0e-6)**0.43 * Zeff**-0.24

        return etalh

    def lower_hybrid_current(self, Ip):
        """
        Lower hybrid current

        Parameters
        ----------
        Ip : float
            Total plasma current

        Returns
        -------
        Ilh : float
            Lower hybrid current

        """
        etalh = self.lower_hybrid_current_drive_effiency(Ip)
        Plh = self.lower_hybrid_power()
        elad = self.electron_line_average_density()

        Ilh = etalh * Plh / (R0 * elad * 1.0e-19)

        return Ilh

    def lower_hybrid_current_density_profile(self, Ip, r):
        """
        Lower hybird current density profile

        Paramters
        ---------
        Ip : float
            Total plasma current
        r : ndarray
            Normalized radius

        Returns
        -------
        jlh : ndarray
            Lower hybrid current density

        """
        Bphi0 = self.Bphi0

        elad = self.electron_line_average_density()
        Plh = self.lower_hybrid_power()
        Nll = self.parallel_refrection_index()
        Ilh = self.lower_hybrid_current(Ip)

        mu = 0.2 * Bphi0**-0.39 * (Ip*1.0e-6)**0.71 * \
                (elad*1.0e-19)**-0.02 * (Plh*1.0e-6)**0.13 * Nll**1.2
        w = 0.53 * Bphi0**-0.24 * (Ip*1.0e-6)**0.57 * \
                (elad*1.0e-19)**-0.08 * (Plh*1.0e-6)**0.13 * \
                Nll**0.39
        sigmalh = ((mu - w)**2)/(2.0 * math.log(2.0))

        def f(x): return x * np.exp(-(mu - x)**2 / (2.0*sigmalh))

        result = spi.quad(f, 0, 1)[0]
        varthetalh = Ilh/(2.0 * math.pi * a**2 * result)
        jlh = varthetalh * np.exp(-(mu - r)**2 / (2.0*sigmalh))

        return jlh

    def temperature_profile_amplitude(self, Ip, Wth, r):
        """
        Temperature profile amplitude

        Parameters
        ----------
        Ip : float
            Total plasma current
        Wth : float
            Plasma thermal energy
        r : ndarray
            Normalized radius

        Returns
        -------
        ATe : float
            Temperature profile amplitude

        """
        e = sc.e

        alphani = self.density_ratio()
        alphaTi = self.ion_ratio_to_electron_temperature(Ip)
        alh = self.alphalh(Ip)
        blh = self.betalh(Ip)
        glh = self.gammalh(Ip)
        edp = self.electron_density_profile

        def f(x): return (edp(x) * 1.0) * x * alh / \
                        (1.0 + np.exp(-blh*(x - glh)))

        nea = spi.quad(f, 0.0, 1.0)[0]
        scrA = 1.0/(6.0 * (math.pi * a)**2 * R0 * \
                    (e*1.0) * (1.0 + alphani * alphaTi) * nea)
        ATe = Wth * scrA

        return ATe

    def electron_temperature_profile(self, Ip, Wth, r):
        """
        Electron temperature profile

        Parameters
        ----------
        Ip : float
            Total plasma current
        Wth : float
            Plasma thermal energy
        r : ndarray
            Normalized radius

        Returns
        -------
        Te : ndarray
            Temerature profile of electron

        """
        alh = self.alphalh(Ip)
        blh = self.betalh(Ip)
        glh = self.gammalh(Ip)

        ATe = self.temperature_profile_amplitude(Ip, Wth, r)

        Te = (alh * ATe)/(1.0 + np.exp(-blh * (r - glh)))

        return Te

    def ion_temperature_profile(self, Ip, Wth, r):
        """
        Ion temperature profile

        Parameters
        ----------
        Ip : float
            Total plasma current
        Wth : float
            Plasma thermal energy
        r : ndarray
            Normalized radius

        Returns
        -------
        Ti : ndarray
            Ion temprature profile

        """
        Te = self.electron_temperature_profile(Ip, Wth, r)
        alphaTi = self.ion_ratio_to_electron_temperature(Ip)

        Ti = alphaTi * Te

        return Ti

    def safety_factor_profile(self, dpsidr, r):
        """
        Safety factor profile

        Parameters
        ----------
        dpsidr : ndarray
            Magnetic flux of the poloidal field
            differentiated by normalized radius
        r : ndarray
            Normalized radius

        Returns
        -------
        q : ndarray
            Safety factor profile

        """
        Bphi0 = self.Bphi0

        q = - Bphi0 * a**2 * r / dpsidr

        return q

    def trapped_banana_regime_particles(self, r):
        """
        The fraction of trapped particles in banana regime

        Parameters
        ----------
        r : ndarray
            Normalized radius

        Returns
        -------
        ft : ndarray
            The fraction of trapped particles in banana regime

        """
        epsilon = a/R0
        ft = 1.0 - (1.0 - r*epsilon)**2 * \
            ((1.0 - (r*epsilon)**2)**-0.5) / (1.0 + 1.46 * np.sqrt(r*epsilon))

        return ft

    def parallel_conductivity(self, Ip, Wth, dpsidr, r):
        """
        Plasma parallel conductivity

        Parameters
        ----------
        Ip : float
            Total plasma current
        Wth : float
            Plasma thermal energy
        dpsidr : ndarray
            Magnetic flux of the poloidal field
            differentiated by normalized radius
        r : ndarray
            Normalized radius

        Returns
        -------
        etall : ndarray
            Plasma parallel conductivity

        """
        e, epsilon_0, m_e = sc.e, sc.epsilon_0, sc.m_e

        Te = self.electron_temperature_profile(Ip, Wth, r)
        edp = self.electron_density_profile(r)
        q = self.safety_factor_profile(dpsidr, r)
        ft = self.trapped_banana_regime_particles(r)
        Zeff = self.plasma_effective_charge()

        epsilon = a/R0

        xi = 0.58 + 0.2 * Zeff
        lambdae = (3.4/Zeff)*((1.13 + Zeff)/(2.67 + Zeff))
        l = 31.318 + np.log(Te/np.sqrt(edp))
        taue = (12.0 * math.pi**1.5 * m_e**0.5 * epsilon_0**2 * Te**1.5) / \
                (e**2.5 * math.sqrt(2.0) * edp * np.log(l))
        s0 = edp * e**2 * taue/m_e
        alphae = np.sqrt(e * Te / m_e)
        nue = (R0 * q)/(((r*epsilon)**1.5) * alphae * taue)
        cr = (0.56/Zeff) * (3.0 - Zeff)/(3.0 + Zeff)
        etall = s0 * lambdae * \
            (1.0 - ft/(1.0 + xi * nue)) * (1.0 - (cr * ft)/(1.0 + xi * nue))

        return etall

    def ohmic_current_profile(self, Ip, Wth, dpsidr, r):
        """
        Ohmic current profile

        Parameters
        ----------
        Ip : float
            Total plasma current
        Wth : float
            Plasma thermal energy
        dpsidr : ndarray
            Maginetic flux of the poloidal field
            differentiated by normalized radius

        Returns
        -------
        joh : ndarray
            Ohmic current profile

        """
        pc = self.parallel_conductivity(Ip, Wth, dpsidr, r)

        joh = - pc * dpsidr / R0

        return joh

    def initial_plasma_thermal_energy(self, Ip0):
        """
        Initial plasma thermal energy

        Paramters
        ---------
        Ip0 : float
            Initial total plasma current

        Returns
        -------
        Wth0 : float
            Initial plasma thermal energy

        """
        Wth0 = self.total_input_power() * \
            self.thermal_energy_confinement_time(Ip0)

        return Wth0

    def parallel_resistivity(self, Ip, Wth, dpsidr, r):
        """
        Plasma parallel resistivity

        Parameters
        ----------
        Ip : float
            Total plasma current
        Wth : float
            Plasma thermal energy
        dpsidr : ndarray
            Maginetic flux of the poloidal field
            differentiated by normalized radius

        Returns
        -------
        etall : ndarray
            Plasma parallel resistivity

        """
        etall = 1.0/self.parallel_conductivity(Ip, Wth, dpsidr, r)

        return etall

    def thermal_energy_confinement_time(self, Ip):
        """
        Thermal energy confinement time

        Paramters
        ---------
        Ip : float
            Total plasma current

        Returns
        -------
        taue : float
            Thermal energy confinement time

        """
        Bphi0 = self.Bphi0

        elad = self.electron_line_average_density()
        Ptot = self.total_input_power()
        Plh = self.lower_hybrid_power()

        taue = 0.135 * (Ip * 1.0e-6)**0.94 * Bphi0**-0.15 * \
                (elad * 1.0e-19)**0.78 * (1 + Plh/Ptot)**0.13 * \
                (Ptot * 1.0e-6)**-0.78

        return taue

    def bootstrap_current_profile(self, Ip, Wth, dpsidr, r):
        """
        Bootstrap current profile

        Parameters
        ----------
        Ip : float
            Total plasma current
        Wth : float
            Plasma thermal energy
        dpsidr : ndarray
            Maginetic flux of the poloidal field
            differentiated by normalized radius
        r : ndarray
            Normalized radius

        Returns
        -------
        jbs : ndarray
            Bootstrap current profile

        """
        gamman = self.gamman
        e = sc.e

        Zeff = self.plasma_effective_charge()
        ft = self.trapped_banana_regime_particles(r)
        alphaTi = self.ion_ratio_to_electron_temperature(Ip)
        alphani = self.density_ratio()
        Te = self.electron_temperature_profile(Ip, Wth, r)
        Ti = self.ion_temperature_profile(Ip, Wth, r)
        ne = self.electron_density_profile(r)
        ni = self.ion_density_profile(r)
        alphalh = self.alphalh(Ip)
        betalh = self.betalh(Ip)
        gammalh = self.gammalh(Ip)

        dTedr = alphalh*betalh*np.exp(betalh*(r - gammalh)) / \
            (1 + np.exp(betalh*(r - gammalh)))**2
        dTidr = alphaTi * dTedr
        dnedr = - (gamman + 1) * r**(gamman - 1) * ne
        dnidr = alphani * dnedr

        xt = ft/(1.0 - ft)
        De = 1.414*Zeff + Zeff**2 + xt*(0.754 + 2.657*Zeff + 2*Zeff**2) + \
                xt**2 * (0.348 + 1.243*Zeff * Zeff**2)
        A1 = xt*(0.754 + 2.21*Zeff + Zeff**2 + \
                xt*(0.348 + 1.243*Zeff + Zeff**2))/De
        A2 = xt*(0.884 + 2.074*Zeff)/De
        alphai = 1.172/(1 - 0.462*xt)
        jbs = e*R0*((A1 - A2)*ne*dTedr + A1*Te*dnedr +
            A1*(1-alphai)*ni*dTidr + A1*Ti*dnidr)/dpsidr

        return jbs

    def bootstrap_current(self, Ip, Wth, dpsidr, r):
        """
        Bootstrap current

        Parameters
        ----------
        Ip : float
            Total plasma current
        Wth : float
            Plasma thermal energy
        dpsidr : ndarray
            Maginetic flux of the poloidal field
            differentiated by normalized radius
        r : ndarray
            Normalized radius

        Returns
        -------
        Ibs : float
            Bootstrap current

        """
        jbs = self.bootstrap_current_profile

        def f(i): return r[i] * jbs(Ip, Wth, dpsidr[i], r[i])

        y = f(np.arange(len(r)))
        Ibs = 2*math.pi*a**2 * spi.simps(y, x=r)

        return Ibs

    def noninductive_effective_current(self, Ip, Wth, dpsidr, r):
        """
        Noninductive effective current

        Paramters
        ---------
        Ip : float
            Total plasma current
        r : ndarray
            Normalized radius

        Returns
        -------
        jin : ndarray
            Noninductive effective current profile

        """
        jlh = self.lower_hybrid_current_density_profile(Ip, r)
        jbs = self.bootstrap_current_profile(Ip, Wth, dpsidr, r)

        jin = jlh + jbs

        return jin

    def total_plasma_current(self, I, Wth, psi, r):
        """
        Total plasma current model

        Parameters
        ----------
        I : ndarray
            Total plasma current and ohmic current
        psi : ndarray
            Magnetic flux of the poloidal field
        r : ndarray
            Normalized radius

        Returns
        -------
        dI : ndarray
            Total plasma current and ohmic current differentiated by time

        """
        Ip, Ioh = I[0], I[1]
        Lp, Loh, Rp, Roh, M = self.Lp, self.Loh, self.Rp, self.Roh, self.M

        if len(r) != len(psi):
            raise ValueError('r and psi length must be equal')

        dpsidr = (psi[2:] - psi[:-2])/(r[2:] - r[:-2])

        Plh = self.lower_hybrid_power()
        Voh = self.ohmic_voltage(Ip, Ioh)

        etalh = self.lower_hybrid_current_drive_effiency(Ip)
        elad = self.electron_line_average_density()
        Ilh = self.lower_hybrid_current(Ip)
        Ibs = self.bootstrap_current(Ip, Wth, dpsidr, r[1:-1])

        LM = [[Lp, M],
            [M, Loh]]
        R = [[-Rp, 0],
            [0, -Roh]]
        G = [[Rp, 0],
            [0, 1]]
        U = [Ilh + Ibs, Voh]

        dI = np.dot(nplin.inv(LM), (np.dot(R, I) + np.dot(G, U)))

        return dI

    def plasma_thermal_energy(self, Wth, Ip):
        """
        Plasma thermal energy model

        Parameters
        ----------
        Wth : float
            Plasma thermal energy
        Ip : float
            Total plasma current

        Returns
        -------
        dWth : float
            Plasma thermal energy differentiated by time

        """
        Ptot = self.total_input_power()
        tauth = self.thermal_energy_confinement_time(Ip)

        dWth = Ptot - Wth/tauth

        return dWth

    def poloidal_flux(self, psi, I, Wth, r):
        """
        Poloidal flux model

        Parameters
        ----------
        psi : ndarray
            Magnetic flux of the poloidal field
        I : ndarray
            Total plasma current and ohmic current
        Wth : float
            Plasma thermal energy
        r : ndarray
            Normalized radius

        Returns
        -------
        dpsidr : ndarray
            Poloidal flux differentiated by time

        """
        mu_0 = sc.mu_0
        M, Lp = self.M, self.Lp
        Ip = I[0]

        if len(psi) != len(r):
            raise ValueError('Invalid length')

        dr = (r[2:] - r[:-2])
        dpsidr = (psi[2:] - psi[:-2])/dr
        ddpsidr = (psi[2:] - 2*psi[1:-1] + psi[:-2])/dr**2
        etall = self.parallel_resistivity(Ip, Wth, dpsidr, r[1:-1])
        jni = self.noninductive_effective_current(Ip, Wth, dpsidr, r[1:-1])
        dI = self.total_plasma_current(I, Wth, psi, r)
        Vloop = M*dI[1] - Lp*dI[0] 
        dpsidt = etall * ((ddpsidr + dpsidr/r[1:-1])/(mu_0*a**2) + R0*jni)
        dpsidt0 = dpsidt[0]

        dpsidt = np.r_[dpsidt0, dpsidt, Vloop]

        return dpsidt
