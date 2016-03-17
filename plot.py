#!/usr/bin/env python3

import configparser
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick

import toresupra as ts

fontsize = 32
plt.rcParams['font.size'] = fontsize

def plotpsi(t, psi, linewidth=8, figsize=(16, 9)):
    fig = plt.figure(figsize=figsize)
    fig.set_tight_layout(True)
    ax = fig.add_subplot(111)
    ax.plot(t, psi, linewidth=linewidth)

    ax.set_xlabel(r'Time[$s$]')
    ax.set_ylabel(r'Poloidal flux[$T/m^2$]')
    ax.tick_params(labelsize=fontsize)
    ax.grid()
    fig = ax.get_figure()
    fig.savefig('toroidal.png')

def plotWth(t, Wth, linewidth=8, figsize=(16, 9)):
    fig = plt.figure(figsize=figsize)
    fig.set_tight_layout(True)
    ax = fig.add_subplot(111)
    ax.plot(t, Wth, linewidth=linewidth)
    ax.set_xlabel(r'Time[$s$]')
    ax.set_ylabel(r'Plasma thermal energy[$J$]')
    ax.axis([0, np.amax(t), 0, 1.25*np.amax(Wth)])
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    ax.tick_params(labelsize=fontsize)
    ax.grid()
    fig = ax.get_figure()
    fig.savefig('Wth.png') 

def plotI(t, I, linewidth=8, figsize=(16, 9)):
    fig = plt.figure(figsize=(16, 9))
    fig.set_tight_layout(True)
    ax = fig.add_subplot(111)
    plotIp, = ax.plot(t, I[:, 0], linewidth=linewidth, label='Total plasma current', linestyle='-')
    plotIoh, = ax.plot(t, I[:, 1], linewidth=linewidth, label='Ohmic current', linestyle='--')
    ax.legend(handles=[plotIp, plotIoh], loc=4)
    ax.set_xlabel(r'Time[$s$]')
    ax.set_ylabel(r'Current[$A$]')
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True)) 
    ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    ax.axis([0, np.amax(t), 1.25*np.amin(I[:,:2]), 1.25*np.amax(I[:, :2])])
    ax.tick_params(labelsize=fontsize)
    ax.grid()
    fig = ax.get_figure()
    fig.savefig('Ip-Ioh.png')

def plotq(t, r, q, linewidth=8.0, figsize=(16, 9)):
    N = len(q[0,:])
    fig = plt.figure(figsize=figsize)
    fig.set_tight_layout(True)
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'Time[$s$]')
    ax.set_ylabel('Safety factor')
    ax.axis([np.amin(t), np.amax(t), 0, np.amax(q[:, int(0.8*N)])])
    r1, = ax.plot(t, q[:, int(0.1*N)], label=r'$r=0.1$',
                    linewidth=linewidth, markersize=fontsize/2.0)
    r2, = ax.plot(t, q[:, int(0.2*N)], label=r'$r=0.2$',
                    linewidth=linewidth, markersize=fontsize/2.0)
    r4, = ax.plot(t, q[:, int(0.4*N)], label=r'$r=0.4$',
                    linewidth=linewidth, markersize=fontsize/2.0)
    r6, = ax.plot(t, q[:, int(0.6*N)], label=r'$r=0.6$',
                    linewidth=linewidth, markersize=fontsize/2.0)
    r8, = ax.plot(t, q[:, int(0.8*N)], label=r'$r=0.8$',
                    linewidth=linewidth, markersize=fontsize/2.0)
    ax.legend(handles=[r1, r2, r4, r6, r8])
    ax.tick_params(labelsize=fontsize)
    ax.grid()
    fig = ax.get_figure()
    fig.savefig('safety-factor-profile-time.png')

    fig = plt.figure(figsize=figsize)
    fig.set_tight_layout(True)
    ax = fig.add_subplot(111)
    plotInit, = ax.plot(r, q[0, :], linewidth=linewidth,
                        label='Initial profile')
    plotSteady, = ax.plot(r, q[-1, :], linewidth=linewidth,
                            label='Equilibrium profile', linestyle='--', 
                            markersize=fontsize/2.0)
    ax.legend(handles=[plotInit, plotSteady], loc=2)
    ax.set_xlabel('Normarized radius')
    ax.set_ylabel('Safety factor')
    ax.axis([np.amin(r), np.amax(r), 0, np.amax(q[-1, :])])
    ax.tick_params(labelsize=fontsize)
    ax.grid()
    fig = ax.get_figure()
    fig.savefig('safety-factor-profile.png')

def main():
    a = ts.a
    config = configparser.ConfigParser()
    config.read('constantpower.conf')

    Bphi0 = 3.69
    N = int(config['Simulation']['Dimension'])
    r = np.linspace(0, 1, num=N)
    t = np.loadtxt('t.txt')

    I = np.loadtxt('I.txt')
    Wth = np.loadtxt('Wth.txt')
    psi = np.loadtxt('psi.txt')

    dpsidr = (psi[:, 2:] - psi[:, :-2])/(r[2:] - r[:-2])
    q = - Bphi0 * a**2 * r[1:-1] / dpsidr

    r = np.linspace(0, 1, num=N-2)

    plotI(t, I)
    plotpsi(t, psi)
    plotWth(t, Wth)
    plotq(t, r, q)

if __name__ == '__main__':
    main()
