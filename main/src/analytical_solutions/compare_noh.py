#!/usr/bin/env python3

# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Command line utility for compare analytical solutions of some SPH-EXA test
simulations.

Reference Noh solution:
- "Errors for Calculations of Strong Shocks Using an Artificial Viscosity and
   an Artificial Heat Flux", W.F. Noh. JCP 72 (1987), 78-120

Usage examples:
    $ python ./compare_noh.py --help
    $ python ./compare_noh.py dump_noh.h5 --time 0.018
    $ python ./compare_noh.py dump_noh.h5 --step 100
"""

__program__ = "compare_noh.py"
__author__ = "Jose A. Escartin (ja.escartin@gmail.com)"
__version__ = "0.1.0"

from argparse import ArgumentParser

import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys


def nohShockFront(gamma, vel0, time):
    """ position of the shock front """
    return 0.5 * (gamma - 1) * abs(vel0) * time


def nohRho(xgeom, gamma, rho0, vel0, r, time):
    """ analytical density at radius r for given time"""
    r2 = nohShockFront(gamma, vel0, time)
    if r > r2:
        return rho0 * (1.0 - vel0 * time / r) ** (xgeom - 1)
    else:
        return rho0 * ((gamma + 1) / (gamma - 1)) ** xgeom


def nohU(gamma, u0, vel0, r, time):
    """ analytical internal energy at radius r for given time"""
    r2 = nohShockFront(gamma, vel0, time)
    if r > r2:
        return u0
    else:
        return 0.5 * vel0**2


def nohP(xgeom, gamma, rho0, u0, p0, vel0, r, time):
    """ analytical pressure at radius r for given time"""
    r2 = nohShockFront(gamma, vel0, time)
    if r > r2:
        return p0
    else:
        return (gamma - 1) * nohRho(xgeom, gamma, rho0, vel0, r, time) * nohU(gamma, u0, vel0, r, time)


def nohVel(gamma, vel0, r, time):
    """ analytical velocity magnitude at radius r for given time"""
    r2 = nohShockFront(gamma, vel0, time)
    if r > r2:
        return abs(vel0)
    else:
        return 0


def nohCs(xgeom, gamma, rho0, u0, p0, vel0, cs0, r, time):
    """ analytical pressure at radius r for given time"""
    r2 = nohShockFront(gamma, vel0, time)
    if r > r2:
        return cs0
    else:
        return np.sqrt(
            gamma * nohP(xgeom, gamma, rho0, u0, p0, vel0, r, time) / nohRho(xgeom, gamma, rho0, vel0, r, time))


def loadH5Field(h5File, what, step):
    """ Load the specified particle field at the given step, returns an array of length numParticles """
    return np.array(h5File["Step#%s/%s" % (step, what)])


def loadTimesteps(h5File):
    """ Load simulation times of each recorded time step """
    return np.array(sorted([h5File[step].attrs["time"][0] for step in list(h5File["/"])]))


def loadStepNumbers(h5File):
    """ Load the iteration count of each recorded time step """
    return np.array(sorted([h5File[step].attrs["iteration"][0] for step in list(h5File["/"])]))


def determineTimestep(time, timesteps):
    """ Return the timestep with simulation time closest to the specified time """
    return np.argmin(np.abs(timesteps - time))


def computeRadii(h5File, step):
    """ Load XYZ coordinates and compute their radii """
    x = loadH5Field(h5File, "x", step)
    y = loadH5Field(h5File, "y", step)
    z = loadH5Field(h5File, "z", step)
    print("Loaded %s particles" % len(x))
    return np.sqrt(x ** 2 + y ** 2 + z ** 2)


def plotRadialProfile(props, xSim, ySim, xSol, ySol):
    plt.scatter(xSim, ySim, s=0.1, label="Simulation, L1 = %3f" % props["L1"], color="C0")
    plt.plot(xSol, ySol, label="Solution", color="C1")
    plt.xlabel("r")
    plt.ylabel(props["ylabel"])
    plt.draw()
    plt.title(props["title"] + ", N = %3e, t = %3f" % (len(xSim), props["time"]))
    plt.legend(loc="upper right")
    plt.savefig(props["fname"], format="png")
    plt.figure().clear()


def createDensityPlot(h5File, attrs, radii, time, step):
    rho = loadH5Field(h5File, "rho", step)

    rSol = np.linspace(attrs["r0"], attrs["r1"], 1000)
    rhoSol = np.vectorize(nohRho)(attrs["dim"], attrs["gamma"], attrs["rho0"], attrs["vr0"], rSol, time)

    rhoSolFull = np.vectorize(nohRho)(attrs["dim"], attrs["gamma"], attrs["rho0"], attrs["vr0"], radii, time)
    L1 = sum(abs(rhoSolFull - rho)) / len(rho)

    props = {"ylabel": "rho", "title": "Density", "fname": "noh_density_%4f.png" % time, "time": time, "L1": L1}
    plotRadialProfile(props, radii, rho, rSol, rhoSol)

    print("Density L1 error", L1)


def createPressurePlot(h5File, attrs, radii, time, step):
    p = loadH5Field(h5File, "p", step)

    rSol = np.linspace(attrs["r0"], attrs["r1"], 1000)
    pSol = np.vectorize(nohP)(attrs["dim"], attrs["gamma"], attrs["rho0"], attrs["u0"], attrs["p0"], attrs["vr0"],
                              rSol, time)

    pSolFull = np.vectorize(nohP)(attrs["dim"], attrs["gamma"], attrs["rho0"], attrs["u0"], attrs["p0"], attrs["vr0"],
                                  radii, time)
    L1 = sum(abs(pSolFull - p)) / len(p)

    props = {"ylabel": "p", "title": "Pressure", "fname": "noh_pressure_%4f.png" % time, "time": time, "L1": L1}
    plotRadialProfile(props, radii, p, rSol, pSol)

    print("Pressure L1 error", L1)


def createVelocityPlot(h5File, attrs, radii, time, step):
    vx = loadH5Field(h5File, "vx", step)
    vy = loadH5Field(h5File, "vy", step)
    vz = loadH5Field(h5File, "vz", step)

    vr = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

    rSol = np.linspace(attrs["r0"], attrs["r1"], 1000)
    vSol = np.vectorize(nohVel)(attrs["gamma"], attrs["vr0"], rSol, time)

    vSolFull = np.vectorize(nohVel)(attrs["gamma"], attrs["vr0"], radii, time)
    L1 = sum(abs(vSolFull - vr)) / len(vr)

    props = {"ylabel": "vel", "title": "Velocity", "fname": "noh_velocity_%4f.png" % time, "time": time, "L1": L1}
    plotRadialProfile(props, radii, vr, rSol, vSol)

    print("Velocity L1 error", L1)


def createEnergyPlot(h5File, attrs, radii, time, step):
    temp = loadH5Field(h5File, "temp", step)
    mui = 10.0
    cv = 1.5 * 8.317e7 / mui
    u = cv * temp

    rSol = np.linspace(attrs["r0"], attrs["r1"], 1000)
    uSol = np.vectorize(nohU)(attrs["gamma"], attrs["u0"], attrs["vr0"], rSol, time)

    uSolFull = np.vectorize(nohU)(attrs["gamma"], attrs["u0"], attrs["vr0"], radii, time)
    L1 = sum(abs(uSolFull - u)) / len(u)

    props = {"ylabel": "u", "title": "Energy", "fname": "noh_energy_%4f.png" % time, "time": time, "L1": L1}
    plotRadialProfile(props, radii, u, rSol, uSol)

    print("Energy L1 error", L1)


if __name__ == "__main__":
    parser = ArgumentParser(description='Plot analytical solutions against SPH simulations')
    parser.add_argument('simFile', help="SPH simulation HDF5 file")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-s', '--step', type=int, dest="step", help="plot solution at the given simulation step")
    group.add_argument('-t', '--time', type=float, dest="time", help="simulation time for which to plot solution")
    args = parser.parse_args()

    h5File = h5py.File(args.simFile, "r")

    step = args.step

    # simulation time of each step that was written to file
    timesteps = loadTimesteps(h5File)
    # the actual iteration number of each step that was written
    stepNumbers = loadStepNumbers(h5File)

    if step is None:
        # output time specified instead of step, locate closest output step
        stepIndex = determineTimestep(args.time, timesteps)
        step = stepNumbers[stepIndex]
        print("The closest timestep to the specified time of %s is step %s at t=%s" % (
            args.time, step, timesteps[stepIndex]))

    hdf5_step = np.searchsorted(stepNumbers, step)
    time = timesteps[hdf5_step]

    attrs = h5File.attrs

    radii = None
    try:
        radii = computeRadii(h5File, hdf5_step)
    except KeyError:
        print("Could not load radii, input file does not contain fields \"x, y, z\"")
        sys.exit(1)

    try:
        createDensityPlot(h5File, attrs, radii, time, hdf5_step)
    except KeyError:
        print("Could not plot density profile, input does not contain field \"rho\"")

    try:
        createPressurePlot(h5File, attrs, radii, time, hdf5_step)
    except KeyError:
        print("Could not plot pressure profile, input does not contain field \"p\"")

    try:
        createVelocityPlot(h5File, attrs, radii, time, hdf5_step)
    except KeyError:
        print("Could not plot velocity profile, input does not contain fields \"vx, vy, vz\"")

    try:
        createEnergyPlot(h5File, attrs, radii, time, hdf5_step)
    except KeyError:
        print("Could not plot velocity profile, input does not contain fields \"temp\"")
