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

__program__ = "compare_greshochan.py"
__author__ = "Lukas Schmidt, Jose A. Escartin (ja.escartin@gmail.com)"
__version__ = "0.1.0"

from argparse import ArgumentParser
import math
import h5py
import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys


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


def analyticalVelocity(R1, radius):
    psi = radius / R1
    if psi <= 1.0:
        return psi
    elif psi <= 2.0:
        return 2 - psi
    return 0


def compute2DRadiiAndVt(h5File, step):
    """ Load XYZ coordinates and compute their radii and Tangential Velocity"""
    x = loadH5Field(h5File, "x", step)
    y = loadH5Field(h5File, "y", step)
    vx = loadH5Field(h5File, "vx", step)
    vy = loadH5Field(h5File, "vy", step)
    radii = np.sqrt(x ** 2 + y ** 2)
    vt = np.sqrt(vx ** 2 + vy ** 2)
    print("Calculated Radii and Tangential Velocity in %s particles" % len(x))
    return radii, vt


def computeL1Error(radii, vt, R1):
    return sum(abs(vt - np.array([analyticalVelocity(R1, num) for num in radii])) / len(radii))


def plotRadialProfile(props, xSim, ySim, xSol, ySol):
    if props["xLogScale"] == "true":
        plt.xscale('log')

    if props["yLogScale"] == "true":
        plt.yscale('log')

    plt.scatter(xSim, ySim, s=0.1, label="Simulation, L1 = %3f" % props["L1"], color="C0")
    plt.plot(xSol, ySol, label="Solution ", color="C1")

    plt.xlabel("r")
    plt.ylabel(props["ylabel"])
    plt.draw()
    plt.title(props["title"] + " : N = %8d, t = %.3f, iteration = %6d" % (len(xSim), props["time"], props["step"]))
    plt.legend(loc="upper right")
    plt.savefig(props["fname"], format="png")
    plt.figure().clear()


def createVelocityPlot(vt, t, step, radii, R1):
    lin = np.linspace(0, 0.7, num=200)
    vtSol = np.array([analyticalVelocity(R1, num) for num in lin])
    L1 = computeL1Error(radii, vt, R1)
    print("Velocity L1 error", L1)

    props = {"ylabel": "vt", "title": "Tangential Velocity", "fname": "greshochan_velocity_%.3f.png" % t, "time": t,
             "step": step, "xLogScale": "false", "yLogScale": "false", "L1": L1}
    plotRadialProfile(props, radii, vt, lin, vtSol)


def createCmapPlot(h5File, vt, t, step):
    x = loadH5Field(h5File, "x", step)
    y = loadH5Field(h5File, "y", step)
    z = loadH5Field(h5File, "z", step)
    h = loadH5Field(h5File, "h", step)

    cond = np.arange(len(x))[abs(z / h) < 2]
    x_ = x[cond]
    y_ = y[cond]
    vt_ = vt[cond]

    plt.scatter(x_, y_, c=vt_, s=0.1, cmap="inferno")
    plt.title("Gresho-Chan Vortex: N = %8d, t = %.3f" % (len(vt), t))
    plt.xlabel("x")
    plt.ylabel("y")
    cbar = plt.colorbar()
    cbar.set_label("tangential velocity")
    plt.savefig("greshochan_colourmap_%.3f.png" % (t), format="png")
    plt.figure().clear()


if __name__ == "__main__":
    parser = ArgumentParser(description='Plot paper solutions against SPH simulations')
    parser.add_argument('simFile', help="SPH simulation HDF5 file")
    parser.add_argument('-t', '--time', type=float, dest="time", help="Physical time for comparison")
    parser.add_argument('-r1', '-R1', type=float, nargs='?', default=0.2, dest="R1",
                        help="Radius at which tangential velocity is one, 0.2 is the default in SPH-EXA")

    args = parser.parse_args()

    time = args.time
    R1 = args.R1

    # Get HDF5 Simulation file
    h5File = h5py.File(args.simFile, "r")

    # simulation time of each step that was written to file
    timesteps = loadTimesteps(h5File)

    # the actual iteration number of each step that was written
    stepNumbers = loadStepNumbers(h5File)

    # output time specified instead of step, locate closest output step
    stepIndex = determineTimestep(time, timesteps)
    step = stepNumbers[stepIndex]
    tReal = timesteps[stepIndex]
    print("The closest timestep to the specified solution time of t=%s is step=%s" % (time, step))

    hdf5_step = np.searchsorted(stepNumbers, step)

    # Calulate Radius and RadialVelocity
    radii = None
    vt = None
    try:
        radii, vt = compute2DRadiiAndVt(h5File, hdf5_step)
    except KeyError:
        print("Could not load radii, input file does not contain all fields \"x, y, z, h, vx, vy, vz\"")
        sys.exit(1)
    try:
        createVelocityPlot(vt, time, step, radii, R1)
    except KeyError:
        print("Could not plot velocity profile, input does not contain fields \"vx, vy, vz\"")
    # try:
    createCmapPlot(h5File, vt, time, hdf5_step)
    # except KeyError:
    #    print("Could not plot Colormap, input does not contain required fields")
