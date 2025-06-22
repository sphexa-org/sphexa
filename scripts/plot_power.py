#!/usr/bin/env python3

"""
SPH-EXA
Copyright (c) 2024 CSCS, ETH Zurich

Usage examples:
    Run a simulation on a system that provides power counters (Cray) with profiling turned on
    $ sphexa --init evrard --glass 50c.h5 -n 200 -s 10 --profile
    $ plot_power.py profile.h5
"""

__author__ = "Sebastian Keller (sebastian.f.keller@gmail.com)"

import numpy as np
import h5py
import sys

import matplotlib
import matplotlib.pyplot as plt

font = {'size': 16}
matplotlib.rc('font', **font)


def convertToIntervals(data):
    """ Convert numRanks x numInterations x numMeasurmentsPerIterations points to intervals """
    return data[:, :, 1:] - data[:, :, :-1]


def convertBarToLine(data):
    """ Convert a series bar plots data points to equivalent line plot data """
    return np.append(np.array([0]), np.repeat(data, 2))[:-1]


def removeZeros(data):
    """ Remove all elements in axis 0 where the sum across axes 1 and 2 is zero """
    return data[np.sum(data, axis=(1, 2)) != 0.0]


def loadField(fname, what, stepnr):
    """ Load /Step#stepnr/what and /Step#stepnr/what_timeStams from file and reshape into
        numRanks x numIterations x numMeasurementsPerIteration
    """
    h5file = h5py.File(fname, "r")
    h5step = h5file["/Step#" + str(stepnr)]
    numRanks = h5step.attrs["numRanks"][0]
    numIterations = h5step.attrs["numIterations"][0]
    raw = np.array(h5step[what])
    numPoints = len(raw)
    ret = np.reshape(raw, (numRanks, numIterations, int(numPoints / numRanks / numIterations)))
    return removeZeros(ret)


def loadEnergy(fname, what, stepnr):
    """ Load what and what_timeStamps from file """

    what_ts = what + "_timeStamps"

    dt = 1e-6 * loadField(fname, what_ts, stepnr)
    # average across ranks
    dt_avg = np.average(convertToIntervals(dt), axis=0)
    # scan to get total time and repeat data points to get flat lines within intervals
    timeline = convertBarToLine(np.cumsum(dt_avg.flatten()))

    energy = loadField(fname, what, stepnr)
    # average across ranks
    energy_avg = np.average(convertToIntervals(energy), axis=0)
    power = (energy_avg / dt_avg)
    # repeat to get flat lines within intervals
    powerline = np.repeat(power.flatten(), 2)

    print("Loaded energy data \"{0}\" with dimensions".format(what), energy.shape,
          "= (numCounters x numIterations x numMeasurementsPerIteration)")

    fields = {}
    fields["x"] = timeline
    fields["y"] = powerline
    fields["numCounters"] = energy.shape[0]

    return fields


def plotModulePower(fname):
    modulePower = loadEnergy(fname, "acc", 2)
    numAccelerators = modulePower["numCounters"]

    fig = plt.figure(figsize=(15, 11))
    plt.title("SPH-EXA, power consumption, averaged over %d accelerators" % numAccelerators)
    plt.xlabel("runtime [s]", fontsize=16)
    plt.ylabel("average module power [W]", fontsize=16)
    plt.plot(modulePower["x"], modulePower["y"], linewidth=2, color="g", label="module power")
    plt.savefig("sphexa-accel-power.png", bbox_inches="tight")


def plotNodePower(fname):
    nodePower = loadEnergy(fname, "node", 1)
    numNodes = nodePower["numCounters"]

    fig = plt.figure(figsize=(15, 11))
    plt.title("SPH-EXA, power consumption, averaged over %d nodes" % numNodes)
    plt.xlabel("runtime [s]", fontsize=16)
    plt.ylabel("average node power [W]", fontsize=16)

    plt.axhline(y=np.max(nodePower["y"]), color="b", linestyle="--", linewidth=1,
                label="%dW max measurement" % np.max(nodePower["y"]))

    simlabel = "simulation, (average power per node for a %d-node simulation)" % numNodes
    plt.plot(nodePower["x"], nodePower["y"], linewidth=2, color="g", label=simlabel)

    plt.legend(loc="lower left")
    plt.savefig("sphexa-node-power.png", bbox_inches="tight")


if __name__ == "__main__":
    plotModulePower(sys.argv[1])
    plotNodePower(sys.argv[1])
