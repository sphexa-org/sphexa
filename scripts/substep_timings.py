#!/usr/bin/env python3

"""Plot timings of time-step components as stacked bars

This script takes an output file of an SPH-EXA simulation and plots
one bar for each time step. Each bar consists of stacked bars for the
substeps of time step, such as domain::sync, halo exchange, SPH, gravity, ...

Parameters to tweak:
    - range for rolling averages (numSmooth)
    - maximum step number to plot (maxPlotStep)

Author: Sebastian Keller, <sebastian.f.keller@gmail.com>
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

clrs = ['darkred',      # domain-sync,
        # 'yellow',      # FindNeighbors,
        'grey',         # XMass,
        'red',          # syncHalos,
        'purple',       # Norm&Grad,
        # 'green',       # EOS,
        'red',          # syncHalos,
        'lime',         # IAD,
        'red',          # syncHalos,
        'olivedrab',    # AVSwitches,
        'red',          # syncHalos,
        'forestgreen',  # momentumEnergy,
        'navy',         # Upsweep,
        'black',        # Gravity,
        'firebrick',    # Timestep,
        # 'gold',        # updateQuantities,
        # 'chocolate',   # updateSmoothingLength
        # 'lightblue'    # TotalTimestep
        ]


def extractValue(linestring, qname):
    # linestring looks like this: # MomentumAndEnergy: 0.071481s
    i1 = linestring.find(qname)
    if i1 > -1:
        i2 = linestring.find(" ", i1 + len(qname))
        return float(linestring[i2:].split()[0].strip('s').strip(','))
    else:
        return None


def extractQuantity(lines, qname):
    return np.array(
        [extractValue(ll, qname) for ll in lines
         if extractValue(ll, qname) is not None]
    )


def extractFromFile(fname, quantities):
    lines = open(fname, 'r').readlines()
    results = {}
    for qname in quantities:
        dataset = extractQuantity(lines, qname)
        if qname == "synchronizeHalos":
            mult = 4
            syncHalos = dataset.reshape(mult,
                                        int(dataset.shape[0] / mult + 0.5))
            for i in range(mult):
                results[qname + str(i)] = syncHalos[i, :]
        else:
            results[qname] = dataset

    return results


def rollingAverage(dataset, numAvg):
    kernel = np.ones(numAvg) / numAvg
    if dataset.size == 0:
        return np.array([0])
    else:
        return np.convolve(dataset, kernel, mode="valid")


def plot(data, quantities):
    fig, ax = plt.subplots(figsize=(10, 7.5))
    xdata = np.arange(len(data[quantities[0]]))
    bottom = np.zeros(len(xdata))
    for i, q in enumerate(quantities):
        plt.bar(xdata, data[q], bottom=bottom, width=1.0, label=q,
                color=clrs[i])
        bottom += data[q]

    ax.legend(loc="upper center", ncol=5)
    plt.show()


if __name__ == "__main__":
    fname = sys.argv[1]
    quantities = ["domain::sync", "XMass", "synchronizeHalos", "Gradh",
                  "IadVelocityDivCurl", "AVswitches", "MomentumAndEnergy",
                  "Upsweep", "Gravity", "Timestep"]
    plotOrder = ["domain::sync", "XMass", "synchronizeHalos0", "Gradh",
                 "synchronizeHalos1", "IadVelocityDivCurl",
                 "synchronizeHalos2", "AVswitches", "synchronizeHalos3",
                 "MomentumAndEnergy", "Upsweep", "Gravity", "Timestep"]
    results = extractFromFile(fname, quantities)

    # rolling average range in steps
    numSmooth = 5
    # maximum time step to include in the plot
    maxPlotStep = len(results["domain::sync"])

    resultsSmoothed = {}
    for k, v in results.items():
        print(k, v.shape)
        resultsSmoothed[k] = rollingAverage(v, numSmooth)[:maxPlotStep]

    plot(resultsSmoothed, plotOrder)
