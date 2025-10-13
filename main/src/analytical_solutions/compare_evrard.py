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

References Evrard solution:

- "Beyond N-body: 3D cosmological gas dynamics", August E. Evrard.
    MNRAS 235 (1988), p. 911-934.
    3.3 Self-gravitationg collapse of a cold gas sphere (p 922)

- "On the capabilities and limits of smoothed particle hydrodynamics",
    M. Steinmetz & E. Muller. A&A 268 (1993), p. 391-410.
    3.3 Adiabatic spherical collapse of an initially isothermal gas cloud (p 405)


This routine produces 1d solutions for the evrard collapse:

    rho = pow(rho, -omega), in spherical geometry(3D)

    Normalized units:

        Time           (timeNorm)  :  pow(M_PI * M_PI / 8., 0.5)
                                    * pow(R, 1.5)
                                    * pow(Mt, -0.5)

        Density         (rhoNorm) : (3. * Mt) / (4. * M_PI * pow(R, 3.))
        Internal energy (uNorm)   : G * Mt / R
        Velocity        (velNorm) : pow(uNorm, 0.5)
        Pressure        (pNorm)   : rhoNorm * uNorm

    Initial density and internal energy distribution:

        gamma  = 5./3.
        u0     = 0.05
        u      = u0 * uNorm
        rho(r) = (M(R) / (2. * M_PI * pow(R, 2.))) * (1. / r)

Usage examples:
    $ python ./compare_evrard.py --help
    $ python ./compare_evrard.py dump_evrard.h5 --time 0.77
    $ python ./compare_evrard.py dump_evrard.h5 --time 1.29
    $ python ./compare_evrard.py dump_evrard.h5 --time 2.58

"""

__program__ = "compare_evrard.py"
__author__ = "Jose A. Escartin (ja.escartin@gmail.com)"
__version__ = "0.1.0"

from argparse import ArgumentParser
from scipy.interpolate import LinearNDInterpolator

import math
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys

""" Time where the outputs with be compare with the solution """
t1 = 0.77
t2 = 1.29
t3 = 2.58
tSolutions = [t1, t2, t3]

""" Evrard density solution at t1,t2,t3 """
Evrard_Density_t1 = np.array([
    [0.0016606542098941835, 2709.7996457527793],
    [0.003929321378361267, 2045.0131033377916],
    [0.008506617198716024, 1449.7406703726315],
    [0.014880187237354103, 996.0983188627732],
    [0.031638456926028326, 532.9118139079258],
    [0.05018317407660172, 303.5101189169387],
    [0.0774908690873875, 143.2837515491685],
    [0.1301457317582952, 46.47641250050049],
    [0.13529756445629482, 14.161314418203443],
    [0.23035591723971707, 6.479552178677312],
    [0.45199493923581874, 2.381825919226254],
    [0.5843245983963044, 1.0562557840784061],
    [0.7618135249344167, 0.302325919211004],
    [0.9371598871024821, 0.05945570708544395],
    [1.0460007248077803, 0.01811609194200417],
    [1.1116878144487818, 0.004870865214687275],
    [1.171158615519198, 0.0014384498882876629],
    [1.244367608944717, 0.00029187388992869703],
    [1.2710221690005559, 0.0001023733127843761]
])
Evrard_Density_t2 = np.array([
    [0.0016448955607539012, 2340.639442460815],
    [0.0034281707095353943, 2045.3965298792039],
    [0.005288627019587109, 1783.7717103711284],
    [0.009482988959966968, 1312.6310107781835],
    [0.014629374056732546, 964.9513747188178],
    [0.02670010494090905, 608.0048300091244],
    [0.05004118317869709, 298.85504033956295],
    [0.08359690554558837, 121.82399727288657],
    [0.20887416474275367, 20.228020348125078],
    [0.27479002311687817, 10.554839565181437],
    [0.388022463304985, 3.053749885777916],
    [0.49710224588307317, 1.0974007695565138],
    [0.71764188947589, 0.12122701339455293],
    [0.71764188947589, 0.032884326420307655],
    [0.978123857764031, 0.00258021946642881],
    [1.126868395873958, 0.00034290531342945715],
    [1.2256784800365277, 0.0001087085275856605]
])
Evrard_Density_t3 = np.array([
    [0.0016803132858173816, 2264.9610543052763],
    [0.004617822938545222, 1898.0819690264645],
    [0.009780054551850909, 1288.2360035913691],
    [0.01695463513342797, 873.7609969117367],
    [0.029389563234801186, 541.6288517143742],
    [0.04993059334450631, 311.4649606997383],
    [0.0782737186028183, 132.65074164391942],
    [0.125177272195588, 52.41614631456304],
    [0.20017354268905696, 19.505710285830137],
    [0.27280345415605467, 9.503774259433104],
    [0.39647346618160306, 2.254331708985578],
    [0.4625886389923515, 0.9448999707063818],
    [0.5477880897169484, 0.33082545787086814],
    [0.5580836345341309, 0.09106776474714436],
    [1.0061659195597998, 0.028317514660760257],
    [1.3576227700751025, 0.014216892397298784],
    [1.9446168703809883, 0.005288659076781498],
    [2.3507274423933886, 0.003177719075805415],
    [2.869211488569923, 0.00137270308421014],
    [3.4675081499521183, 0.000648801196035319],
    [4.10695635435924, 0.00027195741286988804],
    [4.336606023123166, 0.00014927653722116463],
    [4.724253775889861, 0.00024131355941456984],
    [4.915031571801211, 0.0001585722928033268]
])

""" Evrard pressure solution at t1,t2,t3 """
Evrard_Pressure_t1 = np.array([
    [0.001637705988401401, 896.8908249563841],
    [0.009869274427609851, 776.4644537626524],
    [0.02905201141543184, 456.1412665059449],
    [0.055442512947666736, 204.65815808049413],
    [0.08042029012791974, 104.88353722753007],
    [0.12522494820391264, 39.77301447556444],
    [0.1322328654454736, 2.3083989422240148],
    [0.23310028475107358, 0.5859256565512168],
    [0.4489001647842807, 0.16449404376434196],
    [0.6632629737607675, 0.017467116504584906],
    [0.8357676241057913, 0.0016212837670526387],
    [0.9856343797850842, 0.0001190074237208406],
    [1.1021989445439089, 0.000010154526299996846]
])
Evrard_Pressure_t2 = np.array([
    [0.0016937749470421492, 891.9451377929502],
    [0.003047902180129135, 861.4874886651216],
    [0.005030329166939124, 779.6939570690078],
    [0.008231002966248386, 729.1166537940057],
    [0.011830710155600093, 660.2145787075024],
    [0.01775485903091786, 578.5252536054519],
    [0.02486518382980436, 490.7740961164764],
    [0.033060712230837425, 377.5441360528516],
    [0.04550321341170666, 290.4026680585298],
    [0.05843863226984373, 196.06610628087228],
    [0.07440333107584726, 128.12485957546699],
    [0.0899519615447056, 92.36281643335764],
    [0.11451517905630204, 54.72364233555844],
    [0.14206869623144455, 35.76397580753654],
    [0.17775950881675634, 20.51001051616424],
    [0.21303234252512582, 12.974945858656254],
    [0.2552574388891607, 6.747441465626986],
    [0.31121165696697806, 3.869894582018562],
    [0.3793733098199554, 1.885113024021782],
    [0.4624920837570147, 0.9802669331280768],
    [0.5590397966959464, 0.5809008583648277],
    [0.6641657588346738, 0.3442597869669465],
    [0.7270699229378795, 0.2693791364209373],
    [0.7309675565449475, 0.00008045129283566505],
    [0.897668116339834, 0.00001027069024365498]
])
Evrard_Pressure_t3 = np.array([
    [0.0017122554090483423, 846.0791693050612],
    [0.0030369944954524984, 817.2797447844564],
    [0.005122907161400072, 815.7685145258407],
    [0.007736675883231673, 814.5789380951659],
    [0.011679862884718317, 691.0405003433143],
    [0.018174075856678344, 605.6010721164912],
    [0.026353460356688414, 497.34712032072525],
    [0.0374399127140637, 347.03044910096156],
    [0.053182674783014985, 226.8596154945818],
    [0.06970618213797264, 148.34416283725128],
    [0.08776175583472806, 97.01649108031059],
    [0.12333764060680812, 47.29530020620761],
    [0.17157632449736476, 21.60167871745004],
    [0.2409564413245686, 7.600953927503205],
    [0.3217540320515909, 2.5061532283641252],
    [0.4004441984604447, 0.8265237633571388],
    [0.46467268195281336, 0.32092768599290533],
    [0.5607723304637618, 0.07893482273955073],
    [0.5632226808266458, 0.005814582974799403],
    [1.0571040973744534, 0.0009975826370745913],
    [1.6079992519012651, 0.00027035372479656525],
    [2.126317836895269, 0.00010156032125770563],
    [2.7827877497027966, 0.00003348840796513976],
    [3.6037013974081, 0.000008789539392744833],
    [4.106926384984898, 0.000008785473392540442],
    [4.280344026483241, 0.000014799609588260073],
    [4.466486166394138, 0.000043395176015580854],
    [4.713858731888536, 0.00022881393968114788],
    [4.822297328835688, 0.0007644363316902975]
])

""" Evrard velocity solution at t1,t2,t3 """
Evrard_Velocity_t1 = np.array([
    [0.0016633122375051316, -0.007204348880443079],
    [0.005883868003282526, 0.0009767466925368895],
    [0.012459643548010494, -0.012825694500576135],
    [0.023283121179925526, -0.017733004607546676],
    [0.04503581769441411, -0.04419610379112149],
    [0.08748781406143223, -0.12281410826394812],
    [0.10576563210511924, -0.16223425753778753],
    [0.12352654645652737, -0.21966696305876443],
    [0.12567559103428322, -0.2844016921294409],
    [0.13465220919865162, -1.5109665076927283],
    [0.19849647968045844, -1.3200158973944487],
    [0.2851387095413025, -1.1758480855764863],
    [0.4562217961091266, -1.0369917549106813],
    [0.5783254236324091, -0.8695403227979432],
    [0.6931448008704887, -0.7003341147306883],
    [0.8452132411960877, -0.4735604774607274],
    [1.044062727042806, -0.21080551289305416],
    [1.1479556070074288, -0.059652278177455],
    [1.256755910342831, 0.13106593376984232],
    [1.3349597290612056, 0.2678037345404627],
    [1.369948117271132, 0.328975022229413],
    [1.4741420201106197, 0.393780481232997]
])
Evrard_Velocity_t2 = np.array([
    [0.0017080201787886583, 0.0032217359787365396],
    [0.016970392098685866, 0.003208312078824882],
    [0.10968554388906637, 0.0693926132753413],
    [0.1601333512921932, 0.12289580368888675],
    [0.21824061370039197, 0.22417912852041688],
    [0.29488625393952855, 0.38774487430755045],
    [0.3563040249104465, 0.5510779391628846],
    [0.41954876431296395, 0.7622806311022803],
    [0.5113090426145883, 1.088570891615431],
    [0.6020675098040562, 1.4627128806794274],
    [0.715058768643925, 1.951888742717533],
    [0.7212346180545267, -0.9091157229665043],
    [0.9175964556222209, -0.606697631129129],
    [1.1376859872510925, -0.32829489623325525],
    [1.3338798541054784, -0.04042383726653753],
    [1.4599312326348306, 0.15145738806704667]
])
Evrard_Velocity_t3 = np.array([
    [0.0016978296259880093, 0],
    [0.0070732280020383105, -0.004836759371221078],
    [0.05580882697345989, -0.01451027811366501],
    [0.1559816195196327, -0.01451027811366501],
    [0.49135557408456126, 0.033857315598547544],
    [0.5429817520395361, -0.014510278113665898],
    [0.5494935648593446, -0.7932285368802923],
    [0.9980953655222143, -0.043530834340992364],
    [1.4277101699976924, 0.45949214026602103],
    [2.277695860275938, 1.2237001209189824],
    [3.159717442420715, 2.016928657799273],
    [3.988011770379086, 2.6940749697702526],
    [4.275144571642219, 2.824667472793227],
    [4.510643649548783, 1.330108827085851],
    [4.678578103651703, 0.6771463119709793]
])


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


def computeRadiiAndVr(h5File, step):
    """ Load XYZ coordinates and compute their radii and RadialVelocity"""
    x = loadH5Field(h5File, "x", step)
    y = loadH5Field(h5File, "y", step)
    z = loadH5Field(h5File, "z", step)
    vx = loadH5Field(h5File, "vx", step)
    vy = loadH5Field(h5File, "vy", step)
    vz = loadH5Field(h5File, "vz", step)
    radii = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    vr = (vx*x + vy*y + vz*z) / radii
    print("Calculated Radii and RadialVelocity in %s particles" % len(x))
    return radii, vr


def computeL1Error(xSim, ySim, xSol, ySol):
    ySolExpanded = np.interp(xSim, xSol, ySol)
    return sum(abs(ySolExpanded - ySim)) / len(xSim)


def plotRadialProfile(props, xSim, ySim, xSol, ySol):

    if props["xLogScale"] == "true":
        plt.xscale('log')

    if props["yLogScale"] == "true":
        plt.yscale('log')

    plt.scatter(xSim, ySim, s=0.1, label="Simulation, L1 = %3f" % props["L1"], color="C0")
    plt.plot(xSol,
             ySol,
             label="Solution, t = %.3f" % props["tApprox"],
             color="C1")

    plt.xlabel("r")
    plt.ylabel(props["ylabel"])
    plt.draw()
    plt.title(props["title"] + " : N = %8d, t = %.3f, step = %6d" % (len(xSim), props["tReal"], props["step"]))
    plt.legend(loc="lower left")
    plt.savefig(props["fname"], format="png")
    plt.figure().clear()


def createDensityPlot(h5File, hdf5_step, tApprox, tReal, step, radii, rhoNorm, rhoSolX, rhoSolY):
    rho = loadH5Field(h5File, "rho", hdf5_step) / rhoNorm

    L1 = computeL1Error(radii, rho, rhoSolX, rhoSolY)
    print("Density L1 error", L1)

    props = {"ylabel": "rho", "title": "Density", "fname": "evrard_density_%4f.png" % tReal, "tApprox": tApprox, "tReal": tReal, "step": step, "xLogScale": "true", "yLogScale": "true", "L1": L1}
    plotRadialProfile(props, radii, rho, rhoSolX, rhoSolY)


def createPressurePlot(h5File, hdf5_step, tApprox, tReal, step, radii, pNorm, pSolX, pSolY):
    p = loadH5Field(h5File, "p", hdf5_step) / pNorm

    L1 = computeL1Error(radii, p, pSolX, pSolY)
    print("Pressure L1 error", L1)

    props = {"ylabel": "p", "title": "Pressure", "fname": "evrard_pressure_%4f.png" % tReal, "tApprox": tApprox, "tReal": tReal, "step": step, "xLogScale": "true", "yLogScale": "true", "L1": L1}
    plotRadialProfile(props, radii, p, pSolX, pSolY)


def createVelocityPlot(h5File, vr, tApprox, tReal, step, radii, vNorm, velSolX, velSolY):
    vrPlot = vr / vNorm

    L1 = computeL1Error(radii, vrPlot, velSolX, velSolY)
    print("Velocity L1 error", L1)

    props = {"ylabel": "vel", "title": "Velocity", "fname": "evrard_velocity_%4f.png" % tReal, "tApprox": tApprox, "tReal": tReal, "step": step, "xLogScale": "true", "yLogScale": "false", "L1": L1}
    plotRadialProfile(props, radii, vrPlot, velSolX, velSolY)


if __name__ == "__main__":
    parser = ArgumentParser(description='Plot paper solutions against SPH simulations')
    parser.add_argument('simFile', help="SPH simulation HDF5 file")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-t', '--time', type=float, dest="time", choices=tSolutions, help="Valid simulation times (t/t*) to plot the same paper solution graphics")
    args = parser.parse_args()

    # Get time
    time = args.time

    # Get HDF5 Simulation file
    h5File = h5py.File(args.simFile, "r")

    # Get attributes from 'evrard_init.hpp'
    attrs = h5File.attrs
    G  = attrs["gravConstant"]
    R  = attrs["r"]
    Mt = attrs["mTotal"]
    
    # Normalization variables: Steinmetz & Muller (1993)
    tNorm = ((R ** 3.) / G * Mt) ** 0.5
    rhoNorm = (3. * Mt) / (4. * math.pi * (R ** 3.))
    uNorm = G * Mt / R
    vNorm = uNorm ** 0.5
    pNorm = rhoNorm * uNorm

    # Select Solution in function of the time
    if time == t1:
        rhoSolX, rhoSolY = Evrard_Density_t1[:, 0],  Evrard_Density_t1[:, 1]
        pSolX, pSolY = Evrard_Pressure_t1[:, 0], Evrard_Pressure_t1[:, 1]
        velSolX, velSolY = Evrard_Velocity_t1[:, 0], Evrard_Velocity_t1[:, 1]
    elif time == t2:
        rhoSolX, rhoSolY = Evrard_Density_t2[:, 0],  Evrard_Density_t2[:, 1]
        pSolX, pSolY = Evrard_Pressure_t2[:, 0], Evrard_Pressure_t2[:, 1]
        velSolX, velSolY = Evrard_Velocity_t2[:, 0], Evrard_Velocity_t2[:, 1]
    elif time == t3:
        rhoSolX, rhoSolY = Evrard_Density_t3[:, 0],  Evrard_Density_t3[:, 1]
        pSolX, pSolY = Evrard_Pressure_t3[:, 0], Evrard_Pressure_t3[:, 1]
        velSolX, velSolY = Evrard_Velocity_t3[:, 0], Evrard_Velocity_t3[:, 1]
    else:
        print("No valid input time for the solution")
        sys.exit(1)

    # simulation time of each step that was written to file
    timesteps = loadTimesteps(h5File)

    # the actual iteration number of each step that was written
    stepNumbers = loadStepNumbers(h5File)

    # output time specified instead of step, locate closest output step
    tApprox = time * tNorm
    stepIndex = determineTimestep(tApprox, timesteps)
    step = stepNumbers[stepIndex]
    tReal = timesteps[stepIndex]
    print("The closest timestep to the specified solution time of t/t*=%s is step=%s at tReal=%s, where t*=%s" % (time, step, tReal, tNorm))

    hdf5_step = np.searchsorted(stepNumbers, step)

    # Calulate Radius and RadialVelocity
    radii = None
    vr = None
    try:
        radii, vr = computeRadiiAndVr(h5File, hdf5_step)
    except KeyError:
        print("Could not load radii, input file does not contain all fields \"x, y, z, vx, vy, vz\"")
        sys.exit(1)

    try:
        createDensityPlot(h5File, hdf5_step, tApprox, tReal, step, radii, rhoNorm, rhoSolX, rhoSolY)
    except KeyError:
        print("Could not plot density profile, input does not contain field \"rho\"")

    try:
        createPressurePlot(h5File, hdf5_step, tApprox, tReal, step, radii, pNorm, pSolX, pSolY)
    except KeyError:
        print("Could not plot pressure profile, input does not contain field \"p\"")

    try:
        createVelocityPlot(h5File, vr, tApprox, tReal, step, radii, vNorm, velSolX, velSolY)
    except KeyError:
        print("Could not plot velocity profile, input does not contain fields \"vx, vy, vz\"")
