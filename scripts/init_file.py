import h5py
import numpy as np

file = h5py.File("initialConditions.h5", "w")
initStep = file.create_group("/Step#0")

numParticles = 10000

randomArray = lambda x,y : np.array(np.random.rand(x), dtype=y)

# field data types will be converted by the HDF5 library if necessary to match what the SPH-EXA was configured to use
x = randomArray(numParticles, np.float64)
y = randomArray(numParticles, np.float64)
z = randomArray(numParticles, np.float64)

m = np.ones(numParticles, dtype=np.float32) / numParticles
h = np.ones(numParticles, dtype=np.float32) * (0.523 / 100) ** (1. / 3)

vx = randomArray(numParticles, np.float64) * 0.1
vy = randomArray(numParticles, np.float64) * 0.1
vz = randomArray(numParticles, np.float64) * 0.1

du_m1 = np.zeros(numParticles, dtype=np.float32)
temp = np.ones(numParticles, dtype=np.float64) * 273
alphamin = 0.05
alpha = np.ones(numParticles, dtype=np.float32) * alphamin

minDt = 1e-7
initStep["x"] = x
initStep["y"] = y
initStep["z"] = z
initStep["vx"] = vx
initStep["vy"] = vy
initStep["vz"] = vz
# coordinate difference to previous time-step
# x(iteration=n) = x(iteration=n-1) + x_m1
initStep["x_m1"] = vx * minDt
initStep["y_m1"] = vy * minDt
initStep["z_m1"] = vz * minDt
initStep["m"] = m
initStep["du_m1"] = du_m1
initStep["temp"] = temp
initStep["h"] = h
initStep["alpha"] = alpha
initStep["id"] = np.arange(0, numParticles, dtype=np.uint64)

# attributes

initStep.attrs["numParticlesGlobal"] = numParticles
initStep.attrs["time"] = 0.0
initStep.attrs["minDt"] = minDt
initStep.attrs["minDt_m1"] = minDt
initStep.attrs["iteration"] = 1
initStep.attrs["ng0"] = np.array([100], dtype=np.int32)
initStep.attrs["ngmax"] = np.array([150], dtype=np.int32)
# gravity is turned on by a non-zero attribute value
initStep.attrs["gravConstant"] = 1.0
initStep.attrs["gamma"] = 5. / 3.
initStep.attrs["muiConst"] = 1.21 * 4.329e13
initStep.attrs["Kcour"] = 0.4

# BoundaryType: 0 = open, 1 = periodic, 2 = fixed
initStep.attrs["boundaryType"] = np.array([0, 0, 0], dtype=np.int8)
initStep.attrs["box"] = np.array([0, 1, 0, 1, 0, 1], dtype=np.float64)

file.close()
