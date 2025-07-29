import h5py
import numpy as np
import sys

inputFilename = sys.argv[1]
h5File = h5py.File(inputFilename, "r+")


# Get all step names from the HDF5 file
step_names = list(h5File["/"])
# Extract step numbers, convert to integers, and sort
step_numbers = [int(name.split('#')[1]) for name in step_names]
step_numbers.sort()

# Finding the latest step by selecting the maximum step number
latest_step_number = step_numbers[-1]
latestH5Step = f"Step#{latest_step_number}"

# Get the corresponding group in the HDF5 file
stepHandle = h5File[latestH5Step]
print(f"Working on the latest step: {latestH5Step}")

num_particles = stepHandle.attrs["numParticlesGlobal"]  # Get the number of particles

if isinstance(num_particles, np.ndarray):
    num_particles = num_particles.item()  # Convert to integer if it's an array

num_particles = int(num_particles)  # Convert float to integer

# Define the constant mass value for each particle
mass_per_particle = 1. / float(num_particles)

# Print to ensure correct retrieval and conversion
print(f"num_particles type: {type(num_particles)} - value: {num_particles}")
print(f"particles mass: {mass_per_particle}")

# Create the mass dataset
m = np.full((num_particles,), mass_per_particle, dtype=np.float32)

# Ensure minDt is a scalar
minDt = stepHandle.attrs["minDt"]
if isinstance(minDt, np.ndarray):
    minDt = minDt.item()  # Convert numpy array to a scalar
print(f"minDt type: {type(minDt)}")  # Print the type to verify

def safely_multiply(dataset, scalar):
    return np.array(dataset[:], dtype=np.float32) * scalar

if "x_m1" not in stepHandle:
    print(f"Adding x_m1 to SPH iteration {stepHandle.attrs['iteration']}")
    stepHandle["x_m1"] = safely_multiply(stepHandle["vx"], minDt)

if "y_m1" not in stepHandle:
    print(f"Adding y_m1 to SPH iteration {stepHandle.attrs['iteration']}")
    stepHandle["y_m1"] = safely_multiply(stepHandle["vy"], minDt)

if "z_m1" not in stepHandle:
    print(f"Adding z_m1 to SPH iteration {stepHandle.attrs['iteration']}")
    stepHandle["z_m1"] = safely_multiply(stepHandle["vz"], minDt)

if "du_m1" not in stepHandle:
    print(f"Adding du_m1 to SPH iteration {stepHandle.attrs['iteration']}")
    stepHandle["du_m1"] = np.zeros(stepHandle.attrs["numParticlesGlobal"], dtype=np.float32)

if "m" not in stepHandle:
    print(f"Adding m to SPH iteration {stepHandle.attrs['iteration']}")
    stepHandle.create_dataset("m", data=m)

h5File.close()
