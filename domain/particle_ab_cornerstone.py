from cornerstone import iHilbert, iHilbertMixD, hilbertIBoxKeys, hilbertMixDIBoxKeys, spanSfcRange
import random
import matplotlib.pyplot as plt

def create_random_particles(pow2_x_range, pow2_y_range, pow2_z_range, num_particles):
    """
    Generates a list of random particles within a 3D space.
    
    Parameters:
    - pow2_x_range: The range of the x-axis as a power of 2.
    - pow2_y_range: The range of the y-axis as a power of 2.
    - pow2_z_range: The range of the z-axis as a power of 2.
    - num_particles: The number of particles to generate.
    
    Returns:
    - A list of random particles within the specified 3D space.
    """
    random.seed(42)  # Set the seed to a constant value for reproducibility
    particles = []
    for _ in range(num_particles):
        x = random.randint(0, 2**pow2_x_range - 1)
        y = random.randint(0, 2**pow2_y_range - 1)
        z = random.randint(0, 2**pow2_z_range - 1)
        particles.append((x, y, z))
    return particles

def sort_particles_by_distance(particles, origin=(0, 0, 0)):
    """
    Sorts a list of particles based on their distance from a given origin.
    
    Parameters:
    - particles: A list of particles to sort.
    - origin: The origin point to calculate distances from (default is (0, 0, 0)).
    
    Returns:
    - A list of particles sorted by their distance from the origin.
    """
    return sorted(particles, key=lambda p: ((p[0] - origin[0])**2 + (p[1] - origin[1])**2 + (p[2] - origin[2])**2)**0.5)

def plot_combined(pa, pb, particles, pabIBox, pabIBoxMixD):
    """
    Plots the points pa, pb and the IBoxes pabIBox and pabIBoxMixD in 3D space using matplotlib.
    
    Parameters:
    - pa: The first particle point.
    - pb: The second particle point.
    - pabIBox: The first IBox object to plot.
    - pabIBoxMixD: The second IBox object to plot.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot pa and pb
    ax.scatter(*pa, color='r', label='Particle A')
    ax.scatter(*pb, color='g', label='Particle B')
    
    def plot_particles_3d(ax, particles):
        """
        Plots a list of particles in 3D space using matplotlib.
        
        Parameters:
        - particles: A list of particles to plot.
        """
        # Convert the list of points into separate x, y, z arrays
        x_vals, y_vals, z_vals = zip(*particles)
        
        # Plot the particles
        max_range = max(max(x_vals) - min(x_vals), max(y_vals) - min(y_vals), max(z_vals) - min(z_vals))
        mid_x = (max(x_vals) + min(x_vals)) / 2
        mid_y = (max(y_vals) + min(y_vals)) / 2
        mid_z = (max(z_vals) + min(z_vals)) / 2

        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
        ax.scatter(x_vals, y_vals, z_vals, color='gray')
        ax.set_title('Random Particles in 3D Space')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    # Function to plot IBox
    def plot_IBox(ax, ibox, color='b', label='IBox'):
        xmin, xmax = ibox.xmin(), ibox.xmax()
        ymin, ymax = ibox.ymin(), ibox.ymax()
        zmin, zmax = ibox.zmin(), ibox.zmax()
        
        vertices = [
            [xmin, ymin, zmin],
            [xmin, ymin, zmax],
            [xmin, ymax, zmin],
            [xmin, ymax, zmax],
            [xmax, ymin, zmin],
            [xmax, ymin, zmax],
            [xmax, ymax, zmin],
            [xmax, ymax, zmax]
        ]
        
        edges = [
            [vertices[0], vertices[1]],
            [vertices[0], vertices[2]],
            [vertices[0], vertices[4]],
            [vertices[1], vertices[3]],
            [vertices[1], vertices[5]],
            [vertices[2], vertices[3]],
            [vertices[2], vertices[6]],
            [vertices[3], vertices[7]],
            [vertices[4], vertices[5]],
            [vertices[4], vertices[6]],
            [vertices[5], vertices[7]],
            [vertices[6], vertices[7]]
        ]
        
        for edge in edges:
            ax.plot3D(*zip(*edge), color=color)
    
    # Plot particles
    plot_particles_3d(ax, particles)

    # Plot IBoxes
    plot_IBox(ax, pabIBox, color='b')
    ax.text(pabIBox.xmin(), pabIBox.ymin(), pabIBox.zmax(), 'IBox', color='b')
    
    plot_IBox(ax, pabIBoxMixD, color='m')
    ax.text(pabIBoxMixD.xmax(), pabIBoxMixD.ymax(), pabIBoxMixD.zmin(), 'IBoxMixD', color='m')
    
    ax.set_title('Particles and IBoxes in 3D Space')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

bx = 10
by = 1
bz = 10
max_level = 10
assert bx <= max_level and by <= max_level and bz <= max_level
nparticles = 200

particles = create_random_particles(pow2_x_range=bx, pow2_y_range=by, pow2_z_range=bz, num_particles=nparticles)

sorted_particles = sort_particles_by_distance(particles)

pa = sorted_particles[5]
pb = sorted_particles[7]

print(pa, pb)

pa_iHilbert_key = iHilbert(pa[0], pa[1], pa[2], max_level)
pb_iHilbert_key = iHilbert(pb[0], pb[1], pb[2], max_level)

print("Particle A iHilbert key: {}\toct: {}\tbin: {}".format(pa_iHilbert_key, oct(pa_iHilbert_key), bin(pa_iHilbert_key)))
print("Partic;e B iHilbert key: {}\toct: {}\tbin: {}".format(pb_iHilbert_key, oct(pb_iHilbert_key), bin(pb_iHilbert_key)))

num_values, span_sfc = spanSfcRange(min(pa_iHilbert_key, pb_iHilbert_key), max(pa_iHilbert_key, pb_iHilbert_key))

print("Number of values between A and B:", num_values)
print("Octal values of span_sfc:")
print("\n".join([oct(key) for key in span_sfc]))

pa_iHilbertMixD_key = iHilbertMixD(pa[0], pa[1], pa[2], bx, by, bz)
pb_iHilbertMixD_key = iHilbertMixD(pb[0], pb[1], pb[2], bx, by, bz)

print("Particle A iHilbertMixD key: {}\toct: {}\tbin: {}".format(pa_iHilbertMixD_key, oct(pa_iHilbertMixD_key), bin(pa_iHilbertMixD_key)))
print("Partic;e B iHilbertMixD key: {}\toct: {}\tbin: {}".format(pb_iHilbertMixD_key, oct(pb_iHilbertMixD_key), bin(pb_iHilbertMixD_key)))

num_values_mixd, span_sfc_mixd = spanSfcRange(min(pa_iHilbertMixD_key, pb_iHilbertMixD_key), max(pa_iHilbertMixD_key, pb_iHilbertMixD_key))

print("Number of values between A and B:", num_values_mixd)
print("Octal values of span_sfc:")
print("\n".join([oct(key) for key in span_sfc_mixd]))

pabIBox = hilbertIBoxKeys(min(pa_iHilbert_key, pb_iHilbert_key), max(pa_iHilbert_key, pb_iHilbert_key))

pabIBoxMixD = hilbertMixDIBoxKeys(min(pa_iHilbertMixD_key, pb_iHilbertMixD_key), max(pa_iHilbertMixD_key, pb_iHilbertMixD_key), bx, by, bz)

plot_combined(pa, pb, particles, pabIBox, pabIBoxMixD)