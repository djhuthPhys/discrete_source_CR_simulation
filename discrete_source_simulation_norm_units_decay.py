import numpy as np
import matplotlib.pyplot as plt

from itertools import repeat
from scipy.signal import argrelextrema
from joblib import Parallel, delayed


# Seed np.random
np.random.seed(0)

# Plotting parameters
plt.rcParams.update({'font.size': 22})
my_cmap = plt.cm.viridis


# Function for generating arm coordinates
def galactic_arms(r, params):
    return params[0] * np.log(r / params[1]) + params[2]


def create_sources(params):
    """
    Model spiral structure of galaxy based on 'BIRTH AND EVOLUTION OF ISOLATED RADIO PULSARS'
    by Faucher-Gigue`re and Kaspi. Populate the galaxy with source positions and the times at which
    they ignite.

    Returns arrays for the source positions, the Sun's position, and the times the sources ignite
    """
    # Unpack model parameters
    L, K, q, B, r_sol, lam_z, duration = params

    # Initialize parameters for Milky Way arms [k (rad), r_o (100 pc), theta_o (rad)]
    norma = np.array([4.25, 34.8, 1.57])
    carina_sagittarius = np.array([4.25, 34.8, 4.71])
    perseus = np.array([4.89, 49.0, 4.09])
    crux_scutum = np.array([4.89, 49.0, 0.95])
    arms_list = [norma, carina_sagittarius, perseus, crux_scutum]

    # Generate arm centroid coordinates
    r_values = np.arange(0.1, 200, 0.00001)
    z_values = np.arange(0, L / 2, 0.00001)
    theta = np.zeros([r_values.shape[0], 4])
    i = 0
    for arm in arms_list:
        theta[:, i] = galactic_arms(r_values, arm)
        i += 1

    # Calculate source density distributions
    norm_const = 1 / np.sum((r_values / r_sol) ** q * np.exp(-B * (r_values / r_sol)))
    source_density = norm_const * (r_values / r_sol) ** q * np.exp(-B * (r_values / r_sol))

    norm_const = 1 / np.sum(1 / (np.cosh(z_values / lam_z) ** 2))
    z_density = norm_const / (np.cosh(z_values / lam_z) ** 2)

    # Use time bins of 25 yrs over 100 Myrs and sample whether a SN explodes in each time bin
    source_creation_times = np.random.choice([0, 1], size=int(duration), p=[0.75, 0.25])  # assuming ~ 1 SN/century

    # Check that we have ~ 1 million sources if duration = 100 Myrs
    print('Number of sources created: ' + str(np.sum(source_creation_times)))

    # Randomly choose an arm and r value correponding to source positions
    num_sources = np.sum(source_creation_times)
    arm_choices = np.random.choice(4, size=num_sources,
                                   p=[0.25, 0.25, 0.25, 0.25])  # Randomly choose an arm to place the source in
    r_raw = np.random.choice(r_values, size=num_sources, p=source_density, replace=False)

    uniform = True
    if uniform:
        r_source = r_raw
        theta_source = np.random.uniform(0, 2 * np.pi, size=num_sources)
    else:
        # Calculate theta_raw from r_raw values and galactic_arms function
        theta_raw = np.zeros((num_sources,))
        for source in range(num_sources):
            theta_raw[source] = galactic_arms(r_raw[source], arms_list[arm_choices[source]])

        # Make corrections to theta and r to blur distribution
        # theta correction
        theta_corr = np.random.uniform(0, 2 * np.pi, size=num_sources)
        theta_source = theta_raw + theta_corr * np.exp(-0.035 * r_raw)

        # r correction
        r_corr = np.random.normal(scale=0.07 * r_raw, size=num_sources)
        r_source = r_raw + r_corr

    # Get z coordinates of sources from sech(z)^2 distribution
    sign_choice = np.random.choice([-1, 1], size=num_sources, p=[0.5, 0.5])
    z_choice = sign_choice * np.random.choice(z_values, size=num_sources, p=z_density, replace=True) + L / 2

    # Convert to cartesian coordinates
    source_coords = np.concatenate((np.reshape(r_source * np.cos(theta_source), (-1, 1)),
                                    np.reshape(r_source * np.sin(theta_source), (-1, 1)),
                                    np.reshape(z_choice, (-1, 1))), axis=1)  # [x, y, z]

    # Sun location
    sun_location = np.array([0, r_sol, L/2])

    # Check that no source is within 1 pc of the sun
    distances = np.sqrt(np.sum(np.power((sun_location - source_coords), 2), axis=1))
    print('Closest source to sun: ' + str(np.min(distances)) + ' [100 pc]')
    print('Is earth safe?: ' + str(np.min(distances > 0.01)))

    assert np.min(distances > 0.01), 'One or more sources are too close to the sun.'

    # Plot source positions
    # plt.figure(figsize=(11, 11))
    # plt.scatter(source_coords[:, 0], source_coords[:, 1], s=0.8, alpha=0.15, color=my_cmap(50))
    # plt.scatter(sun_location[0], sun_location[1], s=60, c='r')
    # # plt.title('Simulated source positions')
    # plt.xlabel('x (100 pc)')
    # plt.ylabel('y (100 pc)')
    # plt.xlim([-200, 200])
    # plt.ylim([-200, 200])
    # plt.savefig('/Users/dawsonhuth/Documents/Thesis Files/plots/simulated_source_xy_positions.png')

    # Plot source positions
    # plt.figure(figsize=(11, 11))
    # plt.scatter(source_coords[:, 0], source_coords[:, 2], s=0.8, alpha=0.15, color=my_cmap(50))
    # plt.scatter(sun_location[0], sun_location[2], s=60, c='r')
    # # plt.title('Simulated source positions')
    # plt.xlabel('x (100 pc)')
    # plt.ylabel('z (100 pc)')
    # plt.savefig('/Users/dawsonhuth/Documents/Thesis Files/plots/simulated_source_xz_positions.png')

    return source_coords, sun_location, source_creation_times


def calculate_cr_quantities(source_position, t_start, observer_position, L, K, sim_length, tau_eff):
    """
    Calculate the CR density and spatial gradients contribution of a source beginning at some time
    step t_start at source_position(x, y, z). Sum all N term contributions.
    """

    # Initialize variables
    x_s, y_s, z_s = source_position[0], source_position[1], source_position[2]
    x, y, z = observer_position[0], observer_position[1], observer_position[2]
    T = np.reshape(np.arange(t_start, (sim_length+1) * 0.25, 0.25), (1, -1)) - t_start  # Time vector
    N = np.reshape(np.arange(1, 101, 1), (-1, 1))  # Vector for N values
    chunk_factor = 1000

    rho = np.zeros((T.shape[1], 1))
    rho_x = np.zeros((T.shape[1], 1))
    rho_y = np.zeros((T.shape[1], 1))
    rho_z = np.zeros((T.shape[1], 1))

    for i in range(int(T.shape[1] / chunk_factor) + 1):
        if int((i + 1) * chunk_factor) < int(T.shape[1]):
            last_idx = int((i + 1) * chunk_factor)
        else:
            last_idx = int(T.shape[1])
        common = 2 / L * np.sin(N * np.pi * z_s / L) * \
                 np.exp(-(((K * N ** 2 * np.pi ** 2) / L ** 2) + 1/tau_eff) * T[:, int(i * chunk_factor):last_idx]) \
                 * (1 / (4 * np.pi * K * T[:, int(i * chunk_factor):last_idx] + 1e-8)) \
                 * np.exp(-((x - x_s) ** 2 + (y - y_s) ** 2) / (4 * K * T[:, int(i * chunk_factor):last_idx] + 1e-8))

        # Calculate energy density
        rho[int(i * chunk_factor):last_idx, :] += np.reshape(np.sum(common * np.sin(N * np.pi * z / L), axis=0),
                                                             (-1, 1))

        # Calculate gradients
        rho_z[int(i * chunk_factor):last_idx, :] += np.reshape(
            np.sum(((np.pi * N / L) * np.cos(N * np.pi * z / L)) * common, axis=0), (-1, 1))

        rho_x[int(i * chunk_factor):last_idx, :] += np.reshape(np.sum(-common * (np.sin(N * np.pi * z / L) * (
                    (x - x_s) / (2 * K * T[0, int(i * chunk_factor):last_idx] + 1e-8))), axis=0), (-1, 1))

        rho_y[int(i * chunk_factor):last_idx, :] += np.reshape(np.sum(-common * (np.sin(N * np.pi * z / L) * (
                    (y - y_s) / (2 * K * T[0, int(i * chunk_factor):last_idx] + 1e-8))), axis=0), (-1, 1))

    rho_grads = np.concatenate((rho_x, rho_y, rho_z), axis=1)

    return rho, rho_grads


def simulation_worker(func, args_batch, observer_position, L, K, sim_length, tau_eff):
    """
    Call calculate_cr_density() with every packet of arguments received and update
    result array on the run.

    Worker function which runs the job in each spawned process.
    """
    worker_rho = np.zeros((int(sim_length), 1))
    worker_grads = np.zeros((int(sim_length), 3))
    for args_ in args_batch:
        new_rho, new_grads = func(*args_, observer_position, L, K, sim_length, tau_eff)
        ZERO_PADDING_SIZE = worker_rho.shape[0] - new_rho.shape[0]
        new_rho = np.concatenate((np.zeros((ZERO_PADDING_SIZE, 1)), new_rho), axis=0)
        new_grads = np.concatenate((np.zeros((ZERO_PADDING_SIZE, 3)), new_grads), axis=0)
        assert new_rho.shape[0] == worker_rho.shape[0], 'Source density time series length must match total density'
        worker_rho += new_rho
        worker_grads += new_grads

    return worker_rho, worker_grads


def simulate_sources(func, arguments, n_jobs, verbose, indices, observer_position, sim_length, L, K, tau_eff):
    """
    Simulates CR density contributions at the sun from sources whose positions and start times are randomly generated. Jobs
    are given to workers in batches. Results from each source are summed in batches and then returned. The returned batches
    are summed into a final result called cr_density.
    """

    with Parallel(n_jobs=-1, verbose=verbose) as parallel:
        # bundle up jobs:
        funcs = repeat(func, n_jobs)  # functools.partial seems not pickle-able
        L_arg = repeat(L, n_jobs)
        K_arg = repeat(K, n_jobs)
        tau_decay_arg = repeat(tau_eff, n_jobs)
        observer_arg = repeat(observer_position, n_jobs)
        sim_len_arg = repeat(sim_length)
        args_batches = np.array_split(arguments, indices, axis=0)
        jobs = zip(funcs, args_batches, observer_arg, L_arg, K_arg, sim_len_arg, tau_decay_arg)

        cr_quantities = parallel(delayed(simulation_worker)(*job) for job in jobs)

    # Group results and sum over worker results
    cr_density = []
    cr_gradients = []
    for i in range(len(cr_quantities)):
        cr_density.append(cr_quantities[i][0])
        cr_gradients.append(cr_quantities[i][1])

    cr_density = sum(cr_density)
    cr_gradients = sum(cr_gradients)
    del cr_quantities

    return cr_density, cr_gradients


def main(L=160, K=9.945e-4, q=1.69, B=8.3, tau_eff=2.62e4, r_sol=83, lam_z=1, duration=1e4/0.25):

    params = [L, K, q, B, r_sol, lam_z, duration]

    source_coords, sun_location, ignition_times = create_sources(params)

    # Set up arguments
    start_times = np.where(ignition_times == 1)[0] * 0.25
    ARGUMENTS = [*zip(source_coords, start_times)]
    N_JOBS = 100
    VERBOSE = N_JOBS

    # Find start_time indices to evenly distribute jobs
    lengths = duration - (start_times / 0.25)
    cumulative = np.cumsum(lengths) % (np.sum(lengths) / N_JOBS)
    INDICES = argrelextrema(cumulative, np.less)[0]

    # Perform the simulation
    cr_energy_density, cr_gradients = simulate_sources(calculate_cr_quantities, ARGUMENTS, N_JOBS, VERBOSE, INDICES,
                                                       sun_location, duration, L, K, tau_eff)

    # Plot and save data
    time_data = np.arange(0, duration, 1) * 0.25
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(25, 8))
    plt.plot(time_data, cr_energy_density, linewidth=3, label='Energy density')
    plt.title('CR energy density from discrete sources')
    plt.ylabel(r'CR energy density (u/kpc$^3$)')
    plt.xlabel('Time (century)')
    plt.grid('both')
    plt.legend()
    plt.savefig('./plots/discrete_source_sim_cr_energy_density_uniform_norm_Fe60.pdf')
    # plt.show()

    # plt.figure(figsize=(25, 8))
    plt.plot(time_data, cr_gradients[:, 0], linewidth=3, label='X density gradient')
    plt.plot(time_data, cr_gradients[:, 1], linewidth=3, label='Y density gradient')
    plt.plot(time_data, cr_gradients[:, 2], linewidth=3, label='Z density gradient')
    plt.title('CR energy density gradients from discrete sources')
    plt.ylabel(r'CR energy density gradients (u/kpc$^3$/kpc)')
    plt.xlabel('Time (century)')
    plt.grid('both')
    plt.legend()
    plt.savefig('./plots/discrete_source_sim_cr_energy_density_gradients_uniform_norm_Fe60.pdf')
    # plt.show()

    with open('./data/discrete_source_sim_cr_energy_density_uniform_norm_Fe60.npy', 'wb') as file:
        np.save(file, cr_energy_density)

    with open('./data/discrete_source_sim_cr_energy_density_gradients_uniform_norm_Fe60.npy', 'wb') as file:
        np.save(file, cr_gradients)

    with open('./data/discrete_source_positions_uniform_norm_Fe60.npy', 'wb') as file:
        np.save(file, source_coords)

    with open('./data/discrete_source_start_times_uniform_norm_Fe60.npy', 'wb') as file:
        np.save(file, start_times)

    return cr_energy_density


if __name__ == '__main__':
    main(L=160,
         K=9.945e-4,
         q=1.69,
         B=3.33,
         tau_eff=1/((np.log(2)/2.62e4) + (1/4e4)),  # Fe60 2.62e4 centuries, half-life
         r_sol=83,
         lam_z=1,
         duration=round(1e6/0.25))  # 100 million years
