import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import repeat
from scipy.signal import argrelextrema
from joblib import Parallel, delayed

plt.style.use("default")

# Seed np.random
np.random.seed(0)


# Function for generating arm coordinates
def galactic_arms(r, params):
    return params[0] * np.log(r / params[1]) + params[2]


def create_sources(params, dist_type='armless'):
    """
    Model spiral structure of galaxy based on 'BIRTH AND EVOLUTION OF ISOLATED RADIO PULSARS'
    by Faucher-Gigue`re and Kaspi. Populate the galaxy with source positions and the times at which
    they ignite.

    Returns arrays for the source positions, the Sun's position, and the times the sources ignite
    """
    # Unpack model parameters
    L, K, q, B, r_sol, lam_z, duration = params

    if dist_type == 'armless':
        # Generate grid points 
        r_values = np.arange(0.01, 20, 0.000001)  # in 1 kpc units
        z_values = np.arange(0, L / 2, 0.000001)  # in 1 kpc units
        theta_values = np.arange(0, 2*np.pi, 0.000001)

    else:
        # Initialize parameters for Milky Way arms [k (rad), r_o (kpc), theta_o (rad)]
        norma = np.array([4.25, 3.48, 1.57])
        carina_sagittarius = np.array([4.25, 3.48, 4.71])
        perseus = np.array([4.89, 4.90, 4.09])
        crux_scutum = np.array([4.89, 4.90, 0.95])
        arms_list = [norma, carina_sagittarius, perseus, crux_scutum]

        # Generate arm centroid coordinates
        r_values = np.arange(0.01, 20, 0.000001)
        z_values = np.arange(0, 0.3 + L / 2, 0.000001)
        theta_values = np.zeros([r_values.shape[0], 4])
        i = 0
        for arm in arms_list:
            theta_values[:, i] = galactic_arms(r_values, arm)
            i += 1

    # Calculate source density distributions
    norm_const = 1 / np.sum((r_values / r_sol) ** q * np.exp(-B * (r_values / r_sol)))
    source_density = norm_const * (r_values / r_sol) ** q * np.exp(-B * (r_values / r_sol))

    norm_const = 1 / np.sum(1 / (np.cosh(z_values / lam_z) ** 2))
    z_density = norm_const / (np.cosh(z_values / lam_z) ** 2)

    # Use time bins of 25 yrs over 100 Myrs and sample whether a SN explodes in each time bin
    source_creation_times = np.random.choice([0, 1], size=int(duration), p=[3/4, 1/4])  # assuming ~ 1 SN/century

    # Check that we have ~ 1 million sources if duration = 100 Myrs
    num_sources = np.sum(source_creation_times)
    print('Number of sources created: ' + str(num_sources))

    if dist_type == 'armless':
        # Randomly choose an r and theta value corresponding to a source position
        r_raw = np.random.choice(r_values, size=num_sources, p=source_density, replace=False)
        theta_raw = np.random.choice(theta_values, size=num_sources, 
                                     p=1/theta_values.shape[0] * np.ones_like(theta_values),
                                     replace=False)
    else:
        # Randomly choose an arm and r value correponding to source positions
        arm_choices = np.random.choice(4, size=num_sources,
                                    p=[0.25, 0.25, 0.25, 0.25])  # Randomly choose an arm to place the source in
        r_raw = np.random.choice(r_values, size=num_sources, p=source_density, replace=False)

        # Calculate theta_raw from r_raw values and galactic_arms function
        theta_raw = np.zeros((num_sources,))
        for source in range(num_sources):
            theta_raw[source] = galactic_arms(r_raw[source], arms_list[arm_choices[source]])

    # Make corrections to theta and r to blur distribution
    # theta correction
    theta_corr = np.random.uniform(0, 2 * np.pi, size=num_sources)
    theta_source = theta_raw + theta_corr * np.exp(-0.35 * r_raw)

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
    sun_location = np.array([0, r_sol, L / 2])

    # Check that no source is within 1 pc of the sun
    distances = np.sqrt(np.sum(np.power((sun_location - source_coords), 2), axis=1))
    print('Closest source to sun: ' + str(np.min(distances)) + ' kpc')
    print('Is earth safe?: ' + str(np.min(distances > 0.001)))

    assert np.min(distances > 0.001), 'One or more sources are too close to the sun.'

    # # Plot source positions
    # plt.figure(figsize=(10, 10))
    # plt.scatter(source_coords[:, 0], source_coords[:, 1], s=0.8, alpha=0.5)
    # plt.scatter(sun_location[0], sun_location[1], s=20, c='r')
    # plt.title('Simulated source positions')
    # plt.xlabel('x (100 pc)')
    # plt.ylabel('y (100 pc)')
    # plt.xlim([-20, 20])
    # plt.ylim([-20, 20])
    # # plt.savefig('./plots/simulated_source_positions.pdf')
    # plt.show()

    # # Plot source positions
    # plt.figure(figsize=(10, 10))
    # plt.scatter(source_coords[:, 0], source_coords[:, 2], s=0.8, alpha=0.5)
    # plt.scatter(sun_location[0], sun_location[2], s=40, c='r')
    # plt.title('Simulated source positions')
    # plt.xlabel('x (100 pc)')
    # plt.ylabel('z (100 pc)')
    # # plt.savefig('./plots/simulated_source_positions.pdf')
    # plt.show()

    return source_coords, sun_location, source_creation_times


def calculate_cr_quantities(source_position, t_start, observer_position, L, 
                            K, sim_length, species_params, species):
    """
    Calculate the CR density and spatial gradients contribution of a source beginning at some time
    step t_start at source_position(x, y, z). Sum all N term contributions.
    """

    # Convert parameters to cgs units
    L *= 3.086e21  # cm
    t_start *= 3.154e7  # s 

    # Initialize variables
    x_s, y_s, z_s = source_position[0]*3.086e21, source_position[1]*3.086e21, source_position[2]*3.086e21  # cm
    x, y, z = observer_position[0]*3.086e21, observer_position[1]*3.086e21, observer_position[2]*3.086e21  # cm
    T = np.reshape(np.arange(t_start, (sim_length+1) * 25*3.154e7, 
                             25*3.154e7), (1, -1)) - t_start  # Time vector in s
    # N = np.reshape(np.arange(1, 101, 1), (-1, 1))  # Vector for N values
    chunk_factor = 10000

    rho = np.zeros((T.shape[1], 1))

    for i in range(int(T.shape[1] / chunk_factor) + 1):
        if int((i + 1) * chunk_factor) < int(T.shape[1]):
            last_idx = int((i + 1) * chunk_factor)
        else:
            last_idx = int(T.shape[1])
        # common = 2 / L * np.sin(N * np.pi * z_s / L) * np.sin(N * np.pi * z / L) \
        #     * np.exp(-(K * N ** 2 * np.pi ** 2 * (T[:, int(i * chunk_factor):last_idx])) / L ** 2) \
        #     * (1 / (4 * np.pi * K * T[:, int(i * chunk_factor):last_idx] + 1e-8)) \
        #     * np.exp(-((x - x_s) ** 2 + (y - y_s) ** 2) / (4 * K * T[:, int(i * chunk_factor):last_idx] + 1e-8))
        
        # # Sum over N
        # rho[int(i * chunk_factor):last_idx, :] += np.reshape(np.sum(common, axis=0),
        #                                                      (-1, 1))
            
        common = (1 / (4 * np.pi * K * T[:, int(i * chunk_factor):last_idx] + 1e-8)) * \
                 np.exp(-((x - x_s) ** 2 + (y - y_s) ** 2 + (z - z_s) ** 2) / (4 * K * T[:, int(i * chunk_factor):last_idx] + 1e-8))
        
        rho[int(i * chunk_factor):last_idx, :]  += np.reshape(common, (-1, 1))

    if species == 'C':
        [tau_spec, C_s, tau_S] = species_params
        species_rho = rho * np.exp(-T.T/tau_spec) * (C_s/tau_S)
    elif species == 'Be10':
        [tau_CCG, Be10_S, tau_S, n_G, sig_CBe10, tau_spec, c] = species_params
        species_galaxy = rho * (c * n_G * sig_CBe10) * np.exp(-T.T/tau_spec) * \
                            (tau_spec * tau_CCG)/(tau_spec - tau_CCG) * \
                            (1 - np.exp(-T.T*((tau_spec - tau_CCG)/(tau_spec * tau_CCG))))
        species_source = rho * (Be10_S/tau_S) * np.exp(-T.T/tau_spec)
        species_rho = species_galaxy + species_source
    elif species == 'Be9':
        [tau_CCG, Be9_S, tau_S, n_G, sig_CBe9, tau_spec, c] = species_params
        species_galaxy = rho * (c * n_G * sig_CBe9) * np.exp(-T.T/tau_spec) * \
                            (tau_spec * tau_CCG)/(tau_spec - tau_CCG) * \
                            (1 - np.exp(-T.T*((tau_spec - tau_CCG)/(tau_spec * tau_CCG))))
        species_source = rho * (Be9_S/tau_S) * np.exp(-T.T/tau_spec)
        species_rho = species_galaxy + species_source
        
    return species_rho


def simulation_worker(func, args_batch, observer_position, L, 
                      K, sim_length, species_params, species):
    """
    Call calculate_cr_density() with every packet of arguments received and update
    result array on the run.

    Worker function which runs the job in each spawned process.
    """
    worker_rho = np.zeros((int(sim_length), 1))
    for args_ in args_batch:
        new_rho = func(*args_, observer_position, L, K, sim_length, species_params, species)
        ZERO_PADDING_SIZE = worker_rho.shape[0] - new_rho.shape[0]
        new_rho = np.concatenate((np.zeros((ZERO_PADDING_SIZE, 1)), new_rho), axis=0)
        worker_rho += new_rho

    return worker_rho


def simulate_sources(func, arguments, n_jobs, verbose, indices, observer_position, sim_length, 
                     L, K, species_params, species):
    """
    Simulates CR density contributions at the sun from sources whose positions and start times are randomly generated. Jobs
    are given to workers in batches. Results from each source are summed in batches and then returned. The returned batches
    are summed into a final result called cr_density.
    """

    with Parallel(n_jobs=-1, verbose=verbose) as parallel:
        # Bundle up jobs:
        funcs = repeat(func, n_jobs)  # functools.partial seems not pickle-able
        L_arg = repeat(L, n_jobs)
        K_arg = repeat(K, n_jobs)
        observer_arg = repeat(observer_position, n_jobs)
        sim_len_arg = repeat(sim_length, n_jobs)
        species_params_arg = repeat(species_params, n_jobs)
        species_arg = repeat(species, n_jobs)
        args_batches = np.array_split(arguments, indices, axis=0)
        jobs = zip(funcs, args_batches, observer_arg, L_arg, 
                   K_arg, sim_len_arg, species_params_arg, species_arg)

        cr_quantities = parallel(delayed(simulation_worker)(*job) for job in jobs)

        cr_quantities = sum(cr_quantities)

    return cr_quantities


def main(L=15, K=1.274e-7, q=1.69, B=8.3, r_sol=8.3, lam_z=0.2, duration=1e6/25, e_indx=0):
    # Generate source positions and ignition times
    params = [L, K, q, B, r_sol, lam_z, duration]
    source_coords, sun_location, ignition_times = create_sources(params, dist_type='a')

    # Set up simulation arguments
    start_times = np.where(ignition_times == 1)[0] * 25
    ARGUMENTS = [*zip(source_coords, start_times)]
    N_JOBS = 100
    VERBOSE = N_JOBS

    # Find start_time indices to evenly distribute jobs for simulation
    lengths = duration - start_times / 25
    cumulative = np.cumsum(lengths) % (np.sum(lengths) / N_JOBS)
    INDICES = argrelextrema(cumulative, np.less)[0]

    n_G = 1  # H volume density in the Galaxy (cm^-3)
    n_s = 200  # H volume density in sources (cm^-3)
    c = 3e10  # speed of light (cm/s)
    tau_G = 3.64e13 * 10  # Galaxy residence time (s)

    # Carbon source generation
    R_o = 1  # Normalization constant
    decades = np.logspace(-1, 4, 6)
    multiples = np.arange(1, 10, 1)
    E = np.outer(decades, multiples).flatten()  # Energy values (GeV/n)
    q_C = R_o * E**-2.7

    # Source residence time
    tau_o = 3e14
    mu = 0.1
    delta = 0.19
    tau_S = tau_o * E**(- mu - delta * np.log(E))

    # Carbon source density
    sig_CC = 242 * 1e-27  # C total cross section (cm^2)
    C_s = (q_C * tau_S) / (1 + c * sig_CC * n_s * tau_S)
    
    # Simulation of Carbon
    tau_CCG = 1 / (c * n_G * sig_CC + 1/tau_G)  # (s)
    # C_params = [tau_CCG, C_s[0], tau_S[0]]
    # C_observed = simulate_sources(calculate_cr_quantities, ARGUMENTS, N_JOBS, VERBOSE, INDICES,
    #                        sun_location, duration, L, K, C_params, 'C')
    
    # # Plot and save data
    # time_data = np.arange(0, duration, 1) * 25
    # plt.rcParams.update({'font.size': 22})
    # plt.figure(figsize=(25, 8))
    # plt.plot(time_data, C_observed, linewidth=3, label='Energy density')
    # plt.title('Carbon density from discrete sources')
    # plt.ylabel(r'Observed Carbon')
    # plt.xlabel('Time (years)')
    # plt.grid('both')
    # plt.legend()
    # # plt.savefig('./plots/discrete_source_sim_cr_energy_density.pdf')
    # plt.show()
    
    # Beryllium-10 source generation
    sig_CBe10 = 3.5 * 1e-27  # Be10 partial cross section (cm^2)
    sig_Be10Be10 = 200 * 1e-27  # Be10 total cross section (cm^2)

    # Beryllium-10 source density
    Be10_S = (C_s * sig_CBe10 * n_s * tau_S) / (1 + sig_Be10Be10 * c * n_s * tau_S)
    tau_Be10_decay = (2e6 * 3.154e7) * (1 + 1.073 * E)  # Mean lifetime of Be10 (s)
    tau_Be10G = 1/(c * n_G * sig_Be10Be10 + 1 / tau_G + 1/ tau_Be10_decay)

    # Beryllium-10 simulation
    Be10_params = [tau_CCG, Be10_S[e_indx], tau_S[e_indx], n_G, sig_CBe10, tau_Be10G[e_indx], c]
    Be10_observed = simulate_sources(calculate_cr_quantities, ARGUMENTS, N_JOBS, VERBOSE, INDICES,
                                     sun_location, duration, L, K, Be10_params, 'Be10')

    # # Plot and save data
    # time_data = np.arange(0, duration, 1) * 25
    # plt.rcParams.update({'font.size': 22})
    # plt.figure(figsize=(25, 8))
    # plt.plot(time_data, Be10_observed, linewidth=3, label='Energy density')
    # plt.title('Be-10 density from discrete sources')
    # plt.ylabel(r'Observed Carbon')
    # plt.xlabel('Time (years)')
    # plt.grid('both')
    # plt.legend()
    # # plt.savefig('./plots/discrete_source_sim_cr_energy_density.pdf')
    # plt.show()

    # Beryllium-9 source generation
    sig_CBe9 = 6 * 1e-27  # Be9 partial cross section (cm^2)
    sig_Be9Be9 = 200 * 1e-27  # Be9 total cross section (cm^2)

    # Beryllium-9 source density
    Be9_S = (C_s * sig_CBe9 * n_s * tau_S) / (1 + sig_Be9Be9 * n_s * c * tau_S)
    tau_Be9G = 1/(c * n_G * sig_Be9Be9 + 1 / tau_G)

    # Beryllium-10 simulation
    Be9_params = [tau_CCG, Be9_S[e_indx], tau_S[e_indx], n_G, sig_CBe9, tau_Be9G, c]
    Be9_observed = simulate_sources(calculate_cr_quantities, ARGUMENTS, N_JOBS, VERBOSE, INDICES,
                                    sun_location, duration, L, K, Be9_params, 'Be9')
    
    # # Plot and save data
    # time_data = np.arange(0, duration, 1) * 25
    # plt.rcParams.update({'font.size': 22})
    # plt.figure(figsize=(25, 8))
    # plt.plot(time_data, Be9_observed, linewidth=3, label='Energy density')
    # plt.title('Be-9 density from discrete sources')
    # plt.ylabel(r'Observed Carbon')
    # plt.xlabel('Time (years)')
    # plt.grid('both')
    # plt.legend()
    # # plt.savefig('./plots/discrete_source_sim_cr_energy_density.pdf')
    # plt.show()

    Be10_9_ratio = Be10_observed / (Be9_observed + 1e-8 * np.max(Be9_observed))

    # Plot and save data
    time_data = np.arange(0, duration, 1) * 25
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(25, 8))
    plt.plot(time_data, Be10_9_ratio, linewidth=3)
    plt.title('Be-10 / Be-9 ratio from discrete sources')
    plt.ylabel(r'$^{10}$Be/$^{9}$Be')
    plt.xlabel('Time (years)')
    plt.grid('both')
    # plt.savefig('/Users/dawsonhuth/Documents/Cosmic Ray Theory/Be_study/Be10_Be9_ratio.pdf')
    plt.show()



if __name__ == '__main__':
    # Normalized units from Notes 100 pc = 1, 1 century = 1
    main(L=16,  # kpc
         K=1e28,  # cm^2 s^-1
         q=1.69,
         B=3.33,
         r_sol=8.3,  # kpc
         lam_z=0.3,  # kpc
         duration=3e7/25,
         e_indx=0)  # num 25 yr bins
