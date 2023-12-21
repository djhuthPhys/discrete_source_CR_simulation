#include <iostream>
#include <fstream>
#include <tuple>
#include <random>
#include <vector>
#include <algorithm>
#include <execution>
#include <typeinfo>
#include <cmath>
#include <assert.h>
#include "Eigen/Core"
#include "Eigen/Dense"

using namespace Eigen;


// Define format for saving Eigen arrays to CSV
// see https://eigen.tuxfamily.org/dox/structEigen_1_1IOFormat.html
const static IOFormat CSVFormat(StreamPrecision, Eigen::DontAlignCols, ", ", "\n");


// Function taking Eigen object and file name and saves data to .csv file, 
// see https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
template <typename Derived>
void writeToCSVfile(std::string name, const Eigen::ArrayBase<Derived>& matrix)
{
    std::ofstream file(name.c_str());
    file << matrix.format(CSVFormat);
    // file.close() is not necessary, 
    // desctructur closes file, see https://en.cppreference.com/w/cpp/io/basic_ofstream
}


/**
 * @brief Calculates the coordinates of the centroids of the galactic 
 *        arms at radial values r_values
 * @param r_values Eigen array containing radial values
 * @param params Eigen array containing parameter values 
 *               which characterize one of the Milky Way arms
 * @return Eigen array containing theta values of an arm
 */
ArrayXd galactic_arms(ArrayXd r_values, ArrayXXd params)
{
    return params(0) * log(r_values/params(1)) + params(2);
}


/**
 * @brief Calculates the theta coordinate of a source at radial value r_value
 * @param r_value Double containing radial value
 * @param params Eigen array containing parameter values 
 *               which characterize one of the Milky Way arms
 * @return Eigen array containing theta values of an arm
 */
double find_source_theta(double r_value, ArrayXXd params)
{
    return params(0) * log(r_value/params(1)) + params(2);
}


/**
 * @brief Creates a population of supernova sources based on 
 *        model_params and returns their positions, ignition times
 *        and the position of the sun in the simulated Milky Way
 * @param model_params Eigen array containing model parameters
 * @return tuple<ArrayXd> 
 */
std::tuple<ArrayXXd, ArrayXi, ArrayXXd> create_sources(ArrayXd model_params)
{
    // Define model parameters
    double L = model_params(0); // Halo size in kpc
    double K = model_params(1); // Diffusion coefficient in kpc^2/yr
    double q = model_params(2); // Model parameter, unitless
    double B = model_params(3); // Model parameter, unitless
    double r_sol = model_params(4); // Sun distance from Galactic Center
    double lam_z = model_params(5); // Scale height of z source dist.
    double sim_length = model_params(6); // Number of 25 yr time bins

    // Define galactic arms parameters
    ArrayXXd arms_params(4,3); 
    arms_params << 4.25, 3.48, 1.57, // Norma
                   4.25, 3.48, 4.71, // Carina Sagittarius
                   4.89, 4.90, 4.09, // Perseus
                   4.89, 4.90, 0.95; // Crux Scutum

    // Generate arm centroid coordinates
    ArrayXd r_values = VectorXd::LinSpaced(2000000, 0.01, 20);
    ArrayXd z_values = VectorXd::LinSpaced(2000000, 0, L/2);

    ArrayXXd arm_cents(r_values.rows(), arms_params.rows());
    for(int arm=0; arm <= 3; arm++){
        arm_cents(all, arm) = galactic_arms(r_values, arms_params(arm, all));
    }

    // Calculate radial and vertical source density distribution
    double norm_const = 1 / (pow(r_values / r_sol, q) * exp(-B * (r_values / r_sol))).sum();
    ArrayXd radial_source_density = norm_const * pow(r_values / r_sol, q) * exp(-B * (r_values / r_sol));

    norm_const = 1 / (1 / pow(cosh(z_values / lam_z), 2)).sum();
    ArrayXd vertical_source_density = norm_const / pow(cosh(z_values / lam_z), 2);

    // Determine source creation times
    std::random_device rd;
    std::mt19937 engine(rd());
    std::discrete_distribution<double> time_distribution {0.75, 0.25};
    ArrayXd creation_status = ArrayXd::Zero(sim_length);
    for(int i = 0; i < sim_length; i++){
        creation_status(i) = time_distribution(rd);
    }

    int num_sources = creation_status.sum();
    std::cout << num_sources << " sources generated" << std::endl;

    ArrayXi start_times = ArrayXi::Zero(num_sources);
    int idx = 0;
    for(int i=0; i<creation_status.size(); ++i){
        if(creation_status(i) == 1){
            start_times(idx) = i * 25; // 25 year time bins
            idx++;
        }
    }

    // Choose arms for sources to lie in
    std::discrete_distribution<double> arm_distribution {0.25, 0.25, 0.25, 0.25};
    ArrayXi source_arms = ArrayXi::Zero(num_sources);
    for(int i = 0; i < num_sources; i++){
        source_arms(i) = arm_distribution(rd);
    }

    // Choose radial values of sources
    std::discrete_distribution<double> radial_distribution (radial_source_density.begin(), radial_source_density.end());
    ArrayXd source_radii = ArrayXd::Zero(num_sources);
    for(int i = 0; i < num_sources; i++){
        source_radii(i) = radial_distribution(rd);
    }
    source_radii = r_values(source_radii);

    // Determine theta values of each source from choosen arm and radial values
    ArrayXd source_theta = ArrayXd::Zero(num_sources);
    for(int i = 0; i < num_sources; i++){
        source_theta(i) = find_source_theta(source_radii(i), arms_params(source_arms(i), all));
    }
    
    // Make corrections to source r and theta values for a more natural distribution
    // Choose radial values of sources
    std::uniform_real_distribution<double> uniform_dist(0.0, 2 * M_PI);
    ArrayXd theta_corr = ArrayXd::Zero(num_sources);
    for(int i = 0; i < num_sources; i++){
        theta_corr(i) = uniform_dist(rd);
    }
    source_theta += theta_corr * exp(-0.35 * source_radii);

    ArrayXd r_corr = ArrayXd::Zero(num_sources);
    for(int i = 0; i < num_sources; i++){
        std::normal_distribution<double> normal_dist(0, 0.07 * source_radii(i)); // standard deviation proportional to radial value of source
        r_corr(i) = normal_dist(rd);
    }

    source_radii += r_corr;

    // Choose z values of sources
    std::discrete_distribution<int> vertical_distribution (vertical_source_density.begin(), vertical_source_density.end());
    std::discrete_distribution<double> sign_distribution {0.5, 0.5}; // Determines source position above or below Galactic plane
    ArrayXd source_z = ArrayXd::Zero(num_sources);
    for(int i = 0; i < num_sources; i++){
        source_z(i) = z_values(vertical_distribution(rd)) * (-1  + 2 * sign_distribution(rd));
    }

    // Convert source position to cartesian coordinates
    ArrayXd source_x = source_radii * cos(source_theta);
    ArrayXd source_y = source_radii * sin(source_theta);

    // Concatenate into single array and write to file
    ArrayXXd source_coords(num_sources, 3);
    source_coords << source_x, source_y, source_z;
    writeToCSVfile("source_coodinates.csv", source_coords);

    ArrayXXd sun_coords(1, 3);
    sun_coords << 0, 
                  8.3, 
                  0.2;

    // Check that the sun is not within 1 pc of a source
    auto differences = source_coords - sun_coords.replicate(source_coords.rows(), 1);
    ArrayXXd distances = sqrt(pow(differences, 2).colwise().sum());
    if(distances.minCoeff() >= 0.001){
        std::cout << "The sun is safe." << std::endl;
    } else {
        assert (distances.minCoeff() >= 0.001);
        std::cout << "The sun is not safe.";
    }

    return {source_coords, start_times, sun_coords};
}


/**
 * @brief Calculates the energy density and energy density gradients for a source
 * 
 * @param source_position An eigen array containing the x, y, z, coordinates of the source
 * @param observer_position An eigen array containing the x, y, z coordinates of the sun
 * @param t_start The time the source explodes
 * @param L Halo size of the modeled Galaxy
 * @param K Diffusion coefficient for CR diffusion
 * @param sim_length Number of 25 year time bins the simulation represents
 * @return A tuple containing arrays for the energy density of the source and gradients
 *         as a function of 25 year time bins
 */
int calculate_cr_quantities(ArrayXXd source_position, ArrayXXd observer_position, double t_start, double L, double K, double sim_length)
{
    // Initialize relevant parameters
    double x_s = source_position(0);
    double y_s = source_position(1);
    double z_s = source_position(2);
    double x = observer_position(0);
    double y = observer_position(1);
    double z =  observer_position(2);

    // Create arrays for time and N values
    ArrayXd time_arr = VectorXd::LinSpaced((sim_length*25 - t_start)/25 + 1, t_start, sim_length*25);
    ArrayXXd N = VectorXd::LinSpaced(100, 1, 100).transpose().replicate(time_arr.rows(), 1);

    // Calculate energy density and gradients for the CR source
    ArrayXXd common = (2/L * sin(N * M_PI * z_s /L) * exp(-(K * pow(M_PI, 2) * (pow(N, 2).colwise() * (time_arr - t_start)))/pow(L, 2))).colwise()
                      * ((1 / (4 * M_PI * K * (time_arr - t_start) + 1e-8))
                      * exp(-(pow((x - x_s), 2) + pow((y - y_s), 2))/(4 * K * (time_arr - t_start) + 1e-8)));
    
    ArrayXXd rho = (common * sin(M_PI * N * z / L)).rowwise().sum();
    writeToCSVfile("./test_density.csv", rho);

    ArrayXXd rho_z = ((M_PI * N / L) * common * cos(M_PI * N * z / L)).rowwise().sum();

    ArrayXXd rho_x_o = ((- common * sin(M_PI * N * z / L)).colwise() 
                     * ((x - x_s) / (2 * K * (time_arr - t_start) + 1e-8))).rowwise().sum();
    
    // ArrayXd rho_x = ArrayXXd::Zero(time_arr.rows(), 1);
    // for(int i=0; i<rho_x_o.cols(); i++){
    //     rho_x += rho_x_o(all, i);
    // }

    ArrayXXd rho_y_o = ((- common * sin(M_PI * N * z / L)).colwise() 
                     * ((y - y_s) / (2 * K * (time_arr - t_start) + 1e-8))).rowwise().sum();
    
    // ArrayXd rho_y = ArrayXXd::Zero(time_arr.rows(), 1);
    // for(int i=0; i<rho_y_o.cols(); i++){
    //     rho_y += rho_y_o(all, i);
    // }

    // Pad all quantities, concatenate gradients and return results in a tuple to sum to running total

    return 0;
}


int simulate(ArrayXXd source_position, ArrayXXd observer_position, ArrayXi start_times, ArrayXd model_params)
{
    double L = model_params(0); // Halo size in kpc
    double K = model_params(1); // Diffusion coefficient in kpc^2/yr
    int sim_length = model_params(6); // Number of 25 yr time bins

    // ArrayXXd energy_density = ArrayXXd::Zero(sim_length, 1);
    // ArrayXXd density_gradients = ArrayXXd::Zero(sim_length, 3);
    for(int i = 0; i < 1; i++){
        calculate_cr_quantities(source_position(i, all), observer_position, start_times(i), L, K, sim_length);
        std::cout << "Finished source " << i+1 << std::endl;
    }

    return 0;
}


int main()
{
    ArrayXd parameters(7);
    parameters << 15, 1.274e-7, 1.69, 3.3, 8.3, 0.1, 4e6;

    auto [source_coordinates, start_times, sun_coordinates] = create_sources(parameters);

    int test = simulate(source_coordinates, sun_coordinates, start_times, parameters);

    return 0;
}