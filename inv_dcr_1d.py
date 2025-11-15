"""
1D Inversion of DC Resistivity Data for a Single Sounding
University of British Columbia

This tutorial demonstrates three approaches for inverting DC resistivity data:
1. Weighted least-squares inversion with fixed layer thicknesses
2. Iteratively re-weighted least-squares (IRLS) inversion for sparse/blocky models
3. Parametric inversion for layer thicknesses and electrical properties

The tutorial teaches:
- How to carry out 1D geophysical inversion with SimPEG
- How to assign appropriate uncertainties to apparent resistivity data
- Choosing suitable inversion parameters and directives
- Comparing weighted least-squares, sparse-norm and parametric inversions
- Analyzing and visualizing inversion outputs

Keywords: DC resistivity, 1D sounding, inversion, parametric, sparse norm,
          apparent resistivity, wires mapping
"""

# ============================================================================
# Import Modules
# ============================================================================

# SimPEG functionality
from simpeg.electromagnetics.static import resistivity as dc
from simpeg.utils import plot_1d_layer_model, download
from simpeg import (
    maps,
    data,
    data_misfit,
    regularization,
    optimization,
    inverse_problem,
    inversion,
    directives,
)

# discretize functionality
from discretize import TensorMesh

# Basic Python functionality
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tarfile

mpl.rcParams.update({"font.size": 14})


# ============================================================================
# Download and Extract Tutorial Data
# ============================================================================

print("=" * 80)
print("Step 1: Downloading and Extracting Tutorial Data")
print("=" * 80)

# URL to download from repository assets
data_source = "https://github.com/simpeg/user-tutorials/raw/main/assets/05-dcr/inv_dcr_1d_files.tar.gz"

# Download the data
downloaded_data = download(data_source, overwrite=True)

# Unzip the tarfile
tar = tarfile.open(downloaded_data, "r")
tar.extractall()
tar.close()

# Path to the directory containing our data
dir_path = downloaded_data.split(".")[0] + os.path.sep
data_filename = dir_path + "app_res_1d_data.dobs"

print(f"  Downloaded and extracted data to: {dir_path}")
print()


# ============================================================================
# Load and Plot the Data
# ============================================================================

print("=" * 80)
print("Step 2: Loading and Plotting Observed Data")
print("=" * 80)

# Load data
dobs = np.loadtxt(str(data_filename))

# Extract A, B, M and N electrode locations and the observed data
A_electrodes = dobs[:, 0:3]
B_electrodes = dobs[:, 3:6]
M_electrodes = dobs[:, 6:9]
N_electrodes = dobs[:, 9:12]
dobs = dobs[:, -1]

# Compute the AB/2 separation for the 1D Wenner array
AB_separations = np.sqrt(np.sum((A_electrodes - B_electrodes) ** 2, axis=1))

print(f"  Number of data points: {len(dobs)}")
print(f"  AB/2 range: [{AB_separations.min()/2:.1f}, {AB_separations.max()/2:.1f}] m")
print(f"  Apparent resistivity range: [{dobs.min():.1f}, {dobs.max():.1f}] Ohm-m")

# Plot apparent resistivity sounding curve
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_axes([0.15, 0.1, 0.7, 0.85])
ax1.semilogy(AB_separations / 2, dobs, "b-o")
ax1.grid(which="both")
ax1.set_xlabel("AB/2 (m)")
ax1.set_ylabel(r"Apparent Resistivity ($\Omega m$)")
ax1.set_title("Observed Sounding Data")
plt.savefig("dcr_1d_observed_data.png", dpi=150, bbox_inches="tight")
print("  Sounding data plot saved as 'dcr_1d_observed_data.png'")
plt.close()
print()


# ============================================================================
# Assign Uncertainties
# ============================================================================

print("=" * 80)
print("Step 3: Assigning Uncertainties to Data")
print("=" * 80)

# Add 2.5% uncertainties to all data
uncertainties = 0.025 * np.abs(dobs)

print(f"  Applied 2.5% uncertainty to all data")
print(f"  Uncertainty range: [{uncertainties.min():.2f}, {uncertainties.max():.2f}] Ohm-m")
print()


# ============================================================================
# Define the Survey
# ============================================================================

print("=" * 80)
print("Step 4: Defining DC Resistivity Survey")
print("=" * 80)

# Sort by unique current electrode locations
unique_tx, k = np.unique(np.c_[A_electrodes, B_electrodes], axis=0, return_index=True)
n_sources = len(k)
k = np.sort(k)
k = np.r_[k, len(dobs)]

# Define empty list for sources
source_list = []

# Loop over all sources
for ii in range(0, n_sources):
    # MN electrode locations for receivers
    M_locations = M_electrodes[k[ii] : k[ii + 1], :]
    N_locations = N_electrodes[k[ii] : k[ii + 1], :]
    receiver_list = [
        dc.receivers.Dipole(
            M_locations,
            N_locations,
            data_type="apparent_resistivity",
        )
    ]

    # AB electrode locations for source
    A_location = A_electrodes[k[ii], :]
    B_location = B_electrodes[k[ii], :]
    source_list.append(dc.sources.Dipole(receiver_list, A_location, B_location))

# Define survey
survey = dc.Survey(source_list)

print(f"  Number of sources: {n_sources}")
print(f"  Total data points: {survey.nD}")
print()


# ============================================================================
# Define the Data Object
# ============================================================================

print("=" * 80)
print("Step 5: Creating Data Object")
print("=" * 80)

data_object = data.Data(survey, dobs=dobs, standard_deviation=uncertainties)

print(f"  Data object created with {len(data_object.dobs)} observations")
print()


# ============================================================================
# WEIGHTED LEAST-SQUARES INVERSION
# ============================================================================

print("=" * 80)
print("APPROACH 1: WEIGHTED LEAST-SQUARES INVERSION")
print("=" * 80)
print()

# ============================================================================
# Design a 1D Layered Earth (L2)
# ============================================================================

print("=" * 80)
print("Step 6: Designing 1D Layered Earth Model")
print("=" * 80)

# Use Wenner electrode spacings to set discretization parameters
depth_min = 5.0  # top layer thickness
depth_max = np.max(AB_separations / 2)  # depth to lowest layer
geometric_factor = 1.1  # rate of thickness increase

# Increase subsequent layer thicknesses by geometric factor
layer_thicknesses = [depth_min]
while np.sum(layer_thicknesses) < depth_max:
    layer_thicknesses.append(geometric_factor * layer_thicknesses[-1])

n_layers = len(layer_thicknesses) + 1

print(f"  Top layer thickness: {depth_min} m")
print(f"  Maximum depth: {depth_max:.1f} m")
print(f"  Geometric factor: {geometric_factor}")
print(f"  Number of layers: {n_layers}")
print()


# ============================================================================
# Define Model and Mapping (L2)
# ============================================================================

print("=" * 80)
print("Step 7: Defining Model and Mapping for L2 Inversion")
print("=" * 80)

log_resistivity_map = maps.ExpMap(nP=n_layers)

# Starting model is log-resistivity values (Ohm-m)
starting_resistivity_model = np.log(1e3 * np.ones(n_layers))

# Reference model
reference_resistivity_model = starting_resistivity_model.copy()

print(f"  Model parameters: {n_layers}")
print(f"  Starting resistivity: {np.exp(starting_resistivity_model[0]):.1f} Ohm-m")
print()


# ============================================================================
# Define Forward Simulation (L2)
# ============================================================================

print("=" * 80)
print("Step 8: Defining Forward Simulation for L2 Inversion")
print("=" * 80)

simulation_L2 = dc.simulation_1d.Simulation1DLayers(
    survey=survey,
    rhoMap=log_resistivity_map,
    thicknesses=layer_thicknesses,
)

print(f"  Simulation type: 1D Layered Earth")
print(f"  Number of layers: {n_layers}")
print()


# ============================================================================
# Define Data Misfit (L2)
# ============================================================================

print("=" * 80)
print("Step 9: Defining Data Misfit for L2 Inversion")
print("=" * 80)

dmis_L2 = data_misfit.L2DataMisfit(simulation=simulation_L2, data=data_object)

print(f"  Data misfit type: L2 norm")
print()


# ============================================================================
# Define Regularization (L2)
# ============================================================================

print("=" * 80)
print("Step 10: Defining Regularization for L2 Inversion")
print("=" * 80)

# Define 1D cell widths
h = np.r_[layer_thicknesses, layer_thicknesses[-1]]
h = np.flipud(h)

# Create regularization mesh
regularization_mesh = TensorMesh([h], "N")

reg_L2 = regularization.WeightedLeastSquares(
    regularization_mesh,
    length_scale_x=1.0,
    reference_model=reference_resistivity_model,
    reference_model_in_smooth=False,
)

print(f"  Regularization mesh cells: {regularization_mesh.nC}")
print(f"  Regularization type: Weighted Least Squares")
print()


# ============================================================================
# Define Optimization, Inverse Problem, and Directives (L2)
# ============================================================================

print("=" * 80)
print("Step 11: Setting Up L2 Inversion")
print("=" * 80)

opt_L2 = optimization.InexactGaussNewton(
    maxIter=100, maxIterLS=20, maxIterCG=20, tolCG=1e-3
)

inv_prob_L2 = inverse_problem.BaseInvProblem(dmis_L2, reg_L2, opt_L2)

sensitivity_weights = directives.UpdateSensitivityWeights(every_iteration=True)
update_jacobi = directives.UpdatePreconditioner(update_every_iteration=True)
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=5)
beta_schedule = directives.BetaSchedule(coolingFactor=2.0, coolingRate=2)
target_misfit = directives.TargetMisfit(chifact=1.0)

directives_list_L2 = [
    sensitivity_weights,
    update_jacobi,
    starting_beta,
    beta_schedule,
    target_misfit,
]

print(f"  Optimization: Inexact Gauss Newton")
print(f"  Maximum iterations: 100")
print(f"  Number of directives: {len(directives_list_L2)}")
print()


# ============================================================================
# Run L2 Inversion
# ============================================================================

print("=" * 80)
print("Step 12: Running Weighted Least-Squares Inversion")
print("=" * 80)

inv_L2 = inversion.BaseInversion(inv_prob_L2, directives_list_L2)
recovered_model_L2 = inv_L2.run(starting_resistivity_model)

print()
print(f"  L2 inversion completed successfully")
print()


# ============================================================================
# ITERATIVELY RE-WEIGHTED LEAST-SQUARES INVERSION
# ============================================================================

print("=" * 80)
print("APPROACH 2: ITERATIVELY RE-WEIGHTED LEAST-SQUARES (IRLS) INVERSION")
print("=" * 80)
print()

# ============================================================================
# Set Up IRLS Inversion
# ============================================================================

print("=" * 80)
print("Step 13: Setting Up IRLS Inversion")
print("=" * 80)

simulation_irls = dc.simulation_1d.Simulation1DLayers(
    survey=survey,
    rhoMap=log_resistivity_map,
    thicknesses=layer_thicknesses,
)

dmis_irls = data_misfit.L2DataMisfit(simulation=simulation_irls, data=data_object)

reg_irls = regularization.Sparse(
    regularization_mesh,
    alpha_s=0.1,
    alpha_x=1,
    reference_model_in_smooth=False,
    norms=[1.0, 1.0],
)

opt_irls = optimization.InexactGaussNewton(
    maxIter=100, maxIterLS=20, maxIterCG=30, tolCG=1e-3
)

inv_prob_irls = inverse_problem.BaseInvProblem(dmis_irls, reg_irls, opt_irls)

sensitivity_weights_irls = directives.UpdateSensitivityWeights(every_iteration=True)
starting_beta_irls = directives.BetaEstimate_ByEig(beta0_ratio=1)
update_jacobi_irls = directives.UpdatePreconditioner(update_every_iteration=True)
update_irls = directives.UpdateIRLS(
    cooling_factor=2,
    cooling_rate=2,
    f_min_change=1e-4,
    max_irls_iterations=30,
    chifact_start=1.0,
)

directives_list_irls = [
    update_irls,
    sensitivity_weights_irls,
    starting_beta_irls,
    update_jacobi_irls,
]

print(f"  Regularization type: Sparse (IRLS)")
print(f"  Norms: [1.0, 1.0] for sparse/blocky recovery")
print(f"  Maximum IRLS iterations: 30")
print()


# ============================================================================
# Run IRLS Inversion
# ============================================================================

print("=" * 80)
print("Step 14: Running IRLS Inversion")
print("=" * 80)

inv_irls = inversion.BaseInversion(inv_prob_irls, directives_list_irls)
recovered_model_irls = inv_irls.run(starting_resistivity_model)

print()
print(f"  IRLS inversion completed successfully")
print()


# ============================================================================
# PARAMETRIC INVERSION
# ============================================================================

print("=" * 80)
print("APPROACH 3: PARAMETRIC INVERSION")
print("=" * 80)
print()

# ============================================================================
# Set Up Parametric Inversion
# ============================================================================

print("=" * 80)
print("Step 15: Setting Up Parametric Inversion")
print("=" * 80)

# Wire maps to extract log-thicknesses and log-conductivities
wire_map = maps.Wires(("log_thicknesses", 2), ("log_conductivity", 3))

# Mapping for layer thicknesses
log_thicknesses_map = maps.ExpMap() * wire_map.log_thicknesses

# Mapping for conductivities
log_conductivity_map = maps.ExpMap() * wire_map.log_conductivity

# Starting model for 3-layer Earth
starting_parametric_model = np.log(np.r_[125.0, 50.0, 1e-3, 2e-3, 5e-2])
reference_parametric_model = starting_parametric_model.copy()

simulation_parametric = dc.simulation_1d.Simulation1DLayers(
    survey=survey,
    sigmaMap=log_conductivity_map,
    thicknessesMap=log_thicknesses_map,
)

dmis_parametric = data_misfit.L2DataMisfit(
    simulation=simulation_parametric, data=data_object
)

reg_1 = regularization.Smallness(
    TensorMesh([(np.ones(2))], "0"),
    mapping=wire_map.log_thicknesses,
    reference_model=reference_parametric_model,
)

reg_2 = regularization.Smallness(
    TensorMesh([(np.ones(3))], "0"),
    mapping=wire_map.log_conductivity,
    reference_model=reference_parametric_model,
)

reg_parametric = reg_1 + reg_2
reg_parametric.multipliers = [1.0, 0.1]

opt_parametric = optimization.InexactGaussNewton(
    maxIter=100, maxIterLS=20, maxIterCG=20, tolCG=1e-3
)

inv_prob_parametric = inverse_problem.BaseInvProblem(
    dmis_parametric, reg_parametric, opt_parametric
)

sensitivity_weights = directives.UpdateSensitivityWeights(every_iteration=True)
update_jacobi = directives.UpdatePreconditioner(update_every_iteration=True)
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=10)
beta_schedule = directives.BetaSchedule(coolingFactor=2.0, coolingRate=2)
target_misfit = directives.TargetMisfit(chifact=1.0)

directives_list_parametric = [
    sensitivity_weights,
    update_jacobi,
    starting_beta,
    beta_schedule,
    target_misfit,
]

print(f"  Model parameters: 2 thicknesses + 3 conductivities = 5 total")
print(f"  Assumed structure: 3-layer Earth")
print()


# ============================================================================
# Run Parametric Inversion
# ============================================================================

print("=" * 80)
print("Step 16: Running Parametric Inversion")
print("=" * 80)

inv_parametric = inversion.BaseInversion(
    inv_prob_parametric, directives_list_parametric
)
recovered_model_parametric = inv_parametric.run(starting_parametric_model)

print()
print(f"  Parametric inversion completed successfully")
print()


# ============================================================================
# Plot Observed and Predicted Data Comparison
# ============================================================================

print("=" * 80)
print("Step 17: Plotting Data Fit for All Three Approaches")
print("=" * 80)

# Plot the observed and predicted data
fig = plt.figure(figsize=(11, 5))
ax1 = fig.add_axes([0.1, 0.15, 0.85, 0.75])
ax1.semilogy(AB_separations / 2, dobs, "k-o", lw=2, label="Observed")
ax1.semilogy(
    AB_separations / 2,
    simulation_L2.dpred(recovered_model_L2),
    "b-s",
    lw=2,
    label="L2 Inversion",
)
ax1.semilogy(
    AB_separations / 2,
    simulation_irls.dpred(recovered_model_irls),
    "r-^",
    lw=2,
    label="IRLS Inversion",
)
ax1.semilogy(
    AB_separations / 2,
    simulation_parametric.dpred(recovered_model_parametric),
    "g-d",
    lw=2,
    label="Parametric Inversion",
)
ax1.grid(which="both")
ax1.set_xlabel("AB/2 (m)")
ax1.set_ylabel(r"Apparent Resistivity ($\Omega m$)")
ax1.set_title("Data Fit Comparison")
ax1.legend()
plt.savefig("dcr_1d_data_fit_comparison.png", dpi=150, bbox_inches="tight")
print("  Data fit comparison saved as 'dcr_1d_data_fit_comparison.png'")
plt.close()
print()


# ============================================================================
# Plot Recovered Models
# ============================================================================

print("=" * 80)
print("Step 18: Plotting Recovered Models")
print("=" * 80)

# Define true model
true_resistivities = np.r_[1e3, 4e3, 2e2]
true_layers = np.r_[100.0, 100.0]

# Plot all models
fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_axes([0.2, 0.15, 0.7, 0.7])
plot_1d_layer_model(true_layers, true_resistivities, ax=ax1, color="k", label="True Model")
plot_1d_layer_model(
    layer_thicknesses,
    log_resistivity_map * recovered_model_L2,
    ax=ax1,
    color="b",
    label="L2 Model",
)
plot_1d_layer_model(
    layer_thicknesses,
    log_resistivity_map * recovered_model_irls,
    ax=ax1,
    color="r",
    label="IRLS Model",
)
plot_1d_layer_model(
    log_thicknesses_map * recovered_model_parametric,
    1 / (log_conductivity_map * recovered_model_parametric),
    ax=ax1,
    color="g",
    label="Parametric Model",
)
x_min, x_max = true_resistivities.min(), true_resistivities.max()
ax1.set_xlim(0.8 * x_min, 2 * x_max)
ax1.set_ylim([np.sum(layer_thicknesses), 0])
ax1.grid()
ax1.set_xlabel(r"Resistivity ($\Omega m$)")
ax1.set_title("Recovered Models Comparison")
ax1.legend()
plt.savefig("dcr_1d_models_comparison.png", dpi=150, bbox_inches="tight")
print("  Models comparison saved as 'dcr_1d_models_comparison.png'")
plt.close()
print()


# ============================================================================
# Summary
# ============================================================================

print("=" * 80)
print("TUTORIAL COMPLETE")
print("=" * 80)
print()
print("Summary of Results:")
print("-" * 80)
print(f"  Data points:              {len(dobs)}")
print(f"  AB/2 range:               [{AB_separations.min()/2:.1f}, {AB_separations.max()/2:.1f}] m")
print(f"  Number of layers:         {n_layers}")
print()
print("Inversion Results:")
print("-" * 80)
print()
print("Generated Files:")
print("-" * 80)
print("  1. dcr_1d_observed_data.png          - Observed sounding curve")
print("  2. dcr_1d_data_fit_comparison.png    - Data fit for all approaches")
print("  3. dcr_1d_models_comparison.png      - Comparison of recovered models")
print()
print("Key Findings:")
print("-" * 80)
print("  - L2 inversion recovers smooth models")
print("  - IRLS inversion recovers blocky/sparse structures")
print("  - Parametric inversion assumes fixed number of layers")
print("  - All three approaches fit the data reasonably well")
print("=" * 80)
