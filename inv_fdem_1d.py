"""
1D Inversion of Frequency Domain EM Data for a Single Sounding
University of British Columbia

This tutorial demonstrates three approaches for inverting FDEM data:
1. Weighted least-squares inversion with fixed layer thicknesses
2. Iteratively re-weighted least-squares (IRLS) inversion for sparse/blocky models
3. Parametric inversion for layer thicknesses and electrical properties

The tutorial teaches:
- How to carry out 1D FDEM inversion with SimPEG
- How to assign appropriate uncertainties to FDEM data
- Choosing suitable inversion parameters and directives
- Comparing weighted least-squares, sparse-norm and parametric inversions
- Analyzing and visualizing inversion outputs

Keywords: frequency-domain EM, 1D sounding, inversion, parametric, sparse norm,
          wires mapping
"""

# ============================================================================
# Import Modules
# ============================================================================

# SimPEG functionality
import simpeg.electromagnetics.frequency_domain as fdem
from simpeg.utils import plot_1d_layer_model, download, mkvc
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

# URL to assets
data_source = "https://github.com/simpeg/user-tutorials/raw/main/assets/07-fdem/inv_fdem_1d_files.tar.gz"

# Download the data
downloaded_data = download(data_source, overwrite=True)

# Unzip the tarfile
tar = tarfile.open(downloaded_data, "r")
tar.extractall()
tar.close()

# Path to the directory containing our data
dir_path = downloaded_data.split(".")[0] + os.path.sep
data_filename = dir_path + "em1dfm_data.txt"

print(f"  Downloaded and extracted data to: {dir_path}")
print()


# ============================================================================
# Load and Plot the Data
# ============================================================================

print("=" * 80)
print("Step 2: Loading and Plotting Observed Data")
print("=" * 80)

# Load data
dobs = np.loadtxt(str(data_filename), skiprows=1)

# Extract frequency and observed data columns
frequencies = dobs[:, 0]
dobs = mkvc(dobs[:, 1:].T)

print(f"  Number of frequencies: {len(frequencies)}")
print(f"  Frequency range: [{frequencies.min():.1f}, {frequencies.max():.1f}] Hz")
print(f"  Number of data points: {len(dobs)}")
print(f"  Data range: [{dobs.min():.2e}, {dobs.max():.2e}] ppm")

# Plot data
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.loglog(frequencies, np.abs(dobs[0::2]), "k-o", lw=2, label="Real")
ax.loglog(frequencies, np.abs(dobs[1::2]), "k:o", lw=2, label="Imaginary")
ax.grid(which="both")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("|Hs/Hp| (ppm)")
ax.set_title("Sounding Data")
ax.legend()
plt.savefig("fdem_1d_observed_data.png", dpi=150, bbox_inches="tight")
print("  Sounding data plot saved as 'fdem_1d_observed_data.png'")
plt.close()
print()


# ============================================================================
# Define the Survey
# ============================================================================

print("=" * 80)
print("Step 3: Defining FDEM Survey")
print("=" * 80)

source_location = np.array([0.0, 0.0, 30.0])
source_orientation = "z"
moment = 1.0

receiver_location = np.array([10.0, 0.0, 30.0])
receiver_orientation = "z"
data_type = "ppm"

# Receiver list
receiver_list = []
receiver_list.append(
    fdem.receivers.PointMagneticFieldSecondary(
        receiver_location,
        orientation=receiver_orientation,
        data_type=data_type,
        component="real",
    )
)
receiver_list.append(
    fdem.receivers.PointMagneticFieldSecondary(
        receiver_location,
        orientation=receiver_orientation,
        data_type=data_type,
        component="imag",
    )
)

# Define source list
source_list = []
for freq in frequencies:
    source_list.append(
        fdem.sources.MagDipole(
            receiver_list=receiver_list,
            frequency=freq,
            location=source_location,
            orientation=source_orientation,
            moment=moment,
        )
    )

# Survey
survey = fdem.survey.Survey(source_list)

print(f"  Number of sources (frequencies): {len(source_list)}")
print(f"  Source type: Vertical magnetic dipole")
print(f"  Receiver type: Vertical magnetic field (secondary)")
print(f"  Source-receiver offset: 10 m")
print()


# ============================================================================
# Assign Uncertainties
# ============================================================================

print("=" * 80)
print("Step 4: Assigning Uncertainties to Data")
print("=" * 80)

# 5% of the absolute value
uncertainties = 0.05 * np.abs(dobs) * np.ones(np.shape(dobs))

print(f"  Applied 5% uncertainty to all data")
print()


# ============================================================================
# Define the Data Object
# ============================================================================

print("=" * 80)
print("Step 5: Creating Data Object")
print("=" * 80)

data_object = data.Data(survey, dobs=dobs, noise_floor=uncertainties)

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

# Estimated host conductivity (S/m)
estimated_conductivity = 0.1

# Minimum and maximum skin depths
d_min = 500.0 / np.sqrt(estimated_conductivity * frequencies.max())
d_max = 500.0 / np.sqrt(estimated_conductivity * frequencies.min())

print(f"  Estimated conductivity: {estimated_conductivity} S/m")
print(f"  Minimum skin depth: {d_min:.2f} m")
print(f"  Maximum skin depth: {d_max:.2f} m")

depth_min = 0.5  # top layer thickness
depth_max = 200.0  # depth to lowest layer
geometric_factor = 1.1  # rate of thickness increase

# Increase subsequent layer thicknesses
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

log_conductivity_map = maps.ExpMap(nP=n_layers)

# Starting model is log-conductivity values (S/m)
starting_conductivity_model = np.log(1e-1 * np.ones(n_layers))

# Reference model
reference_conductivity_model = starting_conductivity_model.copy()

print(f"  Model parameters: {n_layers}")
print(f"  Starting conductivity: {np.exp(starting_conductivity_model[0]):.2f} S/m")
print()


# ============================================================================
# Define Forward Simulation (L2)
# ============================================================================

print("=" * 80)
print("Step 8: Defining Forward Simulation for L2 Inversion")
print("=" * 80)

simulation_L2 = fdem.Simulation1DLayered(
    survey=survey, thicknesses=layer_thicknesses, sigmaMap=log_conductivity_map
)

print(f"  Simulation type: 1D Layered Earth")
print(f"  Number of layers: {n_layers}")
print()


# ============================================================================
# Define Data Misfit and Regularization (L2)
# ============================================================================

print("=" * 80)
print("Step 9: Defining Data Misfit and Regularization for L2")
print("=" * 80)

dmis_L2 = data_misfit.L2DataMisfit(simulation=simulation_L2, data=data_object)

# Define 1D regularization mesh
h = np.r_[layer_thicknesses, layer_thicknesses[-1]]
h = np.flipud(h)
regularization_mesh = TensorMesh([h], "N")

reg_L2 = regularization.WeightedLeastSquares(
    regularization_mesh,
    length_scale_x=10.0,
    reference_model=reference_conductivity_model,
    reference_model_in_smooth=False,
)

print(f"  Data misfit: L2 norm")
print(f"  Regularization mesh cells: {regularization_mesh.nC}")
print(f"  Regularization type: Weighted Least Squares")
print()


# ============================================================================
# Set Up and Run L2 Inversion
# ============================================================================

print("=" * 80)
print("Step 10: Setting Up and Running L2 Inversion")
print("=" * 80)

opt_L2 = optimization.InexactGaussNewton(
    maxIter=100, maxIterLS=20, maxIterCG=20, tolCG=1e-3
)

inv_prob_L2 = inverse_problem.BaseInvProblem(dmis_L2, reg_L2, opt_L2)

update_jacobi = directives.UpdatePreconditioner(update_every_iteration=True)
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=5)
beta_schedule = directives.BetaSchedule(coolingFactor=2.0, coolingRate=3)
target_misfit = directives.TargetMisfit(chifact=1.0)

directives_list_L2 = [update_jacobi, starting_beta, beta_schedule, target_misfit]

print(f"  Optimization: Inexact Gauss Newton")
print(f"  Number of directives: {len(directives_list_L2)}")

inv_L2 = inversion.BaseInversion(inv_prob_L2, directives_list_L2)
recovered_model_L2 = inv_L2.run(starting_conductivity_model)

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
# Set Up and Run IRLS Inversion
# ============================================================================

print("=" * 80)
print("Step 11: Setting Up IRLS Inversion")
print("=" * 80)

simulation_irls = fdem.simulation_1d.Simulation1DLayered(
    survey=survey,
    sigmaMap=log_conductivity_map,
    thicknesses=layer_thicknesses,
)

dmis_irls = data_misfit.L2DataMisfit(simulation=simulation_irls, data=data_object)

reg_irls = regularization.Sparse(
    regularization_mesh,
    alpha_s=0.01,
    alpha_x=1,
    reference_model_in_smooth=False,
    norms=[1.0, 1.0],
)

opt_irls = optimization.InexactGaussNewton(
    maxIter=100, maxIterLS=20, maxIterCG=30, tolCG=1e-3
)

inv_prob_irls = inverse_problem.BaseInvProblem(dmis_irls, reg_irls, opt_irls)

starting_beta_irls = directives.BetaEstimate_ByEig(beta0_ratio=5)
update_jacobi_irls = directives.UpdatePreconditioner(update_every_iteration=True)
update_irls = directives.UpdateIRLS(
    cooling_factor=2,
    cooling_rate=3,
    f_min_change=1e-4,
    max_irls_iterations=30,
    chifact_start=1.0,
)

directives_list_irls = [update_irls, starting_beta_irls, update_jacobi_irls]

print(f"  Regularization type: Sparse (IRLS)")
print(f"  Norms: [1.0, 1.0] for sparse/blocky recovery")

print("=" * 80)
print("Step 12: Running IRLS Inversion")
print("=" * 80)

inv_irls = inversion.BaseInversion(inv_prob_irls, directives_list_irls)
recovered_model_irls = inv_irls.run(starting_conductivity_model)

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
print("Step 13: Setting Up Parametric Inversion")
print("=" * 80)

# Wire maps to extract log-thicknesses and log-resistivities
wire_map = maps.Wires(("log_thicknesses", 2), ("log_resistivity", 3))

# Mapping for layer thicknesses
log_thicknesses_map = maps.ExpMap() * wire_map.log_thicknesses

# Mapping for resistivities
log_resistivity_map = maps.ExpMap() * wire_map.log_resistivity

# Starting model for 3-layer Earth
starting_parametric_model = np.log(np.r_[30.0, 20.0, 20, 0.5, 5])
reference_parametric_model = starting_parametric_model.copy()

simulation_parametric = fdem.simulation_1d.Simulation1DLayered(
    survey=survey,
    rhoMap=log_resistivity_map,
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
    mapping=wire_map.log_resistivity,
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

update_jacobi = directives.UpdatePreconditioner(update_every_iteration=True)
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=5)
beta_schedule = directives.BetaSchedule(coolingFactor=2.0, coolingRate=3)
target_misfit = directives.TargetMisfit(chifact=1.0)

directives_list_parametric = [
    update_jacobi,
    starting_beta,
    beta_schedule,
    target_misfit,
]

print(f"  Model parameters: 2 thicknesses + 3 resistivities = 5 total")
print(f"  Assumed structure: 3-layer Earth")

print("=" * 80)
print("Step 14: Running Parametric Inversion")
print("=" * 80)

inv_parametric = inversion.BaseInversion(
    inv_prob_parametric, directives_list_parametric
)
recovered_model_parametric = inv_parametric.run(starting_parametric_model)

print()
print(f"  Parametric inversion completed successfully")
print()


# ============================================================================
# Plot Data Fit Comparison
# ============================================================================

print("=" * 80)
print("Step 15: Plotting Data Fit for All Three Approaches")
print("=" * 80)

dpred_L2 = simulation_L2.dpred(recovered_model_L2)
dpred_irls = simulation_irls.dpred(recovered_model_irls)
dpred_parametric = simulation_parametric.dpred(recovered_model_parametric)

fig = plt.figure(figsize=(10, 5))
ax = [fig.add_axes([0.1 + ii * 0.5, 0.1, 0.37, 0.85]) for ii in range(2)]

for ii in range(2):
    ax[ii].loglog(frequencies, np.abs(dobs[ii::2]), "k-o", lw=2, label="Observed")
    ax[ii].loglog(
        frequencies, np.abs(dpred_L2[ii::2]), "b-s", lw=2, label="L2 Inversion"
    )
    ax[ii].loglog(
        frequencies, np.abs(dpred_irls[ii::2]), "r-^", lw=2, label="IRLS Inversion"
    )
    ax[ii].loglog(
        frequencies,
        np.abs(dpred_parametric[ii::2]),
        "g-d",
        lw=2,
        label="Parametric Inversion",
    )
    ax[ii].grid(which="both")
    ax[ii].set_xlabel("Frequency (Hz)")
    ax[ii].set_ylabel("|Hs/Hp| (ppm)")
    ax[ii].legend()
    if ii == 1:
        ax[ii].set_ylabel("")

ax[0].set_title("Real Component")
ax[1].set_title("Imaginary Component")
plt.savefig("fdem_1d_data_fit_comparison.png", dpi=150, bbox_inches="tight")
print("  Data fit comparison saved as 'fdem_1d_data_fit_comparison.png'")
plt.close()
print()


# ============================================================================
# Plot Recovered Models
# ============================================================================

print("=" * 80)
print("Step 16: Plotting Recovered Models")
print("=" * 80)

# True conductivities and layer thicknesses
true_conductivities = np.array([0.1, 1.0, 0.1])
true_layers = np.r_[20.0, 40.0, 160.0]

# Plot all models
fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_axes([0.2, 0.15, 0.7, 0.7])
plot_1d_layer_model(true_layers, true_conductivities, ax=ax1, color="k", label="True Model")
plot_1d_layer_model(
    layer_thicknesses,
    log_conductivity_map * recovered_model_L2,
    ax=ax1,
    color="b",
    label="L2 Model",
)
plot_1d_layer_model(
    layer_thicknesses,
    log_conductivity_map * recovered_model_irls,
    ax=ax1,
    color="r",
    label="IRLS Model",
)
plot_1d_layer_model(
    log_thicknesses_map * recovered_model_parametric,
    1 / (log_resistivity_map * recovered_model_parametric),
    ax=ax1,
    color="g",
    label="Parametric Model",
)
ax1.grid()
ax1.set_xlabel(r"Conductivity (S/m)")
x_min, x_max = true_conductivities.min(), true_conductivities.max()
ax1.set_xlim(0.8 * x_min, 2 * x_max)
ax1.set_ylim([np.sum(layer_thicknesses), 0])
ax1.set_title("Recovered Models Comparison")
ax1.legend()
plt.savefig("fdem_1d_models_comparison.png", dpi=150, bbox_inches="tight")
print("  Models comparison saved as 'fdem_1d_models_comparison.png'")
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
print(f"  Frequencies:              {len(frequencies)}")
print(f"  Frequency range:          [{frequencies.min():.1f}, {frequencies.max():.1f}] Hz")
print(f"  Number of layers:         {n_layers}")
print()
print("Inversion Results:")
print("-" * 80)
print()
print("Generated Files:")
print("-" * 80)
print("  1. fdem_1d_observed_data.png         - Observed sounding data")
print("  2. fdem_1d_data_fit_comparison.png   - Data fit for all approaches")
print("  3. fdem_1d_models_comparison.png     - Comparison of recovered models")
print()
print("Key Findings:")
print("-" * 80)
print("  - L2 inversion recovers smooth conductivity models")
print("  - IRLS inversion recovers blocky/sparse structures")
print("  - Parametric inversion assumes fixed number of layers")
print("  - All three approaches fit the FDEM data well")
print("  - Skin depth considerations guide layer thickness design")
print("=" * 80)
