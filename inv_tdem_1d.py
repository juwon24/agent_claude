"""
1D Inversion for a Single Sounding
University of British Columbia

This tutorial demonstrates how to invert time-domain EM data for a single sounding. We show three
approaches for recovering a conductivity/resistivity model:

1. Weighted least-squares inversion
2. Iteratively re-weighted least-squares (IRLS) inversion
3. Parametric inversion

The weighted least-squares approach recovers smooth structures. The IRLS approach can recover
sparse and/or blocky structures. The parametric approach assumes a specific layered Earth model
and recovers the properties and thicknesses of the layers.

Learning Objectives:
- How to carry out 1D geophysical inversion with SimPEG
- How to assign appropriate uncertainties to TDEM data
- Choosing suitable parameters for the inversion
- Specifying directives that are applied throughout the inversion
- Weighted least-squares, sparse-norm and parametric inversion
- Analyzing inversion outputs

Keywords: time-domain electromagnetics, 1D inversion, layered Earth, parametric inversion
"""

# ============================================================================
# Import Modules
# ============================================================================

# SimPEG functionality
import simpeg.electromagnetics.time_domain as tdem
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
# Load Tutorial Data
# ============================================================================

print("=" * 80)
print("Step 1: Loading Tutorial Data")
print("=" * 80)

# URL to assets folder
data_source = "https://github.com/simpeg/user-tutorials/raw/main/assets/08-tdem/inv_tdem_1d_files.tar.gz"

# download the data
downloaded_data = download(data_source, overwrite=True)

# unzip the tarfile
tar = tarfile.open(downloaded_data, "r")
tar.extractall()
tar.close()

# path to the directory containing our data
dir_path = downloaded_data.split(".")[0] + os.path.sep

# files to work with
data_filename = dir_path + "em1dtm_data.txt"

# Load data
dobs = np.loadtxt(str(data_filename), skiprows=1)

print(f"  Loaded data from: {data_filename}")
print()


# ============================================================================
# Plot Observed Data
# ============================================================================

print("=" * 80)
print("Step 2: Plotting Observed Data")
print("=" * 80)

times = dobs[:, 0]
dobs = mkvc(dobs[:, -1])

fig = plt.figure(figsize=(5, 5))
ax = fig.add_axes([0.15, 0.15, 0.8, 0.75])
ax.loglog(times, np.abs(dobs), "k-o", lw=3)
ax.grid(which="both")
ax.set_xlabel("Times (s)")
ax.set_ylabel("|B| (T)")
ax.set_title("Observed Data")
plt.savefig("tdem_1d_observed_data.png", dpi=150, bbox_inches="tight")
print("✓ Observed data plot saved as 'tdem_1d_observed_data.png'")
plt.close()
print()


# ============================================================================
# Define the Survey
# ============================================================================

print("=" * 80)
print("Step 3: Defining Survey")
print("=" * 80)

# Source loop geometry
source_location = np.array([0.0, 0.0, 1.0])
source_orientation = "z"
source_current = 1.0
source_radius = 10.0

# Receiver geometry
receiver_location = np.array([0.0, 0.0, 1.0])
receiver_orientation = "z"

# Receiver list
receiver_list = []
receiver_list.append(
    tdem.receivers.PointMagneticFluxDensity(
        receiver_location, times, orientation=receiver_orientation
    )
)

# Define the source waveform
waveform = tdem.sources.StepOffWaveform()

# Sources
source_list = [
    tdem.sources.CircularLoop(
        receiver_list=receiver_list,
        location=source_location,
        waveform=waveform,
        current=source_current,
        radius=source_radius,
    )
]

# Survey
survey = tdem.Survey(source_list)

print(f"  Number of time channels: {len(times)}")
print(f"  Source: Circular loop (radius={source_radius} m)")
print(f"  Receiver: Point magnetic flux density")
print()


# ============================================================================
# Assign Uncertainties
# ============================================================================

print("=" * 80)
print("Step 4: Assigning Data Uncertainties")
print("=" * 80)

# 5% of the absolute value
uncertainties = 0.05 * np.abs(dobs) * np.ones(np.shape(dobs))

print(f"  Uncertainty: 5% of |B|")
print()


# ============================================================================
# Define the Data Object
# ============================================================================

print("=" * 80)
print("Step 5: Creating Data Object")
print("=" * 80)

data_object = data.Data(survey, dobs=dobs, standard_deviation=uncertainties)

print(f"  Data object created with {len(dobs)} observations")
print()


# ============================================================================
# WEIGHTED LEAST-SQUARES INVERSION
# ============================================================================

print()
print("=" * 80)
print("WEIGHTED LEAST-SQUARES INVERSION")
print("=" * 80)
print()


# ============================================================================
# Define 1D Layered Earth
# ============================================================================

print("=" * 80)
print("Step 6: Defining 1D Layered Earth Model")
print("=" * 80)

# estimated host conductivity (S/m)
estimated_conductivity = 0.1

# minimum diffusion distance
d_min = 1250 * np.sqrt(times.min() / estimated_conductivity)
print(f"  Minimum diffusion distance: {d_min:.2f} m")

# maximum diffusion distance
d_max = 1250 * np.sqrt(times.max() / estimated_conductivity)
print(f"  Maximum diffusion distance: {d_max:.2f} m")

depth_min = 1  # top layer thickness
depth_max = 800.0  # depth to lowest layer
geometric_factor = 1.15  # rate of thickness increase

# Increase subsequent layer thicknesses by the geometric factors
layer_thicknesses = [depth_min]
while np.sum(layer_thicknesses) < depth_max:
    layer_thicknesses.append(geometric_factor * layer_thicknesses[-1])

n_layers = len(layer_thicknesses) + 1  # Number of layers

print(f"  Number of layers: {n_layers}")
print()


# ============================================================================
# Model Mapping
# ============================================================================

print("=" * 80)
print("Step 7: Creating Model Mapping")
print("=" * 80)

log_conductivity_map = maps.ExpMap(nP=n_layers)

print(f"  Model parameters: {n_layers} log-conductivities")
print()


# ============================================================================
# Starting and Reference Models
# ============================================================================

print("=" * 80)
print("Step 8: Defining Starting and Reference Models")
print("=" * 80)

# Starting model is log-conductivity values (S/m)
starting_conductivity_model = np.log(1e-1 * np.ones(n_layers))

# Reference model is also log-conductivity values (S/m)
reference_conductivity_model = starting_conductivity_model.copy()

print(f"  Starting conductivity: {np.exp(starting_conductivity_model[0]):.2e} S/m")
print()


# ============================================================================
# Define Forward Simulation (L2)
# ============================================================================

print("=" * 80)
print("Step 9: Defining Forward Simulation (L2)")
print("=" * 80)

simulation_L2 = tdem.Simulation1DLayered(
    survey=survey, thicknesses=layer_thicknesses, sigmaMap=log_conductivity_map
)

print("  Simulation type: 1D Layered")
print()


# ============================================================================
# Define Data Misfit (L2)
# ============================================================================

print("=" * 80)
print("Step 10: Defining Data Misfit (L2)")
print("=" * 80)

dmis_L2 = data_misfit.L2DataMisfit(simulation=simulation_L2, data=data_object)

print("  Data misfit type: L2")
print()


# ============================================================================
# Define Regularization (L2)
# ============================================================================

print("=" * 80)
print("Step 11: Defining Regularization (L2)")
print("=" * 80)

# Define 1D cell widths (reversed for regularization)
h = np.r_[layer_thicknesses, layer_thicknesses[-1]]
h = np.flipud(h)

# Create regularization mesh
regularization_mesh = TensorMesh([h], "N")

reg_L2 = regularization.WeightedLeastSquares(
    regularization_mesh,
    length_scale_x=10.0,
    reference_model=reference_conductivity_model,
    reference_model_in_smooth=False,
)

print("  Regularization type: Weighted Least-Squares")
print("  Length scale: 10.0")
print()


# ============================================================================
# Define Optimization (L2)
# ============================================================================

print("=" * 80)
print("Step 12: Defining Optimization Algorithm (L2)")
print("=" * 80)

opt_L2 = optimization.InexactGaussNewton(
    maxIter=100, maxIterLS=20, maxIterCG=20, tolCG=1e-3
)

print("  Algorithm: Inexact Gauss-Newton")
print()


# ============================================================================
# Define Inverse Problem (L2)
# ============================================================================

print("=" * 80)
print("Step 13: Defining Inverse Problem (L2)")
print("=" * 80)

inv_prob_L2 = inverse_problem.BaseInvProblem(dmis_L2, reg_L2, opt_L2)

print("  Inverse problem created")
print()


# ============================================================================
# Define Directives (L2)
# ============================================================================

print("=" * 80)
print("Step 14: Defining Inversion Directives (L2)")
print("=" * 80)

update_jacobi = directives.UpdatePreconditioner(update_every_iteration=True)
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=5)
beta_schedule = directives.BetaSchedule(coolingFactor=2.0, coolingRate=3)
target_misfit = directives.TargetMisfit(chifact=1.0)

directives_list_L2 = [update_jacobi, starting_beta, beta_schedule, target_misfit]

print("  Directives configured")
print()


# ============================================================================
# Run L2 Inversion
# ============================================================================

print("=" * 80)
print("Step 15: Running L2 Inversion")
print("=" * 80)
print()

inv_L2 = inversion.BaseInversion(inv_prob_L2, directives_list_L2)
recovered_model_L2 = inv_L2.run(starting_conductivity_model)

print()
print("✓ L2 inversion completed")
print()


# ============================================================================
# Plot L2 Results
# ============================================================================

print("=" * 80)
print("Step 16: Analyzing L2 Results")
print("=" * 80)

dpred_L2 = simulation_L2.dpred(recovered_model_L2)

fig = plt.figure(figsize=(5, 5))
ax1 = fig.add_axes([0.15, 0.15, 0.8, 0.75])
ax1.loglog(times, np.abs(dobs), "k-o")
ax1.loglog(times, np.abs(dpred_L2), "b-o")
ax1.grid(which="both")
ax1.set_xlabel("times (s)")
ax1.set_ylabel("Bz (T)")
ax1.set_title("Predicted and Observed Data")
ax1.legend(["Observed", "L2 Inversion"], loc="upper right")
plt.savefig("tdem_1d_L2_data_fit.png", dpi=150, bbox_inches="tight")
print("✓ L2 data fit plot saved as 'tdem_1d_L2_data_fit.png'")
plt.close()
print()


# ============================================================================
# ITERATIVELY RE-WEIGHTED LEAST-SQUARES INVERSION
# ============================================================================

print()
print("=" * 80)
print("ITERATIVELY RE-WEIGHTED LEAST-SQUARES INVERSION")
print("=" * 80)
print()


# ============================================================================
# Define Forward Simulation (IRLS)
# ============================================================================

print("=" * 80)
print("Step 17: Defining Forward Simulation (IRLS)")
print("=" * 80)

simulation_irls = tdem.simulation_1d.Simulation1DLayered(
    survey=survey,
    sigmaMap=log_conductivity_map,
    thicknesses=layer_thicknesses,
)

print("  Simulation type: 1D Layered")
print()


# ============================================================================
# Define Data Misfit (IRLS)
# ============================================================================

print("=" * 80)
print("Step 18: Defining Data Misfit (IRLS)")
print("=" * 80)

dmis_irls = data_misfit.L2DataMisfit(simulation=simulation_irls, data=data_object)

print("  Data misfit type: L2")
print()


# ============================================================================
# Define Regularization (IRLS)
# ============================================================================

print("=" * 80)
print("Step 19: Defining Regularization (IRLS)")
print("=" * 80)

reg_irls = regularization.Sparse(
    regularization_mesh,
    alpha_s=0.01,
    alpha_x=1,
    reference_model_in_smooth=False,
    norms=[1.0, 1.0],
)

print("  Regularization type: Sparse")
print("  Norms: [1.0, 1.0] (compact structures)")
print()


# ============================================================================
# Define Optimization (IRLS)
# ============================================================================

print("=" * 80)
print("Step 20: Defining Optimization Algorithm (IRLS)")
print("=" * 80)

opt_irls = optimization.InexactGaussNewton(
    maxIter=100, maxIterLS=20, maxIterCG=30, tolCG=1e-3
)

print("  Algorithm: Inexact Gauss-Newton")
print()


# ============================================================================
# Define Inverse Problem (IRLS)
# ============================================================================

print("=" * 80)
print("Step 21: Defining Inverse Problem (IRLS)")
print("=" * 80)

inv_prob_irls = inverse_problem.BaseInvProblem(dmis_irls, reg_irls, opt_irls)

print("  Inverse problem created")
print()


# ============================================================================
# Define Directives (IRLS)
# ============================================================================

print("=" * 80)
print("Step 22: Defining Inversion Directives (IRLS)")
print("=" * 80)

sensitivity_weights_irls = directives.UpdateSensitivityWeights(every_iteration=True)
starting_beta_irls = directives.BetaEstimate_ByEig(beta0_ratio=1)
update_jacobi_irls = directives.UpdatePreconditioner(update_every_iteration=True)
update_irls = directives.UpdateIRLS(
    cooling_factor=2,
    cooling_rate=2,
    f_min_change=1e-4,
    max_irls_iterations=40,
    chifact_start=1.0,
)

directives_list_irls = [
    update_irls,
    sensitivity_weights_irls,
    starting_beta_irls,
    update_jacobi_irls,
]

print("  Directives configured (including IRLS)")
print()


# ============================================================================
# Run IRLS Inversion
# ============================================================================

print("=" * 80)
print("Step 23: Running IRLS Inversion")
print("=" * 80)
print()

inv_irls = inversion.BaseInversion(inv_prob_irls, directives_list_irls)
recovered_model_irls = inv_irls.run(starting_conductivity_model)

print()
print("✓ IRLS inversion completed")
print()


# ============================================================================
# Plot IRLS Results
# ============================================================================

print("=" * 80)
print("Step 24: Analyzing IRLS Results")
print("=" * 80)

dpred_irls = simulation_irls.dpred(recovered_model_irls)

fig = plt.figure(figsize=(5, 5))
ax1 = fig.add_axes([0.15, 0.15, 0.8, 0.75])
ax1.loglog(times, np.abs(dobs), "k-o")
ax1.loglog(times, np.abs(dpred_L2), "b-o")
ax1.loglog(times, np.abs(dpred_irls), "r-o")
ax1.grid(which="both")
ax1.set_xlabel("times (s)")
ax1.set_ylabel("Bz (T)")
ax1.set_title("Predicted and Observed Data")
ax1.legend(["Observed", "L2 Inversion", "IRLS Inversion"], loc="upper right")
plt.savefig("tdem_1d_data_fit_comparison.png", dpi=150, bbox_inches="tight")
print("✓ Data fit comparison plot saved as 'tdem_1d_data_fit_comparison.png'")
plt.close()
print()


# ============================================================================
# PARAMETRIC INVERSION
# ============================================================================

print()
print("=" * 80)
print("PARAMETRIC INVERSION (3-LAYER EARTH)")
print("=" * 80)
print()


# ============================================================================
# Model and Mapping (Parametric)
# ============================================================================

print("=" * 80)
print("Step 25: Creating Parametric Model and Mapping")
print("=" * 80)

# Wire maps to extract log-thicknesses and log-conductivities
wire_map = maps.Wires(("log_thicknesses", 2), ("log_resistivity", 3))

# Mapping for layer thicknesses
log_thicknesses_map = maps.ExpMap() * wire_map.log_thicknesses

# Mapping for conductivities
log_resistivity_map = maps.ExpMap() * wire_map.log_resistivity

print("  Model: 2 log-thicknesses + 3 log-resistivities")
print()


# ============================================================================
# Starting and Reference Models (Parametric)
# ============================================================================

print("=" * 80)
print("Step 26: Defining Starting Model (Parametric)")
print("=" * 80)

starting_parametric_model = np.log(np.r_[30.0, 20.0, 20, 0.5, 5])
reference_parametric_model = starting_parametric_model.copy()

print(f"  Starting layers: {np.exp(starting_parametric_model[:2])} m")
print(f"  Starting resistivities: {np.exp(starting_parametric_model[2:])} Ω·m")
print()


# ============================================================================
# Define Forward Simulation (Parametric)
# ============================================================================

print("=" * 80)
print("Step 27: Defining Forward Simulation (Parametric)")
print("=" * 80)

simulation_parametric = tdem.simulation_1d.Simulation1DLayered(
    survey=survey,
    rhoMap=log_resistivity_map,
    thicknessesMap=log_thicknesses_map,
)

print("  Simulation type: 1D Layered (parametric)")
print()


# ============================================================================
# Define Data Misfit (Parametric)
# ============================================================================

print("=" * 80)
print("Step 28: Defining Data Misfit (Parametric)")
print("=" * 80)

dmis_parametric = data_misfit.L2DataMisfit(
    simulation=simulation_parametric, data=data_object
)

print("  Data misfit type: L2")
print()


# ============================================================================
# Define Regularization (Parametric)
# ============================================================================

print("=" * 80)
print("Step 29: Defining Regularization (Parametric)")
print("=" * 80)

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

print("  Regularization: Combo (thicknesses + resistivities)")
print("  Multipliers: [1.0, 0.1]")
print()


# ============================================================================
# Define Optimization (Parametric)
# ============================================================================

print("=" * 80)
print("Step 30: Defining Optimization Algorithm (Parametric)")
print("=" * 80)

opt_parametric = optimization.InexactGaussNewton(
    maxIter=100, maxIterLS=20, maxIterCG=20, tolCG=1e-3
)

print("  Algorithm: Inexact Gauss-Newton")
print()


# ============================================================================
# Define Inverse Problem (Parametric)
# ============================================================================

print("=" * 80)
print("Step 31: Defining Inverse Problem (Parametric)")
print("=" * 80)

inv_prob_parametric = inverse_problem.BaseInvProblem(
    dmis_parametric, reg_parametric, opt_parametric
)

print("  Inverse problem created")
print()


# ============================================================================
# Define Directives (Parametric)
# ============================================================================

print("=" * 80)
print("Step 32: Defining Inversion Directives (Parametric)")
print("=" * 80)

update_jacobi_p = directives.UpdatePreconditioner(update_every_iteration=True)
starting_beta_p = directives.BetaEstimate_ByEig(beta0_ratio=10)
beta_schedule_p = directives.BetaSchedule(coolingFactor=2.0, coolingRate=3)
target_misfit_p = directives.TargetMisfit(chifact=1.0)

directives_list_parametric = [
    update_jacobi_p,
    starting_beta_p,
    beta_schedule_p,
    target_misfit_p,
]

print("  Directives configured")
print()


# ============================================================================
# Run Parametric Inversion
# ============================================================================

print("=" * 80)
print("Step 33: Running Parametric Inversion")
print("=" * 80)
print()

inv_parametric = inversion.BaseInversion(
    inv_prob_parametric, directives_list_parametric
)
recovered_model_parametric = inv_parametric.run(starting_parametric_model)

print()
print("✓ Parametric inversion completed")
print()


# ============================================================================
# Compare All Models
# ============================================================================

print("=" * 80)
print("Step 34: Comparing All Models")
print("=" * 80)

# Load the true model and layer thicknesses
true_conductivities = np.array([0.1, 1.0, 0.1])
true_layers = np.r_[40.0, 40.0, 160.0]

# Predicted data
dpred_parametric = simulation_parametric.dpred(recovered_model_parametric)

# Plot data fit
fig = plt.figure(figsize=(5, 5))
ax1 = fig.add_axes([0.15, 0.15, 0.8, 0.75])
ax1.loglog(times, np.abs(dobs), "k-o")
ax1.loglog(times, np.abs(dpred_L2), "b-o")
ax1.loglog(times, np.abs(dpred_irls), "r-o")
ax1.loglog(times, np.abs(dpred_parametric), "g-o")
ax1.grid(which="both")
ax1.set_xlabel("times (s)")
ax1.set_ylabel("Bz (T)")
ax1.set_title("Predicted and Observed Data")
ax1.legend(
    ["Observed", "L2 Inversion", "IRLS Inversion", "Parametric Inversion"],
    loc="upper right",
)
plt.savefig("tdem_1d_all_data_fits.png", dpi=150, bbox_inches="tight")
print("✓ All data fits plot saved as 'tdem_1d_all_data_fits.png'")
plt.close()

# Plot all models
fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_axes([0.2, 0.15, 0.7, 0.7])

x_min, x_max = true_conductivities.min(), true_conductivities.max()

plot_1d_layer_model(true_layers, true_conductivities, ax=ax1, color="k")
plot_1d_layer_model(
    layer_thicknesses, log_conductivity_map * recovered_model_L2, ax=ax1, color="b"
)
plot_1d_layer_model(
    layer_thicknesses, log_conductivity_map * recovered_model_irls, ax=ax1, color="r"
)
plot_1d_layer_model(
    log_thicknesses_map * recovered_model_parametric,
    1 / (log_resistivity_map * recovered_model_parametric),
    ax=ax1,
    color="g",
)
ax1.grid()
ax1.set_xlabel(r"Conductivity (S/m)")
ax1.set_xlim(0.8 * x_min, 2 * x_max)
ax1.set_ylim([np.sum(true_layers), 0])
ax1.legend(["True Model", "L2 Model", "IRLS Model", "Parametric Model"])
plt.savefig("tdem_1d_model_comparison.png", dpi=150, bbox_inches="tight")
print("✓ Model comparison plot saved as 'tdem_1d_model_comparison.png'")
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
print(f"  Number of time channels: {len(times)}")
print(f"  Number of layers (L2/IRLS): {n_layers}")
print(f"  Number of parameters (parametric): {len(starting_parametric_model)}")
print()
print(f"  L2 Model:")
print(f"    Conductivity range: [{np.min(log_conductivity_map * recovered_model_L2):.4f}, " +
      f"{np.max(log_conductivity_map * recovered_model_L2):.4f}] S/m")
print()
print(f"  IRLS Model:")
print(f"    Conductivity range: [{np.min(log_conductivity_map * recovered_model_irls):.4f}, " +
      f"{np.max(log_conductivity_map * recovered_model_irls):.4f}] S/m")
print()
print(f"  Parametric Model:")
print(f"    Layer thicknesses: {log_thicknesses_map * recovered_model_parametric} m")
print(f"    Resistivities: {log_resistivity_map * recovered_model_parametric} Ω·m")
print()
print("Generated Files:")
print("-" * 80)
print("  1. tdem_1d_observed_data.png         - Observed TDEM data")
print("  2. tdem_1d_L2_data_fit.png           - L2 inversion data fit")
print("  3. tdem_1d_data_fit_comparison.png   - L2 vs IRLS data fits")
print("  4. tdem_1d_all_data_fits.png         - All inversion data fits")
print("  5. tdem_1d_model_comparison.png      - Comparison of all models")
print()
print("Key Findings:")
print("-" * 80)
print("  - L2 inversion recovers smooth conductivity structures")
print("  - IRLS inversion recovers more blocky layer boundaries")
print("  - Parametric inversion directly recovers layer properties")
print("  - Choice of inversion method depends on prior knowledge")
print("=" * 80)
