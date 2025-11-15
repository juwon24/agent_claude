"""
2.5D DC Resistivity Inversion
University of British Columbia

This tutorial demonstrates two approaches for 2.5D DC resistivity inversion:
1. Weighted least-squares inversion of normalized voltage data
2. Iteratively re-weighted least-squares (IRLS) inversion of apparent resistivity data

The tutorial teaches:
- How to design suitable tree meshes for 2.5D DC resistivity inversion
- How to assign appropriate uncertainties to voltage and apparent resistivity data
- Choosing suitable inversion parameters and directives
- Applying sensitivity weighting for DC resistivity
- Comparing smooth (L2) and blocky (IRLS) inversion results

Keywords: DC resistivity, 2.5D inversion, sparse norm, tree mesh, sensitivity weighting
"""

# ============================================================================
# Import Modules
# ============================================================================

# SimPEG functionality
from simpeg.electromagnetics.static import resistivity as dc
from simpeg.electromagnetics.static.utils.static_utils import (
    plot_pseudosection,
    generate_survey_from_abmn_locations,
    apparent_resistivity_from_voltage,
)
from simpeg.utils.io_utils.io_utils_electromagnetics import read_dcip2d_ubc
from simpeg.utils import download, model_builder
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
from discretize import TreeMesh
from discretize.utils import active_from_xyz

# Basic Python functionality
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import tarfile

mpl.rcParams.update({"font.size": 14})
cmap = mpl.cm.RdYlBu_r


# ============================================================================
# Download and Extract Tutorial Data
# ============================================================================

print("=" * 80)
print("Step 1: Downloading and Extracting Tutorial Data")
print("=" * 80)

# URL to download from repository assets
data_source = "https://github.com/simpeg/user-tutorials/raw/main/assets/05-dcr/inv_dcr_2d_files.tar.gz"

# Download the data
downloaded_data = download(data_source, overwrite=True)

# Unzip the tarfile
tar = tarfile.open(downloaded_data, "r")
tar.extractall()
tar.close()

# Path to the directory containing our data
dir_path = downloaded_data.split(".")[0] + os.path.sep
topo_filename = dir_path + "topo_2d.txt"
data_filename = dir_path + "dc_data.obs"

print(f"  Downloaded and extracted data to: {dir_path}")
print()


# ============================================================================
# Load and Plot Topography
# ============================================================================

print("=" * 80)
print("Step 2: Loading and Plotting Topography")
print("=" * 80)

# Load 2D topography
topo_2d = np.loadtxt(str(topo_filename))

print(f"  Topography points: {len(topo_2d)}")
print(f"  X-range: [{topo_2d[:, 0].min():.1f}, {topo_2d[:, 0].max():.1f}] m")
print(f"  Z-range: [{topo_2d[:, -1].min():.1f}, {topo_2d[:, -1].max():.1f}] m")

# Plot topography
fig = plt.figure(figsize=(10, 2))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(topo_2d[:, 0], topo_2d[:, -1], color="b", linewidth=1)
ax.set_xlim([topo_2d[:, 0].min(), topo_2d[:, 0].max()])
ax.set_xlabel("x (m)", labelpad=5)
ax.set_ylabel("z (m)", labelpad=5)
ax.grid(True)
ax.set_title("Topography (Exaggerated z-axis)", fontsize=16, pad=10)
plt.savefig("dcr_2d_topography.png", dpi=150, bbox_inches="tight")
print("  Topography plot saved as 'dcr_2d_topography.png'")
plt.close()
print()


# ============================================================================
# Load DC Resistivity Data
# ============================================================================

print("=" * 80)
print("Step 3: Loading DC Resistivity Data")
print("=" * 80)

voltage_data = read_dcip2d_ubc(data_filename, "volt", "general")

print(f"  Number of data: {voltage_data.nD}")
print(f"  Data type: Normalized voltages (V/A)")
print(f"  Data range: [{voltage_data.dobs.min():.2e}, {voltage_data.dobs.max():.2e}] V/A")
print()


# ============================================================================
# Plot Data in Pseudo-Section
# ============================================================================

print("=" * 80)
print("Step 4: Plotting Data in Pseudo-Section")
print("=" * 80)

# Plot voltages pseudo-section
fig = plt.figure(figsize=(8, 2.75))
ax1 = fig.add_axes([0.1, 0.15, 0.75, 0.78])
plot_pseudosection(
    voltage_data,
    plot_type="scatter",
    ax=ax1,
    scale="log",
    cbar_label="V/A",
    scatter_opts={"cmap": mpl.cm.viridis},
)
ax1.set_title("Normalized Voltages")
plt.savefig("dcr_2d_voltage_pseudosection.png", dpi=150, bbox_inches="tight")
print("  Voltage pseudo-section saved as 'dcr_2d_voltage_pseudosection.png'")
plt.close()

# Get apparent resistivities
apparent_resistivities = apparent_resistivity_from_voltage(
    voltage_data.survey, voltage_data.dobs
)

# Plot apparent resistivity pseudo-section
fig = plt.figure(figsize=(8, 2.75))
ax1 = fig.add_axes([0.1, 0.15, 0.75, 0.78])
plot_pseudosection(
    voltage_data.survey,
    apparent_resistivities,
    plot_type="contourf",
    ax=ax1,
    scale="log",
    cbar_label=r"$\Omega m$",
    mask_topography=True,
    contourf_opts={"levels": 20, "cmap": mpl.cm.RdYlBu},
)
ax1.set_title("Apparent Resistivity")
plt.savefig("dcr_2d_appres_pseudosection.png", dpi=150, bbox_inches="tight")
print("  Apparent resistivity pseudo-section saved as 'dcr_2d_appres_pseudosection.png'")
plt.close()

print(f"  Apparent resistivity range: [{apparent_resistivities.min():.1f}, {apparent_resistivities.max():.1f}] Ohm-m")
print()


# ============================================================================
# Assign Uncertainties
# ============================================================================

print("=" * 80)
print("Step 5: Assigning Uncertainties to Voltage Data")
print("=" * 80)

# Apply uncertainties to normalized voltage data
voltage_data.standard_deviation = 1e-7 + 0.05 * np.abs(voltage_data.dobs)

print(f"  Applied floor: 1e-7 V/A")
print(f"  Applied percent: 5%")
print()


# ============================================================================
# Design a Tree Mesh
# ============================================================================

print("=" * 80)
print("Step 6: Designing Tree Mesh")
print("=" * 80)

dh = 4  # base cell width
dom_width_x = 3200.0  # domain width x
dom_width_z = 2400.0  # domain width z
nbcx = 2 ** int(np.round(np.log(dom_width_x / dh) / np.log(2.0)))
nbcz = 2 ** int(np.round(np.log(dom_width_z / dh) / np.log(2.0)))

# Define the base mesh with top at z = 0 m
hx = [(dh, nbcx)]
hz = [(dh, nbcz)]
mesh = TreeMesh([hx, hz], x0="CN", diagonal_balance=True)

# Shift top to maximum topography
mesh.origin = mesh.origin + np.r_[0.0, topo_2d[:, -1].max()]

# Mesh refinement based on topography
mesh.refine_surface(
    topo_2d,
    padding_cells_by_level=[0, 0, 4, 4],
    finalize=False,
)

# Extract unique electrode locations
unique_locations = voltage_data.survey.unique_electrode_locations

# Mesh refinement near electrodes
mesh.refine_points(
    unique_locations, padding_cells_by_level=[8, 12, 6, 6], finalize=False
)

mesh.finalize()

print(f"  Base cell width: {dh} m")
print(f"  Number of cells: {mesh.n_cells}")
print(f"  Mesh extent (x): [{mesh.nodes_x.min():.1f}, {mesh.nodes_x.max():.1f}] m")
print(f"  Mesh extent (z): [{mesh.nodes_z.min():.1f}, {mesh.nodes_z.max():.1f}] m")
print()


# ============================================================================
# Define Active Cells
# ============================================================================

print("=" * 80)
print("Step 7: Defining Active Cells")
print("=" * 80)

# Indices of the active mesh cells from topography
active_cells = active_from_xyz(mesh, topo_2d)
n_active = np.sum(active_cells)

print(f"  Total cells: {mesh.n_cells}")
print(f"  Active cells: {n_active}")
print(f"  Inactive cells: {(~active_cells).sum()}")
print()


# ============================================================================
# Project Electrodes to Discretized Topography
# ============================================================================

print("=" * 80)
print("Step 8: Projecting Electrodes to Discretized Topography")
print("=" * 80)

voltage_data.survey.drape_electrodes_on_topography(mesh, active_cells, option="top")

print(f"  Electrodes draped onto mesh surface")
print()


# ============================================================================
# WEIGHTED LEAST-SQUARES INVERSION
# ============================================================================

print("=" * 80)
print("APPROACH 1: WEIGHTED LEAST-SQUARES INVERSION")
print("=" * 80)
print()

# ============================================================================
# Define Mapping, Starting and Reference Models (L2)
# ============================================================================

print("=" * 80)
print("Step 9: Defining Model Mapping and Starting Model for L2")
print("=" * 80)

# Map model parameters to all cells
log_conductivity_map = maps.InjectActiveCells(mesh, active_cells, 1e-8) * maps.ExpMap(
    nP=n_active
)

# Median apparent resistivity
median_resistivity = np.median(apparent_resistivities)

# Create starting model from log-conductivity
starting_conductivity_model = np.log(1 / median_resistivity) * np.ones(n_active)

# Reference conductivity model
reference_conductivity_model = starting_conductivity_model.copy()

print(f"  Model parameters: {n_active}")
print(f"  Starting conductivity: {1/median_resistivity:.4f} S/m")
print(f"  Starting resistivity: {median_resistivity:.1f} Ohm-m")
print()


# ============================================================================
# Define Forward Simulation (L2)
# ============================================================================

print("=" * 80)
print("Step 10: Defining Forward Simulation for L2")
print("=" * 80)

voltage_simulation = dc.simulation_2d.Simulation2DNodal(
    mesh, survey=voltage_data.survey, sigmaMap=log_conductivity_map, storeJ=True
)

print(f"  Simulation type: 2D Nodal")
print(f"  Store Jacobian: True")
print()


# ============================================================================
# Set Up and Run L2 Inversion
# ============================================================================

print("=" * 80)
print("Step 11: Setting Up L2 Inversion")
print("=" * 80)

dmis_L2 = data_misfit.L2DataMisfit(simulation=voltage_simulation, data=voltage_data)

reg_L2 = regularization.WeightedLeastSquares(
    mesh,
    active_cells=active_cells,
    alpha_s=dh**-2,
    alpha_x=1,
    alpha_y=1,
    reference_model=reference_conductivity_model,
    reference_model_in_smooth=False,
)

opt_L2 = optimization.InexactGaussNewton(
    maxIter=40, maxIterLS=20, maxIterCG=20, tolCG=1e-3
)

inv_prob_L2 = inverse_problem.BaseInvProblem(dmis_L2, reg_L2, opt_L2)

sensitivity_weights = directives.UpdateSensitivityWeights(
    every_iteration=True, threshold_value=1e-2
)
update_jacobi = directives.UpdatePreconditioner(update_every_iteration=True)
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=10)
beta_schedule = directives.BetaSchedule(coolingFactor=2.0, coolingRate=2)
target_misfit = directives.TargetMisfit(chifact=1.0)

directives_list_L2 = [
    sensitivity_weights,
    update_jacobi,
    starting_beta,
    beta_schedule,
    target_misfit,
]

print(f"  Data misfit: L2 norm")
print(f"  Regularization: Weighted Least Squares")
print(f"  Number of directives: {len(directives_list_L2)}")
print()

print("=" * 80)
print("Step 12: Running Weighted Least-Squares Inversion")
print("=" * 80)

inv_L2 = inversion.BaseInversion(inv_prob_L2, directives_list_L2)
recovered_log_conductivity_model = inv_L2.run(starting_conductivity_model)

print()
print(f"  L2 inversion completed successfully")
print(f"  Final data misfit: {inv_prob_L2.dmisfit.phi:.2f}")
print()


# ============================================================================
# Plot L2 Data Misfit
# ============================================================================

print("=" * 80)
print("Step 13: Plotting L2 Data Misfit")
print("=" * 80)

# Predicted data from recovered model
dpred = inv_prob_L2.dpred
dobs = voltage_data.dobs
std = voltage_data.standard_deviation

# Plot
fig = plt.figure(figsize=(9, 11))
data_array = [np.abs(dobs), np.abs(dpred), (dobs - dpred) / std]
plot_title = ["Observed Voltage", "Predicted Voltage", "Normalized Misfit"]
plot_units = ["V/A", "V/A", ""]
scale = ["log", "log", "linear"]
cmap_list = [mpl.cm.viridis, mpl.cm.viridis, mpl.cm.RdYlBu]

for ii in range(0, 3):
    ax1 = fig.add_axes([0.15, 0.72 - 0.33 * ii, 0.65, 0.21])
    cax1 = fig.add_axes([0.81, 0.72 - 0.33 * ii, 0.03, 0.21])
    plot_pseudosection(
        voltage_data.survey,
        data_array[ii],
        "contourf",
        ax=ax1,
        cax=cax1,
        scale=scale[ii],
        cbar_label=plot_units[ii],
        mask_topography=True,
        contourf_opts={"levels": 25, "cmap": cmap_list[ii]},
    )
    ax1.set_title(plot_title[ii])

plt.savefig("dcr_2d_L2_data_misfit.png", dpi=150, bbox_inches="tight")
print("  L2 data misfit plots saved as 'dcr_2d_L2_data_misfit.png'")
plt.close()
print()


# ============================================================================
# Plot L2 Recovered Model
# ============================================================================

print("=" * 80)
print("Step 14: Plotting L2 Recovered Model")
print("=" * 80)

# Convert log-conductivity to conductivity
recovered_conductivity_L2 = np.exp(recovered_log_conductivity_model)

# Define plotting map
plotting_map = maps.InjectActiveCells(mesh, active_cells, np.nan)

norm = LogNorm(vmin=5e-4, vmax=2e-1)

fig = plt.figure(figsize=(9, 4))

ax1 = fig.add_axes([0.14, 0.17, 0.68, 0.7])
mesh.plot_image(
    plotting_map * recovered_conductivity_L2,
    normal="Y",
    ax=ax1,
    grid=False,
    pcolor_opts={"norm": norm, "cmap": cmap},
)
ax1.set_xlim(-500, 500)
ax1.set_ylim(-300, 200)
ax1.set_title("Recovered Conductivity Model (L2)")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")

ax2 = fig.add_axes([0.84, 0.17, 0.03, 0.7])
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=cmap
)
cbar.set_label(r"$\sigma$ (S/m)", rotation=270, labelpad=15, size=12)

plt.savefig("dcr_2d_L2_model.png", dpi=150, bbox_inches="tight")
print("  L2 model plot saved as 'dcr_2d_L2_model.png'")
plt.close()
print()


# ============================================================================
# ITERATIVELY RE-WEIGHTED LEAST-SQUARES INVERSION
# ============================================================================

print("=" * 80)
print("APPROACH 2: ITERATIVELY RE-WEIGHTED LEAST-SQUARES (IRLS) INVERSION")
print("=" * 80)
print()

# ============================================================================
# Define Apparent Resistivity Survey
# ============================================================================

print("=" * 80)
print("Step 15: Defining Apparent Resistivity Survey")
print("=" * 80)

# Extract ABMN electrode locations
locations_a = voltage_data.survey.locations_a.copy()
locations_b = voltage_data.survey.locations_b.copy()
locations_m = voltage_data.survey.locations_m.copy()
locations_n = voltage_data.survey.locations_n.copy()

# Define survey from ABMN locations
resistivity_survey = generate_survey_from_abmn_locations(
    locations_a=locations_a,
    locations_b=locations_b,
    locations_m=locations_m,
    locations_n=locations_n,
    data_type="apparent_resistivity",
)

# Set geometric factor
resistivity_survey.set_geometric_factor()

print(f"  Survey created for apparent resistivity inversion")
print()


# ============================================================================
# Define Data and Uncertainties
# ============================================================================

print("=" * 80)
print("Step 16: Defining Data Object for IRLS")
print("=" * 80)

resistivity_data = data.Data(survey=resistivity_survey, dobs=apparent_resistivities)
resistivity_data.standard_deviation = 1e-3 + 0.05 * np.abs(resistivity_data.dobs)

print(f"  Applied floor: 1e-3 Ohm-m")
print(f"  Applied percent: 5%")
print()


# ============================================================================
# Set Up and Run IRLS Inversion
# ============================================================================

print("=" * 80)
print("Step 17: Setting Up IRLS Inversion")
print("=" * 80)

log_resistivity_map = maps.InjectActiveCells(mesh, active_cells, 1e8) * maps.ExpMap(
    nP=n_active
)

starting_resistivity_model = np.log(median_resistivity) * np.ones(n_active)
reference_resistivity_model = starting_resistivity_model.copy()

resistivity_data.survey.drape_electrodes_on_topography(mesh, active_cells, option="top")

resistivity_simulation = dc.simulation_2d.Simulation2DNodal(
    mesh, survey=resistivity_data.survey, rhoMap=log_resistivity_map, storeJ=True
)

dmis_irls = data_misfit.L2DataMisfit(
    simulation=resistivity_simulation, data=resistivity_data
)

reg_irls = regularization.Sparse(
    mesh,
    active_cells=active_cells,
    length_scale_x=5.0,
    length_scale_y=5.0,
    norms=[0, 2, 2],
    reference_model=reference_resistivity_model,
)

opt_irls = optimization.InexactGaussNewton(
    maxIter=50, maxIterLS=20, maxIterCG=20, tolCG=1e-3
)

inv_prob_irls = inverse_problem.BaseInvProblem(dmis_irls, reg_irls, opt_irls)

sensitivity_weights_irls = directives.UpdateSensitivityWeights(
    every_iteration=True, threshold_value=1e-2
)
update_irls = directives.UpdateIRLS(
    cooling_factor=2,
    cooling_rate=2,
    f_min_change=1e-4,
    max_irls_iterations=30,
    chifact_start=1.0,
)
starting_beta_irls = directives.BetaEstimate_ByEig(beta0_ratio=10)
update_jacobi_irls = directives.UpdatePreconditioner(update_every_iteration=True)

directives_list_irls = [
    update_irls,
    sensitivity_weights_irls,
    starting_beta_irls,
    update_jacobi_irls,
]

print(f"  Regularization: Sparse (IRLS)")
print(f"  Norms: [0, 2, 2] for blocky recovery")
print()

print("=" * 80)
print("Step 18: Running IRLS Inversion")
print("=" * 80)

inv_irls = inversion.BaseInversion(inv_prob_irls, directives_list_irls)
recovered_log_resistivity_model = inv_irls.run(starting_resistivity_model)

print()
print(f"  IRLS inversion completed successfully")
print(f"  Final data misfit: {inv_prob_irls.dmisfit.phi:.2f}")
print()


# ============================================================================
# Plot True, L2 and IRLS Models
# ============================================================================

print("=" * 80)
print("Step 19: Plotting Model Comparison")
print("=" * 80)

# Recreate true model
true_background_conductivity = 1e-2
true_conductor_conductivity = 1e-1
true_resistor_conductivity = 1e-3

true_conductivity_model = true_background_conductivity * np.ones(n_active)

ind_conductor = model_builder.get_indices_sphere(
    np.r_[-120.0, 40.0], 60.0, mesh.cell_centers[active_cells, :]
)
true_conductivity_model[ind_conductor] = true_conductor_conductivity

ind_resistor = model_builder.get_indices_sphere(
    np.r_[120.0, 72.0], 60.0, mesh.cell_centers[active_cells, :]
)
true_conductivity_model[ind_resistor] = true_resistor_conductivity

# Convert IRLS log-resistivity to conductivity
recovered_conductivity_irls = 1 / np.exp(recovered_log_resistivity_model)

# Plot all models
plotting_model = [
    true_conductivity_model,
    recovered_conductivity_L2,
    recovered_conductivity_irls,
]

fig = plt.figure(figsize=(9, 13))
title_str = [
    "True Conductivity Model",
    "Recovered Model (L2)",
    "Recovered Model (IRLS)",
]

for ii in range(0, 3):
    ax1 = fig.add_axes([0.14, 0.75 - 0.3 * ii, 0.68, 0.22])
    mesh.plot_image(
        plotting_map * plotting_model[ii],
        ax=ax1,
        grid=False,
        pcolor_opts={"norm": norm, "cmap": cmap},
    )
    ax1.set_xlim(-500, 500)
    ax1.set_ylim(-300, 200)
    ax1.set_title(title_str[ii])
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("z (m)")

    ax2 = fig.add_axes([0.84, 0.75 - 0.3 * ii, 0.03, 0.22])
    cbar = mpl.colorbar.ColorbarBase(
        ax2, norm=norm, orientation="vertical", cmap=cmap
    )
    cbar.set_label(r"$\sigma$ (S/m)", rotation=270, labelpad=15, size=12)

plt.savefig("dcr_2d_models_comparison.png", dpi=150, bbox_inches="tight")
print("  Model comparison saved as 'dcr_2d_models_comparison.png'")
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
print(f"  Number of data:           {voltage_data.nD}")
print(f"  Mesh cells:               {mesh.n_cells}")
print(f"  Active cells:             {n_active}")
print()
print("Inversion Results:")
print("-" * 80)
print(f"  L2 final misfit:          {inv_prob_L2.dmisfit.phi:.2f}")
print(f"  IRLS final misfit:        {inv_prob_irls.dmisfit.phi:.2f}")
print()
print("Generated Files:")
print("-" * 80)
print("  1. dcr_2d_topography.png             - Survey topography")
print("  2. dcr_2d_voltage_pseudosection.png  - Voltage data pseudo-section")
print("  3. dcr_2d_appres_pseudosection.png   - Apparent resistivity pseudo-section")
print("  4. dcr_2d_L2_data_misfit.png         - L2 inversion data fit")
print("  5. dcr_2d_L2_model.png               - L2 recovered model")
print("  6. dcr_2d_models_comparison.png      - True, L2, and IRLS models")
print()
print("Key Findings:")
print("-" * 80)
print("  - L2 inversion recovers smooth conductivity distribution")
print("  - IRLS inversion recovers blocky structures with sharp boundaries")
print("  - Both approaches fit the data well")
print("  - Sensitivity weighting is critical for DC resistivity inversion")
print("=" * 80)
