"""
2.5D IP Inversion
University of British Columbia

This tutorial demonstrates how to invert IP data to recover the subsurface chargeability distribution
on a tree mesh. We show two inversion approaches:

1. Weighted least-squares inversion
2. Iteratively re-weighted least-squares (IRLS) inversion

This tutorial focuses on content specific to IP inversion and demonstrates the importance of using
an appropriate background conductivity model for IP inversion.

Learning Objectives:
- Assigning appropriate uncertainties to IP data
- Designing a mesh for IP inversion
- Obtaining a background conductivity/resistivity model for the IP inversion
- Analyzing inversion results
- Understanding the impact of background conductivity on IP inversion

Keywords: induced polarization, 2.5D inversion, sparse norm, tree mesh
"""

# ============================================================================
# Import Modules
# ============================================================================

# SimPEG functionality
from simpeg.electromagnetics.static import induced_polarization as ip
from simpeg.electromagnetics.static.utils.static_utils import plot_pseudosection
from simpeg.utils.io_utils.io_utils_electromagnetics import read_dcip2d_ubc
from simpeg.utils import download, model_builder
from simpeg import (
    maps,
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
from matplotlib.colors import Normalize, LogNorm
import tarfile

mpl.rcParams.update({"font.size": 14})
cmap = mpl.cm.RdYlBu_r


# ============================================================================
# Load Tutorial Data
# ============================================================================

print("=" * 80)
print("Step 1: Loading Tutorial Data")
print("=" * 80)

# URL to download from repository assets
data_source = "https://github.com/simpeg/user-tutorials/raw/main/assets/06-ip/inv_ip_2d_files.tar.gz"

# download the data
downloaded_data = download(data_source, overwrite=True)

# unzip the tarfile
tar = tarfile.open(downloaded_data, "r")
tar.extractall()
tar.close()

# path to the directory containing our data
dir_path = downloaded_data.split(".")[0] + os.path.sep

# files to work with
topo_filename = dir_path + "topo_2d.txt"
data_filename = dir_path + "ip_data.obs"

# Load 2D topography
topo_2d = np.loadtxt(str(topo_filename))

print(f"  Loaded topography from: {topo_filename}")
print(f"  Loaded data from: {data_filename}")
print()


# ============================================================================
# Plot Topography
# ============================================================================

print("=" * 80)
print("Step 2: Plotting Topography")
print("=" * 80)

fig = plt.figure(figsize=(10, 2))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(topo_2d[:, 0], topo_2d[:, -1], color="b", linewidth=1)
ax.set_xlim([topo_2d[:, 0].min(), topo_2d[:, 0].max()])
ax.set_xlabel("x (m)", labelpad=5)
ax.set_ylabel("z (m)", labelpad=5)
ax.grid(True)
ax.set_title("Topography (Exaggerated z-axis)", fontsize=16, pad=10)
plt.savefig("ip_2d_topography.png", dpi=150, bbox_inches="tight")
print("✓ Topography plot saved as 'ip_2d_topography.png'")
plt.close()
print()


# ============================================================================
# Load IP Data
# ============================================================================

print("=" * 80)
print("Step 3: Loading IP Data")
print("=" * 80)

ip_data = read_dcip2d_ubc(data_filename, "apparent_chargeability", "general")

print(f"  Number of data points: {ip_data.survey.nD}")
print()


# ============================================================================
# Plot IP Data in Pseudo-Section
# ============================================================================

print("=" * 80)
print("Step 4: Plotting Apparent Chargeability Data")
print("=" * 80)

mpl.rcParams.update({"font.size": 12})
apparent_chargeability = ip_data.dobs

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_axes([0.1, 0.15, 0.75, 0.78])
plot_pseudosection(
    ip_data.survey,
    apparent_chargeability,
    "contourf",
    ax=ax1,
    scale="linear",
    cbar_label="V/V",
    mask_topography=True,
    contourf_opts={"levels": 20, "cmap": mpl.cm.plasma},
)
ax1.set_title("Apparent Chargeability")
plt.savefig("ip_2d_apparent_chargeability.png", dpi=150, bbox_inches="tight")
print("✓ Apparent chargeability plot saved as 'ip_2d_apparent_chargeability.png'")
plt.close()

mpl.rcParams.update({"font.size": 14})
print()


# ============================================================================
# Assign Uncertainties
# ============================================================================

print("=" * 80)
print("Step 5: Assigning Data Uncertainties")
print("=" * 80)

ip_data.standard_deviation = 5e-3 * np.ones_like(ip_data.dobs)

print(f"  Floor uncertainty: 5e-3 V/V")
print()


# ============================================================================
# Design Tree Mesh
# ============================================================================

print("=" * 80)
print("Step 6: Designing Tree Mesh")
print("=" * 80)

dh = 4  # base cell width
dom_width_x = 3200.0  # domain width x
dom_width_z = 2400.0  # domain width z
nbcx = 2 ** int(np.round(np.log(dom_width_x / dh) / np.log(2.0)))
nbcz = 2 ** int(np.round(np.log(dom_width_z / dh) / np.log(2.0)))

# Define the base mesh
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
unique_locations = ip_data.survey.unique_electrode_locations

# Mesh refinement near electrodes
mesh.refine_points(
    unique_locations, padding_cells_by_level=[8, 12, 6, 6], finalize=False
)

mesh.finalize()

print(f"  Minimum cell width: {dh} m")
print(f"  Number of cells: {mesh.n_cells}")
print()


# ============================================================================
# Define Active Cells
# ============================================================================

print("=" * 80)
print("Step 7: Defining Active Cells")
print("=" * 80)

active_cells = active_from_xyz(mesh, topo_2d)
n_active = np.sum(active_cells)

print(f"  Total mesh cells: {mesh.n_cells}")
print(f"  Active cells: {n_active}")
print()


# ============================================================================
# Project Electrodes to Discretized Topography
# ============================================================================

print("=" * 80)
print("Step 8: Projecting Electrodes to Surface")
print("=" * 80)

ip_data.survey.drape_electrodes_on_topography(mesh, active_cells, option="top")

print("✓ Electrodes projected to discretized surface")
print()


# ============================================================================
# Define Background Conductivity Models
# ============================================================================

print("=" * 80)
print("Step 9: Defining Background Conductivity Models")
print("=" * 80)

# Define electrical conductivities in S/m
air_conductivity = 1e-8
background_conductivity = 1e-2
conductor_conductivity = 1e-1
resistor_conductivity = 1e-3

# Define halfspace conductivity model (for L2 inversion)
halfspace_conductivity_model = background_conductivity * np.ones(n_active)

# Define true conductivity model (for IRLS inversion)
true_conductivity_model = halfspace_conductivity_model.copy()

ind_conductor = model_builder.get_indices_sphere(
    np.r_[-120.0, 40.0], 60.0, mesh.cell_centers[active_cells, :]
)
true_conductivity_model[ind_conductor] = conductor_conductivity

ind_resistor = model_builder.get_indices_sphere(
    np.r_[120.0, 72.0], 60.0, mesh.cell_centers[active_cells, :]
)
true_conductivity_model[ind_resistor] = resistor_conductivity

# Mapping from conductivity model to conductivity on all cells
conductivity_map = maps.InjectActiveCells(mesh, active_cells, air_conductivity)

# Mapping to neglect air cells in plot
plotting_map = maps.InjectActiveCells(mesh, active_cells, np.nan)

print(f"  Halfspace conductivity: {background_conductivity} S/m")
print(f"  Conductor conductivity: {conductor_conductivity} S/m")
print(f"  Resistor conductivity: {resistor_conductivity} S/m")
print()

# Plot true background conductivity
fig = plt.figure(figsize=(9, 4))
norm = LogNorm(vmin=1e-3, vmax=1e-1)

ax1 = fig.add_axes([0.14, 0.17, 0.68, 0.7])
mesh.plot_image(
    plotting_map * true_conductivity_model,
    ax=ax1,
    grid=False,
    pcolor_opts={"norm": norm, "cmap": mpl.cm.RdYlBu_r},
)
ax1.set_xlim(-500, 500)
ax1.set_ylim(-300, 200)
ax1.set_title("True Background Conductivity")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")

ax2 = fig.add_axes([0.84, 0.17, 0.03, 0.7])
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.RdYlBu_r
)
cbar.set_label(r"$\sigma$ (S/m)", rotation=270, labelpad=15, size=12)

plt.savefig("ip_2d_background_conductivity.png", dpi=150, bbox_inches="tight")
print("✓ Background conductivity plot saved as 'ip_2d_background_conductivity.png'")
plt.close()


# ============================================================================
# WEIGHTED LEAST-SQUARES INVERSION (HALFSPACE BACKGROUND)
# ============================================================================

print()
print("=" * 80)
print("WEIGHTED LEAST-SQUARES INVERSION (HALFSPACE BACKGROUND)")
print("=" * 80)
print()


# ============================================================================
# Model Mapping
# ============================================================================

print("=" * 80)
print("Step 10: Creating Chargeability Model Mapping")
print("=" * 80)

chargeability_map = maps.InjectActiveCells(mesh, active_cells, 0.0)

print(f"  Model parameters: {n_active}")
print()


# ============================================================================
# Starting and Reference Models
# ============================================================================

print("=" * 80)
print("Step 11: Defining Starting and Reference Models")
print("=" * 80)

starting_chargeability_model = 1e-6 * np.ones(n_active)
reference_chargeability_model = np.zeros_like(starting_chargeability_model)

print(f"  Starting model: {starting_chargeability_model[0]:.2e} V/V")
print()


# ============================================================================
# Define Forward Simulation (L2)
# ============================================================================

print("=" * 80)
print("Step 12: Defining Forward Simulation (L2)")
print("=" * 80)

simulation_L2 = ip.Simulation2DNodal(
    mesh,
    survey=ip_data.survey,
    sigma=conductivity_map * halfspace_conductivity_model,
    etaMap=chargeability_map,
    storeJ=True,
)

print("  Simulation type: 2D Nodal")
print("  Background: Halfspace conductivity")
print()


# ============================================================================
# Define Data Misfit (L2)
# ============================================================================

print("=" * 80)
print("Step 13: Defining Data Misfit (L2)")
print("=" * 80)

dmis_L2 = data_misfit.L2DataMisfit(simulation=simulation_L2, data=ip_data)

print("  Data misfit type: L2")
print()


# ============================================================================
# Define Regularization (L2)
# ============================================================================

print("=" * 80)
print("Step 14: Defining Regularization (L2)")
print("=" * 80)

reg_L2 = regularization.WeightedLeastSquares(
    mesh,
    active_cells=active_cells,
    length_scale_x=5.0,
    length_scale_y=5.0,
    reference_model=reference_chargeability_model,
    reference_model_in_smooth=False,
)

print("  Regularization type: Weighted Least-Squares")
print("  Length scales: x=5.0, y=5.0")
print()


# ============================================================================
# Define Optimization (L2)
# ============================================================================

print("=" * 80)
print("Step 15: Defining Optimization Algorithm (L2)")
print("=" * 80)

opt_L2 = optimization.ProjectedGNCG(
    maxIter=40, lower=0.0, maxIterLS=20, maxIterCG=20, tolCG=1e-2
)

print("  Algorithm: Projected Gauss-Newton CG")
print("  Lower bound: 0.0 V/V")
print()


# ============================================================================
# Define Inverse Problem (L2)
# ============================================================================

print("=" * 80)
print("Step 16: Defining Inverse Problem (L2)")
print("=" * 80)

inv_prob_L2 = inverse_problem.BaseInvProblem(dmis_L2, reg_L2, opt_L2)

print("  Inverse problem created")
print()


# ============================================================================
# Define Directives (L2)
# ============================================================================

print("=" * 80)
print("Step 17: Defining Inversion Directives (L2)")
print("=" * 80)

sensitivity_weights = directives.UpdateSensitivityWeights(
    every_iteration=False, threshold_value=1e-2
)
update_jacobi = directives.UpdatePreconditioner(update_every_iteration=False)
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=20)
beta_schedule = directives.BetaSchedule(coolingFactor=2.0, coolingRate=2)
target_misfit = directives.TargetMisfit(chifact=1.0)

directives_list_L2 = [
    sensitivity_weights,
    update_jacobi,
    starting_beta,
    beta_schedule,
    target_misfit,
]

print("  Directives configured")
print()


# ============================================================================
# Run L2 Inversion
# ============================================================================

print("=" * 80)
print("Step 18: Running L2 Inversion")
print("=" * 80)
print()

inv_L2 = inversion.BaseInversion(inv_prob_L2, directives_list_L2)
recovered_chargeability_L2 = inv_L2.run(starting_chargeability_model)

print()
print("✓ L2 inversion completed")
print()


# ============================================================================
# Plot L2 Results
# ============================================================================

print("=" * 80)
print("Step 19: Analyzing L2 Results")
print("=" * 80)

dpred = inv_prob_L2.dpred
dobs = ip_data.dobs
std = ip_data.standard_deviation

# Plot data fit
fig = plt.figure(figsize=(9, 11))
data_array = [np.abs(dobs), np.abs(dpred), (dobs - dpred)]
plot_title = ["Observed", "Predicted", "Misfit"]
plot_units = ["V/V", "V/V", "V/V"]
scale = ["linear", "linear", "linear"]
cmap_list = [mpl.cm.plasma, mpl.cm.plasma, mpl.cm.RdYlBu_r]

ax1 = 3 * [None]
cax1 = 3 * [None]
cbar = 3 * [None]
cplot = 3 * [None]

for ii in range(0, 3):
    ax1[ii] = fig.add_axes([0.15, 0.72 - 0.33 * ii, 0.65, 0.21])
    cax1[ii] = fig.add_axes([0.81, 0.72 - 0.33 * ii, 0.03, 0.21])
    cplot[ii] = plot_pseudosection(
        ip_data.survey,
        data_array[ii],
        "contourf",
        ax=ax1[ii],
        cax=cax1[ii],
        scale=scale[ii],
        cbar_label=plot_units[ii],
        mask_topography=True,
        contourf_opts={"levels": 25, "cmap": cmap_list[ii]},
    )
    ax1[ii].set_title(plot_title[ii])

plt.savefig("ip_2d_L2_data_fit.png", dpi=150, bbox_inches="tight")
print("✓ L2 data fit plot saved as 'ip_2d_L2_data_fit.png'")
plt.close()

# Plot recovered model
fig = plt.figure(figsize=(9, 4))
norm = Normalize(vmin=0.0, vmax=0.2)

ax1 = fig.add_axes([0.14, 0.17, 0.68, 0.7])
mesh.plot_image(
    plotting_map * recovered_chargeability_L2,
    normal="Y",
    ax=ax1,
    grid=False,
    pcolor_opts={"norm": norm, "cmap": mpl.cm.plasma},
)
ax1.set_xlim(-500, 500)
ax1.set_ylim(-300, 200)
ax1.set_title("Recovered L2-Model (Halfspace Background)")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")

ax2 = fig.add_axes([0.84, 0.17, 0.03, 0.7])
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.plasma
)
cbar.set_label("V/V", rotation=270, labelpad=15, size=12)

plt.savefig("ip_2d_L2_model.png", dpi=150, bbox_inches="tight")
print("✓ L2 model plot saved as 'ip_2d_L2_model.png'")
plt.close()

print(f"  L2 model range: [{np.min(recovered_chargeability_L2):.4f}, {np.max(recovered_chargeability_L2):.4f}] V/V")
print()


# ============================================================================
# ITERATIVELY RE-WEIGHTED LEAST-SQUARES INVERSION (TRUE BACKGROUND)
# ============================================================================

print()
print("=" * 80)
print("ITERATIVELY RE-WEIGHTED LEAST-SQUARES INVERSION (TRUE BACKGROUND)")
print("=" * 80)
print()


# ============================================================================
# Define Forward Simulation (IRLS)
# ============================================================================

print("=" * 80)
print("Step 20: Defining Forward Simulation (IRLS)")
print("=" * 80)

simulation_irls = ip.Simulation2DNodal(
    mesh,
    survey=ip_data.survey,
    sigma=conductivity_map * true_conductivity_model,
    etaMap=chargeability_map,
    storeJ=True,
)

print("  Simulation type: 2D Nodal")
print("  Background: True conductivity model")
print()


# ============================================================================
# Define Data Misfit (IRLS)
# ============================================================================

print("=" * 80)
print("Step 21: Defining Data Misfit (IRLS)")
print("=" * 80)

dmis_irls = data_misfit.L2DataMisfit(simulation=simulation_irls, data=ip_data)

print("  Data misfit type: L2")
print()


# ============================================================================
# Define Regularization (IRLS)
# ============================================================================

print("=" * 80)
print("Step 22: Defining Regularization (IRLS)")
print("=" * 80)

reg_irls = regularization.Sparse(
    mesh,
    active_cells=active_cells,
    length_scale_x=5.0,
    length_scale_y=5.0,
    norms=[0, 2, 2],
    reference_model=reference_chargeability_model,
    reference_model_in_smooth=False,
)

print("  Regularization type: Sparse")
print("  Norms: [0, 2, 2] (compact structures)")
print()


# ============================================================================
# Define Optimization (IRLS)
# ============================================================================

print("=" * 80)
print("Step 23: Defining Optimization Algorithm (IRLS)")
print("=" * 80)

opt_irls = optimization.ProjectedGNCG(
    maxIter=40, lower=0.0, maxIterLS=20, maxIterCG=20, tolCG=1e-2
)

print("  Algorithm: Projected Gauss-Newton CG")
print()


# ============================================================================
# Define Inverse Problem (IRLS)
# ============================================================================

print("=" * 80)
print("Step 24: Defining Inverse Problem (IRLS)")
print("=" * 80)

inv_prob_irls = inverse_problem.BaseInvProblem(dmis_irls, reg_irls, opt_irls)

print("  Inverse problem created")
print()


# ============================================================================
# Define Directives (IRLS)
# ============================================================================

print("=" * 80)
print("Step 25: Defining Inversion Directives (IRLS)")
print("=" * 80)

sensitivity_weights_irls = directives.UpdateSensitivityWeights(
    every_iteration=False, threshold_value=1e-2
)
update_irls = directives.UpdateIRLS(
    cooling_factor=2,
    cooling_rate=2,
    f_min_change=1e-4,
    max_irls_iterations=30,
    chifact_start=1.0,
)
starting_beta_irls = directives.BetaEstimate_ByEig(beta0_ratio=20)
update_jacobi_irls = directives.UpdatePreconditioner(update_every_iteration=True)

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
print("Step 26: Running IRLS Inversion")
print("=" * 80)
print()

inv_irls = inversion.BaseInversion(inv_prob_irls, directives_list_irls)
recovered_chargeability_irls = inv_irls.run(starting_chargeability_model)

print()
print("✓ IRLS inversion completed")
print()


# ============================================================================
# Compare All Models
# ============================================================================

print("=" * 80)
print("Step 27: Comparing All Models")
print("=" * 80)

# Recreate True Model
true_background_chargeability = 0.0
true_conductor_chargeability = 1e-1

true_chargeability_model = true_background_chargeability * np.ones(n_active)

ind_conductor = model_builder.get_indices_sphere(
    np.r_[-120.0, 40.0], 60.0, mesh.cell_centers[active_cells, :]
)
true_chargeability_model[ind_conductor] = true_conductor_chargeability

# Extract the L2 Model from the IRLS Inversion
recovered_chargeability_L2_good = inv_prob_irls.l2model

# Plot all models
norm = Normalize(vmin=0.0, vmax=0.1)

fig = plt.figure(figsize=(9, 16))
ax1 = 4 * [None]
ax2 = 4 * [None]
title_str = [
    "True Chargeability Model",
    "Recovered L2-Model (halfspace background)",
    "Recovered L2-Model (true background)",
    "Recovered IRLS-Model (true background)",
]
plotting_model = [
    true_chargeability_model,
    recovered_chargeability_L2,
    recovered_chargeability_L2_good,
    recovered_chargeability_irls,
]

for ii in range(0, 4):
    ax1[ii] = fig.add_axes([0.14, 0.78 - 0.25 * ii, 0.68, 0.18])
    mesh.plot_image(
        plotting_map * plotting_model[ii],
        ax=ax1[ii],
        grid=False,
        pcolor_opts={"norm": norm, "cmap": mpl.cm.plasma},
    )
    ax1[ii].set_xlim(-500, 500)
    ax1[ii].set_ylim(-300, 200)
    ax1[ii].set_title(title_str[ii])
    ax1[ii].set_xlabel("x (m)")
    ax1[ii].set_ylabel("z (m)")

    ax2[ii] = fig.add_axes([0.84, 0.78 - 0.25 * ii, 0.03, 0.18])
    cbar = mpl.colorbar.ColorbarBase(
        ax2[ii], norm=norm, orientation="vertical", cmap=mpl.cm.plasma
    )
    cbar.set_label("V/V", rotation=270, labelpad=15, size=12)

plt.savefig("ip_2d_model_comparison.png", dpi=150, bbox_inches="tight")
print("✓ Model comparison plot saved as 'ip_2d_model_comparison.png'")
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
print(f"  Mesh cells:          {mesh.n_cells}")
print(f"  Active cells:        {n_active}")
print(f"  Number of data:      {ip_data.survey.nD}")
print()
print(f"  L2 Model (halfspace):")
print(f"    Range: [{np.min(recovered_chargeability_L2):.4f}, {np.max(recovered_chargeability_L2):.4f}] V/V")
print()
print(f"  L2 Model (true background):")
print(f"    Range: [{np.min(recovered_chargeability_L2_good):.4f}, {np.max(recovered_chargeability_L2_good):.4f}] V/V")
print()
print(f"  IRLS Model (true background):")
print(f"    Range: [{np.min(recovered_chargeability_irls):.4f}, {np.max(recovered_chargeability_irls):.4f}] V/V")
print()
print("Generated Files:")
print("-" * 80)
print("  1. ip_2d_topography.png                  - 2D topography profile")
print("  2. ip_2d_apparent_chargeability.png      - Observed apparent chargeability")
print("  3. ip_2d_background_conductivity.png     - True background conductivity")
print("  4. ip_2d_L2_data_fit.png                 - L2 inversion data fit")
print("  5. ip_2d_L2_model.png                    - L2 recovered model")
print("  6. ip_2d_model_comparison.png            - Comparison of all models")
print()
print("Key Findings:")
print("-" * 80)
print("  - Background conductivity model SIGNIFICANTLY affects IP inversion")
print("  - Halfspace background leads to artifacts at depth")
print("  - True background conductivity produces better results")
print("  - IRLS recovers more compact chargeability structures")
print("=" * 80)
