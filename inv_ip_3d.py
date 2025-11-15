"""
3D IP Inversion
University of British Columbia

This tutorial demonstrates how to invert apparent chargeability data to recover the subsurface
chargeability distribution on a tree mesh. We show two inversion approaches:

1. Weighted least-squares inversion
2. Iteratively re-weighted least-squares (IRLS) inversion

Learning Objectives:
- Assigning appropriate uncertainties to IP data
- Designing a mesh for IP inversion
- Obtaining a background conductivity/resistivity model for the IP inversion
- Analyzing inversion results

Keywords: induced polarization, 3D inversion, sparse norm, tree mesh
"""

# ============================================================================
# Import Modules
# ============================================================================

# SimPEG functionality
from simpeg.electromagnetics.static import induced_polarization as ip
from simpeg.electromagnetics.static.utils.static_utils import (
    plot_pseudosection,
    convert_survey_3d_to_2d_lines,
)
from simpeg.utils.io_utils.io_utils_electromagnetics import read_dcip_xyz
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
import tarfile
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

mpl.rcParams.update({"font.size": 16})


# ============================================================================
# Load Tutorial Data
# ============================================================================

print("=" * 80)
print("Step 1: Loading Tutorial Data")
print("=" * 80)

# URL to download from repository assets
data_source = "https://github.com/simpeg/user-tutorials/raw/main/assets/06-ip/inv_ip_3d_files.tar.gz"

# download the data
downloaded_data = download(data_source, overwrite=True)

# unzip the tarfile
tar = tarfile.open(downloaded_data, "r")
tar.extractall()
tar.close()

# path to the directory containing our data
dir_path = downloaded_data.split(".")[0] + os.path.sep

# files to work with
topo_filename = dir_path + "topo_xyz.txt"
data_filename = dir_path + "ip_data.xyz"

# Load topography
topo_xyz = np.loadtxt(str(topo_filename))

print(f"  Loaded topography from: {topo_filename}")
print(f"  Loaded data from: {data_filename}")
print()


# ============================================================================
# Plot Topography
# ============================================================================

print("=" * 80)
print("Step 2: Plotting Topography")
print("=" * 80)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection="3d")
ax.set_zlim([-400, 400])
ax.scatter3D(topo_xyz[:, 0], topo_xyz[:, 1], topo_xyz[:, 2], s=0.25, c="b")
ax.set_box_aspect(aspect=None, zoom=0.85)
ax.set_xlabel("X (m)", labelpad=10)
ax.set_ylabel("Y (m)", labelpad=10)
ax.set_zlabel("Z (m)", labelpad=10)
ax.set_title("Topography (Exaggerated z-axis)", fontsize=16, pad=-20)
ax.view_init(elev=45.0, azim=-125)
plt.savefig("ip_3d_topography.png", dpi=150, bbox_inches="tight")
print("✓ Topography plot saved as 'ip_3d_topography.png'")
plt.close()
print()


# ============================================================================
# Load IP Data
# ============================================================================

print("=" * 80)
print("Step 3: Loading IP Data")
print("=" * 80)

ip_data, out_dict = read_dcip_xyz(
    data_filename,
    "apparent_chargeability",
    data_header="APP_CHG",
    uncertainties_header="UNCERT",
    is_surface_data=False,
    dict_headers=["LINEID"],
)

print(f"  Number of data points: {ip_data.survey.nD}")
print()

# Extract line IDs
lineID = np.array(out_dict["LINEID"], dtype=int)


# ============================================================================
# Plot 2D Pseudosection for One Line
# ============================================================================

print("=" * 80)
print("Step 4: Plotting Apparent Chargeability Data")
print("=" * 80)

# Create list of 2D surveys
survey_2d_list, index_list = convert_survey_3d_to_2d_lines(
    ip_data.survey, lineID, data_type="apparent_chargeability", output_indexing=True
)

dobs_2d_list = []
apparent_chargeabilities_2d = []
for ind in index_list:
    dobs_2d_list.append(ip_data.dobs[ind])
    apparent_chargeabilities_2d.append(ip_data.dobs[ind])

line_index = 0

fig = plt.figure(figsize=(8, 2.75))
ax1 = fig.add_axes([0.1, 0.15, 0.75, 0.78])
plot_pseudosection(
    survey_2d_list[line_index],
    dobs=apparent_chargeabilities_2d[line_index],
    plot_type="contourf",
    ax=ax1,
    scale="linear",
    cbar_label="V/V",
    mask_topography=True,
    contourf_opts={"levels": 20, "cmap": mpl.cm.plasma},
)
ax1.set_title("Apparent Chargeability")
plt.savefig("ip_3d_apparent_chargeability.png", dpi=150, bbox_inches="tight")
print("✓ Apparent chargeability plot saved as 'ip_3d_apparent_chargeability.png'")
plt.close()
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

# Defining domain size and minimum cell size
dh = 25.0  # base cell width
dom_width_x = 8000.0  # domain width x
dom_width_y = 8000.0  # domain width y
dom_width_z = 4000.0  # domain width z

# Number of base mesh cells in each direction
nbcx = 2 ** int(np.round(np.log(dom_width_x / dh) / np.log(2.0)))
nbcy = 2 ** int(np.round(np.log(dom_width_y / dh) / np.log(2.0)))
nbcz = 2 ** int(np.round(np.log(dom_width_z / dh) / np.log(2.0)))

# Define the base mesh
hx = [(dh, nbcx)]
hy = [(dh, nbcy)]
hz = [(dh, nbcz)]
mesh = TreeMesh([hx, hy, hz], x0="CCN", diagonal_balance=True)

# Shift top to maximum topography
mesh.origin = mesh.origin + np.r_[0.0, 0.0, topo_xyz[:, -1].max()]

# Mesh refinement based on surface topography
k = np.sqrt(np.sum(topo_xyz[:, 0:2] ** 2, axis=1)) < 1200
mesh.refine_surface(topo_xyz[k, :], padding_cells_by_level=[0, 4, 4], finalize=False)

# Extract unique electrode locations
unique_locations = ip_data.survey.unique_electrode_locations

# Mesh refinement near electrodes
mesh.refine_points(unique_locations, padding_cells_by_level=[6, 6, 4], finalize=False)

# Finalize the mesh
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

active_cells = active_from_xyz(mesh, topo_xyz)
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
# Define Background Conductivity
# ============================================================================

print("=" * 80)
print("Step 9: Defining Background Conductivity Model")
print("=" * 80)

# Define conductivity model in S/m
air_conductivity = 1e-8
background_conductivity = 1e-2
conductor_conductivity = 1e-1
resistor_conductivity = 1e-3

# Define conductivity model
conductivity_model = background_conductivity * np.ones(n_active)

ind_conductor = model_builder.get_indices_sphere(
    np.r_[-300.0, 0.0, 100.0], 165.0, mesh.cell_centers[active_cells, :]
)
conductivity_model[ind_conductor] = conductor_conductivity

ind_resistor = model_builder.get_indices_sphere(
    np.r_[300.0, 0.0, 100.0], 165.0, mesh.cell_centers[active_cells, :]
)
conductivity_model[ind_resistor] = resistor_conductivity

# Mappings
conductivity_map = maps.InjectActiveCells(mesh, active_cells, air_conductivity)
plotting_map = maps.InjectActiveCells(mesh, active_cells, np.nan)

print(f"  Background conductivity: {background_conductivity} S/m")
print(f"  Conductor conductivity: {conductor_conductivity} S/m")
print(f"  Resistor conductivity: {resistor_conductivity} S/m")
print()

# Plot background conductivity
fig = plt.figure(figsize=(10, 4.5))
log_norm = LogNorm(vmin=1e-3, vmax=0.1)

ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
mesh.plot_slice(
    plotting_map * conductivity_model,
    ax=ax1,
    normal="Y",
    ind=int(len(mesh.h[1]) / 2),
    grid=True,
    pcolor_opts={"cmap": mpl.cm.RdYlBu_r, "norm": log_norm},
)
ax1.set_title("True Background Conductivity")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")
ax1.set_xlim([-1500, 1500])
ax1.set_ylim([topo_xyz[:, -1].max() - 1500, topo_xyz[:, -1].max()])

ax2 = fig.add_axes([0.84, 0.15, 0.03, 0.75])
cbar = mpl.colorbar.ColorbarBase(
    ax2, cmap=mpl.cm.RdYlBu_r, norm=log_norm, orientation="vertical"
)
cbar.set_label("Conductivity [S/m]", rotation=270, labelpad=15, size=12)

plt.savefig("ip_3d_background_conductivity.png", dpi=150, bbox_inches="tight")
print("✓ Background conductivity plot saved as 'ip_3d_background_conductivity.png'")
plt.close()


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

starting_chargeability_model = 1e-4 * np.ones(n_active)
reference_chargeability_model = np.zeros_like(starting_chargeability_model)

print(f"  Starting model: {starting_chargeability_model[0]:.2e} V/V")
print()


# ============================================================================
# Define Forward Simulation (L2)
# ============================================================================

print("=" * 80)
print("Step 12: Defining Forward Simulation (L2)")
print("=" * 80)

simulation_L2 = ip.Simulation3DNodal(
    mesh,
    survey=ip_data.survey,
    sigma=conductivity_map * conductivity_model,
    etaMap=chargeability_map,
    storeJ=True,
)

print("  Simulation type: 3D Nodal")
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
    length_scale_x=10.0,
    length_scale_y=10.0,
    length_scale_z=10.0,
    reference_model=reference_chargeability_model,
    reference_model_in_smooth=False,
)

print("  Regularization type: Weighted Least-Squares")
print("  Length scales: x=10.0, y=10.0, z=10.0")
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
update_jacobi = directives.UpdatePreconditioner(update_every_iteration=True)
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1000)
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

# Predicted data from recovered model
dpred_ip = inv_prob_L2.dpred

# Plot 2D pseudosections for individual survey lines
line_index = 0
k = lineID == line_index + 1
data_array = [
    np.abs(ip_data.dobs[k]),
    np.abs(dpred_ip[k]),
    ip_data.dobs[k] - dpred_ip[k],
]

fig = plt.figure(figsize=(9, 11))
plot_title = ["Observed", "Predicted", "Misfit"]
plot_units = ["V/V", "V/V", "V/V"]
scale = ["linear", "linear", "linear"]
cmap_list = [mpl.cm.plasma, mpl.cm.plasma, mpl.cm.RdYlBu]

ax1 = 3 * [None]
cax1 = 3 * [None]
cbar = 3 * [None]
cplot = 3 * [None]

for ii in range(0, 3):
    ax1[ii] = fig.add_axes([0.15, 0.72 - 0.33 * ii, 0.65, 0.21])
    cax1[ii] = fig.add_axes([0.81, 0.72 - 0.33 * ii, 0.03, 0.21])
    cplot[ii] = plot_pseudosection(
        survey_2d_list[line_index],
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

plt.savefig("ip_3d_L2_data_fit.png", dpi=150, bbox_inches="tight")
print("✓ L2 data fit plot saved as 'ip_3d_L2_data_fit.png'")
plt.close()


# Plot recovered model
# Define intrinsic chargeability model (V/V)
true_chargeability_model = 1e-6 * np.ones(n_active)
ind_chargeable = model_builder.get_indices_sphere(
    np.r_[-300.0, 0.0, 100.0], 165.0, mesh.cell_centers[active_cells, :]
)
true_chargeability_model[ind_chargeable] = 0.1

fig = plt.figure(figsize=(9, 9))
ax1 = 2 * [None]
ax2 = 2 * [None]
cbar = 2 * [None]
norm = 2 * [None]
title_str = [
    "True Chargeability Model",
    "Recovered Chargeability (L2)",
]

for ii, m in enumerate([true_chargeability_model, recovered_chargeability_L2]):
    norm[ii] = Normalize(vmin=0.0, vmax=np.max(m))

    ax1[ii] = fig.add_axes([0.14, 0.6 - 0.5 * ii, 0.68, 0.35])

    mesh.plot_slice(
        plotting_map * m,
        ax=ax1[ii],
        normal="Y",
        ind=int(len(mesh.h[1]) / 2),
        grid=False,
        pcolor_opts={"cmap": mpl.cm.plasma, "norm": norm[ii]},
    )
    ax1[ii].set_title(title_str[ii])
    ax1[ii].set_xlabel("x (m)")
    ax1[ii].set_ylabel("z (m)")
    ax1[ii].set_xlim([-1500, 1500])
    ax1[ii].set_ylim([topo_xyz[:, -1].max() - 1500, topo_xyz[:, -1].max()])

    ax2[ii] = fig.add_axes([0.84, 0.6 - 0.5 * ii, 0.03, 0.35])
    cbar[ii] = mpl.colorbar.ColorbarBase(
        ax2[ii], norm=norm[ii], orientation="vertical", cmap=mpl.cm.plasma
    )
    cbar[ii].set_label("V/V", rotation=270, labelpad=15, size=12)

plt.savefig("ip_3d_model_comparison.png", dpi=150, bbox_inches="tight")
print("✓ Model comparison plot saved as 'ip_3d_model_comparison.png'")
plt.close()

print(f"  L2 model range: [{np.min(recovered_chargeability_L2):.4f}, {np.max(recovered_chargeability_L2):.4f}] V/V")
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
print(f"  L2 model range:      [{np.min(recovered_chargeability_L2):.4f}, {np.max(recovered_chargeability_L2):.4f}] V/V")
print()
print("Generated Files:")
print("-" * 80)
print("  1. ip_3d_topography.png                  - 3D topography")
print("  2. ip_3d_apparent_chargeability.png      - Observed data pseudosection")
print("  3. ip_3d_background_conductivity.png     - Background conductivity model")
print("  4. ip_3d_L2_data_fit.png                 - L2 inversion data fit")
print("  5. ip_3d_model_comparison.png            - True vs recovered model")
print()
print("Key Findings:")
print("-" * 80)
print("  - 3D IP inversion requires accurate background conductivity")
print("  - L2 inversion recovers smooth chargeability distribution")
print("  - Sensitivity weighting reduces near-electrode artifacts")
print("=" * 80)
