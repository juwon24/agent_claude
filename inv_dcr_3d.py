"""
3D DC Resistivity Inversion
University of British Columbia

This tutorial demonstrates weighted least-squares inversion of 3D DC resistivity data
to recover a log-conductivity model on a tree mesh.

The tutorial teaches:
- How to design suitable tree meshes for 3D DC resistivity inversion
- Practical aspects of 3D DC resistivity inversion with SimPEG
- Applying sensitivity weighting for 3D DC resistivity
- Analyzing and visualizing 3D inversion outputs

Keywords: DC resistivity, 3D inversion, weighted least-squares, tree mesh,
          sensitivity weighting
"""

# ============================================================================
# Import Modules
# ============================================================================

# SimPEG functionality
from simpeg.electromagnetics.static import resistivity as dc
from simpeg.electromagnetics.static.utils.static_utils import (
    plot_pseudosection,
    apparent_resistivity_from_voltage,
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
from matplotlib.colors import LogNorm

mpl.rcParams.update({"font.size": 16})


# ============================================================================
# Download and Extract Tutorial Data
# ============================================================================

print("=" * 80)
print("Step 1: Downloading and Extracting Tutorial Data")
print("=" * 80)

# URL to download from repository assets
data_source = "https://github.com/simpeg/user-tutorials/raw/main/assets/05-dcr/inv_dcr_3d_files.tar.gz"

# Download the data
downloaded_data = download(data_source, overwrite=True)

# Unzip the tarfile
tar = tarfile.open(downloaded_data, "r")
tar.extractall()
tar.close()

# Path to the directory containing our data
dir_path = downloaded_data.split(".")[0] + os.path.sep
topo_filename = dir_path + "topo_xyz.txt"
dc_data_filename = dir_path + "dc_data.xyz"

print(f"  Downloaded and extracted data to: {dir_path}")
print()


# ============================================================================
# Load and Plot Topography
# ============================================================================

print("=" * 80)
print("Step 2: Loading and Plotting Topography")
print("=" * 80)

topo_xyz = np.loadtxt(str(topo_filename))

print(f"  Topography points: {len(topo_xyz)}")
print(f"  X-range: [{topo_xyz[:, 0].min():.1f}, {topo_xyz[:, 0].max():.1f}] m")
print(f"  Y-range: [{topo_xyz[:, 1].min():.1f}, {topo_xyz[:, 1].max():.1f}] m")
print(f"  Z-range: [{topo_xyz[:, 2].min():.1f}, {topo_xyz[:, 2].max():.1f}] m")

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
plt.savefig("dcr_3d_topography.png", dpi=150, bbox_inches="tight")
print("  Topography plot saved as 'dcr_3d_topography.png'")
plt.close()
print()


# ============================================================================
# Load DC Resistivity Data
# ============================================================================

print("=" * 80)
print("Step 3: Loading DC Resistivity Data")
print("=" * 80)

dc_data, out_dict = read_dcip_xyz(
    dc_data_filename,
    "volt",
    data_header="V/A",
    uncertainties_header="UNCERT",
    is_surface_data=False,
    dict_headers=["LINEID"],
)

print(f"  Number of data: {dc_data.nD}")
print(f"  Data type: Normalized voltages (V/A)")
print(f"  Data range: [{dc_data.dobs.min():.2e}, {dc_data.dobs.max():.2e}] V/A")
print()


# ============================================================================
# Compute and Plot Apparent Conductivities
# ============================================================================

print("=" * 80)
print("Step 4: Computing Apparent Conductivities")
print("=" * 80)

apparent_conductivities = 1 / apparent_resistivity_from_voltage(
    dc_data.survey,
    dc_data.dobs,
)

print(f"  Apparent conductivity range: [{apparent_conductivities.min():.4f}, {apparent_conductivities.max():.4f}] S/m")
print()


# ============================================================================
# Plot 2D Pseudosection for a Survey Line
# ============================================================================

print("=" * 80)
print("Step 5: Plotting 2D Pseudosection for Survey Line")
print("=" * 80)

# Extract line IDs
lineID = np.array(out_dict["LINEID"], dtype=int)

# Convert to 2D survey lines
survey_2d_list, index_list = convert_survey_3d_to_2d_lines(
    dc_data.survey, lineID, data_type="volt", output_indexing=True
)

dobs_2d_list = []
apparent_conductivities_2d = []
for ind in index_list:
    dobs_2d_list.append(dc_data.dobs[ind])
    apparent_conductivities_2d.append(apparent_conductivities[ind])

# Plot first line
line_index = 0
fig = plt.figure(figsize=(8, 2.75))
ax1 = fig.add_axes([0.1, 0.15, 0.75, 0.78])
plot_pseudosection(
    survey_2d_list[line_index],
    dobs=apparent_conductivities_2d[line_index],
    plot_type="contourf",
    ax=ax1,
    scale="log",
    cbar_label="S/m",
    mask_topography=True,
    contourf_opts={"levels": 20, "cmap": mpl.cm.RdYlBu_r},
)
ax1.set_title("Apparent Conductivity (Line 1)")
plt.savefig("dcr_3d_apparent_conductivity.png", dpi=150, bbox_inches="tight")
print(f"  Apparent conductivity plot saved as 'dcr_3d_apparent_conductivity.png'")
plt.close()
print()


# ============================================================================
# Assign Uncertainties
# ============================================================================

print("=" * 80)
print("Step 6: Assigning Uncertainties to Data")
print("=" * 80)

dc_data.standard_deviation = 1e-7 + 0.1 * np.abs(dc_data.dobs)

print(f"  Applied floor: 1e-7 V/A")
print(f"  Applied percent: 10%")
print()


# ============================================================================
# Design a Tree Mesh
# ============================================================================

print("=" * 80)
print("Step 7: Designing 3D Tree Mesh")
print("=" * 80)

# Defining domain size and minimum cell size
dh = 25.0  # base cell width
dom_width_x = 8000.0  # domain width x
dom_width_y = 8000.0  # domain width y
dom_width_z = 4000.0  # domain width z

# Number of base mesh cells
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
unique_locations = dc_data.survey.unique_electrode_locations

# Mesh refinement near electrodes
mesh.refine_points(unique_locations, padding_cells_by_level=[6, 6, 4], finalize=False)

# Finalize the mesh
mesh.finalize()

print(f"  Base cell width: {dh} m")
print(f"  Number of cells: {mesh.n_cells}")
print(f"  Mesh extent (x): [{mesh.nodes_x.min():.1f}, {mesh.nodes_x.max():.1f}] m")
print(f"  Mesh extent (y): [{mesh.nodes_y.min():.1f}, {mesh.nodes_y.max():.1f}] m")
print(f"  Mesh extent (z): [{mesh.nodes_z.min():.1f}, {mesh.nodes_z.max():.1f}] m")
print()


# ============================================================================
# Define Active Cells
# ============================================================================

print("=" * 80)
print("Step 8: Defining Active Cells")
print("=" * 80)

# Indices of the active mesh cells from topography
active_cells = active_from_xyz(mesh, topo_xyz)
n_active = np.sum(active_cells)

print(f"  Total cells: {mesh.n_cells}")
print(f"  Active cells: {n_active}")
print(f"  Inactive cells: {(~active_cells).sum()}")
print()


# ============================================================================
# Project Electrodes to Discretized Topography
# ============================================================================

print("=" * 80)
print("Step 9: Projecting Electrodes to Discretized Topography")
print("=" * 80)

dc_data.survey.drape_electrodes_on_topography(mesh, active_cells, option="top")

print(f"  Electrodes draped onto mesh surface")
print()


# ============================================================================
# Define Model Mapping and Starting Model
# ============================================================================

print("=" * 80)
print("Step 10: Defining Model Mapping and Starting Model")
print("=" * 80)

# Model parameters to all cells
log_conductivity_map = maps.InjectActiveCells(mesh, active_cells, 1e-8) * maps.ExpMap(
    nP=n_active
)

# Median apparent conductivity
median_conductivity = np.median(apparent_conductivities)

# Create starting model from log-conductivity
starting_conductivity_model = np.log(median_conductivity) * np.ones(n_active)

# Reference conductivity model
reference_conductivity_model = starting_conductivity_model.copy()

print(f"  Model parameters: {n_active}")
print(f"  Starting conductivity: {median_conductivity:.4f} S/m")
print(f"  Starting resistivity: {1/median_conductivity:.1f} Ohm-m")
print()


# ============================================================================
# Define Forward Simulation
# ============================================================================

print("=" * 80)
print("Step 11: Defining Forward Simulation")
print("=" * 80)

dc_simulation = dc.simulation.Simulation3DNodal(
    mesh, survey=dc_data.survey, sigmaMap=log_conductivity_map, storeJ=True
)

print(f"  Simulation type: 3D Nodal")
print(f"  Store Jacobian: True")
print()


# ============================================================================
# Define Data Misfit and Regularization
# ============================================================================

print("=" * 80)
print("Step 12: Defining Data Misfit and Regularization")
print("=" * 80)

dmis_L2 = data_misfit.L2DataMisfit(simulation=dc_simulation, data=dc_data)

reg_L2 = regularization.WeightedLeastSquares(
    mesh,
    active_cells=active_cells,
    length_scale_x=100.0,
    length_scale_y=100.0,
    length_scale_z=100.0,
    reference_model=reference_conductivity_model,
)

print(f"  Data misfit: L2 norm")
print(f"  Regularization: Weighted Least Squares")
print(f"  Length scales: [100, 100, 100] m")
print()


# ============================================================================
# Define Optimization and Inverse Problem
# ============================================================================

print("=" * 80)
print("Step 13: Defining Optimization and Inverse Problem")
print("=" * 80)

opt_L2 = optimization.InexactGaussNewton(
    maxIter=40, maxIterLS=20, maxIterCG=30, tolCG=1e-3
)

inv_prob_L2 = inverse_problem.BaseInvProblem(dmis_L2, reg_L2, opt_L2)

print(f"  Optimization: Inexact Gauss Newton")
print(f"  Maximum iterations: 40")
print()


# ============================================================================
# Define Inversion Directives
# ============================================================================

print("=" * 80)
print("Step 14: Defining Inversion Directives")
print("=" * 80)

if dc_simulation.storeJ:
    sensitivity_weights = directives.UpdateSensitivityWeights(
        every_iteration=True, threshold_value=1e-2
    )
    update_jacobi = directives.UpdatePreconditioner(update_every_iteration=True)
    directives_list_L2 = [
        sensitivity_weights,
        update_jacobi,
    ]
else:
    directives_list_L2 = []

starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=100)
beta_schedule = directives.BetaSchedule(coolingFactor=2.0, coolingRate=2)
target_misfit = directives.TargetMisfit(chifact=1.0)

directives_list_L2 += [starting_beta, beta_schedule, target_misfit]

print(f"  Number of directives: {len(directives_list_L2)}")
print(f"  Sensitivity weighting: Enabled")
print()


# ============================================================================
# Run Inversion
# ============================================================================

print("=" * 80)
print("Step 15: Running 3D DC Resistivity Inversion")
print("=" * 80)

inv_L2 = inversion.BaseInversion(inv_prob_L2, directives_list_L2)
recovered_log_conductivity_model = inv_L2.run(starting_conductivity_model)

print()
print(f"  Inversion completed successfully")
print(f"  Final data misfit: {inv_prob_L2.dmisfit.phi:.2f}")
print()


# ============================================================================
# Plot Normalized Data Misfit in 2D Pseudosection
# ============================================================================

print("=" * 80)
print("Step 16: Plotting Normalized Data Misfit")
print("=" * 80)

# Predicted data from recovered model
dpred_dc = inv_prob_L2.dpred

# Compute normalized misfit
dc_normalized_misfit = (dc_data.dobs - dpred_dc) / dc_data.standard_deviation

# Plot for line 0
line_index = 0
k = lineID == line_index + 1
data_array = [
    np.abs(dc_data.dobs[k]),
    np.abs(dpred_dc[k]),
    (dc_data.dobs[k] - dpred_dc[k]) / dc_data.standard_deviation[k],
]

# Plot 2D pseudosections
fig = plt.figure(figsize=(9, 11))
plot_title = ["Observed Voltage", "Predicted Voltage", "Normalized Misfit"]
plot_units = ["V/A", "V/A", ""]
scale = ["log", "log", "linear"]
cmap_list = [mpl.cm.viridis, mpl.cm.viridis, mpl.cm.RdYlBu]

for ii in range(0, 3):
    ax1 = fig.add_axes([0.15, 0.72 - 0.33 * ii, 0.65, 0.21])
    cax1 = fig.add_axes([0.81, 0.72 - 0.33 * ii, 0.03, 0.21])
    plot_pseudosection(
        survey_2d_list[line_index],
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

plt.savefig("dcr_3d_data_misfit.png", dpi=150, bbox_inches="tight")
print("  Data misfit plot saved as 'dcr_3d_data_misfit.png'")
plt.close()
print()


# ============================================================================
# Plot True and Recovered Models
# ============================================================================

print("=" * 80)
print("Step 17: Plotting True and Recovered Models")
print("=" * 80)

# Define true model
background_value = 1e-2
conductor_value = 1e-1
resistor_value = 1e-3

true_conductivity_model = background_value * np.ones(n_active)

ind_conductor = model_builder.get_indices_sphere(
    np.r_[-300.0, 0.0, 100.0], 165.0, mesh.cell_centers[active_cells, :]
)
true_conductivity_model[ind_conductor] = conductor_value

ind_resistor = model_builder.get_indices_sphere(
    np.r_[300.0, 0.0, 100.0], 165.0, mesh.cell_centers[active_cells, :]
)
true_conductivity_model[ind_resistor] = resistor_value

# Convert log-conductivity to conductivity
recovered_conductivity_L2 = np.exp(recovered_log_conductivity_model)

# Define plotting map
plotting_map = maps.InjectActiveCells(mesh, active_cells, np.nan)

norm = LogNorm(vmin=1e-3, vmax=1e-1)

fig = plt.figure(figsize=(10, 9))
title_str = [
    "True Conductivity Model",
    "Recovered Model (L2)",
]
plotting_model = [
    true_conductivity_model,
    recovered_conductivity_L2,
]

for ii in range(0, 2):
    ax1 = fig.add_axes([0.14, 0.6 - 0.5 * ii, 0.68, 0.35])

    temp = plotting_map * plotting_model[ii]

    mesh.plot_slice(
        temp,
        ax=ax1,
        normal="Y",
        ind=int(len(mesh.h[1]) / 2),
        grid=False,
        pcolor_opts={"cmap": mpl.cm.RdYlBu_r, "norm": norm},
    )
    ax1.set_title(title_str[ii])
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("z (m)")
    ax1.set_xlim([-1200, 1200])
    ax1.set_ylim([topo_xyz[:, -1].max() - 1200, topo_xyz[:, -1].max()])

    ax2 = fig.add_axes([0.84, 0.6 - 0.5 * ii, 0.03, 0.35])
    cbar = mpl.colorbar.ColorbarBase(
        ax2, norm=norm, orientation="vertical", cmap=mpl.cm.RdYlBu_r
    )
    cbar.set_label(r"$\sigma$ (S/m)", rotation=270, labelpad=15, size=16)

plt.savefig("dcr_3d_models_comparison.png", dpi=150, bbox_inches="tight")
print("  Model comparison saved as 'dcr_3d_models_comparison.png'")
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
print(f"  Number of data:           {dc_data.nD}")
print(f"  Mesh cells:               {mesh.n_cells}")
print(f"  Active cells:             {n_active}")
print(f"  Survey lines:             {len(survey_2d_list)}")
print()
print("Inversion Results:")
print("-" * 80)
print(f"  Final data misfit:        {inv_prob_L2.dmisfit.phi:.2f}")
print(f"  Target misfit:            {dc_data.nD}")
print()
print("Generated Files:")
print("-" * 80)
print("  1. dcr_3d_topography.png             - 3D topography visualization")
print("  2. dcr_3d_apparent_conductivity.png  - Apparent conductivity pseudo-section")
print("  3. dcr_3d_data_misfit.png            - Observed, predicted, and misfit")
print("  4. dcr_3d_models_comparison.png      - True and recovered models")
print()
print("Key Findings:")
print("-" * 80)
print("  - 3D inversion successfully recovered subsurface structures")
print("  - Sensitivity weighting is critical for 3D DC resistivity")
print("  - Tree mesh allows efficient discretization of large domains")
print("  - Model fits the data well with normalized misfit near 1")
print("=" * 80)
