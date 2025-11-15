"""
3D Forward Simulation of DC Resistivity Data
University of British Columbia

This tutorial teaches intermediate level functionality within SimPEG and demonstrates:
- How to simulate 3D DC resistivity data on a tree mesh
- Specific aspects of designing 3D DC resistivity surveys
- Generating and plotting models on 3D meshes
- How to plot 3D data in pseudosection
- Computational resource considerations for 3D DC resistivity simulations

Keywords: DC Resistivity, forward simulation, 3D, apparent resistivity, tree mesh
"""

# ============================================================================
# Import Modules
# ============================================================================

# SimPEG functionality
from simpeg import maps, data
from simpeg.utils import model_builder
from simpeg.utils.io_utils.io_utils_electromagnetics import write_dcip_xyz
from simpeg.electromagnetics.static import resistivity as dc
from simpeg.electromagnetics.static.utils.static_utils import (
    generate_dcip_sources_line,
    pseudo_locations,
    plot_pseudosection,
    apparent_resistivity_from_voltage,
    convert_survey_3d_to_2d_lines,
)

# discretize functionality
from discretize import TreeMesh
from discretize.utils import mkvc, active_from_xyz

# Common Python functionality
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

mpl.rcParams.update({"font.size": 14})

save_output = False  # Set to True to save outputs


# ============================================================================
# Define the Topography
# ============================================================================

print("=" * 80)
print("Step 1: Defining Topography")
print("=" * 80)

# Generate some topography
x_topo, y_topo = np.meshgrid(
    np.linspace(-2100, 2100, 141), np.linspace(-2100, 2100, 141)
)
z_topo = 410.0 + 140.0 * (1 / np.pi) * (
    np.arctan((x_topo - 500 * np.sin(np.pi * y_topo / 2800) - 400.0) / 200.0)
    - np.arctan((x_topo - 500 * np.sin(np.pi * y_topo / 2800) + 400.0) / 200.0)
)

# Turn into a (N, 3) numpy.ndarray
x_topo, y_topo, z_topo = mkvc(x_topo), mkvc(y_topo), mkvc(z_topo)
topo_xyz = np.c_[mkvc(x_topo), mkvc(y_topo), mkvc(z_topo)]

print(f"  Topography points: {len(x_topo)}")
print(f"  X-range: [{x_topo.min():.1f}, {x_topo.max():.1f}] m")
print(f"  Y-range: [{y_topo.min():.1f}, {y_topo.max():.1f}] m")
print(f"  Z-range: [{z_topo.min():.1f}, {z_topo.max():.1f}] m")
print()

# Plot the topography
fig = plt.figure(figsize=(6, 6))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection="3d")
ax.set_zlim([z_topo.min() - 100, z_topo.max() + 100])
ax.scatter3D(topo_xyz[:, 0], topo_xyz[:, 1], topo_xyz[:, 2], s=0.25, c="b")
ax.set_box_aspect(aspect=None, zoom=0.85)
ax.set_xlabel("X (m)", labelpad=10)
ax.set_ylabel("Y (m)", labelpad=10)
ax.set_zlabel("Z (m)", labelpad=10)
ax.set_title("Topography (Exaggerated z-axis)", fontsize=16, pad=-20)
ax.view_init(elev=45.0, azim=-125)
plt.savefig("dcr_3d_topography.png", dpi=150, bbox_inches="tight")
print("✓ Topography plot saved as 'dcr_3d_topography.png'")
plt.close()


# ============================================================================
# Define the Survey
# ============================================================================

print("=" * 80)
print("Step 2: Defining Survey")
print("=" * 80)

# Define the parameters for each survey line
survey_type = "dipole-dipole"
data_type = "volt"
dimension_type = "3D"
end_locations_list = [
    np.r_[-1000.0, 1000.0, 0.0, 0.0],
    np.r_[-600.0, -600.0, -1000.0, 1000.0],
    np.r_[-300.0, -300.0, -1000.0, 1000.0],
    np.r_[0.0, 0.0, -1000.0, 1000.0],
    np.r_[300.0, 300.0, -1000.0, 1000.0],
    np.r_[600.0, 600.0, -1000.0, 1000.0],
]  # [x0, x1, y0, y1]
station_separation = 100.0
num_rx_per_src = 8

# The source lists for each line can be appended to create the source
# list for the whole survey.
source_list = []
for ii in range(0, len(end_locations_list)):
    source_list += generate_dcip_sources_line(
        survey_type,
        data_type,
        dimension_type,
        end_locations_list[ii],
        topo_xyz,
        num_rx_per_src,
        station_separation,
    )

# Define the survey
survey = dc.survey.Survey(source_list)

print(f"  Survey type: {survey_type}")
print(f"  Number of survey lines: {len(end_locations_list)}")
print(f"  Station separation: {station_separation} m")
print(f"  Max receivers per source: {num_rx_per_src}")
print(f"  Total sources: {survey.nSrc}")
print(f"  Total data points: {survey.nD}")
print()

# Plot electrode locations
unique_locations = survey.unique_electrode_locations
fig = plt.figure(figsize=(12, 2.75))
ax1 = fig.add_axes([0.1, 0.1, 0.2, 0.8])
ax1.scatter(unique_locations[:, 0], unique_locations[:, 1], 8, "r")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")
ax1.set_title("Horizontal locations")

pseudo_locations_xyz = pseudo_locations(survey)
inds = pseudo_locations_xyz[:, 1] == 0.0
ax2 = fig.add_axes([0.4, 0.1, 0.55, 0.8])
ax2.scatter(pseudo_locations_xyz[inds, 0], pseudo_locations_xyz[inds, -1], 8, "r")
ax2.set_xlabel("x (m)")
ax2.set_ylabel("z (m)")
ax2.set_title("Pseudo-locations (EW line)")
plt.savefig("dcr_3d_electrode_locations.png", dpi=150, bbox_inches="tight")
print("✓ Electrode locations plot saved as 'dcr_3d_electrode_locations.png'")
plt.close()


# ============================================================================
# Design a Tree Mesh
# ============================================================================

print("=" * 80)
print("Step 3: Designing Tree Mesh")
print("=" * 80)

# Defining domain size and minimum cell size
dh = 25.0  # base cell width
dom_width_x = 8000.0  # domain width x
dom_width_y = 8000.0  # domain width y
dom_width_z = 4000.0  # domain width z

# Number of base mesh cells in each direction. Must be a power of 2
nbcx = 2 ** int(np.round(np.log(dom_width_x / dh) / np.log(2.0)))  # num. base cells x
nbcy = 2 ** int(np.round(np.log(dom_width_y / dh) / np.log(2.0)))  # num. base cells y
nbcz = 2 ** int(np.round(np.log(dom_width_z / dh) / np.log(2.0)))  # num. base cells z

# Define the base mesh
hx = [(dh, nbcx)]
hy = [(dh, nbcy)]
hz = [(dh, nbcz)]
mesh = TreeMesh([hx, hy, hz], x0="CCN", diagonal_balance=True)

# Shift top to maximum topography
mesh.origin = mesh.origin + np.r_[0.0, 0.0, z_topo.max()]

# Mesh refinement based on surface topography
k = np.sqrt(np.sum(topo_xyz[:, 0:2] ** 2, axis=1)) < 1200
mesh.refine_surface(topo_xyz[k, :], padding_cells_by_level=[0, 4, 4], finalize=False)

# Mesh refinement near electrodes.
mesh.refine_points(unique_locations, padding_cells_by_level=[6, 6, 4], finalize=False)

# Finalize the mesh
mesh.finalize()

print(f"  Base cell width: {dh} m")
print(f"  Domain width: {dom_width_x} x {dom_width_y} x {dom_width_z} m³")
print(f"  Number of cells: {mesh.n_cells}")
print(f"  Origin: {mesh.origin}")
print(f"  Max cell volume: {mesh.cell_volumes.max():.2f} m³")
print()


# ============================================================================
# Define the Active Cells
# ============================================================================

print("=" * 80)
print("Step 4: Defining Active Cells")
print("=" * 80)

# Indices of the active mesh cells from topography (e.g. cells below surface)
active_cells = active_from_xyz(mesh, topo_xyz)

# number of active cells
n_active = np.sum(active_cells)

print(f"  Total mesh cells: {mesh.n_cells}")
print(f"  Active cells: {n_active}")
print(f"  Inactive cells: {mesh.n_cells - n_active}")
print()


# ============================================================================
# Define the Model
# ============================================================================

print("=" * 80)
print("Step 5: Defining Conductivity Model")
print("=" * 80)

# Define conductivity values in S/m (take reciprocal for resistivities in Ohm m)
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

# Define log-resistivity model
log_resistivity_model = np.log(1 / conductivity_model)

print(f"  Background conductivity: {background_conductivity} S/m")
print(f"  Conductor: {conductor_conductivity} S/m")
print(f"  Resistor: {resistor_conductivity} S/m")
print(f"  Conductor cells: {ind_conductor.sum()}")
print(f"  Resistor cells: {ind_resistor.sum()}")
print()

# Define mappings
conductivity_map = maps.InjectActiveCells(mesh, active_cells, air_conductivity)
log_resistivity_map = maps.InjectActiveCells(
    mesh, active_cells, 1 / air_conductivity
) * maps.ExpMap(nP=n_active)

# Generate a mapping to ignore inactive cells in plot
plotting_map = maps.InjectActiveCells(mesh, active_cells, np.nan)

# Plot conductivity model
fig = plt.figure(figsize=(10, 4.5))

norm = LogNorm(vmin=1e-3, vmax=1e-1)

ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
mesh.plot_slice(
    plotting_map * conductivity_model,
    ax=ax1,
    normal="Y",
    ind=int(len(mesh.h[1]) / 2),
    grid=True,
    pcolor_opts={"cmap": mpl.cm.RdYlBu_r, "norm": norm},
)
ax1.set_title("Conductivity Model (Y=0 slice)")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")
ax1.set_xlim([-1500, 1500])
ax1.set_ylim([z_topo.max() - 1500, z_topo.max()])

ax2 = fig.add_axes([0.84, 0.15, 0.03, 0.75])
cbar = mpl.colorbar.ColorbarBase(
    ax2, cmap=mpl.cm.RdYlBu_r, norm=norm, orientation="vertical"
)
cbar.set_label(r"$\sigma$ [S/m]", rotation=270, labelpad=15, size=16)

plt.savefig("dcr_3d_conductivity_model.png", dpi=150, bbox_inches="tight")
print("✓ Conductivity model plot saved as 'dcr_3d_conductivity_model.png'")
plt.close()


# ============================================================================
# Project Electrodes to Discretized Topography
# ============================================================================

print("=" * 80)
print("Step 6: Projecting Electrodes to Discretized Topography")
print("=" * 80)

survey.drape_electrodes_on_topography(mesh, active_cells, option="top")

print("  Electrodes successfully draped on discretized topography")
print()


# ============================================================================
# Define the Forward Simulation
# ============================================================================

print("=" * 80)
print("Step 7: Defining Forward Simulation")
print("=" * 80)

# DC simulation for a conductivity model
simulation_con = dc.simulation.Simulation3DNodal(
    mesh, survey=survey, sigmaMap=conductivity_map
)

# DC simulation for a log-resistivity model
simulation_res = dc.simulation.Simulation3DNodal(
    mesh, survey=survey, rhoMap=log_resistivity_map
)

print("  Simulation type: 3D Nodal")
print("  Two simulations defined:")
print("    1. Conductivity model")
print("    2. Log-resistivity model")
print()


# ============================================================================
# Predict DC Resistivity Data
# ============================================================================

print("=" * 80)
print("Step 8: Predicting DC Resistivity Data")
print("=" * 80)

dpred_con = simulation_con.dpred(conductivity_model)
dpred_res = simulation_res.dpred(log_resistivity_model)

print(f"  Conductivity model predictions:")
print(f"    Data range: [{dpred_con.min():.4e}, {dpred_con.max():.4e}] V/A")

print(f"  Log-resistivity model predictions:")
print(f"    Data range: [{dpred_res.min():.4e}, {dpred_res.max():.4e}] V/A")

print(f"  Maximum absolute error: {np.max(np.abs(dpred_con - dpred_res)):.2e} V/A")
print()


# ============================================================================
# Convert to Apparent Conductivities
# ============================================================================

print("=" * 80)
print("Step 9: Converting to Apparent Conductivities")
print("=" * 80)

apparent_conductivity = apparent_resistivity_from_voltage(survey, dpred_con) ** -1

print(f"  Apparent conductivity range: [{apparent_conductivity.min():.4e}, {apparent_conductivity.max():.4e}] S/m")
print()


# ============================================================================
# Plot Individual Lines in 2D Pseudosection
# ============================================================================

print("=" * 80)
print("Step 10: Plotting 2D Pseudosections for Individual Lines")
print("=" * 80)

# line IDs
n_lines = len(end_locations_list)
n_data_per_line = int(survey.nD / n_lines)
lineID = np.hstack([(ii + 1) * np.ones(n_data_per_line) for ii in range(n_lines)])

# Convert 3D survey to 2D lines
survey_2d_list, index_list = convert_survey_3d_to_2d_lines(
    survey, lineID, data_type="volt", output_indexing=True
)

# Create list of 2D apparent conductivities
dobs_2d_list = []
apparent_conductivities_2d = []
for ind in index_list:
    dobs_2d_list.append(dpred_con[ind])
    apparent_conductivities_2d.append(apparent_conductivity[ind])

print(f"  Number of survey lines: {len(survey_2d_list)}")
print(f"  Data per line: {n_data_per_line}")
print()

# Plot apparent conductivity pseudo-section for EW line (line 0)
line_ind = 0

fig = plt.figure(figsize=(8, 2.75))
ax1 = fig.add_axes([0.1, 0.15, 0.75, 0.78])
plot_pseudosection(
    survey_2d_list[line_ind],
    dobs=apparent_conductivities_2d[line_ind],
    plot_type="contourf",
    ax=ax1,
    scale="log",
    cbar_label="S/m",
    mask_topography=True,
    contourf_opts={"levels": 20, "cmap": mpl.cm.RdYlBu_r},
)
ax1.set_title("Apparent Conductivity (EW Line)")
plt.savefig("dcr_3d_pseudosection_line0.png", dpi=150, bbox_inches="tight")
print("✓ Pseudosection (EW line) saved as 'dcr_3d_pseudosection_line0.png'")
plt.close()

# Plot apparent conductivity pseudo-section for NS line (line 3)
line_ind = 3

fig = plt.figure(figsize=(8, 2.75))
ax1 = fig.add_axes([0.1, 0.15, 0.75, 0.78])
plot_pseudosection(
    survey_2d_list[line_ind],
    dobs=apparent_conductivities_2d[line_ind],
    plot_type="contourf",
    ax=ax1,
    scale="log",
    cbar_label="S/m",
    mask_topography=True,
    contourf_opts={"levels": 20, "cmap": mpl.cm.RdYlBu_r},
)
ax1.set_title("Apparent Conductivity (NS Line at x=0)")
plt.savefig("dcr_3d_pseudosection_line3.png", dpi=150, bbox_inches="tight")
print("✓ Pseudosection (NS line) saved as 'dcr_3d_pseudosection_line3.png'")
plt.close()


# ============================================================================
# Optional: Export Data
# ============================================================================

if save_output:
    print("=" * 80)
    print("Step 11: Exporting Data")
    print("=" * 80)

    dir_path = os.path.sep.join([".", "fwd_dcr_3d_outputs"]) + os.path.sep
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print(f"  Created directory: {dir_path}")

    # Add 10% Gaussian noise to each datum
    rng = np.random.default_rng(seed=433)
    std = 0.1 * np.abs(dpred_con)
    noise = rng.normal(scale=std, size=len(dpred_con))
    dobs = dpred_con + noise

    # Create dictionary that stores line IDs
    out_dict = {"LINEID": lineID}

    # Create a survey with the original electrode locations
    # and not the shifted ones
    source_list = []
    for ii in range(n_lines):
        source_list += generate_dcip_sources_line(
            survey_type,
            data_type,
            dimension_type,
            end_locations_list[ii],
            topo_xyz,
            num_rx_per_src,
            station_separation,
        )
    survey_original = dc.survey.Survey(source_list)

    # Write out data at their original electrode locations (not shifted)
    data_obj = data.Data(survey_original, dobs=dobs, standard_deviation=std)

    fname = dir_path + "dc_data.xyz"
    write_dcip_xyz(
        fname,
        data_obj,
        data_header="V/A",
        uncertainties_header="UNCERT",
        out_dict=out_dict,
    )
    print(f"  ✓ Data saved to: {fname}")

    fname = dir_path + "topo_xyz.txt"
    np.savetxt(fname, topo_xyz, fmt="%.4e")
    print(f"  ✓ Topography saved to: {fname}")
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
print(f"  Survey lines:        {n_lines}")
print(f"  Total data points:   {survey.nD}")
print(f"  Base cell size:      {dh} m")
print(f"  Voltage range:       [{dpred_con.min():.4e}, {dpred_con.max():.4e}] V/A")
print(f"  App. cond. range:    [{apparent_conductivity.min():.4e}, {apparent_conductivity.max():.4e}] S/m")
print()
print("Generated Files:")
print("-" * 80)
print("  1. dcr_3d_topography.png              - 3D topography visualization")
print("  2. dcr_3d_electrode_locations.png     - Electrode locations (plan + pseudo)")
print("  3. dcr_3d_conductivity_model.png      - Conductivity model (Y=0 slice)")
print("  4. dcr_3d_pseudosection_line0.png     - Pseudosection for EW line")
print("  5. dcr_3d_pseudosection_line3.png     - Pseudosection for NS line at x=0")
if save_output:
    print("  6. fwd_dcr_3d_outputs/dc_data.xyz")
    print("  7. fwd_dcr_3d_outputs/topo_xyz.txt")
print()
print("Key Learning Points:")
print("-" * 80)
print("  - 3D simulations require coarser cells than 2.5D due to computational limits")
print("  - Tree mesh adaptively refines near electrodes and topography")
print("  - Multiple survey lines can be combined into a single 3D survey")
print("  - 3D surveys can be parsed into individual 2D lines for visualization")
print("  - Computational cost scales as h^-3 where h is minimum cell size")
print()
print("Note: For 3D pseudosection visualization, install plotly:")
print("      pip install plotly")
print("=" * 80)
