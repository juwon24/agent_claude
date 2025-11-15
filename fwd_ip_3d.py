"""
3D Forward Simulation of Induced Polarization Data
University of British Columbia

This tutorial demonstrates intermediate functionality within SimPEG and focuses on:
- Simulating 3D DC induced polarization (IP) data on a tree mesh
- Defining chargeability models and understanding units
- How to simulate IP data using the linearized formulation
- Understanding the relationship between background conductivity and IP response
- Visualizing 3D IP data in pseudosection format

Keywords: induced polarization, 3D forward simulation, apparent chargeability, tree mesh
"""

# ============================================================================
# Import Modules
# ============================================================================

# SimPEG functionality
from simpeg import maps, data
from simpeg.utils import model_builder
from simpeg.utils.io_utils.io_utils_electromagnetics import write_dcip_xyz
from simpeg.electromagnetics.static import induced_polarization as ip
from simpeg.electromagnetics.static.utils.static_utils import (
    generate_dcip_sources_line,
    pseudo_locations,
    plot_pseudosection,
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
from matplotlib.colors import LogNorm, Normalize

mpl.rcParams.update({"font.size": 14})

write_output = False  # Set to True to save outputs


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

# Plot the topography
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
plt.savefig("topography_ip_3d.png", dpi=150, bbox_inches="tight")
print("✓ Topography plot saved as 'topography_ip_3d.png'")
plt.close()

print(f"  Topography points: {len(x_topo)}")
print(f"  X-range: [{x_topo.min():.2f}, {x_topo.max():.2f}] m")
print(f"  Y-range: [{y_topo.min():.2f}, {y_topo.max():.2f}] m")
print(f"  Z-range: [{z_topo.min():.2f}, {z_topo.max():.2f}] m")
print()


# ============================================================================
# Define the Survey
# ============================================================================

print("=" * 80)
print("Step 2: Defining IP Survey")
print("=" * 80)

# Define the parameters for each survey line
survey_type = "dipole-dipole"
dimension_type = "3D"
data_type = "apparent_chargeability"
end_locations_list = [
    np.r_[-1000.0, 1000.0, 0.0, 0.0],
    np.r_[-600.0, -600.0, -1000.0, 1000.0],
    np.r_[-300.0, -300.0, -1000.0, 1000.0],
    np.r_[0.0, 0.0, -1000.0, 1000.0],
    np.r_[300.0, 300.0, -1000.0, 1000.0],
    np.r_[600.0, 600.0, -1000.0, 1000.0],
]
station_separation = 100.0
num_rx_per_src = 8

# Generate source list for all lines
ip_source_list = []
for ii in range(0, len(end_locations_list)):
    ip_source_list += generate_dcip_sources_line(
        survey_type,
        "apparent_chargeability",
        dimension_type,
        end_locations_list[ii],
        topo_xyz,
        num_rx_per_src,
        station_separation,
    )

# Define the survey
survey = ip.survey.Survey(ip_source_list)

print(f"  Survey type: {survey_type}")
print(f"  Number of lines: {len(end_locations_list)}")
print(f"  Total sources: {len(ip_source_list)}")
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
inds = (pseudo_locations_xyz[:, 1] == 0.0) & (np.abs(pseudo_locations_xyz[:, 0]) != 350)
ax2 = fig.add_axes([0.4, 0.1, 0.55, 0.8])
ax2.scatter(pseudo_locations_xyz[inds, 0], pseudo_locations_xyz[inds, -1], 8, "r")
ax2.set_xlabel("x (m)")
ax2.set_ylabel("z (m)")
ax2.set_title("Pseudo-locations (EW line)")
plt.savefig("survey_locations_ip_3d.png", dpi=150, bbox_inches="tight")
print("✓ Survey locations plot saved as 'survey_locations_ip_3d.png'")
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
nbcx = 2 ** int(np.round(np.log(dom_width_x / dh) / np.log(2.0)))
nbcy = 2 ** int(np.round(np.log(dom_width_y / dh) / np.log(2.0)))
nbcz = 2 ** int(np.round(np.log(dom_width_z / dh) / np.log(2.0)))

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

# Mesh refinement near electrodes
mesh.refine_points(unique_locations, padding_cells_by_level=[6, 6, 4], finalize=False)

# Finalize the mesh
mesh.finalize()

print(f"  Number of cells: {mesh.n_cells}")
print(f"  Number of unique electrode locations: {len(unique_locations)}")
print(f"  Origin: {mesh.origin}")
print(f"  Max cell volume: {mesh.cell_volumes.max():.2f} m³")
print()


# ============================================================================
# Define the Active Cells
# ============================================================================

print("=" * 80)
print("Step 4: Defining Active Cells")
print("=" * 80)

# Indices of the active mesh cells from topography
active_cells = active_from_xyz(mesh, topo_xyz)
n_active = np.sum(active_cells)

print(f"  Total mesh cells: {mesh.n_cells}")
print(f"  Active cells: {n_active}")
print(f"  Inactive cells: {(~active_cells).sum()}")
print()


# ============================================================================
# Define the Background Conductivity Model
# ============================================================================

print("=" * 80)
print("Step 5: Defining Background Conductivity Model")
print("=" * 80)

# Define electrical conductivities in S/m
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

print(f"  Air conductivity: {air_conductivity} S/m")
print(f"  Background conductivity: {background_conductivity} S/m")
print(f"  Conductor conductivity: {conductor_conductivity} S/m")
print(f"  Resistor conductivity: {resistor_conductivity} S/m")
print(f"  Conductor cells: {ind_conductor.sum()}")
print(f"  Resistor cells: {ind_resistor.sum()}")
print()

# Mapping from conductivity to all mesh cells
conductivity_map = maps.InjectActiveCells(mesh, active_cells, air_conductivity)

# Mapping to neglect air cells when plotting
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
ax1.set_title("Conductivity Model")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")
ax1.set_xlim([-1500, 1500])
ax1.set_ylim([z_topo.max() - 1500, z_topo.max()])

ax2 = fig.add_axes([0.84, 0.15, 0.03, 0.75])
cbar = mpl.colorbar.ColorbarBase(
    ax2, cmap=mpl.cm.RdYlBu_r, norm=norm, orientation="vertical"
)
cbar.set_label(r"$\sigma$ [S/m]", rotation=270, labelpad=15, size=16)

plt.savefig("conductivity_model_ip_3d.png", dpi=150, bbox_inches="tight")
print("✓ Conductivity model plot saved as 'conductivity_model_ip_3d.png'")
plt.close()


# ============================================================================
# Define the Chargeability Model
# ============================================================================

print("=" * 80)
print("Step 6: Defining Chargeability Model")
print("=" * 80)

# Define intrinsic chargeability model (V/V)
air_value = 0.0
background_value = 1e-6
chargeable_value = 0.1

# Define chargeability model
chargeability_model = background_value * np.ones(n_active)

ind_chargeable = model_builder.get_indices_sphere(
    np.r_[-350.0, 0.0, 100.0], 160.0, mesh.cell_centers[active_cells, :]
)

chargeability_model[ind_chargeable] = chargeable_value

print(f"  Background chargeability: {background_value} V/V")
print(f"  Chargeable sphere: {chargeable_value} V/V")
print(f"  Chargeable cells: {ind_chargeable.sum()}")
print()

# Define mapping from model to mesh cells
chargeability_map = maps.InjectActiveCells(mesh, active_cells, air_value)

# Plot Chargeability Model
fig = plt.figure(figsize=(10, 4))

norm = Normalize(vmin=0.0, vmax=0.1)

ax1 = fig.add_axes([0.15, 0.15, 0.67, 0.75])
mesh.plot_slice(
    plotting_map * chargeability_model,
    ax=ax1,
    normal="Y",
    ind=int(len(mesh.h[1]) / 2),
    grid=True,
    pcolor_opts={"cmap": mpl.cm.plasma, "norm": norm},
)
ax1.set_title("Chargeability Model")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")
ax1.set_xlim([-1500, 1500])
ax1.set_ylim([z_topo.max() - 1500, z_topo.max()])

ax2 = fig.add_axes([0.84, 0.15, 0.03, 0.75])
cbar = mpl.colorbar.ColorbarBase(
    ax2, cmap=mpl.cm.plasma, norm=norm, orientation="vertical", format="%.2f"
)
cbar.set_label("Intrinsic Chargeability [V/V]", rotation=270, labelpad=15, size=12)

plt.savefig("chargeability_model_ip_3d.png", dpi=150, bbox_inches="tight")
print("✓ Chargeability model plot saved as 'chargeability_model_ip_3d.png'")
plt.close()


# ============================================================================
# Project Electrodes to Discretized Topography
# ============================================================================

print("=" * 80)
print("Step 7: Projecting Electrodes to Topography")
print("=" * 80)

survey.drape_electrodes_on_topography(mesh, active_cells, option="top")

print("  ✓ Electrodes draped on discretized topography")
print()


# ============================================================================
# Define the IP Simulation
# ============================================================================

print("=" * 80)
print("Step 8: Defining IP Simulation")
print("=" * 80)

ip_simulation = ip.Simulation3DNodal(
    mesh,
    survey=survey,
    etaMap=chargeability_map,
    sigma=conductivity_map * conductivity_model,
)

print("  Simulation type: 3D Nodal Formulation")
print("  Chargeability mapping: InjectActiveCells")
print("  Background conductivity: Defined on mesh")
print()


# ============================================================================
# Simulate IP Data
# ============================================================================

print("=" * 80)
print("Step 9: Simulating IP Data")
print("=" * 80)

dpred_ip = ip_simulation.dpred(chargeability_model)

print(f"  Simulated data points: {len(dpred_ip)}")
print(f"  Data range: [{dpred_ip.min():.6f}, {dpred_ip.max():.6f}] V/V")
print(f"  Mean absolute value: {np.abs(dpred_ip).mean():.6f} V/V")
print()


# ============================================================================
# Plot IP Data in Pseudosection
# ============================================================================

print("=" * 80)
print("Step 10: Plotting IP Data")
print("=" * 80)

# Define the line IDs for all data
n_lines = len(end_locations_list)
n_data_per_line = int(survey.nD / n_lines)
lineID = np.hstack([(ii + 1) * np.ones(n_data_per_line) for ii in range(n_lines)])

# Convert 3D survey to 2D lines
survey_2d_list, index_list = convert_survey_3d_to_2d_lines(
    survey, lineID, data_type="apparent_chargeability", output_indexing=True
)

# Create list of 2D apparent chargeabilities
dobs_2d_list = []
apparent_chargeability_2d = []
for ind in index_list:
    dobs_2d_list.append(dpred_ip[ind])
    apparent_chargeability_2d.append(dpred_ip[ind])

print(f"  Converted survey to {len(survey_2d_list)} 2D lines")
print()

# Plot the first line (EW line)
line_index = 0

fig = plt.figure(figsize=(8, 2.75))

ax1 = fig.add_axes([0.1, 0.1, 0.7, 0.8])
cax1 = fig.add_axes([0.82, 0.1, 0.025, 0.8])
plot_pseudosection(
    survey_2d_list[line_index],
    apparent_chargeability_2d[line_index],
    "contourf",
    ax=ax1,
    cax=cax1,
    scale="linear",
    cbar_label="V/V",
    mask_topography=True,
    contourf_opts={"levels": 20, "cmap": mpl.cm.plasma},
)
ax1.set_title("Apparent Chargeability (V/V) - EW Line")

plt.savefig("ip_data_pseudosection_3d.png", dpi=150, bbox_inches="tight")
print("✓ IP pseudosection plot saved as 'ip_data_pseudosection_3d.png'")
plt.close()


# ============================================================================
# Optional: Export Data and Topography
# ============================================================================

if write_output:
    print("=" * 80)
    print("Step 11: Exporting Data and Topography")
    print("=" * 80)

    dir_path = os.path.sep.join([".", "fwd_ip_3d_outputs"]) + os.path.sep
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print(f"  Created directory: {dir_path}")

    # Add 10% Gaussian noise to each datum
    rng = np.random.default_rng(seed=433)
    std = 5e-3 * np.ones_like(dpred_ip)
    noise = rng.normal(scale=std, size=len(dpred_ip))
    dobs = dpred_ip + noise

    # Create dictionary that stores line IDs
    out_dict = {"LINEID": lineID}

    # Create a survey with the original electrode locations
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
    survey_original = ip.survey.Survey(source_list)

    # Write out data at their original electrode locations
    data_obj = data.Data(survey_original, dobs=dobs, standard_deviation=std)

    fname = dir_path + "ip_data.xyz"
    write_dcip_xyz(
        fname,
        data_obj,
        data_header="APP_CHG",
        uncertainties_header="UNCERT",
        out_dict=out_dict,
    )
    print(f"  ✓ IP data saved to: {fname}")

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
print(f"  Survey type:         {survey_type}")
print(f"  Number of lines:     {n_lines}")
print(f"  Data points:         {survey.nD}")
print(f"  Chargeability:       [{np.min(chargeability_model):.6f}, {np.max(chargeability_model):.6f}] V/V")
print(f"  Apparent charge.:    [{dpred_ip.min():.6f}, {dpred_ip.max():.6f}] V/V")
print()
print("Generated Files:")
print("-" * 80)
print("  1. topography_ip_3d.png            - 3D topography visualization")
print("  2. survey_locations_ip_3d.png      - Electrode and pseudo-locations")
print("  3. conductivity_model_ip_3d.png    - Background conductivity model")
print("  4. chargeability_model_ip_3d.png   - Intrinsic chargeability model")
print("  5. ip_data_pseudosection_3d.png    - Apparent chargeability pseudosection")
if write_output:
    print("  6. fwd_ip_3d_outputs/ip_data.xyz")
    print("  7. fwd_ip_3d_outputs/topo_xyz.txt")
print()
print("Note: SimPEG uses linearized formulation for IP simulation")
print("      Apparent chargeability data are in same units as model (V/V)")
print("      Conductive sphere is chargeable; resistive sphere is not")
print("      Survey consists of 6 dipole-dipole lines (1 EW, 5 NS)")
print("=" * 80)
