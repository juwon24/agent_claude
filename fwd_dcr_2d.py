"""
2.5D Forward Simulation of DC Resistivity Data
University of British Columbia

This tutorial teaches basic functionality within SimPEG and demonstrates:
- How to define DC resistivity lines manually or by using utility functions
- How to design a 2D tree mesh for accurately simulating DC resistivity data
- How to define the Earth's electrical properties according to conductivity OR resistivity
- How to include surface topography in the forward simulation
- How to plot simulated data in pseudosection

Keywords: DC resistivity, forward simulation, 2.5D, apparent resistivity, tree mesh
"""

# ============================================================================
# Import Modules
# ============================================================================

# SimPEG functionality
from simpeg.electromagnetics.static import resistivity as dc
from simpeg.utils import model_builder
from simpeg.utils.io_utils.io_utils_electromagnetics import write_dcip2d_ubc
from simpeg import maps, data
from simpeg.electromagnetics.static.utils.static_utils import (
    generate_dcip_sources_line,
    pseudo_locations,
    plot_pseudosection,
    apparent_resistivity_from_voltage,
)

# discretize functionality
from discretize import TreeMesh
from discretize.utils import active_from_xyz

# Common Python functionality
import os
import numpy as np
from scipy.interpolate import interp1d
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

# Along-line locations
x_topo = np.linspace(-2000, 2000, 401)

# Elevation as a function of along-line location
T = 800.0
z_topo = 20.0 * np.sin(2 * np.pi * x_topo / T) + 140.0
z_topo[x_topo < -3 * T / 4] = 160.0
z_topo[x_topo > 3 * T / 4] = 120.0
z_topo += 50.0 * (1.0 + np.tanh(-3 * (x_topo + 1200.0) / T))
z_topo -= 50.0 * (1.0 + np.tanh(3 * (x_topo - 1200.0) / T))

# Define full 2D topography
topo_2d = np.c_[x_topo, z_topo]

print(f"  Topography points: {len(x_topo)}")
print(f"  X-range: [{x_topo.min():.1f}, {x_topo.max():.1f}] m")
print(f"  Z-range: [{z_topo.min():.1f}, {z_topo.max():.1f}] m")
print()

# Plot 2D topography
fig = plt.figure(figsize=(10, 2))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(x_topo, z_topo, color="b", linewidth=2)
ax.set_xlabel("x (m)", labelpad=5)
ax.set_ylabel("z (m)", labelpad=5)
ax.grid(True)
ax.set_title("Topography (Exaggerated z-axis)", fontsize=16, pad=10)
plt.savefig("dcr_2d_topography.png", dpi=150, bbox_inches="tight")
print("✓ Topography plot saved as 'dcr_2d_topography.png'")
plt.close()


# ============================================================================
# Define the Survey
# ============================================================================

print("=" * 80)
print("Step 2: Defining Survey")
print("=" * 80)

# Define survey line parameters
survey_type = "dipole-dipole"
dimension_type = "2D"
data_type = "volt"
end_locations = np.r_[-400.0, 400.0]
station_separation = 40.0
num_rx_per_src = 10

# Define linear interpolation function for elevation
interp_fun = interp1d(x_topo, z_topo)

# Define electrode locations
electrode_locations_x = np.arange(
    end_locations[0], end_locations[1] + station_separation, station_separation
)
electrode_locations_z = interp_fun(electrode_locations_x)
electrode_locations = np.c_[electrode_locations_x, electrode_locations_z]

# Number of electrode locations
n_electrodes = len(electrode_locations_x)

# Instantiate empty list for sources
source_list = []

ii = 0
while ii < n_electrodes - 3:
    # A and B electrode locations
    location_a = electrode_locations[ii, :]
    location_b = electrode_locations[ii + 1, :]

    # M and N electrode locations
    ii_max = np.min([ii + 3 + num_rx_per_src, n_electrodes])
    locations_m = electrode_locations[ii + 2 : ii_max - 1]
    locations_n = electrode_locations[ii + 3 : ii_max]

    # Define receivers for source ii
    receivers_list = [
        dc.receivers.Dipole(
            locations_m=locations_m, locations_n=locations_n, data_type=data_type
        )
    ]

    # Append source ii to list
    source_list.append(
        dc.sources.Dipole(receivers_list, location_a=location_a, location_b=location_b)
    )

    ii += 1

# Define survey
survey = dc.Survey(source_list)

print(f"  Survey type: {survey_type}")
print(f"  Data type: {data_type}")
print(f"  Number of electrodes: {n_electrodes}")
print(f"  Number of sources: {survey.nSrc}")
print(f"  Number of data points: {survey.nD}")
print(f"  Station separation: {station_separation} m")
print(f"  Max receivers per source: {num_rx_per_src}")
print()

# Plot electrode locations
fig = plt.figure(figsize=(10, 2))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(x_topo, z_topo, color="b", linewidth=1)
ax.scatter(electrode_locations_x, electrode_locations_z, 8, "r")
ax.set_xlim([x_topo.min(), x_topo.max()])
ax.set_xlabel("x (m)", labelpad=5)
ax.set_ylabel("z (m)", labelpad=5)
ax.grid(True)
ax.set_title("Topography and electrode locations", fontsize=16, pad=10)
plt.savefig("dcr_2d_electrodes.png", dpi=150, bbox_inches="tight")
print("✓ Electrode locations plot saved as 'dcr_2d_electrodes.png'")
plt.close()

# Plot pseudo-locations
pseudo_locations_xz = pseudo_locations(survey)
fig = plt.figure(figsize=(8, 2.75))
ax = fig.add_axes([0.1, 0.1, 0.85, 0.8])
ax.scatter(pseudo_locations_xz[:, 0], pseudo_locations_xz[:, -1], 8, "r")
ax.set_xlabel("x (m)")
ax.set_ylabel("z (m)")
ax.set_title("Pseudo-locations")
plt.savefig("dcr_2d_pseudo_locations.png", dpi=150, bbox_inches="tight")
print("✓ Pseudo-locations plot saved as 'dcr_2d_pseudo_locations.png'")
plt.close()


# ============================================================================
# Design a Tree Mesh
# ============================================================================

print("=" * 80)
print("Step 3: Designing Tree Mesh")
print("=" * 80)

dh = 4  # base cell width
dom_width_x = 3200.0  # domain width x
dom_width_z = 2400.0  # domain width z
nbcx = 2 ** int(np.round(np.log(dom_width_x / dh) / np.log(2.0)))  # num. base cells x
nbcz = 2 ** int(np.round(np.log(dom_width_z / dh) / np.log(2.0)))  # num. base cells z

# Define the base mesh with top at z = 0 m.
hx = [(dh, nbcx)]
hz = [(dh, nbcz)]
mesh = TreeMesh([hx, hz], x0="CN", diagonal_balance=True)

# Shift top to maximum topography
mesh.origin = mesh.origin + np.r_[0.0, z_topo.max()]

# Mesh refinement based on topography
mesh.refine_surface(
    topo_2d,
    padding_cells_by_level=[0, 0, 4, 4],
    finalize=False,
)

# Extract unique electrode locations.
unique_locations = survey.unique_electrode_locations

# Mesh refinement near electrodes.
mesh.refine_points(
    unique_locations, padding_cells_by_level=[8, 12, 6, 6], finalize=False
)

mesh.finalize()

print(f"  Base cell width: {dh} m")
print(f"  Number of cells: {mesh.n_cells}")
print(f"  Number of x-faces: {mesh.n_faces_x}")
print(f"  Origin: {mesh.origin}")
print(f"  Max cell volume: {mesh.cell_volumes.max():.2f} m²")
print()

# Plot mesh
fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_axes([0.14, 0.17, 0.8, 0.7])
mesh.plot_grid(ax=ax1, linewidth=1)
ax1.grid(False)
ax1.set_xlim(-1500, 1500)
ax1.set_ylim(np.max(z_topo) - 1000, np.max(z_topo))
ax1.set_title("Tree Mesh")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")
plt.savefig("dcr_2d_mesh.png", dpi=150, bbox_inches="tight")
print("✓ Mesh plot saved as 'dcr_2d_mesh.png'")
plt.close()


# ============================================================================
# Define the Active Cells
# ============================================================================

print("=" * 80)
print("Step 4: Defining Active Cells")
print("=" * 80)

# Indices of the active mesh cells from topography (e.g. cells below surface)
active_cells = active_from_xyz(mesh, topo_2d)

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

air_conductivity = 1e-8
background_conductivity = 1e-2
conductor_conductivity = 1e-1
resistor_conductivity = 1e-3

# Define conductivity model
conductivity_model = background_conductivity * np.ones(n_active)

ind_conductor = model_builder.get_indices_sphere(
    np.r_[-120.0, 40.0], 60.0, mesh.cell_centers[active_cells, :]
)
conductivity_model[ind_conductor] = conductor_conductivity

ind_resistor = model_builder.get_indices_sphere(
    np.r_[120.0, 72.0], 60.0, mesh.cell_centers[active_cells, :]
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
fig = plt.figure(figsize=(9, 4))

norm = LogNorm(vmin=1e-3, vmax=1e-1)

ax1 = fig.add_axes([0.14, 0.17, 0.68, 0.7])
mesh.plot_image(
    plotting_map * conductivity_model,
    ax=ax1,
    grid=False,
    pcolor_opts={"norm": norm, "cmap": mpl.cm.RdYlBu_r},
)
ax1.set_xlim(-500, 500)
ax1.set_ylim(-300, 200)
ax1.set_title("Conductivity Model")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")

ax2 = fig.add_axes([0.84, 0.17, 0.03, 0.7])
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.RdYlBu_r
)
cbar.set_label(r"$\sigma$ (S/m)", rotation=270, labelpad=15, size=12)

plt.savefig("dcr_2d_conductivity_model.png", dpi=150, bbox_inches="tight")
print("✓ Conductivity model plot saved as 'dcr_2d_conductivity_model.png'")
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
simulation_con = dc.simulation_2d.Simulation2DNodal(
    mesh, survey=survey, sigmaMap=conductivity_map
)

# DC simulation for a log-resistivity model
simulation_res = dc.simulation_2d.Simulation2DNodal(
    mesh, survey=survey, rhoMap=log_resistivity_map
)

print("  Simulation type: 2D Nodal (2.5D approach)")
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
# Plot Data in Pseudosection
# ============================================================================

print("=" * 80)
print("Step 9: Plotting Data in Pseudosection")
print("=" * 80)

# Plot voltages pseudo-section
fig = plt.figure(figsize=(8, 2.75))
ax1 = fig.add_axes([0.1, 0.15, 0.75, 0.78])
plot_pseudosection(
    survey,
    dobs=np.abs(dpred_con),
    plot_type="scatter",
    ax=ax1,
    scale="log",
    cbar_label="V/A",
    scatter_opts={"cmap": mpl.cm.viridis},
)
ax1.set_title("Normalized Voltages")
plt.savefig("dcr_2d_voltages.png", dpi=150, bbox_inches="tight")
print("✓ Voltage pseudosection saved as 'dcr_2d_voltages.png'")
plt.close()

# Get apparent conductivities from volts and survey geometry
apparent_conductivities = 1 / apparent_resistivity_from_voltage(survey, dpred_con)

# Plot apparent conductivity pseudo-section
fig = plt.figure(figsize=(8, 2.75))
ax1 = fig.add_axes([0.1, 0.15, 0.75, 0.78])
plot_pseudosection(
    survey,
    dobs=apparent_conductivities,
    plot_type="contourf",
    ax=ax1,
    scale="log",
    cbar_label="S/m",
    mask_topography=True,
    contourf_opts={"levels": 20, "cmap": mpl.cm.RdYlBu_r},
)
ax1.set_title("Apparent Conductivity")
plt.savefig("dcr_2d_apparent_conductivity.png", dpi=150, bbox_inches="tight")
print("✓ Apparent conductivity pseudosection saved as 'dcr_2d_apparent_conductivity.png'")
plt.close()


# ============================================================================
# Optional: Export Data
# ============================================================================

if save_output:
    print("=" * 80)
    print("Step 10: Exporting Data")
    print("=" * 80)

    dir_path = os.path.sep.join([".", "fwd_dcr_2d_outputs"]) + os.path.sep
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print(f"  Created directory: {dir_path}")

    # Add 5% Gaussian noise to each datum
    rng = np.random.default_rng(seed=225)
    std = 0.05 * np.abs(dpred_con)
    dc_noise = rng.normal(scale=std, size=len(dpred_con))
    dobs = dpred_con + dc_noise

    # Create a survey with the original electrode locations
    # and not the shifted ones
    # Generate source list for DC survey line
    source_list = generate_dcip_sources_line(
        survey_type,
        data_type,
        dimension_type,
        end_locations,
        topo_2d,
        num_rx_per_src,
        station_separation,
    )
    survey_original = dc.survey.Survey(source_list)

    # Write out data at their original electrode locations (not shifted)
    data_obj = data.Data(survey_original, dobs=dobs, standard_deviation=std)
    fname = dir_path + "dc_data.obs"
    write_dcip2d_ubc(fname, data_obj, "volt", "dobs")
    print(f"  ✓ Data saved to: {fname}")

    fname = dir_path + "topo_2d.txt"
    np.savetxt(fname, topo_2d, fmt="%.4e")
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
print(f"  Data points:         {survey.nD}")
print(f"  Survey line length:  {end_locations[1] - end_locations[0]:.0f} m")
print(f"  Voltage range:       [{dpred_con.min():.4e}, {dpred_con.max():.4e}] V/A")
print(f"  App. cond. range:    [{apparent_conductivities.min():.4e}, {apparent_conductivities.max():.4e}] S/m")
print()
print("Generated Files:")
print("-" * 80)
print("  1. dcr_2d_topography.png              - 2D topography profile")
print("  2. dcr_2d_electrodes.png              - Electrode locations on topography")
print("  3. dcr_2d_pseudo_locations.png        - Pseudo-location plot")
print("  4. dcr_2d_mesh.png                    - Tree mesh visualization")
print("  5. dcr_2d_conductivity_model.png      - Conductivity model cross-section")
print("  6. dcr_2d_voltages.png                - Voltage pseudosection")
print("  7. dcr_2d_apparent_conductivity.png   - Apparent conductivity pseudosection")
if save_output:
    print("  8. fwd_dcr_2d_outputs/dc_data.obs")
    print("  9. fwd_dcr_2d_outputs/topo_2d.txt")
print()
print("Key Learning Points:")
print("-" * 80)
print("  - 2.5D approach leverages symmetry to avoid 3D computational costs")
print("  - Tree mesh provides adaptive refinement near electrodes and topography")
print("  - Electrodes must be draped onto discretized topography surface")
print("  - Both conductivity and resistivity formulations yield identical results")
print("=" * 80)
