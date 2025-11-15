"""
2.5D Forward Simulation of Induced Polarization Data
University of British Columbia

This tutorial demonstrates intermediate functionality within SimPEG and focuses on:
- Simulating 2.5D DC induced polarization (IP) data on a tree mesh
- Defining chargeability models and understanding units
- How to simulate IP data using the linearized formulation
- Understanding the relationship between background conductivity and IP response
- Visualizing data in pseudosection format

Keywords: induced polarization, 2.5D forward simulation, apparent chargeability, tree mesh
"""

# ============================================================================
# Import Modules
# ============================================================================

# SimPEG functionality
from simpeg.electromagnetics.static import induced_polarization as ip
from simpeg.utils import model_builder
from simpeg.utils.io_utils.io_utils_electromagnetics import write_dcip2d_ubc
from simpeg import maps, data
from simpeg.electromagnetics.static.utils.static_utils import (
    generate_dcip_sources_line,
    pseudo_locations,
    plot_pseudosection,
)

# discretize functionality
from discretize import TreeMesh
from discretize.utils import active_from_xyz

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

# Plot 2D topography
fig = plt.figure(figsize=(10, 2))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(x_topo, z_topo, color="b", linewidth=2)
ax.set_xlabel("x (m)", labelpad=5)
ax.set_ylabel("z (m)", labelpad=5)
ax.grid(True)
ax.set_title("Topography (Exaggerated z-axis)", fontsize=16, pad=10)
plt.savefig("topography_ip_2d.png", dpi=150, bbox_inches="tight")
print("✓ Topography plot saved as 'topography_ip_2d.png'")
plt.close()

print(f"  Topography points: {len(x_topo)}")
print(f"  X-range: [{x_topo.min():.2f}, {x_topo.max():.2f}] m")
print(f"  Z-range: [{z_topo.min():.2f}, {z_topo.max():.2f}] m")
print()


# ============================================================================
# Define the Survey
# ============================================================================

print("=" * 80)
print("Step 2: Defining IP Survey")
print("=" * 80)

# Define survey line parameters
survey_type = "dipole-dipole"
dimension_type = "2D"
data_type = "apparent_chargeability"
end_locations = np.r_[-400.0, 400.0]
station_separation = 40.0
num_rx_per_src = 10

# Generate IP source list
ip_source_list = generate_dcip_sources_line(
    survey_type,
    data_type,
    dimension_type,
    end_locations,
    topo_2d,
    num_rx_per_src,
    station_separation,
)

# Create survey
ip_survey = ip.survey.Survey(ip_source_list)

print(f"  Survey type: {survey_type}")
print(f"  Number of sources: {len(ip_source_list)}")
print(f"  Total data points: {ip_survey.nD}")
print(f"  Data type: {data_type}")
print()

# Plot pseudo-locations
pseudo_locations_xz = pseudo_locations(ip_survey)
fig = plt.figure(figsize=(8, 2.75))
ax = fig.add_axes([0.1, 0.1, 0.85, 0.8])
ax.scatter(pseudo_locations_xz[:, 0], pseudo_locations_xz[:, -1], 8, "r")
ax.set_xlabel("x (m)")
ax.set_ylabel("z (m)")
ax.set_title("Pseudo-locations")
plt.savefig("pseudo_locations_ip_2d.png", dpi=150, bbox_inches="tight")
print("✓ Pseudo-locations plot saved as 'pseudo_locations_ip_2d.png'")
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

# Define the base mesh with top at z = 0 m
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

# Extract unique electrode locations
unique_locations = ip_survey.unique_electrode_locations

# Mesh refinement near electrodes
mesh.refine_points(
    unique_locations, padding_cells_by_level=[8, 12, 6, 6], finalize=False
)

mesh.finalize()

print(f"  Number of cells: {mesh.n_cells}")
print(f"  Number of unique electrode locations: {len(unique_locations)}")
print(f"  Origin: {mesh.origin}")
print(f"  Max cell volume: {mesh.cell_volumes.max():.2f} m²")
print()


# ============================================================================
# Define the Active Cells
# ============================================================================

print("=" * 80)
print("Step 4: Defining Active Cells")
print("=" * 80)

# Indices of the active mesh cells from topography
active_cells = active_from_xyz(mesh, topo_2d)
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
    np.r_[-120.0, 40.0], 60.0, mesh.cell_centers[active_cells, :]
)
conductivity_model[ind_conductor] = conductor_conductivity

ind_resistor = model_builder.get_indices_sphere(
    np.r_[120.0, 72.0], 60.0, mesh.cell_centers[active_cells, :]
)
conductivity_model[ind_resistor] = resistor_conductivity

print(f"  Air conductivity: {air_conductivity} S/m")
print(f"  Background conductivity: {background_conductivity} S/m")
print(f"  Conductor conductivity: {conductor_conductivity} S/m")
print(f"  Resistor conductivity: {resistor_conductivity} S/m")
print(f"  Conductor cells: {ind_conductor.sum()}")
print(f"  Resistor cells: {ind_resistor.sum()}")
print()

# Mapping from conductivity model to mesh
conductivity_map = maps.InjectActiveCells(mesh, active_cells, air_conductivity)

# Mapping to neglect air cells when plotting
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
ax1.set_title("Background Conductivity Model")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")

ax2 = fig.add_axes([0.84, 0.17, 0.03, 0.7])
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.RdYlBu_r
)
cbar.set_label(r"$\sigma$ (S/m)", rotation=270, labelpad=15, size=12)

plt.savefig("conductivity_model_ip_2d.png", dpi=150, bbox_inches="tight")
print("✓ Conductivity model plot saved as 'conductivity_model_ip_2d.png'")
plt.close()


# ============================================================================
# Define the Chargeability Model
# ============================================================================

print("=" * 80)
print("Step 6: Defining Chargeability Model")
print("=" * 80)

# Intrinsic chargeability in V/V (unitless)
air_chargeability = 0.0
background_chargeability = 0.0
sphere_chargeability = 1e-1

# Define chargeability model
chargeability_model = background_chargeability * np.ones(n_active)

ind_chargeable = model_builder.get_indices_sphere(
    np.r_[-120.0, 40.0], 60.0, mesh.cell_centers[active_cells, :]
)
chargeability_model[ind_chargeable] = sphere_chargeability

print(f"  Background chargeability: {background_chargeability} V/V")
print(f"  Sphere chargeability: {sphere_chargeability} V/V")
print(f"  Chargeable cells: {ind_chargeable.sum()}")
print()

# Define mapping for chargeability
chargeability_map = maps.InjectActiveCells(mesh, active_cells, air_chargeability)

# Plot Chargeability Model
fig = plt.figure(figsize=(9, 4))

norm = Normalize(vmin=0.0, vmax=0.1)

ax1 = fig.add_axes([0.14, 0.17, 0.68, 0.7])
mesh.plot_image(
    plotting_map * chargeability_model,
    ax=ax1,
    grid=False,
    pcolor_opts={"cmap": mpl.cm.plasma, "norm": norm},
)
ax1.set_xlim(-500, 500)
ax1.set_ylim(-300, 200)
ax1.set_title("Intrinsic Chargeability")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")

ax2 = fig.add_axes([0.84, 0.17, 0.03, 0.7])
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.plasma
)
cbar.set_label("Intrinsic Chargeability (V/V)", rotation=270, labelpad=15, size=12)

plt.savefig("chargeability_model_ip_2d.png", dpi=150, bbox_inches="tight")
print("✓ Chargeability model plot saved as 'chargeability_model_ip_2d.png'")
plt.close()


# ============================================================================
# Project Electrodes to Discretized Topography
# ============================================================================

print("=" * 80)
print("Step 7: Projecting Electrodes to Topography")
print("=" * 80)

ip_survey.drape_electrodes_on_topography(mesh, active_cells, option="top")

print("  ✓ Electrodes draped on discretized topography")
print()


# ============================================================================
# Define the IP Simulation
# ============================================================================

print("=" * 80)
print("Step 8: Defining IP Simulation")
print("=" * 80)

ip_simulation = ip.Simulation2DNodal(
    mesh,
    survey=ip_survey,
    etaMap=chargeability_map,
    sigma=conductivity_map * conductivity_model,
)

print("  Simulation type: 2D Nodal Formulation")
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

# Plot IP Data in Pseudosection
fig = plt.figure(figsize=(9, 4))

ax1 = fig.add_axes([0.1, 0.1, 0.7, 0.8])
cax1 = fig.add_axes([0.82, 0.1, 0.025, 0.8])
plot_pseudosection(
    ip_survey,
    dpred_ip,
    "contourf",
    ax=ax1,
    cax=cax1,
    scale="linear",
    cbar_label="V/V",
    mask_topography=True,
    contourf_opts={"levels": 20, "cmap": mpl.cm.plasma},
)
ax1.set_title("Apparent Chargeability (V/V)")

plt.savefig("ip_data_pseudosection_2d.png", dpi=150, bbox_inches="tight")
print("✓ IP pseudosection plot saved as 'ip_data_pseudosection_2d.png'")
plt.close()


# ============================================================================
# Optional: Export Data and Topography
# ============================================================================

if write_output:
    print("=" * 80)
    print("Step 10: Exporting Data and Topography")
    print("=" * 80)

    dir_path = os.path.sep.join([".", "fwd_ip_2d_outputs"]) + os.path.sep
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print(f"  Created directory: {dir_path}")

    # Add 5% Gaussian noise to each datum
    rng = np.random.default_rng(seed=225)
    std = 5e-3 * np.ones_like(dpred_ip)
    ip_noise = rng.normal(scale=std, size=len(dpred_ip))
    dobs = dpred_ip + ip_noise

    # Create a survey with the original electrode locations
    source_list = generate_dcip_sources_line(
        survey_type,
        data_type,
        dimension_type,
        end_locations,
        topo_2d,
        num_rx_per_src,
        station_separation,
    )
    survey_original = ip.survey.Survey(source_list)

    # Write out data at their original electrode locations
    data_obj = data.Data(survey_original, dobs=dobs, standard_deviation=std)
    fname = dir_path + "ip_data.obs"
    write_dcip2d_ubc(fname, data_obj, "apparent_chargeability", "dobs")
    print(f"  ✓ IP data saved to: {fname}")

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
print(f"  Data points:         {ip_survey.nD}")
print(f"  Chargeability:       [{np.min(chargeability_model):.4f}, {np.max(chargeability_model):.4f}] V/V")
print(f"  Apparent charge.:    [{dpred_ip.min():.6f}, {dpred_ip.max():.6f}] V/V")
print()
print("Generated Files:")
print("-" * 80)
print("  1. topography_ip_2d.png            - 2D topography profile")
print("  2. pseudo_locations_ip_2d.png      - Pseudo-locations of data")
print("  3. conductivity_model_ip_2d.png    - Background conductivity model")
print("  4. chargeability_model_ip_2d.png   - Intrinsic chargeability model")
print("  5. ip_data_pseudosection_2d.png    - Apparent chargeability pseudosection")
if write_output:
    print("  6. fwd_ip_2d_outputs/ip_data.obs")
    print("  7. fwd_ip_2d_outputs/topo_2d.txt")
print()
print("Note: SimPEG uses linearized formulation for IP simulation")
print("      Apparent chargeability data are in same units as model (V/V)")
print("      Conductive sphere is chargeable; resistive sphere is not")
print("=" * 80)
