"""
3D Forward Simulation of Total Magnetic Intensity Data
University of British Columbia

This tutorial teaches basic functionality within SimPEG and demonstrates:
- How to simulate magnetic data for 3D structures with SimPEG
- How to create magnetics surveys
- How to design tensor meshes for magnetic simulations using the integral formulation
- How to predict total magnetic intensity data for a susceptibility model
- How to include surface topography in the forward simulation
- Units of the susceptibility model and resulting data

Keywords: total magnetic intensity, forward simulation, integral formulation, tensor mesh
"""

# ============================================================================
# Import Modules
# ============================================================================

# SimPEG functionality
from simpeg.potential_fields import magnetics
from simpeg.utils import plot2Ddata, model_builder
from simpeg import maps

# discretize functionality
from discretize import TensorMesh
from discretize.utils import mkvc, active_from_xyz

# Common Python functionality
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

mpl.rcParams.update({"font.size": 14})

save_output = False  # Set to True to save outputs


# ============================================================================
# Define the Topography
# ============================================================================

print("=" * 80)
print("Step 1: Defining Topography")
print("=" * 80)

# Create topography grid
[x_topo, y_topo] = np.meshgrid(np.linspace(-200, 200, 41), np.linspace(-200, 200, 41))

# Generate synthetic topography with random noise
rng = np.random.default_rng(seed=737)
z_topo = (
    -15 * np.exp(-(x_topo**2 + y_topo**2) / 80**2)
    + 100.0
    + rng.uniform(low=0.0, high=0.5, size=x_topo.shape)
)

# Plot topography
fig = plt.figure(figsize=(6, 6))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection="3d")
ax.set_zlim([z_topo.max() - 40, z_topo.max()])
ax.plot_surface(x_topo, y_topo, z_topo, color="r", edgecolor="k", linewidth=0.5)
ax.set_box_aspect(aspect=None, zoom=0.85)
ax.set_xlabel("X (m)", labelpad=10)
ax.set_ylabel("Y (m)", labelpad=10)
ax.set_zlabel("Z (m)", labelpad=10)
ax.set_title("Topography (Exaggerated z-axis)", fontsize=16, pad=-20)
ax.view_init(elev=20.0)
plt.savefig("topography_magnetics.png", dpi=150, bbox_inches="tight")
print("✓ Topography plot saved as 'topography_magnetics.png'")
plt.close()

# Convert to array format
x_topo, y_topo, z_topo = mkvc(x_topo), mkvc(y_topo), mkvc(z_topo)
topo_xyz = np.c_[x_topo, y_topo, z_topo]

print(f"  Topography points: {len(x_topo)}")
print(f"  Z-range: [{z_topo.min():.2f}, {z_topo.max():.2f}] m")
print()


# ============================================================================
# Define the Survey
# ============================================================================

print("=" * 80)
print("Step 2: Defining Survey")
print("=" * 80)

# Define observation locations 10 m above topography
x = np.linspace(-80.0, 80.0, 17)
y = np.linspace(-80.0, 80.0, 17)
x, y = np.meshgrid(x, y)
x, y = mkvc(x.T), mkvc(y.T)
fun_interp = LinearNDInterpolator(np.c_[x_topo, y_topo], z_topo)
z = fun_interp(np.c_[x, y]) + 10  # Flight height 10 m above surface
receiver_locations = np.c_[x, y, z]

# Define the component(s) of the field we want to simulate
# Here we simulate total magnetic intensity data
components = ["tmi"]

# Define receivers
receiver_list = magnetics.receivers.Point(receiver_locations, components=components)
receiver_list = [receiver_list]

# Define the inducing field
inclination = 90  # inclination [deg]
declination = 0  # declination [deg]
amplitude = 50000  # amplitude [nT]

source_field = magnetics.sources.UniformBackgroundField(
    receiver_list=receiver_list,
    amplitude=amplitude,
    inclination=inclination,
    declination=declination,
)

# Define the survey
survey = magnetics.survey.Survey(source_field)

print(f"  Number of data points: {survey.nD}")
print(f"  Inducing field inclination: {inclination}°")
print(f"  Inducing field declination: {declination}°")
print(f"  Inducing field amplitude: {amplitude} nT")
print(f"  Source field: {survey.source_field}")
print(f"  Receiver type: {survey.source_field.receiver_list[0]}")
print(f"  First 5 receiver locations:")
print(receiver_list[0].locations[:5, :])
print()


# ============================================================================
# Design a Tensor Mesh
# ============================================================================

print("=" * 80)
print("Step 3: Designing Tensor Mesh")
print("=" * 80)

# Generate tensor mesh with top at z = 0 m
dh = 5.0
hx = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
hy = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
hz = [(dh, 5, -1.3), (dh, 15)]
mesh = TensorMesh([hx, hy, hz], "CCN")

# Shift vertically to top same as maximum topography
mesh.origin += np.r_[0.0, 0.0, z_topo.max()]

print(f"  Number of cells: {mesh.n_cells}")
print(f"  Number of x-faces: {mesh.n_faces_x}")
print(f"  Origin: {mesh.origin}")
print(f"  Max cell volume: {mesh.cell_volumes.max():.2f} m³")
print(f"  First 5 cell centers:")
print(mesh.cell_centers[0:5, :])
print()


# ============================================================================
# Define the Active Cells
# ============================================================================

print("=" * 80)
print("Step 4: Defining Active Cells")
print("=" * 80)

# Indices of the active mesh cells from topography (cells below surface)
active_cells = active_from_xyz(mesh, topo_xyz)

print(f"  Total mesh cells: {mesh.n_cells}")
print(f"  Active cells: {active_cells.sum()}")
print(f"  Inactive cells: {(~active_cells).sum()}")
print()


# ============================================================================
# Mapping from the Model to Active Cells
# ============================================================================

print("=" * 80)
print("Step 5: Creating Model Mapping")
print("=" * 80)

# Define mapping from model to active cells
n_active = int(active_cells.sum())
model_map = maps.IdentityMap(nP=n_active)

print(f"  Number of active cells: {n_active}")
print(f"  Model parameters: {model_map.nP}")
print()


# ============================================================================
# Define the Model
# ============================================================================

print("=" * 80)
print("Step 6: Defining Susceptibility Model")
print("=" * 80)

# Define susceptibility values for each unit in SI
background_susceptibility = 0.0001
sphere_susceptibility = 0.01

# Instantiate a vector array
model = background_susceptibility * np.ones(n_active)

# Add a susceptible sphere using SimPEG utilities
ind_sphere = model_builder.get_indices_sphere(
    np.r_[0.0, 0.0, 55.0], 16.0, mesh.cell_centers
)
ind_sphere = ind_sphere[active_cells]
model[ind_sphere] = sphere_susceptibility

print(f"  Background susceptibility: {background_susceptibility} SI")
print(f"  Sphere susceptibility: {sphere_susceptibility} SI")
print(f"  Sphere cells: {ind_sphere.sum()}")
print()

# Plot Susceptibility Model
plotting_map = maps.InjectActiveCells(mesh, active_cells, np.nan)

fig = plt.figure(figsize=(8, 3.5))
ax1 = fig.add_axes([0.1, 0.12, 0.73, 0.78])

norm = mpl.colors.Normalize(vmin=0, vmax=np.max(model))
mesh.plot_slice(
    plotting_map * model,
    normal="Y",
    ax=ax1,
    ind=int(mesh.shape_cells[1] / 2),
    grid=True,
    pcolor_opts={"cmap": mpl.cm.plasma, "norm": norm},
)
ax1.set_title("Model slice at y = 0 m")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")

ax2 = fig.add_axes([0.85, 0.12, 0.03, 0.78])
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.plasma
)
cbar.set_label("$SI$", rotation=270, labelpad=15, size=16)

plt.savefig("susceptibility_model.png", dpi=150, bbox_inches="tight")
print("✓ Susceptibility model plot saved as 'susceptibility_model.png'")
plt.close()


# ============================================================================
# Define the Forward Simulation
# ============================================================================

print("=" * 80)
print("Step 7: Defining Forward Simulation")
print("=" * 80)

simulation = magnetics.simulation.Simulation3DIntegral(
    survey=survey,
    mesh=mesh,
    model_type="scalar",
    chiMap=model_map,
    active_cells=active_cells,
    store_sensitivities="forward_only",
    engine="choclo",
)

print("  Simulation type: 3D Integral Formulation")
print("  Model type: scalar (susceptibility)")
print("  Engine: Choclo (fast Numba implementation)")
print("  Store sensitivities: forward_only")
print()


# ============================================================================
# Simulate Total Magnetic Intensity Data
# ============================================================================

print("=" * 80)
print("Step 8: Simulating Total Magnetic Intensity Data")
print("=" * 80)

dpred = simulation.dpred(model)

print(f"  Simulated data points: {len(dpred)}")
print(f"  Data range: [{dpred.min():.4f}, {dpred.max():.4f}] nT")
print(f"  Mean absolute value: {np.abs(dpred).mean():.4f} nT")
print()

# Plot the simulated TMI data
fig = plt.figure(figsize=(7, 5))
ax1 = fig.add_axes([0.1, 0.1, 0.75, 0.85])

norm = mpl.colors.Normalize(vmin=-np.max(np.abs(dpred)), vmax=np.max(np.abs(dpred)))
plot2Ddata(
    receiver_list[0].locations,
    dpred,
    ax=ax1,
    ncontour=40,
    contourOpts={"cmap": mpl.cm.bwr, "norm": norm},
)
ax1.set_title("Total Magnetic Intensity")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")

ax2 = fig.add_axes([0.81, 0.1, 0.04, 0.85])
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.bwr
)
cbar.set_label("$nT$", rotation=270, labelpad=20, size=16)

plt.savefig("tmi_data.png", dpi=150, bbox_inches="tight")
print("✓ TMI data plot saved as 'tmi_data.png'")
plt.close()


# ============================================================================
# Optional: Export Data and Topography
# ============================================================================

if save_output:
    print("=" * 80)
    print("Step 9: Exporting Data and Topography")
    print("=" * 80)

    dir_path = os.path.sep.join([".", "fwd_magnetics_induced_3d_outputs"]) + os.path.sep
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print(f"  Created directory: {dir_path}")

    # Save topography
    fname = dir_path + "magnetics_topo.txt"
    np.savetxt(fname, np.c_[topo_xyz], fmt="%.4e")
    print(f"  ✓ Topography saved to: {fname}")

    # Save magnetic data with noise
    rng = np.random.default_rng(seed=211)
    maximum_anomaly = np.max(np.abs(dpred))
    noise = rng.normal(scale=0.02 * maximum_anomaly, size=len(dpred))
    fname = dir_path + "magnetics_data.obs"
    np.savetxt(fname, np.c_[receiver_locations, dpred + noise], fmt="%.4e")
    print(f"  ✓ Magnetic data saved to: {fname}")
    print(f"  Noise level: 2% of maximum anomaly ({0.02 * maximum_anomaly:.4f} nT)")
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
print(f"  Observation points:  {survey.nD}")
print(f"  Susceptibility:      [{np.min(model):.5f}, {np.max(model):.5f}] SI")
print(f"  TMI anomaly:         [{dpred.min():.4f}, {dpred.max():.4f}] nT")
print(f"  Inducing field:      {amplitude} nT at Inc={inclination}°, Dec={declination}°")
print()
print("Generated Files:")
print("-" * 80)
print("  1. topography_magnetics.png        - 3D visualization of surface topography")
print("  2. susceptibility_model.png        - Cross-section of susceptibility model")
print("  3. tmi_data.png                    - Map view of total magnetic intensity")
if save_output:
    print("  4. fwd_magnetics_induced_3d_outputs/magnetics_topo.txt")
    print("  5. fwd_magnetics_induced_3d_outputs/magnetics_data.obs")
print()
print("Note: SimPEG uses right-handed coordinate system with Z positive upward!")
print("      Total magnetic intensity data are in nT (nanoTesla)")
print("      Integral formulation is accurate for susceptibilities < 0.1 SI")
print("      For higher susceptibilities, self-demagnetization becomes significant")
print("=" * 80)
