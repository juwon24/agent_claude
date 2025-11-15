"""
3D Forward Simulation of Gravity Gradiometry Data
University of British Columbia

This tutorial teaches basic functionality within SimPEG and demonstrates:
- How to simulate gravity gradiometry data for 3D structures with SimPEG
- How to create gravity gradiometry surveys and manage multiple data components
- How to design a tree mesh for gravity simulation using the integral solution
- How to predict gravity gradiometry data for a density contrast model
- How to include surface topography in the forward simulation
- Units of the density contrast model and resulting data

Keywords: gravity gradiometry, forward simulation, integral formulation, tree mesh
"""

# ============================================================================
# Import Modules
# ============================================================================

# SimPEG functionality
from simpeg.potential_fields import gravity
from simpeg.utils import plot2Ddata, model_builder
from simpeg import maps

# discretize functionality
from discretize import TreeMesh
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
plt.savefig("topography_gradiometry.png", dpi=150, bbox_inches="tight")
print("✓ Topography plot saved as 'topography_gradiometry.png'")
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

# Define observation locations 5 m above topography
x = np.linspace(-80.0, 80.0, 17)
y = np.linspace(-80.0, 80.0, 17)
x, y = np.meshgrid(x, y)
x, y = mkvc(x.T), mkvc(y.T)
fun_interp = LinearNDInterpolator(np.c_[x_topo, y_topo], z_topo)
z = fun_interp(np.c_[x, y]) + 5.0
receiver_locations = np.c_[x, y, z]

# Define the component(s) of the field we want to simulate
# Measure the x, y and z derivatives of the vertical gravity anomaly
components = ["gxz", "gyz", "gzz"]

# Define receivers
receiver_list = gravity.receivers.Point(receiver_locations, components=components)
receiver_list = [receiver_list]

# Define the source field
source_field = gravity.sources.SourceField(receiver_list=receiver_list)

# Define the survey
survey = gravity.survey.Survey(source_field)

print(f"  Number of receiver locations: {survey.nRx}")
print(f"  Number of data components: {len(components)}")
print(f"  Total data points: {survey.nD}")
print(f"  Source field: {survey.source_field}")
print(f"  Receiver type: {survey.source_field.receiver_list[0]}")
print(f"  First 5 receiver locations:")
print(receiver_list[0].locations[:5, :])
print()


# ============================================================================
# Design a Tree Mesh
# ============================================================================

print("=" * 80)
print("Step 3: Designing Tree Mesh")
print("=" * 80)

# Define domain parameters
dx = 5  # minimum cell width (base mesh cell width) in x
dy = 5  # minimum cell width (base mesh cell width) in y
dz = 5  # minimum cell width (base mesh cell width) in z

x_length = 240.0  # domain width in x
y_length = 240.0  # domain width in y
z_length = 120.0  # domain width in z

# Compute number of base mesh cells required in x and y
nbcx = 2 ** int(np.round(np.log(x_length / dx) / np.log(2.0)))
nbcy = 2 ** int(np.round(np.log(y_length / dy) / np.log(2.0)))
nbcz = 2 ** int(np.round(np.log(z_length / dz) / np.log(2.0)))

# Define the base mesh with top at z = 0 m
hx = [(dx, nbcx)]
hy = [(dy, nbcy)]
hz = [(dz, nbcz)]
mesh = TreeMesh([hx, hy, hz], x0="CCN", diagonal_balance=True)

# Shift vertically to top same as maximum topography
mesh.origin += np.r_[0.0, 0.0, z_topo.max()]

# Refine based on surface topography
mesh.refine_surface(topo_xyz, padding_cells_by_level=[2, 2], finalize=False)

# Refine box based on region of interest
wsb_corner = np.c_[-100, -100, 20]
ent_corner = np.c_[100, 100, 100]
# Note -1 is a flag for smallest cell size
mesh.refine_box(wsb_corner, ent_corner, levels=[-1], finalize=False)

mesh.finalize()

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
print("Step 6: Defining Density Contrast Model")
print("=" * 80)

# Define density contrast values for each unit in g/cc
background_density = 0.0
block_density = -0.2
sphere_density = 0.2

# Instantiate a vector array
model = background_density * np.ones(n_active)

# Add a less dense block
ind_block = (
    (mesh.cell_centers[active_cells, 0] > -50.0)
    & (mesh.cell_centers[active_cells, 0] < -20.0)
    & (mesh.cell_centers[active_cells, 1] > -15.0)
    & (mesh.cell_centers[active_cells, 1] < 15.0)
    & (mesh.cell_centers[active_cells, 2] > 50.0)
    & (mesh.cell_centers[active_cells, 2] < 70.0)
)
model[ind_block] = block_density

# Add a more dense sphere using SimPEG utilities
ind_sphere = model_builder.get_indices_sphere(
    np.r_[35.0, 0.0, 60.0], 14.0, mesh.cell_centers
)
ind_sphere = ind_sphere[active_cells]
model[ind_sphere] = sphere_density

print(f"  Background density: {background_density} g/cc")
print(f"  Block density: {block_density} g/cc")
print(f"  Sphere density: {sphere_density} g/cc")
print(f"  Block cells: {ind_block.sum()}")
print(f"  Sphere cells: {ind_sphere.sum()}")
print()

# Plot Density Contrast Model
plotting_map = maps.InjectActiveCells(mesh, active_cells, np.nan)

fig = plt.figure(figsize=(8, 3.5))

ax1 = fig.add_axes([0.1, 0.12, 0.73, 0.78])
norm = mpl.colors.Normalize(vmin=np.min(model), vmax=np.max(model))

mesh.plot_slice(
    plotting_map * model,
    normal="Y",
    ax=ax1,
    ind=int(mesh.shape_cells[1] / 2),
    grid=True,
    pcolor_opts={"cmap": mpl.cm.RdYlBu_r, "norm": norm},
)
ax1.set_title("Model slice at y = 0 m")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")

ax2 = fig.add_axes([0.85, 0.12, 0.03, 0.78])
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.RdYlBu_r
)
cbar.set_label("$g/cm^3$", rotation=270, labelpad=15, size=16)

plt.savefig("density_model_gradiometry.png", dpi=150, bbox_inches="tight")
print("✓ Density model plot saved as 'density_model_gradiometry.png'")
plt.close()


# ============================================================================
# Define the Forward Simulation
# ============================================================================

print("=" * 80)
print("Step 7: Defining Forward Simulation")
print("=" * 80)

simulation = gravity.simulation.Simulation3DIntegral(
    survey=survey,
    mesh=mesh,
    rhoMap=model_map,
    active_cells=active_cells,
    store_sensitivities="forward_only",
    engine="choclo",
)

print("  Simulation type: 3D Integral Formulation")
print("  Engine: Choclo (fast Numba implementation)")
print("  Store sensitivities: forward_only")
print()


# ============================================================================
# Simulate Gravity Gradiometry Data
# ============================================================================

print("=" * 80)
print("Step 8: Simulating Gravity Gradiometry Data")
print("=" * 80)

dpred = simulation.dpred(model)

print(f"  Simulated data points: {len(dpred)}")
print(f"  Data range: [{dpred.min():.4f}, {dpred.max():.4f}] Eotvos")
print(f"  Mean absolute value: {np.abs(dpred).mean():.4f} Eotvos")
print()

# Reshape data for plotting
n_loc = survey.nRx  # Number of receiver locations
n_comp = len(components)  # Number of data components
dpred_plotting = np.reshape(dpred, (n_loc, n_comp))

# Plot gravity gradiometry data
fig = plt.figure(figsize=(10, 3))
v_max = np.max(np.abs(dpred))

ax = 3 * [None]
cplot = 3 * [None]
comp_list = ["x", "y", "z"]

norm = mpl.colors.Normalize(vmin=-v_max, vmax=v_max)

for ii in range(0, 3):
    ax[ii] = fig.add_axes([0.1 + ii * 0.26, 0.15, 0.25, 0.78])
    cplot[ii] = plot2Ddata(
        receiver_locations,
        dpred_plotting[:, ii],
        ax=ax[ii],
        ncontour=30,
        contourOpts={"cmap": "bwr", "norm": norm},
    )
    ax[ii].set_title(r"$\partial g_z /\partial {}$".format(comp_list[ii]))
    ax[ii].set_xlabel("x (m)")
    if ii == 0:
        ax[ii].set_ylabel("y (m)")
    else:
        ax[ii].set_yticks([])

cx = fig.add_axes([0.89, 0.13, 0.02, 0.79])
cbar = mpl.colorbar.ColorbarBase(cx, norm=norm, orientation="vertical", cmap=mpl.cm.bwr)
cbar.set_label("$Eotvos$", rotation=270, labelpad=10, size=12)

plt.savefig("gravity_gradiometry_data.png", dpi=150, bbox_inches="tight")
print("✓ Gravity gradiometry plot saved as 'gravity_gradiometry_data.png'")
plt.close()


# ============================================================================
# Optional: Export Data and Topography
# ============================================================================

if save_output:
    print("=" * 80)
    print("Step 9: Exporting Data and Topography")
    print("=" * 80)

    dir_path = os.path.sep.join([".", "fwd_gravity_gradiometry_3d_outputs"]) + os.path.sep
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print(f"  Created directory: {dir_path}")

    # Save topography
    fname = dir_path + "gravity_topo.txt"
    np.savetxt(fname, np.c_[topo_xyz], fmt="%.4e")
    print(f"  ✓ Topography saved to: {fname}")

    # Save gravity gradiometry data
    fname = dir_path + "gravity_gradiometry_data.obs"
    np.savetxt(fname, np.c_[receiver_locations, dpred_plotting[:, 0], dpred_plotting[:, 1], dpred_plotting[:, 2]], fmt="%.4e")
    print(f"  ✓ Gravity gradiometry data saved to: {fname}")
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
print(f"  Observation points:  {survey.nRx}")
print(f"  Data components:     {n_comp} (gxz, gyz, gzz)")
print(f"  Total data:          {survey.nD}")
print(f"  Density contrast:    [{np.min(model):.2f}, {np.max(model):.2f}] g/cc")
print(f"  Gradiometry range:   [{dpred.min():.4f}, {dpred.max():.4f}] Eotvos")
print()
print("Generated Files:")
print("-" * 80)
print("  1. topography_gradiometry.png      - 3D visualization of surface topography")
print("  2. density_model_gradiometry.png   - Cross-section of density contrast model")
print("  3. gravity_gradiometry_data.png    - Maps of gravity gradient components")
if save_output:
    print("  4. fwd_gravity_gradiometry_3d_outputs/gravity_topo.txt")
    print("  5. fwd_gravity_gradiometry_3d_outputs/gravity_gradiometry_data.obs")
print()
print("Note: SimPEG uses right-handed coordinate system with Z positive upward!")
print("      Gravity gradiometry values are in Eotvos (1 E = 10^-9 s^-2 = 10^-4 mGal/m)")
print("      gxz, gyz, gzz represent derivatives of vertical gravity in x, y, z directions")
print("=" * 80)
