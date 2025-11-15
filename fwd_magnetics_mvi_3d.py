"""
3D Forward Simulation of Magnetic Gradiometry Data for Magnetic Vector Models
University of British Columbia

This tutorial teaches basic functionality within SimPEG and demonstrates:
- How to simulate magnetic data for 3D structures with SimPEG
- How to create magnetic gradiometry surveys (managing multiple data components)
- How to design tree meshes for magnetic simulations using the integral formulation
- How to construct a magnetic vector model (MVI)
- How to predict magnetic gradiometry data for a magnetic vector model
- How to include surface topography in the forward simulation
- Units of the magnetic vector model and resulting data

Keywords: gradiometry, magnetic vector model, forward simulation, integral formulation, tree mesh
"""

# ============================================================================
# Import Modules
# ============================================================================

# SimPEG functionality
from simpeg.potential_fields import magnetics
from simpeg.utils import plot2Ddata, model_builder, mat_utils
from simpeg import maps

# discretize functionality
from discretize import TreeMesh
from discretize.utils import mkvc, active_from_xyz

# Common Python functionality
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.constants import mu_0
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
rng = np.random.default_rng(seed=42)
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
plt.savefig("mvi_topography.png", dpi=150, bbox_inches="tight")
print("✓ Topography plot saved as 'mvi_topography.png'")
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

# Define the components of the field we want to simulate
# Here we measure the x, y and z derivatives of the Bz anomaly
components = ["bxz", "byz", "bzz"]

# Define receivers
receiver_list = magnetics.receivers.Point(receiver_locations, components=components)
receiver_list = [receiver_list]

# Define the inducing field
field_inclination = 90  # inclination [deg]
field_declination = 0  # declination [deg]
field_amplitude = 50000  # amplitude [nT]

source_field = magnetics.sources.UniformBackgroundField(
    receiver_list=receiver_list,
    amplitude=field_amplitude,
    inclination=field_inclination,
    declination=field_declination,
)

# Define the survey
survey = magnetics.survey.Survey(source_field)

print(f"  Number of locations: {survey.nRx}")
print(f"  Number of data: {survey.nD}")
print(f"  Inducing field inclination: {field_inclination}°")
print(f"  Inducing field declination: {field_declination}°")
print(f"  Inducing field amplitude: {field_amplitude} nT")
print()


# ============================================================================
# Design a Tree Mesh
# ============================================================================

print("=" * 80)
print("Step 3: Designing Tree Mesh")
print("=" * 80)

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

# Define the base mesh. Top defined at z = 0 m.
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
mesh.refine_box(wsb_corner, ent_corner, levels=[-1], finalize=False)

mesh.finalize()

print(f"  Number of cells: {mesh.n_cells}")
print(f"  Number of x-faces: {mesh.n_faces_x}")
print(f"  Origin: {mesh.origin}")
print(f"  Max cell volume: {mesh.cell_volumes.max():.2f} m³")
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
model_map = maps.IdentityMap(nP=3 * n_active)

print(f"  Number of active cells: {n_active}")
print(f"  Model parameters (3-component vector): {model_map.nP}")
print()


# ============================================================================
# Define the Magnetic Vector Model
# ============================================================================

print("=" * 80)
print("Step 6: Defining Magnetic Vector Model (MVI)")
print("=" * 80)

# Define susceptibility values for each unit in SI
background_susceptibility = 0.0001
sphere_susceptibility = 0.01

# Compute the induced magnetization vector (A/m) for every active cell
susceptibility_model = background_susceptibility * np.ones(n_active)
ind_sphere = model_builder.get_indices_sphere(
    np.r_[0.0, 0.0, 55.0], 16.0, mesh.cell_centers
)
ind_sphere = ind_sphere[active_cells]
susceptibility_model[ind_sphere] = sphere_susceptibility

# Compute the unit direction of the inducing field in Cartesian coordinates
field_direction = mat_utils.dip_azimuth2cartesian(field_inclination, field_declination)

# Inducing magnetic field intensity (A/m)
H0 = 1e-9 * field_amplitude * field_direction / mu_0

# Compute induced magnetization
induced_magnetization = np.outer(susceptibility_model, H0)  # (n_active, 3) array

# Define the remanent magnetization vector (A/m) for every active cell
remanence_inclination = 45.0
remanence_declination = 210.0
remanence_amplitude = 0.39788735751313814

remanent_magnetization = np.zeros_like(induced_magnetization)
remanent_magnetization_sphere = remanence_amplitude * mat_utils.dip_azimuth2cartesian(
    remanence_inclination, remanence_declination
)
remanent_magnetization[ind_sphere, :] = remanent_magnetization_sphere

# Compute total magnetization (A/m) for all active cells
total_magnetization = induced_magnetization + remanent_magnetization

# Define effective susceptibility model as a vector np.r_[chi_x, chi_y, chi_z]
model = mkvc(total_magnetization) / np.linalg.norm(H0)

print(f"  Background susceptibility: {background_susceptibility} SI")
print(f"  Sphere susceptibility: {sphere_susceptibility} SI")
print(f"  Sphere cells: {ind_sphere.sum()}")
print(f"  Remanence inclination: {remanence_inclination}°")
print(f"  Remanence declination: {remanence_declination}°")
print()

# Plot Magnetic Vector Model
plotting_map = maps.InjectActiveCells(mesh, active_cells, np.nan)
plotting_model = total_magnetization / np.linalg.norm(H0)
model_amplitude = np.sqrt(np.sum(plotting_model**2, axis=1))

fig = plt.figure(figsize=(14, 4))
norm = mpl.colors.Normalize(vmin=0, vmax=np.max(model_amplitude))

ax1 = fig.add_axes([0.05, 0.12, 0.5, 0.78])
mesh.plot_slice(
    plotting_map * model_amplitude,
    normal="Y",
    ax=ax1,
    ind=int(mesh.h[1].size / 2),
    grid=True,
    pcolor_opts={"cmap": mpl.cm.plasma, "norm": norm},
)
ax1.set_title("MVI Model at y = 0 m (amplitude)")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")

ax2 = fig.add_axes([0.62, 0.12, 0.25, 0.78])
mesh.plot_slice(
    plotting_map * plotting_model,
    v_type="CCv",
    view="vec",
    normal="Y",
    ax=ax2,
    ind=int(mesh.h[1].size / 2),
    grid=True,
    pcolor_opts={"cmap": mpl.cm.plasma, "norm": norm},
    quiver_opts={
        "pivot": "mid",
        "width": 0.01,
        "headwidth": 3.0,
        "headlength": 3.0,
        "headaxislength": 3.0,
        "scale": 0.25,
    },
)
ax2.set_title("MVI Model at y = 0 m")
ax2.set_xlim([-25, 25])
ax2.set_ylim([30, 80])
ax2.set_xlabel("x (m)")
ax2.set_ylabel("z (m)")

cx = fig.add_axes([0.89, 0.12, 0.02, 0.78])
cbar = mpl.colorbar.ColorbarBase(
    cx, norm=norm, orientation="vertical", cmap=mpl.cm.plasma
)
cbar.set_label("SI", rotation=270, labelpad=15)

plt.savefig("mvi_model.png", dpi=150, bbox_inches="tight")
print("✓ MVI model plot saved as 'mvi_model.png'")
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
    model_type="vector",
    chiMap=model_map,
    active_cells=active_cells,
    store_sensitivities="forward_only",
    engine="choclo",
)

print("  Simulation type: 3D Integral Formulation")
print("  Model type: Magnetic Vector (MVI)")
print("  Engine: Choclo (fast Numba implementation)")
print("  Store sensitivities: forward_only")
print()


# ============================================================================
# Simulate Magnetic Gradiometry Data
# ============================================================================

print("=" * 80)
print("Step 8: Simulating Magnetic Gradiometry Data")
print("=" * 80)

dpred = simulation.dpred(model)

print(f"  Simulated data points: {len(dpred)}")
print(f"  Data range: [{dpred.min():.4f}, {dpred.max():.4f}] nT/m")
print(f"  Mean absolute value: {np.abs(dpred).mean():.4f} nT/m")
print()

# Reshape data for plotting
n_loc = survey.nRx
n_comp = len(components)
dpred_plotting = np.reshape(dpred, (n_loc, n_comp))

# Plot the simulated magnetic gradiometry data
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
        ncontour=60,
        contourOpts={"cmap": "bwr", "norm": norm},
    )
    ax[ii].set_title(r"$\partial B_z /\partial {}$".format(comp_list[ii]))
    ax[ii].set_xlabel("x (m)")
    if ii == 0:
        ax[ii].set_ylabel("y (m)")
    else:
        ax[ii].set_yticks([])

cx = fig.add_axes([0.89, 0.13, 0.02, 0.79])
cbar = mpl.colorbar.ColorbarBase(cx, norm=norm, orientation="vertical", cmap=mpl.cm.bwr)
cbar.set_label("$nT/m$", rotation=270, labelpad=10, size=12)

plt.savefig("mvi_gradiometry_data.png", dpi=150, bbox_inches="tight")
print("✓ Gradiometry data plot saved as 'mvi_gradiometry_data.png'")
plt.close()


# ============================================================================
# Optional: Export Data and Topography
# ============================================================================

if save_output:
    print("=" * 80)
    print("Step 9: Exporting Data and Topography")
    print("=" * 80)

    dir_path = os.path.sep.join([".", "fwd_magnetics_mvi_3d_outputs"]) + os.path.sep
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print(f"  Created directory: {dir_path}")

    # Save topography
    fname = dir_path + "magnetics_topo.txt"
    np.savetxt(fname, np.c_[topo_xyz], fmt="%.4e")
    print(f"  ✓ Topography saved to: {fname}")

    # Save magnetic gradiometry data with noise
    rng = np.random.default_rng(seed=42)
    maximum_anomaly = np.max(np.abs(dpred))
    noise = rng.normal(scale=0.02 * maximum_anomaly, size=len(dpred))
    fname = dir_path + "magnetics_mvi_data.obs"
    np.savetxt(fname, np.c_[receiver_locations, dpred + noise], fmt="%.4e")
    print(f"  ✓ Gradiometry data saved to: {fname}")
    print(f"  Noise level: 2% of maximum anomaly ({0.02 * maximum_anomaly:.4f} nT/m)")
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
print(f"  Data components:     {n_comp}")
print(f"  Total data points:   {survey.nD}")
print(f"  MVI amplitude range: [0.0, {np.max(model_amplitude):.4f}] SI")
print(f"  Gradiometry data:    [{dpred.min():.4f}, {dpred.max():.4f}] nT/m")
print()
print("Generated Files:")
print("-" * 80)
print("  1. mvi_topography.png          - 3D visualization of surface topography")
print("  2. mvi_model.png               - Magnetic vector model (amplitude and vectors)")
print("  3. mvi_gradiometry_data.png    - Magnetic gradiometry data (3 components)")
if save_output:
    print("  4. fwd_magnetics_mvi_3d_outputs/magnetics_topo.txt")
    print("  5. fwd_magnetics_mvi_3d_outputs/magnetics_mvi_data.obs")
print()
print("Note: Magnetic vector models (MVI) allow for both induced and remanent magnetization")
print("      Data units are in nT/m (nanotesla per meter) for gradiometry measurements")
print("=" * 80)
