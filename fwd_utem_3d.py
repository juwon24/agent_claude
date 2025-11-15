"""
3D Forward Simulation for On-Time Large-Loop TDEM Data (UTEM)
University of British Columbia

This tutorial teaches intermediate functionality within SimPEG and demonstrates:
- Designing appropriate meshes for 3D large-loop TDEM simulation
- Simulating TDEM data for user-defined waveforms
- Simulating on-time dB/dt data
- Plotting data collected during the on-time
- Computing and plotting primary-reduced UTEM data

UTEM (University of Toronto Electromagnetic) surveys use large transmitter loops with
a triangular waveform and 100% duty cycle. This tutorial shows how to simulate
3-component on-time data for such systems.

Keywords: UTEM, forward simulation, large current loop, on-time db/dt data
"""

# ============================================================================
# Import Modules
# ============================================================================

# SimPEG functionality
import simpeg.electromagnetics.time_domain as tdem
from simpeg import maps
from simpeg.utils import plot2Ddata

# discretize functionality
from discretize import TreeMesh
from discretize.utils import mkvc, ndgrid, active_from_xyz

# Common Python functionality
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

mpl.rcParams.update({"font.size": 14})

write_output = False  # Set to True to save outputs


# ============================================================================
# Define the Topography
# ============================================================================

print("=" * 80)
print("Step 1: Defining Topography")
print("=" * 80)

# Generate topography
x_topo, y_topo = np.meshgrid(
    np.linspace(-2100, 2100, 141), np.linspace(-2100, 2100, 141)
)
z_topo = 400.0 + 200.0 * (1 / np.pi) * (
    np.arctan((x_topo - 500 * np.sin(np.pi * y_topo / 2800) - 1000.0) / 300.0)
    - np.arctan((x_topo - 500 * np.sin(np.pi * y_topo / 2800) + 1000.0) / 300.0)
)

# Turn into a (N, 3) numpy.ndarray
x_topo, y_topo, z_topo = mkvc(x_topo), mkvc(y_topo), mkvc(z_topo)
topo_xyz = np.c_[mkvc(x_topo), mkvc(y_topo), mkvc(z_topo)]

print(f"  Topography points: {len(x_topo)}")
print(f"  Z-range: [{z_topo.min():.2f}, {z_topo.max():.2f}] m")
print()

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
plt.savefig("utem_topography.png", dpi=150, bbox_inches="tight")
print("✓ Topography plot saved as 'utem_topography.png'")
plt.close()


# ============================================================================
# Define the UTEM Survey
# ============================================================================

print("=" * 80)
print("Step 2: Defining UTEM Survey")
print("=" * 80)

# WAVEFORM PROPERTIES
period = 1.0  # Period
I_max = 1.0  # Maximum current amplitude

# TRANSMITTER LOOP PROPERTIES
loop_width = 1400

# RECEIVER LOCATIONS
ds = 125.0
s_min = -562.5
s_max = 562.5
receiver_height = 1.0

# TIME CHANNELS
time_channels = 0.25 * period * 2 ** np.linspace(-7, 0, 8)

print(f"  Loop width: {loop_width} m")
print(f"  Waveform period: {period} s")
print(f"  Receiver spacing: {ds} m")
print(f"  Number of time channels: {len(time_channels)}")
print()


# ============================================================================
# Define the Waveform
# ============================================================================

print("=" * 80)
print("Step 3: Defining Custom Waveform")
print("=" * 80)

# Slope of the triangular waveform
slope = 4 / period


# Function handle
def wave_fun(t):
    ta, tb = -0.5 * period, 0.0
    w = (
        np.heaviside(ta - t, 0.5) * (slope * t + 3.0 * I_max)
        + np.heaviside(t - ta, 0.5) * np.heaviside(tb - t, 0.5) * (-slope * t - I_max)
        + np.heaviside(t - tb, 0.5) * (slope * t - I_max)
    )
    return w


# Define the waveform
general_waveform = tdem.sources.RawWaveform(
    off_time=period / 2, waveform_function=wave_fun
)

print("  ✓ Triangular waveform defined (100% duty cycle)")
print()

# Plot the waveform
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.85])
plot_times = np.linspace(-3 * period / 4, period / 2.0, 26)
ax1.plot(plot_times, wave_fun(plot_times), "b")
ax1.plot(time_channels, wave_fun(time_channels), "ro")
ax1.grid()
ax1.set_xlabel("Times [s]")
ax1.set_ylabel("Current [A]")
ax1.set_title("Current Waveform")
plt.savefig("utem_waveform.png", dpi=150, bbox_inches="tight")
print("✓ Waveform plot saved as 'utem_waveform.png'")
plt.close()


# ============================================================================
# Define Source and Receivers
# ============================================================================

print("=" * 80)
print("Step 4: Defining Source Loop and Receivers")
print("=" * 80)

# Interpolation function for topography
topo_interp = LinearNDInterpolator(np.c_[x_topo, y_topo], z_topo)

# Define the loop source
half_width = 0.5 * loop_width
u = np.arange(-half_width, half_width, 50.0)
v = half_width * np.ones_like(u)

xtx = np.r_[u, v, -u, -v, -half_width]
ytx = np.r_[-v, u, v, -u, -half_width]
ztx = topo_interp(np.c_[xtx, ytx])
xyz_loop = np.c_[xtx, ytx, ztx]

# Define receiver locations
receiver_locations = ndgrid(
    np.arange(s_min, s_max + 1, ds),
    np.arange(s_min, s_max + 1, ds),
)
receiver_locations = np.c_[receiver_locations, topo_interp(receiver_locations)]

# Define receivers (3-component dB/dt)
receivers_list = [
    tdem.receivers.PointMagneticFluxTimeDerivative(receiver_locations, time_channels, s)
    for s in ["x", "y", "z"]
]

# Define loop source
source_list = [
    tdem.sources.LineCurrent(
        receivers_list, location=xyz_loop, waveform=general_waveform
    )
]

# Define the survey
survey = tdem.Survey(source_list)

print(f"  Loop nodes: {xyz_loop.shape[0]}")
print(f"  Receiver locations: {receiver_locations.shape[0]}")
print(f"  Total data points: {survey.nD}")
print()

# Plot survey geometry
fig = plt.figure(figsize=(5, 5))
ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax1.plot(xyz_loop[:, 0], xyz_loop[:, 1], "r-o", lw=1, markersize=2)
ax1.scatter(receiver_locations[:, 0], receiver_locations[:, 1], 8, "b")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")
ax1.set_title("Survey Geometry")
plt.savefig("utem_survey_geometry.png", dpi=150, bbox_inches="tight")
print("✓ Survey geometry plot saved as 'utem_survey_geometry.png'")
plt.close()


# ============================================================================
# Define Subsurface Structures
# ============================================================================

print("=" * 80)
print("Step 5: Defining Subsurface Structures")
print("=" * 80)

air_conductivity = 1e-8
host_conductivity = 0.001
slab_conductivity = 0.5

slab_width = 500.0
slab_thickness = 130.0
slab_center = np.c_[0.0, 0.0, z_topo.min() - 225]
slab_strike = 0.0
slab_dip = 30.0

# Compute diffusion distances
host_diffusion_distance = 1260 * np.sqrt(time_channels / host_conductivity)
slab_diffusion_distance = 1260 * np.sqrt(time_channels / slab_conductivity)

print(f"  Host conductivity: {host_conductivity} S/m")
print(f"  Slab conductivity: {slab_conductivity} S/m")
print(f"  Slab dimensions: {slab_width} x {slab_width} x {slab_thickness} m")
print(f"  Slab dip: {slab_dip}°")
print(f"  Host diffusion distances: {np.round(host_diffusion_distance, 1)} m")
print(f"  Slab diffusion distances: {np.round(slab_diffusion_distance, 1)} m")
print()


# ============================================================================
# Design a Tree Mesh
# ============================================================================

print("=" * 80)
print("Step 6: Designing Tree Mesh")
print("=" * 80)

dh = 25.0
L = 80000.0
nbc = 2 ** int(np.round(np.log(L / dh) / np.log(2.0)))
h = [(dh, nbc)]
mesh = TreeMesh([h, h, h], origin="CCC", diagonal_balance=True)

# Shift vertically to top same as maximum topography
mesh.origin += np.r_[0.0, 0.0, z_topo.min()]

# Refine along the wire path
n_seg = np.shape(xyz_loop)[0] - 1
n_pts_per_seg = 20

pts = [
    np.c_[
        np.linspace(xyz_loop[ii, 0], xyz_loop[ii + 1, 0], n_pts_per_seg),
        np.linspace(xyz_loop[ii, 1], xyz_loop[ii + 1, 1], n_pts_per_seg),
        np.linspace(xyz_loop[ii, 2], xyz_loop[ii + 1, 2], n_pts_per_seg),
    ]
    for ii in range(n_seg)
]

pts = np.vstack(pts)
mesh.refine_points(pts, level=-1, padding_cells_by_level=[2, 2, 2, 2], finalize=False)

# Refine in the region of the slab
pts = ndgrid(
    np.r_[-600, 600], np.r_[-600, 600], np.r_[z_topo.min() - 400, z_topo.min() + 50]
)
mesh.refine_bounding_box(pts, -1, padding_cells_by_level=[2, 4, 4], finalize=False)

# Refine surface topography
inds = (np.abs(topo_xyz[:, 0]) < 1500.0) & (np.abs(topo_xyz[:, 1]) < 1500.0)
mesh.refine_surface(
    topo_xyz[inds, :], -2, padding_cells_by_level=[1, 2], finalize=False
)

mesh.finalize()

print(f"  Number of cells: {mesh.nC}")
print(f"  Minimum cell size: {dh} m")
print(f"  Mesh origin: {mesh.origin}")
print()


# ============================================================================
# Define the Active Cells
# ============================================================================

print("=" * 80)
print("Step 7: Defining Active Cells")
print("=" * 80)

# Indices of the active mesh cells from topography
active_cells = active_from_xyz(mesh, topo_xyz)
n_active = np.sum(active_cells)

print(f"  Total mesh cells: {mesh.nC}")
print(f"  Active cells: {n_active}")
print(f"  Inactive cells: {(~active_cells).sum()}")
print()


# ============================================================================
# Define the Conductivity Model
# ============================================================================

print("=" * 80)
print("Step 8: Defining Conductivity Model")
print("=" * 80)


def indices_from_polygon(poly_pts, xyzc):
    """Find indices of points within a polygon."""
    delaunay_object = Delaunay(poly_pts)
    k = delaunay_object.find_simplex(xyzc) >= 0
    return k


# Define host conductivity on all active cells
conductivity_model = host_conductivity * np.ones(n_active)

# Define the slab
xyz_slab = ndgrid(
    np.r_[-slab_width / 2, slab_width / 2],
    np.r_[-slab_width / 2, slab_width / 2],
    np.r_[-slab_thickness / 2, slab_thickness / 2],
)

# y-axis rotation
THETA = slab_dip * np.pi / 180
Ay = np.r_[
    np.c_[np.cos(THETA), 0.0, -np.sin(THETA)],
    np.c_[0.0, 1.0, 0.0],
    np.c_[np.sin(THETA), 0.0, np.cos(THETA)],
]

# z-axis rotation
PHI = slab_strike * np.pi / 180
Az = np.r_[
    np.c_[np.cos(PHI), -np.sin(PHI), 0.0],
    np.c_[np.sin(PHI), np.cos(PHI), 0.0],
    np.c_[0.0, 0.0, 1.0],
]

# Apply the rotation
A = np.dot(Ay, Az)
xyz_slab = np.dot(xyz_slab, A)
xyz_slab = xyz_slab + slab_center

# Assign slab conductivity
ind_slab = indices_from_polygon(xyz_slab, mesh.cell_centers[active_cells, :])
conductivity_model[ind_slab] = slab_conductivity

print(f"  Host conductivity: {host_conductivity} S/m")
print(f"  Slab conductivity: {slab_conductivity} S/m")
print(f"  Slab cells: {ind_slab.sum()}")
print()

# Plot the model
plotting_map = maps.InjectActiveCells(mesh, active_cells, np.nan)

fig = plt.figure(figsize=(10, 4.5))
norm = LogNorm(vmin=conductivity_model.min(), vmax=conductivity_model.max())

ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
mesh.plot_slice(
    plotting_map * conductivity_model,
    ax=ax1,
    normal="Y",
    ind=int(len(mesh.h[1]) / 2),
    grid=True,
    pcolor_opts={"cmap": mpl.cm.plasma, "norm": norm},
)
ax1.set_title("Conductivity Model")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")
ax1.set_xlim([-1600, 1600])
ax1.set_ylim([z_topo.max() - 1600, z_topo.max()])

ax2 = fig.add_axes([0.84, 0.15, 0.03, 0.75])
cbar = mpl.colorbar.ColorbarBase(
    ax2, cmap=mpl.cm.plasma, norm=norm, orientation="vertical"
)
cbar.set_label(r"$\sigma$ [S/m]", rotation=270, labelpad=15, size=16)
plt.savefig("utem_conductivity_model.png", dpi=150, bbox_inches="tight")
print("✓ Conductivity model plot saved as 'utem_conductivity_model.png'")
plt.close()

# Define mapping
conductivity_map = maps.InjectActiveCells(mesh, active_cells, air_conductivity)


# ============================================================================
# Define Time Discretization
# ============================================================================

print("=" * 80)
print("Step 9: Defining Time Discretization")
print("=" * 80)

n_steps = 20
min_dt = time_channels[0] / 20.0
min_dt = round(min_dt, 1 - int(np.floor(np.log10(min_dt))) - 1)

tc_min = np.min(time_channels)
tc_max = np.max(time_channels)
ti = [None, tc_min, 6 * tc_min, 36 * tc_min]
dt = [period / 60, min_dt, 6 * min_dt, 36 * min_dt]

wave_times = np.arange(-0.75 * period, 0, dt[0])
wave_times = np.r_[wave_times, np.arange(0, ti[1] + dt[1], dt[1])]
wave_times = np.r_[wave_times, np.arange(wave_times[-1] + dt[2], ti[2] + dt[2], dt[2])]
wave_times = np.r_[wave_times, np.arange(wave_times[-1] + dt[3], ti[3] + dt[3], dt[3])]
wave_times = np.r_[
    wave_times, np.arange(wave_times[-1] + dt[0], tc_max + 2 * dt[0], dt[0])
]

time_steps = np.diff(wave_times)
t0 = wave_times[0]

print(f"  Number of time steps: {len(time_steps)}")
print(f"  Initial time (t0): {t0} s")
print(f"  Final time: {wave_times[-1]} s")
print()

# Plot time discretization
fig = plt.figure(figsize=(14, 4))

ax1 = fig.add_axes([0.1, 0.1, 0.5, 0.85])
ax1.plot(wave_times, wave_fun(wave_times), "b-o", markersize=3)
ax1.plot(time_channels, wave_fun(time_channels), "ro", markersize=5)
ax1.grid()
ax1.set_xlabel("Times [s]")
ax1.set_ylabel("Current [A]")
ax1.set_title("Full Waveform")

ax2 = fig.add_axes([0.7, 0.1, 0.3, 0.85])
k = wave_times > 0.0
ax2.semilogx(wave_times[k], wave_fun(wave_times[k]), "b-o", markersize=3)
ax2.semilogx(time_channels, wave_fun(time_channels), "ro", markersize=5)
ax2.grid()
ax2.set_xlabel("Times [s]")
ax2.set_ylabel("Current [A]")
ax2.set_title("t>0 Discretization")

plt.savefig("utem_time_discretization.png", dpi=150, bbox_inches="tight")
print("✓ Time discretization plot saved as 'utem_time_discretization.png'")
plt.close()


# ============================================================================
# Define the Simulation
# ============================================================================

print("=" * 80)
print("Step 10: Defining Forward Simulation")
print("=" * 80)

simulation = tdem.simulation.Simulation3DElectricField(
    mesh, survey=survey, sigmaMap=conductivity_map
)
simulation.time_steps = time_steps
simulation.t0 = t0

print("  Simulation type: 3D Electric Field (E-formulation)")
print("  Time steps: {} unique step lengths".format(len(np.unique(time_steps))))
print()


# ============================================================================
# Predict Total Field Data
# ============================================================================

print("=" * 80)
print("Step 11: Simulating UTEM Data")
print("=" * 80)

dpred = simulation.dpred(conductivity_model)

print(f"  Total field data simulated")
print(f"  Data range: [{dpred.min():.4e}, {dpred.max():.4e}] T/s")
print()


# ============================================================================
# Predict Primary Field Data
# ============================================================================

print("=" * 80)
print("Step 12: Simulating Primary Field Data")
print("=" * 80)

vacuum_model = air_conductivity * np.ones_like(conductivity_model)
dpred0 = simulation.dpred(vacuum_model)

print(f"  Primary field data simulated")
print(f"  Data range: [{dpred0.min():.4e}, {dpred0.max():.4e}] T/s")
print()


# ============================================================================
# Compute Primary-Reduced Data
# ============================================================================

print("=" * 80)
print("Step 13: Computing Primary-Reduced Data")
print("=" * 80)

n_comp = 3  # Simulated x, y and z components
n_rx = np.shape(receiver_locations)[0]
n_times = len(time_channels)

# Reshape and convert to b-field representation
B = (4 / period) * np.reshape(dpred, (n_comp, n_times, n_rx))
B0 = (4 / period) * np.reshape(dpred0, (n_comp, n_times, n_rx))

# Primary field amplitude
B0abs = np.sqrt(B0[0, :, :] ** 2 + B0[1, :, :] ** 2 + B0[2, :, :] ** 2)

# Primary reduced data (%)
x_data = 100 * (B[0, :, :] - B0[0, :, :]) / B0abs
y_data = 100 * (B[1, :, :] - B0[1, :, :]) / B0abs
z_data = 100 * (B[2, :, :] - B0[2, :, :]) / B0abs

data_plotting = [x_data, y_data, z_data]

print(f"  Primary-reduced data computed")
print(f"  X-component range: [{x_data.min():.2f}, {x_data.max():.2f}] %")
print(f"  Y-component range: [{y_data.min():.2f}, {y_data.max():.2f}] %")
print(f"  Z-component range: [{z_data.min():.2f}, {z_data.max():.2f}] %")
print()


# ============================================================================
# Plot Primary-Reduced Data Maps
# ============================================================================

print("=" * 80)
print("Step 14: Plotting Primary-Reduced Data Maps")
print("=" * 80)

time_index = 0  # Index for the time channel being plotted

fig = plt.figure(figsize=(15, 6))
ax1 = [fig.add_axes([0.05 + 0.3 * ii, 0.2, 0.28, 0.75]) for ii in range(0, 3)]
cax1 = [fig.add_axes([0.05 + 0.3 * ii, 0.05, 0.28, 0.05]) for ii in range(0, 3)]
norm1 = 3 * [None]
cbar1 = 3 * [None]
cplot1 = 3 * [None]

comp_list = ["X", "Y", "Z"]

for ii in range(0, 3):
    d_temp = data_plotting[ii][time_index, :]

    max_val = np.max(np.abs(d_temp))
    norm1[ii] = mpl.colors.Normalize(vmin=-max_val, vmax=max_val)
    levels = np.linspace(-max_val, max_val, 19)

    cplot1[ii] = plot2Ddata(
        receiver_locations[:, 0:2],
        d_temp,
        ax=ax1[ii],
        nx=200,
        ny=200,
        ncontour=60,
        shade=False,
        level=True,
        dataloc=True,
        levelOpts={"levels": levels, "colors": "k", "linewidths": 0.75},
        contourOpts={"cmap": mpl.cm.bwr_r, "norm": norm1[ii]},
    )

    ax1[ii].ticklabel_format(axis="both", scilimits=(0, 3))
    ax1[ii].set_xlabel("Easting (m)")
    if ii > 0:
        ax1[ii].set_ylabel("")
        ax1[ii].set_yticks([])
    else:
        ax1[ii].set_ylabel("Northing (m)")
    ax1[ii].set_title("Primary Reduced ({}-comp)".format(comp_list[ii]))

    cbar1[ii] = mpl.colorbar.ColorbarBase(
        cax1[ii], norm=norm1[ii], orientation="horizontal", cmap=mpl.cm.bwr_r
    )
    cbar1[ii].set_label("%", labelpad=5, size=12)

plt.savefig("utem_data_maps.png", dpi=150, bbox_inches="tight")
print("✓ Data maps plot saved as 'utem_data_maps.png'")
plt.close()


# ============================================================================
# Plot TDEM Profile
# ============================================================================

print("=" * 80)
print("Step 15: Plotting TDEM Profile")
print("=" * 80)

EW_line_index = 6

y_unique = np.unique(receiver_locations[:, 1])
locations_indices = receiver_locations[:, 1] == y_unique[EW_line_index]

fig = plt.figure(figsize=(15, 5))
ax1 = [fig.add_axes([0.05 + 0.3 * ii, 0.2, 0.24, 0.75]) for ii in range(0, 3)]

comp_list = ["X", "Y", "Z"]

for ii in range(0, 3):
    d_temp = data_plotting[ii][:, locations_indices]

    for jj in range(n_times):
        ax1[ii].plot(
            receiver_locations[locations_indices, 0],
            d_temp[jj, :],
            marker="o",
            color=[jj / (n_times - 1), 0, 1 - jj / (n_times - 1)],
            markersize=4,
            label="_nolegend_",
        )

    ax1[ii].grid()
    ax1[ii].ticklabel_format(axis="both", scilimits=(0, 3))
    ax1[ii].set_xlabel("Easting (m)")
    ax1[ii].set_ylabel("%")
    ax1[ii].set_title("Primary Reduced ({}-comp)".format(comp_list[ii]))

plt.savefig("utem_profile.png", dpi=150, bbox_inches="tight")
print("✓ Profile plot saved as 'utem_profile.png'")
plt.close()


# ============================================================================
# Optional: Export Data
# ============================================================================

if write_output:
    print("=" * 80)
    print("Step 16: Exporting Data")
    print("=" * 80)

    import os

    dir_path = os.path.sep.join([".", "fwd_utem_3d_outputs"]) + os.path.sep
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print(f"  Created directory: {dir_path}")

    # Save topography
    fname = dir_path + "utem_topo.txt"
    np.savetxt(fname, topo_xyz, fmt="%.4e")
    print(f"  ✓ Topography saved to: {fname}")

    # Save data
    fname = dir_path + "utem_data.txt"
    np.savetxt(fname, np.c_[receiver_locations, dpred], fmt="%.4e")
    print(f"  ✓ UTEM data saved to: {fname}")
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
print(f"  Mesh cells:          {mesh.nC}")
print(f"  Active cells:        {n_active}")
print(f"  Receiver locations:  {n_rx}")
print(f"  Time channels:       {n_times}")
print(f"  Total data points:   {survey.nD}")
print(f"  Loop width:          {loop_width} m")
print(f"  Waveform period:     {period} s")
print()
print("Generated Files:")
print("-" * 80)
print("  1. utem_topography.png              - 3D visualization of topography")
print("  2. utem_waveform.png                - UTEM triangular waveform")
print("  3. utem_survey_geometry.png         - Survey layout (loop and receivers)")
print("  4. utem_conductivity_model.png      - Cross-section of conductivity model")
print("  5. utem_time_discretization.png     - Time discretization scheme")
print("  6. utem_data_maps.png               - Primary-reduced data maps (3 components)")
print("  7. utem_profile.png                 - UTEM profile along E-W line")
if write_output:
    print("  8. fwd_utem_3d_outputs/utem_topo.txt")
    print("  9. fwd_utem_3d_outputs/utem_data.txt")
print()
print("Key Concepts:")
print("-" * 80)
print("  - UTEM uses large loops with triangular waveforms (100% duty cycle)")
print("  - On-time data requires time discretization during waveform on-time")
print("  - Primary-reduced data = 100 * (B_total - B_primary) / |B_primary|")
print("  - 3D tree meshes require refinement near sources and conductive structures")
print("  - Numerical primary field removal cancels discretization errors")
print("=" * 80)
