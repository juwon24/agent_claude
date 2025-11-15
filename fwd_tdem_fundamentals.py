"""
Fundamentals of Finite Volume for TDEM Simulations
University of British Columbia

This tutorial teaches fundamental concepts for TDEM forward simulation and demonstrates:
- Using observation times and subsurface conductivity to generate appropriate meshes
- Defining time discretization during the off-time
- Defining time discretization during a waveform's on-time
- Balancing numerical accuracy with computational efficiency

Successful forward simulation using mimetic finite volume requires reasonable values for
parameters that determine spatial and temporal discretization. This tutorial provides
guidelines for discretizing TDEM problems in space and time.

The tutorial is organized into 3 parts:
Part 1: Discretization in Space (cell size)
Part 2: Discretization in Time (off-time)
Part 3: Time-Discretization (on-time)

Keywords: finite volume fundamentals, TDEM, forward simulation, time discretization, mesh discretization
"""

# ============================================================================
# Import Modules
# ============================================================================

# SimPEG functionality
import simpeg.electromagnetics.time_domain as tdem
from simpeg import maps

# discretize functionality
from discretize import CylindricalMesh, TensorMesh

# Common Python functionality
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({"font.size": 14})


# ============================================================================
# Define the Simulation Geometry
# ============================================================================

print("=" * 80)
print("DEFINING SIMULATION GEOMETRY")
print("=" * 80)

# Source properties
source_location = np.array([0.0, 0.0, 0.0])
source_orientation = "z"
source_current = 1.0
source_radius = 25.0

# Receiver properties
receiver_locations = np.array([0.0, 0.0, 0.0])
receiver_orientation = "z"
time_channels = np.logspace(-5, -3, 21)

# Model properties
halfspace_conductivity = 1e-2

print(f"  Source radius: {source_radius} m")
print(f"  Receiver at loop center")
print(f"  Time channels: {len(time_channels)} (from {time_channels.min():.2e} to {time_channels.max():.2e} s)")
print(f"  Halfspace conductivity: {halfspace_conductivity} S/m")
print()


# ============================================================================
# Define Useful Functions
# ============================================================================

print("=" * 80)
print("Step 1: Defining Helper Functions")
print("=" * 80)


def generate_survey(waveform_object):
    """Generate TDEM survey for given waveform."""
    receiver_list = [
        tdem.receivers.PointMagneticFluxTimeDerivative(
            receiver_locations, time_channels, orientation=receiver_orientation
        )
    ]

    source_list = [
        tdem.sources.CircularLoop(
            receiver_list=receiver_list,
            location=source_location,
            waveform=waveform_object,
            current=source_current,
            radius=source_radius,
        )
    ]

    return tdem.Survey(source_list)


def generate_discretization(dh, d_min, d_max):
    """Generate cylindrical mesh for given parameters."""
    # Number of core mesh cells
    n_core = np.floor(8 * d_min / dh)

    # Number of padding cells
    n_pad = 1
    pad_factor = 1.25
    while sum(dh * pad_factor ** np.arange(n_pad)) < 2 * d_max:
        n_pad += 1

    # Radial and vertical discretization
    hr = [(dh, n_core), (dh, n_pad, 1.2)]
    hz = [(dh, n_pad, -pad_factor), (dh, 2 * n_core), (dh, n_pad, pad_factor)]

    # Generate mesh
    mesh = CylindricalMesh([hr, 1, hz], x0="00C")

    # Active cells
    active_cells = mesh.cell_centers[:, -1] < 0.0

    # Halfspace conductivity model
    model = halfspace_conductivity * np.ones(np.sum(active_cells))

    # Mapping from conductivity model to mesh
    mapping = maps.InjectActiveCells(mesh, active_cells, 1e-8)

    return mesh, model, mapping


print("  ✓ Helper functions defined")
print()


# ============================================================================
# PART 1: Discretization in Space (Cell Size)
# ============================================================================

print("=" * 80)
print("PART 1: DISCRETIZATION IN SPACE (CELL SIZE)")
print("=" * 80)
print()


# ============================================================================
# Compute Diffusion Distances
# ============================================================================

print("=" * 80)
print("Step 2: Computing Diffusion Distances")
print("=" * 80)

diffusion_distances = 1260 * np.sqrt(time_channels / halfspace_conductivity)
d_min = diffusion_distances.min()
d_max = diffusion_distances.max()

print(f"  Minimum diffusion distance: {d_min:.2f} m")
print(f"  Maximum diffusion distance: {d_max:.2f} m")
print(f"  All diffusion distances: {np.round(diffusion_distances, 1)}")
print()


# ============================================================================
# Define Minimum Cell Widths
# ============================================================================

print("=" * 80)
print("Step 3: Testing Different Minimum Cell Widths")
print("=" * 80)

size_factor = [4.0, 2.0, 1.0, 0.5, 0.25]
dh_min = [c * d_min for c in size_factor]

print(f"  Testing cell widths: {[f'{c}*d_min' for c in size_factor]}")
print()


# ============================================================================
# Define Step-off Waveform
# ============================================================================

stepoff_waveform = tdem.sources.StepOffWaveform(off_time=0.0)


# ============================================================================
# Compute Numerical Solutions
# ============================================================================

print("=" * 80)
print("Step 4: Computing Numerical Solutions for Different Cell Sizes")
print("=" * 80)

dpred_1 = []

for ii, dh in enumerate(dh_min):
    mesh, model, mapping = generate_discretization(dh, d_min, d_max)

    simulation = tdem.simulation.Simulation3DElectricField(
        mesh,
        survey=generate_survey(stepoff_waveform),
        sigmaMap=mapping,
        time_steps=[(5e-07, 40), (2.5e-06, 40), (1.25e-05, 81)],
    )

    dpred_1.append(simulation.dpred(model))

    print(f"  [{ii+1}/{len(dh_min)}] N_CELLS: {mesh.nC:6d} | dh = {dh:.2f} m (factor: {size_factor[ii]})")

print()


# ============================================================================
# Compute Semi-Analytic 1D Solution
# ============================================================================

print("=" * 80)
print("Step 5: Computing Semi-Analytic 1D Solution")
print("=" * 80)

simulation_1d = tdem.simulation_1d.Simulation1DLayered(
    survey=generate_survey(stepoff_waveform),
    thicknesses=[],
    sigmaMap=maps.IdentityMap(nP=1),
)

dtrue_stepoff = simulation_1d.dpred(np.array([halfspace_conductivity]))

print(f"  ✓ Analytic solution computed")
print()


# ============================================================================
# Plot Results for Part 1
# ============================================================================

print("=" * 80)
print("Step 6: Plotting Part 1 Results")
print("=" * 80)

fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.85])
ax1.loglog(time_channels, -dtrue_stepoff, "k", lw=3)
for d in dpred_1:
    ax1.loglog(time_channels, -d, "-o", lw=1, markersize=4)
ax1.set_xlim((np.min(time_channels), np.max(time_channels)))
ax1.grid()
ax1.set_xlabel("time [s]")
ax1.set_ylabel("-dB/dz [T/s]")
ax1.set_title("Transient Response vs Cell Size")
ax1.legend(["Analytic"] + ["dh = {} x d_min".format(c) for c in size_factor])
plt.savefig("tdem_fundamentals_part1_cell_size.png", dpi=150, bbox_inches="tight")
print("✓ Part 1 plot saved as 'tdem_fundamentals_part1_cell_size.png'")
plt.close()
print()


# ============================================================================
# PART 2: Discretization in Time (Off-Time)
# ============================================================================

print("=" * 80)
print("PART 2: DISCRETIZATION IN TIME (OFF-TIME)")
print("=" * 80)
print()


# ============================================================================
# Define Minimum Time-Step Lengths
# ============================================================================

print("=" * 80)
print("Step 7: Testing Different Minimum Time-Step Lengths")
print("=" * 80)

t_min = time_channels.min()

size_factor = [5, 10, 20, 40, 80]
dt_min = [t_min / c for c in size_factor]

print(f"  Earliest time channel: {t_min:.2e} s")
print(f"  Testing time-steps: {[f't_min/{c}' for c in size_factor]}")
print()


# ============================================================================
# Compute Numerical Solutions
# ============================================================================

print("=" * 80)
print("Step 8: Computing Numerical Solutions for Different Time-Steps")
print("=" * 80)

dpred_2 = []
simulation_times = []

# Define the mesh, model and mapping
mesh, model, mapping = generate_discretization(d_min, d_min, d_max)

for ii, dt in enumerate(dt_min):
    simulation = tdem.simulation.Simulation3DElectricField(
        mesh,
        survey=generate_survey(stepoff_waveform),
        sigmaMap=mapping,
    )

    # Set the time steps
    time_steps = [
        (dt, int(1.05 * t_min / dt)),
        (5 * dt, int(10 * t_min / (5 * dt) + 1)),
        (25 * dt, int(100 * t_min / (25 * dt) + 1)),
    ]
    simulation.time_steps = time_steps

    # Store the time steps
    simulation_times.append(simulation.times[1:])

    # Simulate the data
    dpred_2.append(simulation.dpred(model))

    print(f"  [{ii+1}/{len(dt_min)}] N_TIMES: {simulation.nT:4d} | dt_min = t_min/{size_factor[ii]}")

print()


# ============================================================================
# Plot Results for Part 2
# ============================================================================

print("=" * 80)
print("Step 9: Plotting Part 2 Results")
print("=" * 80)

fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_axes([0.05, 0.1, 0.4, 0.85])
for ii, tvec in enumerate(simulation_times):
    ax1.semilogx(tvec, ii * np.ones_like(tvec), "|", markersize=10)
ax1.set_yticklabels([])
ax1.set_title("Time Steps")
ax1.grid()

ax2 = fig.add_axes([0.55, 0.1, 0.4, 0.85])
ax2.loglog(time_channels, -dtrue_stepoff, "k", lw=3)
for d in dpred_2:
    ax2.loglog(time_channels, -d, "-o", lw=1, markersize=3)
ax2.set_xlim((np.min(time_channels), np.max(time_channels)))
ax2.grid()
ax2.set_xlabel("time [s]")
ax2.set_ylabel("-dB/dz [T/s]")
ax2.set_title("Transient Response vs Time-Step Size")
ax2.legend(["Analytic"] + ["t_min / {}".format(c) for c in size_factor])
plt.savefig("tdem_fundamentals_part2_time_steps.png", dpi=150, bbox_inches="tight")
print("✓ Part 2 plot saved as 'tdem_fundamentals_part2_time_steps.png'")
plt.close()
print()


# ============================================================================
# PART 3: Time-Discretization (On-Time)
# ============================================================================

print("=" * 80)
print("PART 3: TIME-DISCRETIZATION (ON-TIME)")
print("=" * 80)
print()


# ============================================================================
# Define On-Time and Time-Steps
# ============================================================================

print("=" * 80)
print("Step 10: Defining On-Time Discretization")
print("=" * 80)

t0 = -1e-3  # Start of the on-time

n_on_time_steps = [10, 20, 40, 80, 160]
on_time_steps_list = [(np.abs(t0) / n, n) for n in n_on_time_steps]

# Off-time steps
off_time_steps = [(5e-07, 40), (2.5e-06, 40), (1.25e-05, 81)]

print(f"  On-time start: {t0} s")
print(f"  Testing on-time steps: {n_on_time_steps}")
print()


# ============================================================================
# Define Waveforms
# ============================================================================

print("=" * 80)
print("Step 11: Defining Triangular and Trapezoidal Waveforms")
print("=" * 80)

triangular_waveform = tdem.sources.TriangularWaveform(
    start_time=t0, peak_time=t0 / 2, off_time=0.0
)

ramp_time = np.abs(t0) / 40
trapezoidal_waveform = tdem.sources.TrapezoidWaveform(
    (t0, t0 + ramp_time), (-ramp_time, 0.0), off_time=0.0
)

print("  ✓ Triangular waveform defined")
print("  ✓ Trapezoidal waveform defined")
print()

# Plot time-steps and waveforms
fig = plt.figure(figsize=(10, 4))

ax1 = fig.add_axes([0.05, 0.1, 0.4, 0.85])
for ii in range(len(on_time_steps_list)):
    t_mesh = TensorMesh([[on_time_steps_list[ii]]], origin=np.array([t0]))
    ax1.plot(t_mesh.nodes_x, ii * np.ones_like(t_mesh.nodes_x), "|", markersize=10)

ax1.set_xticks([-1e-3, -5e-4, 0])
ax1.set_ylim([-0.5, 10])
ax1.set_yticklabels([])
ax1.set_xlabel("Times [s]")
ax1.set_title("Time-Steps (on-times)")
ax1.legend(["dt = {} s".format(x[0]) for x in on_time_steps_list], loc="upper left")
ax1.grid()

plotting_times = np.r_[t_mesh.nodes_x, simulation_times[0]]
ax2 = fig.add_axes([0.55, 0.1, 0.4, 0.85])
ax2.plot(plotting_times, [triangular_waveform.eval(t) for t in plotting_times], "k-")
ax2.plot(plotting_times, [trapezoidal_waveform.eval(t) for t in plotting_times], "k--")
ax2.set_xticks([-1e-3, 0, 1e-3])
ax2.set_xlabel("Times [s]")
ax2.set_ylabel("Current [A]")
ax2.set_title("Waveforms")
ax2.grid()
ax2.legend(["Triangular", "Trapezoidal"], loc="upper right")

plt.savefig("tdem_fundamentals_part3_waveforms.png", dpi=150, bbox_inches="tight")
print("✓ Waveforms plot saved as 'tdem_fundamentals_part3_waveforms.png'")
plt.close()


# ============================================================================
# Generate Mesh for Part 3
# ============================================================================

print("=" * 80)
print("Step 12: Generating Mesh for Part 3")
print("=" * 80)

mesh, model, mapping = generate_discretization(d_min, d_min, d_max)

print(f"  ✓ Mesh generated with {mesh.nC} cells")
print()


# ============================================================================
# Simulate for Triangular Waveform
# ============================================================================

print("=" * 80)
print("Step 13: Simulating Data for Triangular Waveform")
print("=" * 80)

simulation_temp = tdem.simulation_1d.Simulation1DLayered(
    survey=generate_survey(triangular_waveform),
    thicknesses=[],
    sigmaMap=maps.IdentityMap(nP=1),
)

dpred_triangular = [simulation_temp.dpred(np.array([halfspace_conductivity]))]

for ii, w in enumerate(on_time_steps_list):
    full_time_steps = [w] + off_time_steps

    simulation_temp = tdem.simulation.Simulation3DElectricField(
        mesh,
        survey=generate_survey(triangular_waveform),
        sigmaMap=mapping,
        time_steps=full_time_steps,
        t0=t0,
    )

    dpred_triangular.append(simulation_temp.dpred(model))
    print(f"  [{ii+1}/{len(on_time_steps_list)}] On-time steps: {n_on_time_steps[ii]}")

print()


# ============================================================================
# Simulate for Trapezoidal Waveform
# ============================================================================

print("=" * 80)
print("Step 14: Simulating Data for Trapezoidal Waveform")
print("=" * 80)

simulation_temp = tdem.simulation_1d.Simulation1DLayered(
    survey=generate_survey(trapezoidal_waveform),
    thicknesses=[],
    sigmaMap=maps.IdentityMap(nP=1),
)

dpred_trapezoidal = [simulation_temp.dpred(np.array([halfspace_conductivity]))]

for ii, w in enumerate(on_time_steps_list):
    full_time_steps = [w] + off_time_steps

    simulation_temp = tdem.simulation.Simulation3DElectricField(
        mesh,
        survey=generate_survey(trapezoidal_waveform),
        sigmaMap=mapping,
        time_steps=full_time_steps,
        t0=t0,
    )

    dpred_trapezoidal.append(simulation_temp.dpred(model))
    print(f"  [{ii+1}/{len(on_time_steps_list)}] On-time steps: {n_on_time_steps[ii]}")

print()


# ============================================================================
# Plot Results for Part 3
# ============================================================================

print("=" * 80)
print("Step 15: Plotting Part 3 Results")
print("=" * 80)

fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_axes([0.05, 0.1, 0.4, 0.85])
ax1.loglog(time_channels, -dpred_triangular[0], "k", lw=3)
for d in dpred_triangular[1:]:
    ax1.loglog(time_channels, -d, "-o", lw=1, markersize=4)
ax1.set_xlim((np.min(time_channels), np.max(time_channels)))
ax1.grid()
ax1.set_xlabel("time [s]")
ax1.set_ylabel("-dB/dz [T/s]")
ax1.set_title("Triangular Waveform")
ax1.legend(
    ["Analytic"] + ["dt = {} s".format(x[0]) for x in on_time_steps_list],
    loc="upper right",
)

ax2 = fig.add_axes([0.55, 0.1, 0.4, 0.85])
ax2.loglog(time_channels, -dpred_trapezoidal[0], "k", lw=3)
for d in dpred_trapezoidal[1:]:
    ax2.loglog(time_channels, -d, "-o", lw=1, markersize=4)
ax2.set_xlim((np.min(time_channels), np.max(time_channels)))
ax2.grid()
ax2.set_xlabel("time [s]")
ax2.set_ylabel("-dB/dz [T/s]")
ax2.set_title("Trapezoidal Waveform")
ax2.legend(
    ["Analytic"] + ["dt = {} s".format(x[0]) for x in on_time_steps_list],
    loc="upper right",
)

plt.savefig("tdem_fundamentals_part3_results.png", dpi=150, bbox_inches="tight")
print("✓ Part 3 results plot saved as 'tdem_fundamentals_part3_results.png'")
plt.close()
print()


# ============================================================================
# Summary
# ============================================================================

print("=" * 80)
print("TUTORIAL COMPLETE")
print("=" * 80)
print()
print("Summary of Key Findings:")
print("-" * 80)
print("Part 1: Spatial Discretization")
print(f"  - Minimum diffusion distance: {d_min:.2f} m")
print(f"  - Recommended: dh_min ≤ d_min for accurate results")
print()
print("Part 2: Temporal Discretization (Off-Time)")
print(f"  - Earliest time channel: {t_min:.2e} s")
print(f"  - Recommended: dt_min ≤ t_min/20 for accurate results")
print()
print("Part 3: Temporal Discretization (On-Time)")
print("  - Triangular waveform: coarser discretization acceptable")
print("  - Trapezoidal waveform: finer discretization needed for ramps")
print()
print("Generated Files:")
print("-" * 80)
print("  1. tdem_fundamentals_part1_cell_size.png    - Spatial discretization impact")
print("  2. tdem_fundamentals_part2_time_steps.png   - Temporal discretization (off-time)")
print("  3. tdem_fundamentals_part3_waveforms.png    - Waveform shapes and discretization")
print("  4. tdem_fundamentals_part3_results.png      - On-time discretization impact")
print()
print("Key Guidelines:")
print("-" * 80)
print("  1. Cell size should be ≤ minimum diffusion distance")
print("  2. Time-step size should be 5-10% of earliest time channel")
print("  3. Increase time-step size by factor of 3-6 each decade")
print("  4. On-time discretization depends on waveform features")
print("  5. Balance accuracy with computational efficiency")
print("=" * 80)
