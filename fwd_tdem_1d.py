"""
1D Forward Simulation for a Single TDEM Sounding
University of British Columbia

This tutorial teaches basic functionality within SimPEG and demonstrates:
- The fundamentals of simulating TDEM data with SimPEG
- Understanding TDEM survey creation (receivers, sources, waveforms)
- Defining Earth's electrical properties (conductivity or resistivity)
- Ways to define 1D layered Earth models
- Simulating TDEM data for different transmitter waveforms
- Simulating TDEM data for dispersive electromagnetic properties (IP, SPM)

The tutorial is organized into 3 parts:
Part 1: Step-off response for a conductive layer
Part 2: Magnetic permeability and dispersive physical properties
Part 3: Simulation for different waveforms

Keywords: TDEM, forward simulation, waveforms, 1D sounding, wires mapping
"""

# ============================================================================
# Import Modules
# ============================================================================

# SimPEG functionality
import simpeg.electromagnetics.time_domain as tdem
from simpeg import maps
from simpeg.utils import plot_1d_layer_model

# Common Python functionality
import os
import numpy as np
from scipy.constants import mu_0
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({"font.size": 14})

write_output = False  # Set to True to save outputs


# ============================================================================
# PART 1: Step-Off Response for a Conductive Layer
# ============================================================================

print("=" * 80)
print("PART 1: STEP-OFF RESPONSE FOR A CONDUCTIVE LAYER")
print("=" * 80)
print()


# ============================================================================
# Define the Survey
# ============================================================================

print("=" * 80)
print("Step 1: Defining Survey")
print("=" * 80)

# Source properties
source_location = np.array([0.0, 0.0, 1.0])  # (3, ) numpy.array_like
source_orientation = "z"  # "x", "y" or "z"
source_current = 1.0  # maximum on-time current (A)
source_radius = 10.0  # source loop radius (m)

# Receiver properties
receiver_locations = np.array([0.0, 0.0, 1.0])  # or (N, 3) numpy.ndarray
receiver_orientation = "z"  # "x", "y" or "z"
times = np.logspace(-5, -2, 31)  # time channels (s)

# Define a step-off waveform
stepoff_waveform = tdem.sources.StepOffWaveform(off_time=0.0)

# Define receiver list
receiver_list = []
receiver_list.append(
    tdem.receivers.PointMagneticFluxDensity(
        receiver_locations, times, orientation=receiver_orientation
    )
)

# Define source list
source_list = [
    tdem.sources.CircularLoop(
        receiver_list=receiver_list,
        location=source_location,
        waveform=stepoff_waveform,
        current=source_current,
        radius=source_radius,
    )
]

# Define the survey
survey = tdem.Survey(source_list)

print(f"  Number of data points: {survey.nD}")
print(f"  Source location: {source_location}")
print(f"  Source radius: {source_radius} m")
print(f"  Receiver orientation: {receiver_orientation}")
print(f"  Time channels: {len(times)}")
print()


# ============================================================================
# Define a 1D Layered Earth Model
# ============================================================================

print("=" * 80)
print("Step 2: Defining 1D Layered Earth Model")
print("=" * 80)

# Layer conductivities
layer_conductivities = np.r_[0.1, 1.0, 0.1]

# Layer thicknesses
layer_thicknesses = np.r_[40.0, 40.0]

# Number of layers
n_layers = len(layer_conductivities)

print(f"  Number of layers: {n_layers}")
print(f"  Layer conductivities: {layer_conductivities} S/m")
print(f"  Layer thicknesses: {layer_thicknesses} m")
print()

# Plot the 1D model
fig = plt.figure(figsize=(4, 5))
ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax1 = plot_1d_layer_model(layer_thicknesses, layer_conductivities, scale="log", ax=ax1)
ax1.grid(which="both")
ax1.set_xlabel(r"Conductivity ($S/m$)")
plt.savefig("tdem_1d_model.png", dpi=150, bbox_inches="tight")
print("✓ 1D model plot saved as 'tdem_1d_model.png'")
plt.close()


# ============================================================================
# Define Models and Mappings
# ============================================================================

print("=" * 80)
print("Step 3: Defining Models and Mappings")
print("=" * 80)

# Conductivity model
conductivity_model = layer_conductivities.copy()
conductivity_map = maps.IdentityMap(nP=n_layers)

# Parametric model (thicknesses + log-resistivities)
parametric_model = np.r_[layer_thicknesses, np.log(1 / layer_conductivities)]
wire_map = maps.Wires(("thicknesses", n_layers - 1), ("log_resistivity", n_layers))
thicknesses_map = wire_map.thicknesses
log_resistivity_map = maps.ExpMap() * wire_map.log_resistivity

print("  Conductivity model parameters: {}".format(len(conductivity_model)))
print("  Parametric model parameters: {}".format(len(parametric_model)))
print()


# ============================================================================
# Define Forward Simulations
# ============================================================================

print("=" * 80)
print("Step 4: Defining Forward Simulations")
print("=" * 80)

# Conductivity model simulation
simulation_conductivity = tdem.simulation_1d.Simulation1DLayered(
    survey=survey,
    sigmaMap=conductivity_map,
    thicknesses=layer_thicknesses,
)

# Parametric model simulation
simulation_parametric = tdem.simulation_1d.Simulation1DLayered(
    survey=survey,
    rhoMap=log_resistivity_map,
    thicknessesMap=thicknesses_map,
)

print("  Conductivity simulation: defined")
print("  Parametric simulation: defined")
print()


# ============================================================================
# Predict 1D TDEM Data
# ============================================================================

print("=" * 80)
print("Step 5: Simulating TDEM Data")
print("=" * 80)

dpred_conductivity = simulation_conductivity.dpred(conductivity_model)
dpred_parametric = simulation_parametric.dpred(parametric_model)

print(f"  Conductivity model data range: [{dpred_conductivity.min():.4e}, {dpred_conductivity.max():.4e}] T")
print(f"  Parametric model data range: [{dpred_parametric.min():.4e}, {dpred_parametric.max():.4e}] T")
print()

# Plot the results
fig = plt.figure(figsize=(5, 6))
ax = fig.add_axes([0.2, 0.15, 0.75, 0.78])
ax.loglog(times, dpred_conductivity, "b-", lw=3)
ax.loglog(times, dpred_parametric, "r--", lw=3)
ax.set_xlim([times.min(), times.max()])
ax.grid()
ax.set_xlabel("Times (s)")
ax.set_ylabel("B (T)")
ax.set_title("Magnetic Flux Density")
ax.legend(["Conductivity model", "Parametric model"])
plt.savefig("tdem_1d_part1_response.png", dpi=150, bbox_inches="tight")
print("✓ Part 1 response plot saved as 'tdem_1d_part1_response.png'")
plt.close()


# ============================================================================
# PART 2: Magnetic Permeability and Dispersive Physical Properties
# ============================================================================

print()
print("=" * 80)
print("PART 2: MAGNETIC PERMEABILITY AND DISPERSIVE PROPERTIES")
print("=" * 80)
print()


# ============================================================================
# Case 1: Magnetically Permeable Layer
# ============================================================================

print("=" * 80)
print("Step 6: Case 1 - Magnetically Permeable Layer")
print("=" * 80)

layer_susceptibilities = np.r_[0.0, 9.0, 0.0]
layer_permeabilities = mu_0 * (1 + layer_susceptibilities)

simulation_permeable = tdem.simulation_1d.Simulation1DLayered(
    survey=survey,
    sigmaMap=conductivity_map,
    thicknesses=layer_thicknesses,
    mu=layer_permeabilities,
)

dpred_permeable = simulation_permeable.dpred(conductivity_model)

print(f"  Layer susceptibilities: {layer_susceptibilities}")
print(f"  Data range: [{dpred_permeable.min():.4e}, {dpred_permeable.max():.4e}] T")
print()


# ============================================================================
# Case 2: Chargeable Layer (Induced Polarization)
# ============================================================================

print("=" * 80)
print("Step 7: Case 2 - Chargeable Layer (IP)")
print("=" * 80)

eta = 0.5  # intrinsic chargeability [0, 1]
tau = 0.001  # central time-relaxation constant in seconds
c = 0.8  # phase constant [0, 1]

layer_eta = np.r_[1e-10, eta, 1e-10]
layer_tau = np.r_[0.0, tau, 0.0]
layer_c = np.r_[0.0, c, 0.0]

simulation_ip = tdem.Simulation1DLayered(
    survey=survey,
    thicknesses=layer_thicknesses,
    sigmaMap=conductivity_map,
    eta=layer_eta,
    tau=layer_tau,
    c=layer_c,
)

dpred_ip = simulation_ip.dpred(conductivity_model)

print(f"  Chargeability (eta): {eta}")
print(f"  Time constant (tau): {tau} s")
print(f"  Phase constant (c): {c}")
print(f"  Data range: [{dpred_ip.min():.4e}, {dpred_ip.max():.4e}] T")
print()


# ============================================================================
# Case 3: Superparamagnetic Layer (SPM)
# ============================================================================

print("=" * 80)
print("Step 8: Case 3 - Superparamagnetic Layer (SPM)")
print("=" * 80)

chi = 0.01  # infinite susceptibility in SI
dchi = 0.01  # amplitude of frequency-dependent susceptibility contribution
tau1 = 1e-7  # lower limit for time relaxation constants in seconds
tau2 = 1.0  # upper limit for time relaxation constants in seconds

layer_mu = mu_0 * (1 + chi * np.r_[1.0, 0.0, 0.0])
layer_dchi = chi * np.r_[1.0, 0.0, 0.0]
layer_tau1 = tau1 * np.r_[1.0, 0.0, 0.0]
layer_tau2 = tau1 * np.r_[1.0, 0.0, 0.0]

simulation_spm = tdem.Simulation1DLayered(
    survey=survey,
    thicknesses=layer_thicknesses,
    sigmaMap=conductivity_map,
    mu=mu_0,
    dchi=dchi,
    tau1=tau1,
    tau2=tau2,
)

dpred_spm = simulation_spm.dpred(conductivity_model)

print(f"  Susceptibility (chi): {chi} SI")
print(f"  Delta chi (dchi): {dchi} SI")
print(f"  Tau1: {tau1} s, Tau2: {tau2} s")
print(f"  Data range: [{dpred_spm.min():.4e}, {dpred_spm.max():.4e}] T")
print()

# Plot all cases
fig = plt.figure(figsize=(5, 6))
ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.85])
ax1.loglog(times, np.abs(dpred_conductivity[0 : len(times)]), "k", lw=2)
ax1.loglog(times, np.abs(dpred_permeable[0 : len(times)]), "r", lw=2)
ax1.loglog(times, np.abs(dpred_ip[0 : len(times)]), "b", lw=2)
ax1.loglog(times, np.abs(dpred_spm[0 : len(times)]), "g", lw=2)
ax1.set_xlim([times.min(), times.max()])
ax1.grid()
ax1.legend(
    [
        "Conductive layer",
        "Conductive and permeable",
        "Conductive and chargeable",
        "Superparamagnetic case",
    ]
)
ax1.set_xlabel("Times (s)")
ax1.set_ylabel("B (T)")
ax1.set_title("Magnetic Flux Density")
plt.savefig("tdem_1d_part2_comparison.png", dpi=150, bbox_inches="tight")
print("✓ Part 2 comparison plot saved as 'tdem_1d_part2_comparison.png'")
plt.close()


# ============================================================================
# PART 3: Simulation for Different Waveforms
# ============================================================================

print()
print("=" * 80)
print("PART 3: SIMULATION FOR DIFFERENT WAVEFORMS")
print("=" * 80)
print()


# ============================================================================
# Define Different Waveforms
# ============================================================================

print("=" * 80)
print("Step 9: Defining Different Waveforms")
print("=" * 80)

# Rectangular waveform
eps = 1e-6
ramp_on = np.r_[-0.004, -0.004 + eps]
ramp_off = np.r_[-eps, 0.0]
rectangular_waveform = tdem.sources.TrapezoidWaveform(
    ramp_on=ramp_on, ramp_off=ramp_off
)

# Triangular waveform
eps = 1e-8
start_time = -0.02
peak_time = -0.01
off_time = 0.0
triangle_waveform = tdem.sources.TriangularWaveform(
    start_time=start_time, peak_time=peak_time, off_time=off_time
)

# General waveform
def custom_waveform(t, tmax):
    out = np.cos(0.5 * np.pi * (t - tmax) / (tmax + 0.02))
    out[t >= tmax] = 1 + (t[t >= tmax] - tmax) / tmax
    return out


waveform_times = np.r_[np.linspace(-0.02, -0.011, 10), -np.logspace(-2, -6, 61), 0.0]
waveform_current = custom_waveform(waveform_times, -0.0055)
general_waveform = tdem.sources.PiecewiseLinearWaveform(
    times=waveform_times, currents=waveform_current
)

print("  Step-off waveform: defined")
print("  Rectangular waveform: defined")
print("  Triangular waveform: defined")
print("  General waveform: defined")
print()

# Plot the waveforms
fig = plt.figure(figsize=(6, 4))
ax = fig.add_axes([0.1, 0.1, 0.85, 0.8])

ax.plot(np.r_[-2e-2, 0.0, 1e-10, 1e-3], np.r_[1.0, 1.0, 0.0, 0.0], "k", lw=2)
plotting_current = [rectangular_waveform.eval(t) for t in waveform_times]
ax.plot(waveform_times, plotting_current, "b", lw=2)
plotting_current = [triangle_waveform.eval(t) for t in waveform_times]
ax.plot(waveform_times, plotting_current, "r", lw=2)
plotting_current = [general_waveform.eval(t) for t in waveform_times]
ax.plot(waveform_times, plotting_current, "g", lw=2)

ax.grid()
ax.set_xlim([waveform_times.min(), 1e-3])
ax.set_xlabel("Time (s)")
ax.set_ylabel("Current (A)")
ax.set_title("Waveforms")
ax.legend(["Step-off", "Rectangular", "Triangle", "General"], loc="lower left")
plt.savefig("tdem_1d_waveforms.png", dpi=150, bbox_inches="tight")
print("✓ Waveforms plot saved as 'tdem_1d_waveforms.png'")
plt.close()


# ============================================================================
# Design Survey with Multiple Sources
# ============================================================================

print("=" * 80)
print("Step 10: Designing Survey with Multiple Sources")
print("=" * 80)

waveforms_list = [
    stepoff_waveform,
    rectangular_waveform,
    triangle_waveform,
    general_waveform,
]

source_list_multi = []

for w in waveforms_list:
    receiver_list_multi = [
        tdem.receivers.PointMagneticFluxDensity(
            receiver_locations, times, orientation=receiver_orientation
        )
    ]

    source_list_multi.append(
        tdem.sources.CircularLoop(
            receiver_list=receiver_list_multi,
            location=source_location,
            waveform=w,
            current=source_current,
            radius=source_radius,
        )
    )

# Define the survey
survey_multi = tdem.Survey(source_list_multi)

print(f"  Number of sources: {len(source_list_multi)}")
print(f"  Total data points: {survey_multi.nD}")
print()


# ============================================================================
# Simulate Data for Multiple Waveforms
# ============================================================================

print("=" * 80)
print("Step 11: Simulating Data for Multiple Waveforms")
print("=" * 80)

simulation_waveforms = tdem.Simulation1DLayered(
    survey=survey_multi, thicknesses=layer_thicknesses, sigmaMap=conductivity_map
)

dpred_waveforms = simulation_waveforms.dpred(conductivity_model)

print(f"  Total simulated data: {len(dpred_waveforms)}")
print(f"  Data range: [{dpred_waveforms.min():.4e}, {dpred_waveforms.max():.4e}] T")
print()

# Plot the results
fig = plt.figure(figsize=(5, 6))
d = np.reshape(dpred_waveforms, (len(source_list_multi), len(times))).T
ax = fig.add_axes([0.15, 0.15, 0.8, 0.75])
colorlist = ["k", "b", "r", "g"]
for ii, k in enumerate(colorlist):
    ax.loglog(times, np.abs(d[:, ii]), k, lw=2)

ax.set_xlim([times.min(), times.max()])
ax.grid()
ax.legend(["Step-off", "Rectangular", "Triangle", "General"])
ax.set_xlabel("Times (s)")
ax.set_ylabel("B (T)")
ax.set_title("Magnetic Flux Density")
plt.savefig("tdem_1d_part3_waveforms.png", dpi=150, bbox_inches="tight")
print("✓ Part 3 waveforms plot saved as 'tdem_1d_part3_waveforms.png'")
plt.close()


# ============================================================================
# Optional: Export Data
# ============================================================================

if write_output:
    print("=" * 80)
    print("Step 12: Exporting Data")
    print("=" * 80)

    dir_path = os.path.sep.join([".", "fwd_tdem_1d_outputs"]) + os.path.sep
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print(f"  Created directory: {dir_path}")

    rng = np.random.default_rng(seed=347)
    noise = rng.normal(
        scale=0.05 * np.abs(dpred_conductivity),
        size=len(dpred_conductivity),
    )
    dpred_conductivity += noise
    fname = dir_path + "em1dtm_data.txt"
    np.savetxt(fname, np.c_[times, dpred_conductivity], fmt="%.4e", header="TIME BZ")
    print(f"  ✓ Data saved to: {fname}")
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
print(f"  Number of layers:    {n_layers}")
print(f"  Layer conductivity:  {layer_conductivities} S/m")
print(f"  Time channels:       {len(times)}")
print(f"  Waveforms tested:    4 (step-off, rectangular, triangular, general)")
print()
print("Generated Files:")
print("-" * 80)
print("  1. tdem_1d_model.png               - 1D layered Earth model")
print("  2. tdem_1d_part1_response.png      - Part 1: Step-off response")
print("  3. tdem_1d_part2_comparison.png    - Part 2: Dispersive properties comparison")
print("  4. tdem_1d_waveforms.png           - Part 3: Waveform shapes")
print("  5. tdem_1d_part3_waveforms.png     - Part 3: Response for different waveforms")
if write_output:
    print("  6. fwd_tdem_1d_outputs/em1dtm_data.txt")
print()
print("Key Concepts:")
print("-" * 80)
print("  - 1D TDEM simulations use semi-analytic Hankel transform solutions")
print("  - Models can be defined as conductivities or resistivities")
print("  - Dispersive properties (IP, SPM) can be included in simulations")
print("  - Different waveforms produce different transient responses")
print("=" * 80)
