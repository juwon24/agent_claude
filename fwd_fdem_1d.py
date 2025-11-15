"""
1D Forward Simulation of Frequency Domain EM Data for a Single Sounding
University of British Columbia

This tutorial teaches basic functionality within SimPEG and demonstrates:
- The fundamentals of simulating FDEM data with SimPEG
- Understanding the way in which FDEM surveys are created in SimPEG
- Defining receivers for different field measurements
- Defining controlled sources (dipole, loop, etc.)
- Organizing sources and receivers into a survey object
- Defining the Earth's electrical properties in terms of conductivity OR resistivity
- The ways in which we can define 1D layered Earth models
- Defining appropriate mappings from model parameters to simulation parameters

Keywords: FDEM, forward simulation, 1D sounding, inductive source, wires mapping
"""

# ============================================================================
# Import Modules
# ============================================================================

# SimPEG functionality
import simpeg.electromagnetics.frequency_domain as fdem
from simpeg import maps
from simpeg.utils import plot_1d_layer_model

# Common Python functionality
import os
import numpy as np
from scipy.constants import mu_0
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({"font.size": 14})

save_output = False  # Set to True to save outputs


# ============================================================================
# Define the Survey
# ============================================================================

print("=" * 80)
print("Step 1: Defining FDEM Survey")
print("=" * 80)

# Source properties
frequencies = np.r_[382, 1822, 7970, 35920, 130100]  # frequencies in Hz
source_location = np.array([0.0, 0.0, 30.0])  # (3, ) numpy.array_like
source_orientation = "z"  # "x", "y" or "z"
moment = 1.0  # dipole moment in Am^2

# Receiver properties
receiver_locations = np.array([10.0, 0.0, 30.0])  # or (N, 3) numpy.ndarray
receiver_orientation = "z"  # "x", "y" or "z"
data_type = "ppm"  # "secondary", "total" or "ppm"

print(f"  Source type: Vertical magnetic dipole")
print(f"  Source location: {source_location}")
print(f"  Source moment: {moment} Am²")
print(f"  Receiver type: Magnetic field (secondary)")
print(f"  Receiver orientation: {receiver_orientation}")
print(f"  Data type: {data_type}")
print(f"  Frequencies: {frequencies}")
print()

source_list = []  # create empty list for source objects

# loop over all sources
for freq in frequencies:
    # Define receivers that measure real and imaginary component
    # magnetic field data in ppm.
    receiver_list = []
    receiver_list.append(
        fdem.receivers.PointMagneticFieldSecondary(
            receiver_locations,
            orientation=receiver_orientation,
            data_type=data_type,
            component="real",
        )
    )
    receiver_list.append(
        fdem.receivers.PointMagneticFieldSecondary(
            receiver_locations,
            orientation=receiver_orientation,
            data_type=data_type,
            component="imag",
        )
    )

    # Define a magnetic dipole source at each frequency
    source_list.append(
        fdem.sources.MagDipole(
            receiver_list=receiver_list,
            frequency=freq,
            location=source_location,
            orientation=source_orientation,
            moment=moment,
        )
    )

# Define the FDEM survey
survey = fdem.survey.Survey(source_list)

print(f"  Number of sources: {len(source_list)}")
print(f"  Number of data points: {survey.nD}")
print(f"  Receivers per source: 2 (real + imaginary)")
print()


# ============================================================================
# Define a 1D Layered Earth
# ============================================================================

print("=" * 80)
print("Step 2: Defining 1D Layered Earth Model")
print("=" * 80)

# Define layer thicknesses (m)
layer_thicknesses = np.array([20.0, 40.0])

# Define layer conductivities (S/m)
layer_conductivities = np.r_[0.1, 1.0, 0.1]

# Define layer susceptibilities (SI)
layer_susceptibilities = np.r_[0.0, 4.0, 0.0]

print(f"  Number of layers: {len(layer_conductivities)}")
print(f"  Layer thicknesses: {layer_thicknesses}")
print(f"  Layer conductivities: {layer_conductivities}")
print(f"  Layer susceptibilities: {layer_susceptibilities}")
print()

# Plot 1D layer models
fig = plt.figure(figsize=(8, 5))

ax1 = fig.add_axes([0.1, 0.1, 0.3, 0.8])
ax1 = plot_1d_layer_model(layer_thicknesses, layer_conductivities, scale="log", ax=ax1)
ax1.grid(which="both")
ax1.set_xlabel(r"Conductivity ($S/m$)")

ax2 = fig.add_axes([0.6, 0.1, 0.3, 0.8])
ax2 = plot_1d_layer_model(
    layer_thicknesses, layer_susceptibilities, scale="linear", ax=ax2
)
ax2.grid(which="both")
ax2.set_xlim([-0.05, 1.1 * np.max(layer_susceptibilities)])
ax2.set_xlabel(r"Susceptibility ($SI$)")

plt.savefig("fdem_1d_layer_model.png", dpi=150, bbox_inches="tight")
print("✓ Layer model plot saved as 'fdem_1d_layer_model.png'")
plt.close()


# ============================================================================
# Define Models and Mappings
# ============================================================================

print("=" * 80)
print("Step 3: Defining Models and Mappings")
print("=" * 80)

n_layers = len(layer_conductivities)

# CASE 1: CONDUCTIVITY MODEL
conductivity_model = layer_conductivities.copy()
conductivity_map = maps.IdentityMap()

print("  Case 1: Conductivity model")
print(f"    Model parameters: {len(conductivity_model)}")
print(f"    Model values: {conductivity_model}")

# CASE 2: LOG-RESISTIVITY MODEL
log_resistivity_model = np.log(1 / layer_conductivities)
log_resistivity_map = maps.ExpMap()

print("  Case 2: Log-resistivity model")
print(f"    Model parameters: {len(log_resistivity_model)}")
print(f"    Model values (log-resistivity): {log_resistivity_model}")

# CASE 3: LOG-CONDUCTIVITY, MAGNETIC PERMEABILITY AND LAYER THICKNESSES
parametric_model = np.r_[
    np.log(layer_conductivities), mu_0 * (1 + layer_susceptibilities), layer_thicknesses
]

wire_map = maps.Wires(
    ("log_conductivity", n_layers),
    ("permeability", n_layers),
    ("thicknesses", n_layers - 1),
)
log_conductivity_map = maps.ExpMap() * wire_map.log_conductivity
permeability_map = wire_map.permeability
thicknesses_map = wire_map.thicknesses

print("  Case 3: Parametric model (log-σ, μ, thicknesses)")
print(f"    Model parameters: {len(parametric_model)}")
print(f"    Log-conductivities: {parametric_model[:n_layers]}")
print(f"    Permeabilities: {parametric_model[n_layers:2*n_layers]}")
print(f"    Thicknesses: {parametric_model[2*n_layers:]}")
print()


# ============================================================================
# Define the Forward Simulations
# ============================================================================

print("=" * 80)
print("Step 4: Defining Forward Simulations")
print("=" * 80)

# CASE 1: Conductivity model
simulation_conductivity = fdem.Simulation1DLayered(
    survey=survey,
    thicknesses=layer_thicknesses,
    sigmaMap=conductivity_map,
)

print("  Case 1: Conductivity model simulation")
print("    Simulation type: 1D Layered (Hankel transform)")
print("    Fixed thicknesses: Yes")
print("    Fixed permeability: Yes (non-permeable)")

# CASE 2: Log-resistivity model with fixed magnetic permeability
simulation_log_resistivity = fdem.Simulation1DLayered(
    survey=survey,
    thicknesses=layer_thicknesses,
    rhoMap=log_resistivity_map,
)

print("  Case 2: Log-resistivity model simulation")
print("    Simulation type: 1D Layered (Hankel transform)")
print("    Fixed thicknesses: Yes")
print("    Fixed permeability: Yes (non-permeable)")

# CASE 3: Log-conductivity, magnetic permeability and layer thicknesses
simulation_parametric = fdem.Simulation1DLayered(
    survey=survey,
    thicknessesMap=thicknesses_map,
    sigmaMap=log_conductivity_map,
    muMap=permeability_map,
)

print("  Case 3: Parametric model simulation")
print("    Simulation type: 1D Layered (Hankel transform)")
print("    Fixed thicknesses: No")
print("    Fixed permeability: No")
print()


# ============================================================================
# Predict FDEM Data
# ============================================================================

print("=" * 80)
print("Step 5: Predicting FDEM Data")
print("=" * 80)

dpred_conductivity = simulation_conductivity.dpred(conductivity_model)
dpred_log_resistivity = simulation_log_resistivity.dpred(log_resistivity_model)
dpred_parametric = simulation_parametric.dpred(parametric_model)

print(f"  Conductivity model predictions:")
print(f"    Data range: [{dpred_conductivity.min():.2f}, {dpred_conductivity.max():.2f}] ppm")

print(f"  Log-resistivity model predictions:")
print(f"    Data range: [{dpred_log_resistivity.min():.2f}, {dpred_log_resistivity.max():.2f}] ppm")

print(f"  Parametric model predictions:")
print(f"    Data range: [{dpred_parametric.min():.2f}, {dpred_parametric.max():.2f}] ppm")

print(f"  Max difference (cases 1&2): {np.max(np.abs(dpred_conductivity - dpred_log_resistivity)):.2e} ppm")
print()

# Plot results
ylim = [np.min(dpred_parametric), 1.1 * np.max(dpred_parametric)]

fig = plt.figure(figsize=(10, 5))

ax1 = fig.add_axes([0.1, 0.1, 0.4, 0.85])
ax1.semilogx(frequencies, dpred_conductivity[0::2], "b-o", lw=3, ms=10, label="Conductivity model")
ax1.semilogx(frequencies, dpred_log_resistivity[0::2], "r--s", lw=3, ms=6, label="Log-resistivity model")
ax1.semilogx(frequencies, dpred_parametric[0::2], "g:d", lw=3, ms=6, label="Parametric model")
ax1.grid()
ax1.set_ylim(ylim)
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Hs/Hp (ppm)")
ax1.set_title("Real Component")
ax1.legend(fontsize=10)

ax2 = fig.add_axes([0.55, 0.1, 0.4, 0.85])
ax2.semilogx(frequencies, dpred_conductivity[1::2], "b-o", lw=3, ms=10, label="Conductivity model")
ax2.semilogx(frequencies, dpred_log_resistivity[1::2], "r--s", lw=3, ms=6, label="Log-resistivity model")
ax2.semilogx(frequencies, dpred_parametric[1::2], "g:d", lw=3, ms=6, label="Parametric model")
ax2.set_ylim(ylim)
ax2.grid()
ax2.set_xlabel("Frequency (Hz)")
ax2.set_yticklabels("")
ax2.set_title("Imaginary Component")
ax2.legend(fontsize=10)

plt.savefig("fdem_1d_response.png", dpi=150, bbox_inches="tight")
print("✓ FDEM response plot saved as 'fdem_1d_response.png'")
plt.close()


# ============================================================================
# Optional: Export Data
# ============================================================================

if save_output:
    print("=" * 80)
    print("Step 6: Exporting Data")
    print("=" * 80)

    dir_path = os.path.sep.join([".", "fwd_fdem_1d_outputs"]) + os.path.sep
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print(f"  Created directory: {dir_path}")

    rng = np.random.default_rng(seed=222)
    noise = rng.normal(
        scale=0.05 * np.abs(dpred_conductivity),
        size=len(dpred_conductivity),
    )
    dpred_out = dpred_conductivity + noise

    fname = dir_path + "em1dfm_data.txt"
    np.savetxt(
        fname,
        np.c_[frequencies, dpred_out[0::2], dpred_out[1::2]],
        fmt="%.4e",
        header="FREQUENCY HZ_REAL HZ_IMAG",
    )
    print(f"  ✓ Data saved to: {fname}")
    print(f"  Noise level: 5% of data")
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
print(f"  Number of layers:    {len(layer_conductivities)}")
print(f"  Frequencies:         {len(frequencies)}")
print(f"  Data points:         {survey.nD}")
print(f"  Source type:         Vertical magnetic dipole")
print(f"  Receiver offset:     10 m")
print(f"  Flight height:       30 m")
print()
print("Model Comparisons:")
print("-" * 80)
print(f"  Conductivity vs Log-resistivity max diff: {np.max(np.abs(dpred_conductivity - dpred_log_resistivity)):.2e} ppm")
print("  → Both non-permeable models yield identical results")
print(f"  Parametric model includes magnetic permeability (χ={layer_susceptibilities[1]} SI)")
print("  → Shows significantly different response due to magnetic effects")
print()
print("Generated Files:")
print("-" * 80)
print("  1. fdem_1d_layer_model.png     - 1D layer model (conductivity + susceptibility)")
print("  2. fdem_1d_response.png        - FDEM response curves (real + imaginary)")
if save_output:
    print("  3. fwd_fdem_1d_outputs/em1dfm_data.txt")
print()
print("Key Learning Points:")
print("-" * 80)
print("  - FDEM surveys require defining sources (frequency, location, type)")
print("  - Receivers can measure different components (real/imaginary, x/y/z)")
print("  - Multiple model parameterizations possible (conductivity, resistivity, etc.)")
print("  - Wires mapping allows complex parametric models (σ, μ, thicknesses)")
print("  - Magnetic susceptibility significantly affects FDEM response")
print()
print("Data Organization:")
print("-" * 80)
print("  - Data organized by source, then receiver, then location")
print("  - For this example: [freq1_real, freq1_imag, freq2_real, freq2_imag, ...]")
print("=" * 80)
