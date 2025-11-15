"""
1D Forward Simulation for DC Resistivity Single Sounding
University of British Columbia

This tutorial teaches basic functionality within SimPEG and demonstrates:
- How to simulate DC resistivity data for a 1D Wenner array
- How to define DC resistivity surveys in SimPEG
- Various approaches for defining a 1D layered Earth model
- How the Earth's electrical properties can be defined according to conductivity OR resistivity
- How to simulate data as normalized voltages OR apparent resistivities

Keywords: DC resistivity, forward simulation, apparent resistivity, 1D sounding, wires mapping
"""

# ============================================================================
# Import Modules
# ============================================================================

# SimPEG functionality
from simpeg.electromagnetics.static import resistivity as dc
from simpeg import maps
from simpeg.utils import plot_1d_layer_model

# Common Python functionality
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({"font.size": 16})

save_output = False  # Set to True to save outputs


# ============================================================================
# Define the Survey
# ============================================================================

print("=" * 80)
print("Step 1: Defining DC Resistivity Survey")
print("=" * 80)

# Define the 'a' spacings for Wenner array and number of soundings
a_min = 20.0
a_max = 500.0
n_stations = 25
electrode_separations = np.linspace(a_min, a_max, n_stations)

source_list = []  # create empty array for sources to live
for ii in range(0, len(electrode_separations)):
    # Extract separation parameter for sources and receivers
    a = electrode_separations[ii]

    # AB electrode locations for source. Each is a (1, 3) numpy array
    A_location = np.r_[-1.5 * a, 0.0, 0.0]
    B_location = np.r_[1.5 * a, 0.0, 0.0]

    # MN electrode locations for receivers. Each is an (N, 3) numpy array
    M_location = np.r_[-0.5 * a, 0.0, 0.0]
    N_location = np.r_[0.5 * a, 0.0, 0.0]

    # Create receivers list. Define as pole or dipole.
    receiver_list = dc.receivers.Dipole(
        M_location, N_location, data_type="apparent_resistivity"
    )
    receiver_list = [receiver_list]

    # Define the source properties and associated receivers
    source_list.append(dc.sources.Dipole(receiver_list, A_location, B_location))

# Define survey
survey = dc.Survey(source_list)

print(f"  Number of sources: {survey.nSrc}")
print(f"  Number of data points: {survey.nD}")
print(f"  Data type: apparent_resistivity")
print(f"  Survey type: Wenner array")
print(f"  Electrode separation range: [{a_min}, {a_max}] m")
print()


# ============================================================================
# Define a 1D Layered Earth Model
# ============================================================================

print("=" * 80)
print("Step 2: Defining 1D Layered Earth Model")
print("=" * 80)

# Define layer thicknesses.
layer_thicknesses = np.r_[100.0, 100.0]

# Define layer resistivities.
layer_resistivities = np.r_[1e3, 4e3, 2e2]

print(f"  Number of layers: {len(layer_resistivities)}")
print(f"  Layer thicknesses: {layer_thicknesses}")
print(f"  Layer resistivities: {layer_resistivities}")
print()

# Plot the 1D layer model
ax = plot_1d_layer_model(layer_thicknesses, layer_resistivities)
ax.grid(which="both")
ax.set_xlabel(r"Resistivity ($\Omega m$)")
plt.savefig("dcr_1d_layer_model.png", dpi=150, bbox_inches="tight")
print("✓ Layer model plot saved as 'dcr_1d_layer_model.png'")
plt.close()
print()


# ============================================================================
# Define Models and Mappings
# ============================================================================

print("=" * 80)
print("Step 3: Defining Models and Mappings")
print("=" * 80)

# 1. Resistivity model: Model as layer resistivities with hard-coded thicknesses
resistivity_model = layer_resistivities.copy()
resistivity_map = maps.IdentityMap(nP=len(layer_resistivities))

print("  Resistivity model approach:")
print(f"    Model parameters: {len(resistivity_model)}")
print(f"    Model values: {resistivity_model}")

# 2. Parametric layered Earth model: Model defines thicknesses and log-conductivities
parametric_model = np.r_[layer_thicknesses, np.log(1 / layer_resistivities)]
wire_map = maps.Wires(
    ("thicknesses", len(layer_thicknesses)),
    ("log_conductivity", len(layer_resistivities)),
)
thicknesses_map = wire_map.thicknesses
log_conductivity_map = maps.ExpMap() * wire_map.log_conductivity

print("  Parametric model approach:")
print(f"    Model parameters: {len(parametric_model)}")
print(f"    Thicknesses: {parametric_model[:len(layer_thicknesses)]}")
print(f"    Log-conductivities: {parametric_model[len(layer_thicknesses):]}")
print()


# ============================================================================
# Define the Forward Simulations
# ============================================================================

print("=" * 80)
print("Step 4: Defining Forward Simulations")
print("=" * 80)

# Resistivity model simulation
simulation_resistivity = dc.simulation_1d.Simulation1DLayers(
    survey=survey,
    rhoMap=resistivity_map,
    thicknesses=layer_thicknesses,
)

print("  Resistivity model simulation:")
print("    Simulation type: 1D Layers (Hankel transform solution)")
print("    Model type: Layer resistivities")
print("    Fixed thicknesses: Yes")

# Parametric model simulation
simulation_parametric = dc.simulation_1d.Simulation1DLayers(
    survey=survey,
    sigmaMap=log_conductivity_map,
    thicknessesMap=thicknesses_map,
)

print("  Parametric model simulation:")
print("    Simulation type: 1D Layers (Hankel transform solution)")
print("    Model type: Thicknesses + log-conductivities")
print("    Fixed thicknesses: No")
print()


# ============================================================================
# Predict DC Resistivity Data
# ============================================================================

print("=" * 80)
print("Step 5: Predicting DC Resistivity Data")
print("=" * 80)

dpred_resistivity = simulation_resistivity.dpred(resistivity_model)
dpred_parametric = simulation_parametric.dpred(parametric_model)

print(f"  Resistivity model predictions:")
print(f"    Number of data: {len(dpred_resistivity)}")
print(f"    Data range: [{dpred_resistivity.min():.2f}, {dpred_resistivity.max():.2f}] Ω·m")

print(f"  Parametric model predictions:")
print(f"    Number of data: {len(dpred_parametric)}")
print(f"    Data range: [{dpred_parametric.min():.2f}, {dpred_parametric.max():.2f}] Ω·m")

print(f"  Maximum difference: {np.max(np.abs(dpred_resistivity - dpred_parametric)):.2e} Ω·m")
print()

# Plot sounding curves
fig = plt.figure(figsize=(9, 5))
ax1 = fig.add_axes([0.1, 0.1, 0.75, 0.85])
ax1.semilogy(1.5 * electrode_separations, dpred_resistivity, "b-o", lw=2, ms=8, label="Resistivity Model")
ax1.semilogy(1.5 * electrode_separations, dpred_parametric, "r*", ms=10, label="Parametric Model")
ax1.grid(True, which="both")
ax1.set_xlabel("AB/2 (m)")
ax1.set_ylabel(r"Apparent Resistivity ($\Omega m$)")
ax1.set_title("DC Resistivity Sounding Curve")
ax1.legend()

plt.savefig("dcr_1d_sounding_curve.png", dpi=150, bbox_inches="tight")
print("✓ Sounding curve plot saved as 'dcr_1d_sounding_curve.png'")
plt.close()


# ============================================================================
# Optional: Export Data
# ============================================================================

if save_output:
    print("=" * 80)
    print("Step 6: Exporting Data")
    print("=" * 80)

    dir_path = os.path.sep.join([".", "fwd_dcr_1d_outputs"]) + os.path.sep
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print(f"  Created directory: {dir_path}")

    rng = np.random.default_rng(seed=145)
    noise = rng.normal(scale=0.025 * dpred_resistivity, size=len(dpred_resistivity))

    data_array = np.c_[
        survey.locations_a,
        survey.locations_b,
        survey.locations_m,
        survey.locations_n,
        dpred_resistivity + noise,
    ]

    fname = dir_path + "app_res_1d_data.dobs"
    np.savetxt(fname, data_array, fmt="%.4e")
    print(f"  ✓ Data saved to: {fname}")
    print(f"  Noise level: 2.5% of data")
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
print(f"  Survey type:         Wenner array (1D sounding)")
print(f"  Number of layers:    {len(layer_resistivities)}")
print(f"  Data points:         {survey.nD}")
print(f"  Electrode spacing:   [{a_min:.1f}, {a_max:.1f}] m")
print(f"  Resistivity range:   [{dpred_resistivity.min():.2f}, {dpred_resistivity.max():.2f}] Ω·m")
print()
print("Generated Files:")
print("-" * 80)
print("  1. dcr_1d_layer_model.png      - 1D layer model visualization")
print("  2. dcr_1d_sounding_curve.png   - Apparent resistivity sounding curve")
if save_output:
    print("  3. fwd_dcr_1d_outputs/app_res_1d_data.dobs")
print()
print("Key Learning Points:")
print("-" * 80)
print("  - Two approaches demonstrated: resistivity model vs. parametric model")
print("  - Both models produce identical results for the same Earth structure")
print("  - Parametric model allows optimization of layer thicknesses")
print("  - SimPEG supports both conductivity and resistivity definitions")
print("=" * 80)
