"""
Cross-gradient Joint Inversion of Gravity and Magnetic Anomaly Data
University of British Columbia

This tutorial demonstrates:
- Simultaneous inversion of gravity and magnetic data using cross-gradient constraint
- Structural similarity enforcement between density and susceptibility models
- Joint inverse problem formulation with coupling term
- Cross-gradient regularization for multi-physics inversion
- Comparing joint vs separate inversions

The cross-gradient constraint ensures that recovered models have structural
similarity, meaning boundaries in density and susceptibility models align.
This is useful when different physical properties respond to the same
geological structures.

Although this tutorial uses gravity and magnetic data, the same approach
applies to other geophysical data types like gradiometry, seismic, EM, etc.

Keywords: cross-gradient, joint inversion, gravity, magnetics, structural coupling
"""

# ============================================================================
# Import Modules
# ============================================================================

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tarfile

from discretize import TensorMesh
from discretize.utils import active_from_xyz
from simpeg.utils import plot2Ddata
from simpeg.potential_fields import gravity, magnetics
from simpeg import (
    maps,
    data,
    data_misfit,
    inverse_problem,
    regularization,
    optimization,
    directives,
    inversion,
    utils,
)

mpl.rcParams.update({"font.size": 12})

# ============================================================================
# Step 1: Download and Extract Data
# ============================================================================

print("=" * 80)
print("Step 1: Downloading and Extracting Data")
print("=" * 80)

data_source = (
    "https://storage.googleapis.com/simpeg/doc-assets/cross_gradient_data.tar.gz"
)

downloaded_data = utils.download(data_source, overwrite=True)

# Unzip the tarfile
tar = tarfile.open(downloaded_data, "r")
tar.extractall()
tar.close()

dir_path = downloaded_data.split(".")[0] + os.path.sep
topo_filename = dir_path + "topo.txt"
model_filename = dir_path + "true_model.txt"

print(f"  Data downloaded and extracted to: {dir_path}")
print(f"  Topography file: {topo_filename}")
print(f"  Model file: {model_filename}")
print()

# ============================================================================
# Step 2: Load and Plot Data
# ============================================================================

print("=" * 80)
print("Step 2: Loading and Plotting Data")
print("=" * 80)

# Load topography
xyz_topo = np.loadtxt(topo_filename)

# Load field data
dobs_grav = np.loadtxt(dir_path + "gravity_data.obs")
dobs_mag = np.loadtxt(dir_path + "magnetic_data.obs")

# Define receiver locations and observed data
receiver_locations = dobs_grav[:, 0:3]
dobs_grav = dobs_grav[:, -1]
dobs_mag = dobs_mag[:, -1]

print(f"  Topography points: {len(xyz_topo)}")
print(f"  Receiver locations: {len(receiver_locations)}")
print(f"  Gravity data range: [{dobs_grav.min():.2e}, {dobs_grav.max():.2e}] mGal")
print(f"  Magnetic data range: [{dobs_mag.min():.2e}, {dobs_mag.max():.2e}] nT")
print()

# Plot gravity data
fig = plt.figure(figsize=(7, 5))
ax1 = fig.add_axes([0.1, 0.1, 0.73, 0.85])
plot2Ddata(receiver_locations, dobs_grav, ax=ax1, contourOpts={"cmap": "bwr"})
ax1.set_title("Gravity Anomaly")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")
ax2 = fig.add_axes([0.8, 0.1, 0.03, 0.85])
norm = mpl.colors.Normalize(
    vmin=-np.max(np.abs(dobs_grav)), vmax=np.max(np.abs(dobs_grav))
)
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.bwr, format="%.1e"
)
cbar.set_label("$mGal$", rotation=270, labelpad=15, size=12)
plt.savefig("cross_gradient_gravity_data.png", dpi=150, bbox_inches="tight")
print("✓ Gravity data plot saved as 'cross_gradient_gravity_data.png'")
plt.close()

# Plot magnetic data
fig = plt.figure(figsize=(7, 5))
ax1 = fig.add_axes([0.1, 0.1, 0.73, 0.85])
plot2Ddata(receiver_locations, dobs_mag, ax=ax1, contourOpts={"cmap": "bwr"})
ax1.set_title("Magnetic Anomaly")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")
ax2 = fig.add_axes([0.8, 0.1, 0.03, 0.85])
norm = mpl.colors.Normalize(
    vmin=-np.max(np.abs(dobs_mag)), vmax=np.max(np.abs(dobs_mag))
)
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.bwr, format="%.1e"
)
cbar.set_label("$nT$", rotation=270, labelpad=15, size=12)
plt.savefig("cross_gradient_magnetic_data.png", dpi=150, bbox_inches="tight")
print("✓ Magnetic data plot saved as 'cross_gradient_magnetic_data.png'")
plt.close()

# ============================================================================
# Step 3: Assign Uncertainties
# ============================================================================

print("=" * 80)
print("Step 3: Assigning Uncertainties")
print("=" * 80)

maximum_anomaly_grav = np.max(np.abs(dobs_grav))
uncertainties_grav = 0.01 * maximum_anomaly_grav * np.ones(np.shape(dobs_grav))

maximum_anomaly_mag = np.max(np.abs(dobs_mag))
uncertainties_mag = 0.01 * maximum_anomaly_mag * np.ones(np.shape(dobs_mag))

print(f"  Gravity uncertainty: {uncertainties_grav[0]:.4e} mGal (1% of max)")
print(f"  Magnetic uncertainty: {uncertainties_mag[0]:.4e} nT (1% of max)")
print()

# ============================================================================
# Step 4: Define Surveys
# ============================================================================

print("=" * 80)
print("Step 4: Defining Surveys")
print("=" * 80)

# Gravity survey
receiver_grav = gravity.receivers.Point(receiver_locations, components="gz")
source_field_grav = gravity.sources.SourceField(receiver_list=[receiver_grav])
survey_grav = gravity.survey.Survey(source_field_grav)

# Magnetic survey
components = ["tmi"]
receiver_mag = magnetics.receivers.Point(receiver_locations, components=components)

inclination = 90
declination = 0
strength = 50000

source_field_mag = magnetics.sources.UniformBackgroundField(
    receiver_list=[receiver_mag],
    amplitude=strength,
    declination=declination,
    inclination=inclination,
)
survey_mag = magnetics.survey.Survey(source_field_mag)

print(f"  Gravity survey: {survey_grav.nD} data points")
print(f"  Magnetic survey: {survey_mag.nD} data points")
print(f"  Magnetic field: inclination={inclination}°, declination={declination}°, strength={strength} nT")
print()

# ============================================================================
# Step 5: Define Data Objects
# ============================================================================

print("=" * 80)
print("Step 5: Creating Data Objects")
print("=" * 80)

data_object_grav = data.Data(
    survey_grav, dobs=dobs_grav, standard_deviation=uncertainties_grav
)
data_object_mag = data.Data(
    survey_mag, dobs=dobs_mag, standard_deviation=uncertainties_mag
)

print(f"  Gravity data object created")
print(f"  Magnetic data object created")
print()

# ============================================================================
# Step 6: Define Tensor Mesh
# ============================================================================

print("=" * 80)
print("Step 6: Defining Tensor Mesh")
print("=" * 80)

dh = 5.0
hx = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
hy = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
hz = [(dh, 5, -1.3), (dh, 15)]
mesh = TensorMesh([hx, hy, hz], "CCN")

print(f"  Mesh created")
print(f"  Number of cells: {mesh.nC}")
print(f"  Cell size: {dh} m")
print(f"  Mesh dimensions: {mesh.shape_cells}")
print()

# ============================================================================
# Step 7: Define Active Cells and Mapping
# ============================================================================

print("=" * 80)
print("Step 7: Defining Active Cells and Mapping")
print("=" * 80)

background_dens, background_susc = 1e-6, 1e-6

# Find the active cells
ind_active = active_from_xyz(mesh, xyz_topo)
nC = int(ind_active.sum())

# Define mapping from model to active cells
model_map = maps.IdentityMap(nP=nC)

# Create Wires Map
wires = maps.Wires(("density", nC), ("susceptibility", nC))

# Define starting model
starting_model = np.r_[background_dens * np.ones(nC), background_susc * np.ones(nC)]

print(f"  Total mesh cells: {mesh.nC}")
print(f"  Active cells: {nC}")
print(f"  Inactive cells: {mesh.nC - nC}")
print(f"  Model parameters: {len(starting_model)} (density + susceptibility)")
print()

# ============================================================================
# Step 8: Define Physics (Simulations)
# ============================================================================

print("=" * 80)
print("Step 8: Defining Physics")
print("=" * 80)

simulation_grav = gravity.simulation.Simulation3DIntegral(
    survey=survey_grav,
    mesh=mesh,
    rhoMap=wires.density,
    active_cells=ind_active,
    engine="choclo",
)

simulation_mag = magnetics.simulation.Simulation3DIntegral(
    survey=survey_mag,
    mesh=mesh,
    model_type="scalar",
    chiMap=wires.susceptibility,
    active_cells=ind_active,
    engine="choclo",
)

print(f"  Gravity simulation created (engine: choclo)")
print(f"  Magnetic simulation created (engine: choclo)")
print()

# ============================================================================
# Step 9: Define Inverse Problem Components
# ============================================================================

print("=" * 80)
print("Step 9: Defining Inverse Problem Components")
print("=" * 80)

# Data misfits
dmis_grav = data_misfit.L2DataMisfit(data=data_object_grav, simulation=simulation_grav)
dmis_mag = data_misfit.L2DataMisfit(data=data_object_mag, simulation=simulation_mag)

# Regularization
reg_grav = regularization.WeightedLeastSquares(
    mesh, active_cells=ind_active, mapping=wires.density
)
reg_mag = regularization.WeightedLeastSquares(
    mesh, active_cells=ind_active, mapping=wires.susceptibility
)

# Cross-gradient coupling term
lamda = 2e12  # Weight for coupling term
cross_grad = regularization.CrossGradient(mesh, wires, active_cells=ind_active)

# Combined objective functions
dmis = dmis_grav + dmis_mag
reg = reg_grav + reg_mag + lamda * cross_grad

print(f"  Data misfits created: gravity + magnetic")
print(f"  Regularization created: gravity + magnetic + cross-gradient")
print(f"  Cross-gradient coupling weight: {lamda:.2e}")
print()

# ============================================================================
# Step 10: Define Optimization
# ============================================================================

print("=" * 80)
print("Step 10: Defining Optimization")
print("=" * 80)

opt = optimization.ProjectedGNCG(
    maxIter=10,
    lower=-2.0,
    upper=2.0,
    maxIterLS=20,
    maxIterCG=100,
    tolCG=1e-3,
    tolX=1e-3,
)

# Create inverse problem
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

print(f"  Optimization: Projected GNCG")
print(f"  Max iterations: {opt.maxIter}")
print(f"  Bounds: [{opt.lower}, {opt.upper}]")
print()

# ============================================================================
# Step 11: Define Directives
# ============================================================================

print("=" * 80)
print("Step 11: Defining Inversion Directives")
print("=" * 80)

# Starting beta
starting_beta = directives.PairedBetaEstimate_ByEig(beta0_ratio=1e0)

# Beta schedule
beta_schedule = directives.PairedBetaSchedule(cooling_factor=5, cooling_rate=1)

# Save iteration outputs
save_iteration = directives.SimilarityMeasureSaveOutputEveryIteration(save_txt=False)

# Joint inversion directive
joint_inv_dir = directives.SimilarityMeasureInversionDirective()

# Stopping criteria
stopping = directives.MovingAndMultiTargetStopping(tol=1e-6)

# Sensitivity weights
sensitivity_weights = directives.UpdateSensitivityWeights(every_iteration=False)

# Update preconditioner
update_jacobi = directives.UpdatePreconditioner()

# Directives list
directives_list = [
    joint_inv_dir,
    sensitivity_weights,
    stopping,
    starting_beta,
    beta_schedule,
    save_iteration,
    update_jacobi,
]

print(f"  Directives configured:")
print(f"    - Paired beta estimation")
print(f"    - Beta cooling schedule (factor={beta_schedule.cooling_factor})")
print(f"    - Similarity measure directive")
print(f"    - Sensitivity weighting")
print(f"    - Stopping criteria")
print()

# ============================================================================
# Step 12: Run Inversion
# ============================================================================

print("=" * 80)
print("Step 12: Running Cross-Gradient Joint Inversion")
print("=" * 80)

# Create inversion
inv = inversion.BaseInversion(inv_prob, directives_list)

print(f"  Starting inversion...")
print(f"  This may take several minutes...")
print()

# Run inversion
recovered_model = inv.run(starting_model)

print()
print(f"  Inversion complete!")
print()

# ============================================================================
# Step 13: Extract Results
# ============================================================================

print("=" * 80)
print("Step 13: Extracting Results")
print("=" * 80)

m_dens_joint, m_susc_joint = wires * recovered_model

print(f"  Recovered density range: [{m_dens_joint.min():.4f}, {m_dens_joint.max():.4f}] g/cc")
print(f"  Recovered susceptibility range: [{m_susc_joint.min():.6f}, {m_susc_joint.max():.6f}] SI")
print()

# ============================================================================
# Step 14: Load True Model
# ============================================================================

print("=" * 80)
print("Step 14: Loading True Model for Comparison")
print("=" * 80)

true_model_dens = np.loadtxt(dir_path + "true_model_dens.txt")
true_model_dens[~ind_active] = np.nan

true_model_susc = np.loadtxt(dir_path + "true_model_susc.txt")
true_model_susc[~ind_active] = np.nan

print(f"  True density range: [{np.nanmin(true_model_dens):.4f}, {np.nanmax(true_model_dens):.4f}] g/cc")
print(f"  True susceptibility range: [{np.nanmin(true_model_susc):.6f}, {np.nanmax(true_model_susc):.6f}] SI")
print()

# ============================================================================
# Step 15: Plot True Models
# ============================================================================

print("=" * 80)
print("Step 15: Plotting True Models")
print("=" * 80)

fig = plt.figure(figsize=(9, 8))
ax1 = plt.subplot(211)
(im,) = mesh.plot_slice(true_model_dens, normal="Y", ax=ax1, grid=True)
ax1.set_title("True density model slice at y = 0 m")
cbar = plt.colorbar(im, format="%.1e")
cbar.set_label("g/cc", rotation=270, labelpad=15, size=12)

ax2 = plt.subplot(212)
(im,) = mesh.plot_slice(
    true_model_susc, normal="Y", ax=ax2, grid=True, pcolor_opts={"cmap": "inferno"}
)
ax2.set_title("True susceptibility model slice at y = 0 m")
cbar = plt.colorbar(im, format="%.1e")
cbar.set_label("SI", rotation=270, labelpad=15, size=12)
plt.tight_layout()
plt.savefig("cross_gradient_true_models.png", dpi=150, bbox_inches="tight")
print("✓ True models plot saved as 'cross_gradient_true_models.png'")
plt.close()

# ============================================================================
# Step 16: Plot Recovered Models
# ============================================================================

print("=" * 80)
print("Step 16: Plotting Recovered Models")
print("=" * 80)

plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

fig = plt.figure(figsize=(9, 8))
ax1 = plt.subplot(211)
(im,) = mesh.plot_slice(
    plotting_map * m_dens_joint,
    normal="Y",
    ax=ax1,
    clim=(-0.04, 0.03),
)
ax1.set_title("Recovered density model slice at y = 0 m")
cbar = plt.colorbar(im)
cbar.set_label("g/cc", rotation=270, labelpad=15, size=12)

ax2 = plt.subplot(212)
(im,) = mesh.plot_slice(
    plotting_map * m_susc_joint, normal="Y", ax=ax2, pcolor_opts={"cmap": "inferno"}
)
ax2.set_title("Recovered susceptibility model slice at y = 0 m")
cbar = plt.colorbar(im)
cbar.set_label("SI", rotation=270, labelpad=15, size=12)

plt.tight_layout()
plt.savefig("cross_gradient_recovered_models.png", dpi=150, bbox_inches="tight")
print("✓ Recovered models plot saved as 'cross_gradient_recovered_models.png'")
plt.close()

# ============================================================================
# Step 17: Analyze Cross-Gradient
# ============================================================================

print("=" * 80)
print("Step 17: Analyzing Cross-Gradient")
print("=" * 80)

# Normalized cross-gradient of jointly recovered models
ncg = cross_grad.calculate_cross_gradient(recovered_model, normalized=True)

print(f"  Normalized cross-gradient range: [{ncg.min():.4e}, {ncg.max():.4e}]")
print(f"  Mean normalized cross-gradient: {ncg.mean():.4e}")
print()

fig = plt.figure(figsize=(9, 4))
ax = plt.subplot(111)
(im,) = mesh.plot_slice(
    plotting_map * ncg,
    normal="Y",
    ax=ax,
    grid=True,
)
ax.set_title("Normalized cross-gradient for joint inversion slice at y = 0 m")
cbar = plt.colorbar(im, format="%.1e")
cbar.set_label("|cross grad|", rotation=270, labelpad=15, size=12)
plt.savefig("cross_gradient_joint_ncg.png", dpi=150, bbox_inches="tight")
print("✓ Cross-gradient plot saved as 'cross_gradient_joint_ncg.png'")
plt.close()

# ============================================================================
# Step 18: Compare with Separate Inversions
# ============================================================================

print("=" * 80)
print("Step 18: Comparing with Separate Inversions")
print("=" * 80)

# Load separately inverted models
m_dens_single = np.loadtxt(dir_path + "single_model_dens.txt")
m_susc_single = np.loadtxt(dir_path + "single_model_susc.txt")
m_separate = np.r_[m_dens_single[ind_active], m_susc_single[ind_active]]

ncg_single = cross_grad.calculate_cross_gradient(m_separate, normalized=True)

print(f"  Separate inversion cross-gradient: [{ncg_single.min():.4e}, {ncg_single.max():.4e}]")
print(f"  Joint inversion cross-gradient: [{ncg.min():.4e}, {ncg.max():.4e}]")
print(f"  Improvement: {((ncg_single.mean() - ncg.mean()) / ncg_single.mean() * 100):.1f}% reduction in mean cross-gradient")
print()

fig = plt.figure(figsize=(9, 4))
ax = plt.subplot(111)
(im,) = mesh.plot_slice(
    plotting_map * ncg_single,
    normal="Y",
    ax=ax,
    grid=True,
)
ax.set_title("Normalized cross-gradient for separate inversion slice at y = 0 m")
cbar = plt.colorbar(im, format="%.1e")
cbar.set_label("|cross grad|", rotation=270, labelpad=15, size=12)
plt.savefig("cross_gradient_separate_ncg.png", dpi=150, bbox_inches="tight")
print("✓ Separate inversion cross-gradient plot saved as 'cross_gradient_separate_ncg.png'")
plt.close()

# ============================================================================
# Step 19: Cross-Plots
# ============================================================================

print("=" * 80)
print("Step 19: Creating Cross-Plots")
print("=" * 80)

fig = plt.figure(figsize=(14, 5))
ax0 = plt.subplot(121)
ax0.scatter(
    plotting_map * m_dens_joint, plotting_map * m_susc_joint, s=4, c="black", alpha=0.1
)
ax0.set_xlabel("Density", size=12)
ax0.set_ylabel("Susceptibility", size=12)
ax0.tick_params(labelsize=12)
ax0.set_title("Joint inversion")

ax1 = plt.subplot(122)
ax1.scatter(m_dens_single, m_susc_single, s=4, c="black", alpha=0.1)
ax1.set_xlabel("Density", size=12)
ax1.set_ylabel("Susceptibility", size=12)
ax1.tick_params(labelsize=12)
ax1.set_title("Separate inversion")

plt.savefig("cross_gradient_crossplots.png", dpi=150, bbox_inches="tight")
print("✓ Cross-plots saved as 'cross_gradient_crossplots.png'")
plt.close()

# ============================================================================
# Summary
# ============================================================================

print("=" * 80)
print("TUTORIAL COMPLETE")
print("=" * 80)
print()
print("Summary of Results:")
print("-" * 80)
print(f"  Mesh cells:              {mesh.nC}")
print(f"  Active cells:            {nC}")
print(f"  Gravity data points:     {survey_grav.nD}")
print(f"  Magnetic data points:    {survey_mag.nD}")
print(f"  Recovered density:       [{m_dens_joint.min():.4f}, {m_dens_joint.max():.4f}] g/cc")
print(f"  Recovered susceptibility:[{m_susc_joint.min():.6f}, {m_susc_joint.max():.6f}] SI")
print(f"  Cross-gradient coupling: λ = {lamda:.2e}")
print()
print("Cross-Gradient Analysis:")
print("-" * 80)
print(f"  Joint inversion mean |∇ρ × ∇χ|:      {ncg.mean():.4e}")
print(f"  Separate inversion mean |∇ρ × ∇χ|:   {ncg_single.mean():.4e}")
print(f"  Improvement:                          {((ncg_single.mean() - ncg.mean()) / ncg_single.mean() * 100):.1f}% reduction")
print()
print("Generated Files:")
print("-" * 80)
print("  1. cross_gradient_gravity_data.png     - Gravity anomaly data")
print("  2. cross_gradient_magnetic_data.png    - Magnetic anomaly data")
print("  3. cross_gradient_true_models.png      - True density and susceptibility models")
print("  4. cross_gradient_recovered_models.png - Recovered joint inversion models")
print("  5. cross_gradient_joint_ncg.png        - Normalized cross-gradient (joint)")
print("  6. cross_gradient_separate_ncg.png     - Normalized cross-gradient (separate)")
print("  7. cross_gradient_crossplots.png       - Density vs susceptibility scatter plots")
print()
print("Key Features:")
print("-" * 80)
print("  - Joint inversion with cross-gradient structural coupling")
print("  - Enforced structural similarity between density and susceptibility")
print("  - Lower cross-gradient values indicate better structural alignment")
print("  - Joint inversion produces more geologically consistent models")
print("  - Comparison with separate inversions shows coupling benefit")
print()
print("Note: The cross-gradient constraint ensures that boundaries in the")
print("      density and susceptibility models align, producing models with")
print("      consistent geological structure across different physical properties.")
print("=" * 80)
