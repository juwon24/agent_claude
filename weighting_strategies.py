"""
Compare Weighting Strategies with Inversion of Surface Gravity Anomaly Data
University of British Columbia

This tutorial demonstrates:
- Comparing different regularization weighting strategies
- Depth weighting to counteract kernel decay with distance
- Distance weighting as an alternative to depth weighting
- Sensitivity weighting based on the Jacobian matrix
- Visualizing the impact of different weighting schemes on inversion results
- IRLS (Iteratively Reweighted Least Squares) optimization
- Setting sparse and blocky norms for regularization

The tutorial inverts gravity anomaly data to recover a density contrast model
using three different weighting strategies. Although we use gravity data here,
the same approach applies to magnetics, gradiometry, and other geophysical methods.

Keywords: gravity inversion, depth weighting, distance weighting, sensitivity weighting, IRLS
"""

# ============================================================================
# Import Modules
# ============================================================================

import os
import tarfile
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from discretize import TensorMesh
from discretize.utils import active_from_xyz
from simpeg import (
    data,
    data_misfit,
    directives,
    inverse_problem,
    inversion,
    maps,
    optimization,
    regularization,
    utils,
)
from simpeg.potential_fields import gravity
from simpeg.utils import model_builder, plot2Ddata

mpl.rcParams.update({"font.size": 12})

# ============================================================================
# Step 1: Download and Extract Data
# ============================================================================

print("=" * 80)
print("Step 1: Downloading and Extracting Data")
print("=" * 80)

data_source = "https://storage.googleapis.com/simpeg/doc-assets/gravity.tar.gz"
downloaded_data = utils.download(data_source, overwrite=True)

# Unzip the tarfile
tar = tarfile.open(downloaded_data, "r")
tar.extractall()
tar.close()

dir_path = downloaded_data.split(".")[0] + os.path.sep
topo_filename = dir_path + "gravity_topo.txt"
data_filename = dir_path + "gravity_data.obs"

print(f"  Data downloaded and extracted to: {dir_path}")
print(f"  Topography file: {topo_filename}")
print(f"  Data file: {data_filename}")
print()

# ============================================================================
# Step 2: Load and Plot Data
# ============================================================================

print("=" * 80)
print("Step 2: Loading and Plotting Data")
print("=" * 80)

# Load topography
xyz_topo = np.loadtxt(str(topo_filename))

# Load field data
dobs = np.loadtxt(str(data_filename))

# Define receiver locations and observed data
receiver_locations = dobs[:, 0:3]
dobs = dobs[:, -1]

print(f"  Topography points: {len(xyz_topo)}")
print(f"  Data points: {len(dobs)}")
print(f"  Data range: [{dobs.min():.4e}, {dobs.max():.4e}] mGal")
print()

# Plot
fig = plt.figure(figsize=(7, 5))
ax1 = fig.add_axes([0.1, 0.1, 0.73, 0.85])
plot2Ddata(
    receiver_locations,
    dobs,
    ax=ax1,
    contourOpts={"cmap": "bwr"},
    shade=True,
    nx=20,
    ny=20,
    dataloc=True,
)
ax1.set_title("Gravity Anomaly")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")

ax2 = fig.add_axes([0.8, 0.1, 0.03, 0.85])
norm = mpl.colors.Normalize(vmin=-np.max(np.abs(dobs)), vmax=np.max(np.abs(dobs)))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.bwr, format="%.1e"
)
cbar.set_label("$mGal$", rotation=270, labelpad=15, size=12)
plt.savefig("weighting_data.png", dpi=150, bbox_inches="tight")
print("✓ Data plot saved as 'weighting_data.png'")
plt.close()

# ============================================================================
# Step 3: Assign Uncertainties
# ============================================================================

print("=" * 80)
print("Step 3: Assigning Uncertainties")
print("=" * 80)

maximum_anomaly = np.max(np.abs(dobs))
uncertainties = 0.01 * maximum_anomaly * np.ones(np.shape(dobs))

print(f"  Maximum anomaly: {maximum_anomaly:.4e} mGal")
print(f"  Uncertainty: {uncertainties[0]:.4e} mGal (1% of maximum)")
print()

# ============================================================================
# Step 4: Define Survey
# ============================================================================

print("=" * 80)
print("Step 4: Defining Survey")
print("=" * 80)

receiver_list = gravity.receivers.Point(receiver_locations, components="gz")
receiver_list = [receiver_list]
source_field = gravity.sources.SourceField(receiver_list=receiver_list)
survey = gravity.survey.Survey(source_field)

print(f"  Survey created with {survey.nD} data points")
print(f"  Component: vertical (gz)")
print()

# ============================================================================
# Step 5: Define Data Object
# ============================================================================

print("=" * 80)
print("Step 5: Creating Data Object")
print("=" * 80)

data_object = data.Data(survey, dobs=dobs, standard_deviation=uncertainties)

print(f"  Data object created")
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
print(f"  Mesh shape: {mesh.shape_cells}")
print()

# ============================================================================
# Step 7: Define Active Cells and Starting Model
# ============================================================================

print("=" * 80)
print("Step 7: Defining Active Cells and Starting Model")
print("=" * 80)

# Find the active cells
ind_active = active_from_xyz(mesh, xyz_topo)
nC = int(ind_active.sum())
model_map = maps.IdentityMap(nP=nC)

# Define starting model
starting_model = np.zeros(nC)

print(f"  Total mesh cells: {mesh.nC}")
print(f"  Active cells: {nC}")
print(f"  Inactive cells: {mesh.nC - nC}")
print(f"  Starting model: all zeros")
print()

# ============================================================================
# Step 8: Define Simulation and Data Misfit
# ============================================================================

print("=" * 80)
print("Step 8: Defining Simulation and Data Misfit")
print("=" * 80)

simulation = gravity.simulation.Simulation3DIntegral(
    survey=survey, mesh=mesh, rhoMap=model_map, active_cells=ind_active
)

dmis = data_misfit.L2DataMisfit(data=data_object, simulation=simulation)

print(f"  Simulation created (3D Integral)")
print(f"  Data misfit: L2 norm")
print()

# ============================================================================
# Step 9: Inversion with Depth Weighting
# ============================================================================

print("=" * 80)
print("Step 9: Running Inversion with Depth Weighting")
print("=" * 80)

# Directives
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e0)
update_IRLS = directives.UpdateIRLS(
    f_min_change=1e-4,
    max_irls_iterations=30,
    irls_cooling_factor=1.5,
    misfit_tolerance=1e-2,
)
save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
update_jacobi = directives.UpdatePreconditioner()

directives_list = [
    update_IRLS,
    starting_beta,
    save_iteration,
    update_jacobi,
]

# Regularization with depth weighting
reg_dpth = regularization.Sparse(mesh, active_cells=ind_active, mapping=model_map)
reg_dpth.norms = [0, 2, 2, 2]
depth_weights = utils.depth_weighting(
    mesh, receiver_locations, active_cells=ind_active, exponent=2
)
reg_dpth.set_weights(depth_weights=depth_weights)

# Optimization
opt = optimization.ProjectedGNCG(
    maxIter=100, lower=-1.0, upper=1.0, maxIterLS=20, maxIterCG=10, tolCG=1e-3
)

# Inverse problem
inv_prob = inverse_problem.BaseInvProblem(dmis, reg_dpth, opt)

# Inversion
inv = inversion.BaseInversion(inv_prob, directives_list)

print(f"  Starting depth weighting inversion...")
print(f"  Regularization: sparse with depth weights")
print(f"  Max iterations: {opt.maxIter}")
print()

# Run inversion
recovered_model_dpth = inv.run(starting_model)

print()
print(f"  Depth weighting inversion complete!")
print(f"  Model range: [{recovered_model_dpth.min():.4f}, {recovered_model_dpth.max():.4f}] g/cc")
print()

# ============================================================================
# Step 10: Inversion with Distance Weighting
# ============================================================================

print("=" * 80)
print("Step 10: Running Inversion with Distance Weighting")
print("=" * 80)

# Regularization with distance weighting
reg_dist = regularization.Sparse(mesh, active_cells=ind_active, mapping=model_map)
reg_dist.norms = [0, 2, 2, 2]
distance_weights = utils.distance_weighting(
    mesh, receiver_locations, active_cells=ind_active, exponent=2
)
reg_dist.set_weights(distance_weights=distance_weights)

# Optimization
opt = optimization.ProjectedGNCG(
    maxIter=100, lower=-1.0, upper=1.0, maxIterLS=20, maxIterCG=10, tolCG=1e-3
)

# Inverse problem
inv_prob = inverse_problem.BaseInvProblem(dmis, reg_dist, opt)

# Inversion
inv = inversion.BaseInversion(inv_prob, directives_list)

print(f"  Starting distance weighting inversion...")
print(f"  Regularization: sparse with distance weights")
print()

# Run inversion
recovered_model_dist = inv.run(starting_model)

print()
print(f"  Distance weighting inversion complete!")
print(f"  Model range: [{recovered_model_dist.min():.4f}, {recovered_model_dist.max():.4f}] g/cc")
print()

# ============================================================================
# Step 11: Inversion with Sensitivity Weighting
# ============================================================================

print("=" * 80)
print("Step 11: Running Inversion with Sensitivity Weighting")
print("=" * 80)

# Add sensitivity weights directive
sensitivity_weights = directives.UpdateSensitivityWeights(every_iteration=False)

directives_list_sensw = [
    update_IRLS,
    sensitivity_weights,
    starting_beta,
    save_iteration,
    update_jacobi,
]

# Regularization for sensitivity weighting
reg_sensw = regularization.Sparse(mesh, active_cells=ind_active, mapping=model_map)
reg_sensw.norms = [0, 2, 2, 2]

# Optimization
opt = optimization.ProjectedGNCG(
    maxIter=100, lower=-1.0, upper=1.0, maxIterLS=20, maxIterCG=10, tolCG=1e-3
)

# Inverse problem
inv_prob = inverse_problem.BaseInvProblem(dmis, reg_sensw, opt)

# Inversion
inv = inversion.BaseInversion(inv_prob, directives_list_sensw)

print(f"  Starting sensitivity weighting inversion...")
print(f"  Regularization: sparse with sensitivity weights (computed from Jacobian)")
print()

# Run inversion
recovered_model_sensw = inv.run(starting_model)

print()
print(f"  Sensitivity weighting inversion complete!")
print(f"  Model range: [{recovered_model_sensw.min():.4f}, {recovered_model_sensw.max():.4f}] g/cc")
print()

# ============================================================================
# Step 12: Recreate True Model
# ============================================================================

print("=" * 80)
print("Step 12: Recreating True Model for Comparison")
print("=" * 80)

background_density = 0.0
block_density = -0.2
sphere_density = 0.2

true_model = background_density * np.ones(nC)

# Add block
ind_block = (
    (mesh.gridCC[ind_active, 0] > -50.0)
    & (mesh.gridCC[ind_active, 0] < -20.0)
    & (mesh.gridCC[ind_active, 1] > -15.0)
    & (mesh.gridCC[ind_active, 1] < 15.0)
    & (mesh.gridCC[ind_active, 2] > -50.0)
    & (mesh.gridCC[ind_active, 2] < -30.0)
)
true_model[ind_block] = block_density

# Add sphere
ind_sphere = model_builder.get_indices_sphere(
    np.r_[35.0, 0.0, -40.0], 15.0, mesh.gridCC
)
ind_sphere = ind_sphere[ind_active]
true_model[ind_sphere] = sphere_density

print(f"  True model created")
print(f"  Background: {background_density} g/cc")
print(f"  Block: {block_density} g/cc")
print(f"  Sphere: {sphere_density} g/cc")
print()

# ============================================================================
# Step 13: Plot Recovered Models
# ============================================================================

print("=" * 80)
print("Step 13: Plotting Recovered Models")
print("=" * 80)

fig, ax = plt.subplots(2, 2, figsize=(20, 10), sharex=True, sharey=True)
ax = ax.flatten()
plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
cmap = "coolwarm"
slice_y_loc = 0.0

# True model
mm = mesh.plot_slice(
    plotting_map * true_model,
    normal="Y",
    ax=ax[0],
    grid=False,
    slice_loc=slice_y_loc,
    pcolor_opts={"cmap": cmap},
)
ax[0].set_title(f"True model slice at y = {slice_y_loc} m")
plt.colorbar(mm[0], label="$g/cm^3$", ax=ax[0])

# Depth weighting result
vmax = np.abs(recovered_model_dpth).max()
norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
mm = mesh.plot_slice(
    plotting_map * recovered_model_dpth,
    normal="Y",
    ax=ax[1],
    grid=False,
    slice_loc=slice_y_loc,
    pcolor_opts={"cmap": cmap, "norm": norm},
)
ax[1].set_title(f"Depth weighting model slice at y = {slice_y_loc} m")
plt.colorbar(mm[0], label="$g/cm^3$", ax=ax[1])

# Distance weighting result
vmax = np.abs(recovered_model_dist).max()
norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
mm = mesh.plot_slice(
    plotting_map * recovered_model_dist,
    normal="Y",
    ax=ax[2],
    grid=False,
    slice_loc=slice_y_loc,
    pcolor_opts={"cmap": cmap, "norm": norm},
)
ax[2].set_title(f"Distance weighting model slice at y = {slice_y_loc} m")
plt.colorbar(mm[0], label="$g/cm^3$", ax=ax[2])

# Sensitivity weighting result
vmax = np.abs(recovered_model_sensw).max()
norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
mm = mesh.plot_slice(
    plotting_map * recovered_model_sensw,
    normal="Y",
    ax=ax[3],
    grid=False,
    slice_loc=slice_y_loc,
    pcolor_opts={"cmap": cmap, "norm": norm},
)
ax[3].set_title(f"Sensitivity weighting model slice at y = {slice_y_loc} m")
plt.colorbar(mm[0], label="$g/cm^3$", ax=ax[3])

# Overlay true model contours
plotting_map_contour = maps.InjectActiveCells(mesh, ind_active, 0.0)
slice_y_ind = (
    mesh.cell_centers[:, 1] == np.abs(mesh.cell_centers[:, 1] - slice_y_loc).min()
)
for axx in ax:
    utils.plot2Ddata(
        mesh.cell_centers[slice_y_ind][:, [0, 2]],
        (plotting_map_contour * true_model)[slice_y_ind],
        contourOpts={"alpha": 0},
        level=True,
        ncontour=2,
        levelOpts={"colors": "grey", "linewidths": 2, "linestyles": "--"},
        method="nearest",
        ax=axx,
    )
    axx.set_aspect(1)

plt.tight_layout()
plt.savefig("weighting_recovered_models.png", dpi=150, bbox_inches="tight")
print("✓ Recovered models plot saved as 'weighting_recovered_models.png'")
plt.close()

# ============================================================================
# Step 14: Visualize Weights
# ============================================================================

print("=" * 80)
print("Step 14: Visualizing Weight Functions")
print("=" * 80)

fig, ax = plt.subplots(1, 3, figsize=(20, 4), sharex=True, sharey=True)
plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
cmap = "magma"

# Depth weights
mm = mesh.plot_slice(
    plotting_map * np.log10(depth_weights),
    normal="Y",
    ax=ax[0],
    grid=False,
    slice_loc=slice_y_loc,
    pcolor_opts={"cmap": cmap},
)
ax[0].set_title(f"log10(depth weights) slice at y = {slice_y_loc} m")
plt.colorbar(mm[0], label="log10(depth weights)", ax=ax[0])

# Distance weights
mm = mesh.plot_slice(
    plotting_map * np.log10(distance_weights),
    normal="Y",
    ax=ax[1],
    grid=False,
    slice_loc=slice_y_loc,
    pcolor_opts={"cmap": cmap},
)
ax[1].set_title(f"log10(distance weights) slice at y = {slice_y_loc} m")
plt.colorbar(mm[0], label="log10(distance weights)", ax=ax[1])

# Sensitivity weights
mm = mesh.plot_slice(
    plotting_map * np.log10(reg_sensw.objfcts[0].get_weights(key="sensitivity")),
    normal="Y",
    ax=ax[2],
    grid=False,
    slice_loc=slice_y_loc,
    pcolor_opts={"cmap": cmap},
)
ax[2].set_title(f"log10(sensitivity weights) slice at y = {slice_y_loc} m")
plt.colorbar(mm[0], label="log10(sensitivity weights)", ax=ax[2])

# Set aspect ratio
for axx in ax:
    axx.set_aspect(1)

plt.tight_layout()
plt.savefig("weighting_weight_functions.png", dpi=150, bbox_inches="tight")
print("✓ Weight functions plot saved as 'weighting_weight_functions.png'")
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
print(f"  Data points:             {survey.nD}")
print(f"  True model range:        [{true_model.min():.2f}, {true_model.max():.2f}] g/cc")
print()
print("Recovered Model Ranges:")
print("-" * 80)
print(f"  Depth weighting:         [{recovered_model_dpth.min():.4f}, {recovered_model_dpth.max():.4f}] g/cc")
print(f"  Distance weighting:      [{recovered_model_dist.min():.4f}, {recovered_model_dist.max():.4f}] g/cc")
print(f"  Sensitivity weighting:   [{recovered_model_sensw.min():.4f}, {recovered_model_sensw.max():.4f}] g/cc")
print()
print("Generated Files:")
print("-" * 80)
print("  1. weighting_data.png                  - Observed gravity anomaly data")
print("  2. weighting_recovered_models.png      - Comparison of all three inversions")
print("  3. weighting_weight_functions.png      - Visualization of weight functions")
print()
print("Weighting Strategy Comparison:")
print("-" * 80)
print("  Depth Weighting:")
print("    - Uses vertical distance from observation points")
print("    - Formula: w(z) = (z - z_obs)^(-n), typically n=2")
print("    - Counters natural decay of gravity kernel with depth")
print("    - Most commonly used for gravity/magnetic inversions")
print()
print("  Distance Weighting:")
print("    - Uses 3D Euclidean distance from observation points")
print("    - Formula: w(r) = r^(-n), where r = sqrt(dx² + dy² + dz²)")
print("    - More general than depth weighting")
print("    - Accounts for lateral distance as well as depth")
print()
print("  Sensitivity Weighting:")
print("    - Uses the Jacobian (sensitivity) matrix directly")
print("    - Formula: w = sqrt(sum(J²)) / cell_volume")
print("    - Data-driven approach based on actual sensitivities")
print("    - Automatically adapts to survey geometry")
print()
print("Note: All three approaches aim to counteract the decay of sensitivity")
print("      with distance from observation points, preventing the inversion")
print("      from concentrating structure near the surface.")
print("=" * 80)
