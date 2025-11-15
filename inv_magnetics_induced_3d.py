"""
3D Inversion of TMI Data to Recover a Susceptibility Model
University of British Columbia

This tutorial demonstrates how to invert total magnetic intensity data to recover a susceptibility model.
We show two approaches:

1. Weighted least-squares inversion for a tensor mesh
2. Iteratively re-weighted least-squares (IRLS) inversion for a tree mesh

The weighted least-squares approach is a great introduction to geophysical inversion with SimPEG.
One drawback however, is that it recovers smooth structures which may not be representative of the
true model. To recover sparse and/or blocky structures, we demonstrate the iteratively re-weighted
least-squares approach.

Learning Objectives:
- Introduce geophysical inversion with SimPEG
- Assigning appropriate uncertainties to total magnetic intensity data
- Designing a suitable mesh for magnetic inversion
- Choosing suitable parameters for the inversion
- Specifying directives that are applied throughout the inversion
- Apply the sensitivity weighting commonly used when inverting magnetic data
- Inversion with weighted least-squares and sparse-norm regularizations
- Analyzing inversion outputs

Keywords: total magnetic intensity, integral formulation, inversion, sparse norm, tensor mesh, tree mesh
"""

# ============================================================================
# Import Modules
# ============================================================================

# SimPEG functionality
from simpeg.potential_fields import magnetics
from simpeg.utils import plot2Ddata, model_builder, download
from simpeg import (
    maps,
    data,
    data_misfit,
    inverse_problem,
    regularization,
    optimization,
    directives,
    inversion,
)

# discretize functionality
from discretize import TensorMesh, TreeMesh
from discretize.utils import active_from_xyz

# Common Python functionality
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tarfile

mpl.rcParams.update({"font.size": 14})

save_output = False  # Set to True to save outputs


# ============================================================================
# Load Tutorial Files
# ============================================================================

print("=" * 80)
print("Step 1: Loading Tutorial Data")
print("=" * 80)

# URL to download from repository assets
data_source = "https://github.com/simpeg/user-tutorials/raw/main/assets/04-magnetics/inv_magnetics_induced_3d_files.tar.gz"

# download the data
downloaded_data = download(data_source, overwrite=True)

# unzip the tarfile
tar = tarfile.open(downloaded_data, "r")
tar.extractall()
tar.close()

# path to the directory containing our data
dir_path = downloaded_data.split(".")[0] + os.path.sep

# files to work with
topo_filename = dir_path + "magnetics_topo.txt"
data_filename = dir_path + "magnetics_data.obs"

# Load topography (xyz file)
topo_xyz = np.loadtxt(str(topo_filename))

# Load field data (xyz file)
dobs = np.loadtxt(str(data_filename))

print(f"  Loaded topography from: {topo_filename}")
print(f"  Loaded data from: {data_filename}")
print()


# ============================================================================
# Plot Observed Data and Topography
# ============================================================================

print("=" * 80)
print("Step 2: Plotting Observed Data and Topography")
print("=" * 80)

# Define receiver locations and observed data
receiver_locations = dobs[:, 0:3]
dobs = dobs[:, -1]

fig = plt.figure(figsize=(8, 5))
ax1 = fig.add_axes([0.05, 0.35, 0.35, 0.6])

v_max = np.max(np.abs(dobs))
norm1 = mpl.colors.Normalize(vmin=-v_max, vmax=v_max)

plot2Ddata(
    receiver_locations,
    dobs,
    ax=ax1,
    dataloc=True,
    ncontour=40,
    contourOpts={"cmap": mpl.cm.bwr, "norm": norm1},
)
ax1.set_title("TMI Anomaly")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")

cx1 = fig.add_axes([0.05, 0.18, 0.35, 0.04])
cbar1 = mpl.colorbar.ColorbarBase(
    cx1, norm=norm1, orientation="horizontal", cmap=mpl.cm.bwr
)
cbar1.set_label("$nT$", size=16)

ax2 = fig.add_axes([0.55, 0.35, 0.35, 0.6])
plot2Ddata(
    topo_xyz[:, 0:2],
    topo_xyz[:, -1],
    ax=ax2,
    ncontour=50,
    contourOpts={"cmap": "gist_earth"},
)
ax2.set_title("Topography", pad=15)
ax2.set_xlabel("x (m)")
ax2.set_ylabel("y (m)")

cx2 = fig.add_axes([0.55, 0.18, 0.35, 0.04])
norm2 = mpl.colors.Normalize(vmin=np.min(topo_xyz[:, -1]), vmax=np.max(topo_xyz[:, -1]))
cbar2 = mpl.colorbar.ColorbarBase(
    cx2, norm=norm2, orientation="horizontal", cmap=mpl.cm.gist_earth
)
cbar2.set_label("$m$", size=16)

plt.savefig("magnetics_data_and_topo.png", dpi=150, bbox_inches="tight")
print("✓ Data and topography plot saved as 'magnetics_data_and_topo.png'")
plt.close()
print()


# ============================================================================
# Assign Uncertainties
# ============================================================================

print("=" * 80)
print("Step 3: Assigning Data Uncertainties")
print("=" * 80)

maximum_anomaly = np.max(np.abs(dobs))
floor_uncertainty = 0.02 * maximum_anomaly
uncertainties = floor_uncertainty * np.ones(np.shape(dobs))

print(f"  Maximum anomaly: {maximum_anomaly:.4f} nT")
print(f"  Floor uncertainty: {floor_uncertainty:.4f} nT (2% of max)")
print()


# ============================================================================
# Define the Survey
# ============================================================================

print("=" * 80)
print("Step 4: Defining Survey")
print("=" * 80)

# Define the component(s) of the field we are inverting
components = ["tmi"]

# Use the observation locations and components to define the receivers
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
print(f"  Component: tmi (total magnetic intensity)")
print(f"  Inducing field: Inc={inclination}°, Dec={declination}°, F={amplitude} nT")
print()


# ============================================================================
# Define the Data Object
# ============================================================================

print("=" * 80)
print("Step 5: Creating Data Object")
print("=" * 80)

data_object = data.Data(survey, dobs=dobs, standard_deviation=uncertainties)

print(f"  Data object created")
print()


# ============================================================================
# WEIGHTED LEAST-SQUARES INVERSION ON TENSOR MESH
# ============================================================================

print()
print("=" * 80)
print("WEIGHTED LEAST-SQUARES INVERSION")
print("=" * 80)
print()


# ============================================================================
# Design Tensor Mesh
# ============================================================================

print("=" * 80)
print("Step 6: Designing Tensor Mesh")
print("=" * 80)

# Generate tensor mesh with top at z = 0 m
dh = 5.0  # minimum cell size
hx = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]  # discretization along x
hy = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]  # discretization along y
hz = [(dh, 5, -1.3), (dh, 15)]  # discretization along z
tensor_mesh = TensorMesh([hx, hy, hz], "CCN")

# Shift vertically to top same as maximum topography
tensor_mesh.origin += np.r_[0.0, 0.0, topo_xyz[:, -1].max()]

print(f"  Minimum cell size: {dh} m")
print(f"  Number of cells: {tensor_mesh.n_cells}")
print()


# ============================================================================
# Define Active Cells (Tensor)
# ============================================================================

print("=" * 80)
print("Step 7: Defining Active Cells (Tensor)")
print("=" * 80)

active_tensor_cells = active_from_xyz(tensor_mesh, topo_xyz)
n_tensor_active = int(active_tensor_cells.sum())

print(f"  Total mesh cells: {tensor_mesh.n_cells}")
print(f"  Active cells: {n_tensor_active}")
print()


# ============================================================================
# Model Mapping (Tensor)
# ============================================================================

print("=" * 80)
print("Step 8: Creating Model Mapping (Tensor)")
print("=" * 80)

tensor_model_map = maps.IdentityMap(nP=n_tensor_active)

print(f"  Model parameters: {tensor_model_map.nP}")
print()


# ============================================================================
# Starting and Reference Models (Tensor)
# ============================================================================

print("=" * 80)
print("Step 9: Defining Starting and Reference Models")
print("=" * 80)

starting_tensor_model = 1e-6 * np.ones(n_tensor_active)
reference_tensor_model = np.zeros_like(starting_tensor_model)

print(f"  Starting model: {starting_tensor_model[0]:.2e} SI")
print()

# Mapping to ignore inactive cells when plotting
tensor_plotting_map = maps.InjectActiveCells(tensor_mesh, active_tensor_cells, np.nan)


# ============================================================================
# Define Forward Simulation (L2)
# ============================================================================

print("=" * 80)
print("Step 10: Defining Forward Simulation (L2)")
print("=" * 80)

simulation_L2 = magnetics.simulation.Simulation3DIntegral(
    survey=survey,
    mesh=tensor_mesh,
    model_type="scalar",
    chiMap=tensor_model_map,
    active_cells=active_tensor_cells,
    engine="choclo",
)

print("  Simulation type: 3D Integral Formulation")
print("  Engine: Choclo (fast Numba implementation)")
print()


# ============================================================================
# Define Data Misfit (L2)
# ============================================================================

print("=" * 80)
print("Step 11: Defining Data Misfit (L2)")
print("=" * 80)

dmis_L2 = data_misfit.L2DataMisfit(data=data_object, simulation=simulation_L2)

print("  Data misfit type: L2")
print()


# ============================================================================
# Define Regularization (L2)
# ============================================================================

print("=" * 80)
print("Step 12: Defining Regularization (L2)")
print("=" * 80)

reg_L2 = regularization.WeightedLeastSquares(
    tensor_mesh,
    active_cells=active_tensor_cells,
    length_scale_x=1.0,
    length_scale_y=1.0,
    length_scale_z=1.0,
    reference_model=reference_tensor_model,
    reference_model_in_smooth=False,
)

print("  Regularization type: Weighted Least-Squares")
print("  Length scales: x=1.0, y=1.0, z=1.0")
print()


# ============================================================================
# Define Optimization (L2)
# ============================================================================

print("=" * 80)
print("Step 13: Defining Optimization Algorithm (L2)")
print("=" * 80)

opt_L2 = optimization.ProjectedGNCG(
    maxIter=100, lower=0.0, maxIterLS=20, maxIterCG=10, tolCG=1e-2
)

print("  Algorithm: Projected Gauss-Newton CG")
print("  Lower bound: 0.0 SI")
print()


# ============================================================================
# Define Inverse Problem (L2)
# ============================================================================

print("=" * 80)
print("Step 14: Defining Inverse Problem (L2)")
print("=" * 80)

inv_prob_L2 = inverse_problem.BaseInvProblem(dmis_L2, reg_L2, opt_L2)

print("  Inverse problem created")
print()


# ============================================================================
# Define Directives (L2)
# ============================================================================

print("=" * 80)
print("Step 15: Defining Inversion Directives (L2)")
print("=" * 80)

sensitivity_weights = directives.UpdateSensitivityWeights(every_iteration=False)
update_jacobi = directives.UpdatePreconditioner(update_every_iteration=True)
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=10)
beta_schedule = directives.BetaSchedule(coolingFactor=2.0, coolingRate=1)
target_misfit = directives.TargetMisfit(chifact=1.0)

directives_list_L2 = [
    sensitivity_weights,
    update_jacobi,
    starting_beta,
    beta_schedule,
    target_misfit,
]

print("  Directives configured")
print()


# ============================================================================
# Run L2 Inversion
# ============================================================================

print("=" * 80)
print("Step 16: Running L2 Inversion")
print("=" * 80)
print()

inv_L2 = inversion.BaseInversion(inv_prob_L2, directives_list_L2)
recovered_tensor_model = inv_L2.run(starting_tensor_model)

print()
print("✓ L2 inversion completed")
print()


# ============================================================================
# Plot L2 Results
# ============================================================================

print("=" * 80)
print("Step 17: Analyzing L2 Results")
print("=" * 80)

dpred = inv_prob_L2.dpred
data_array = np.c_[dobs, dpred, (dobs - dpred)]

fig = plt.figure(figsize=(12, 5))
plot_title = ["Observed", "Predicted", "Data Misfit"]
plot_units = ["nT", "nT", "nT"]

ax1 = 3 * [None]
ax2 = 3 * [None]
norm = 3 * [None]
cbar = 3 * [None]
cplot = 3 * [None]
v_lim = [np.max(np.abs(dobs)), np.max(np.abs(dobs)), np.max(np.abs(dobs - dpred))]

for ii in range(0, 3):
    ax1[ii] = fig.add_axes([0.3 * ii + 0.1, 0.2, 0.27, 0.75])
    norm[ii] = mpl.colors.Normalize(vmin=-v_lim[ii], vmax=v_lim[ii])
    cplot[ii] = plot2Ddata(
        receiver_list[0].locations,
        data_array[:, ii],
        ax=ax1[ii],
        ncontour=30,
        contourOpts={"cmap": "bwr", "norm": norm[ii]},
    )
    ax1[ii].set_title(plot_title[ii])
    ax1[ii].set_xlabel("x (m)")
    if ii == 0:
        ax1[ii].set_ylabel("y (m)")
    else:
        ax1[ii].set_yticks([])

    ax2[ii] = fig.add_axes([0.3 * ii + 0.1, 0.05, 0.27, 0.05])
    cbar[ii] = mpl.colorbar.ColorbarBase(
        ax2[ii], norm=norm[ii], orientation="horizontal", cmap=mpl.cm.bwr
    )
    cbar[ii].set_label(plot_units[ii], labelpad=5)

plt.savefig("magnetics_L2_data_misfit.png", dpi=150, bbox_inches="tight")
print("✓ L2 data misfit plot saved as 'magnetics_L2_data_misfit.png'")
plt.close()

# Plot L2 Recovered Model
fig = plt.figure(figsize=(7, 3))
ax1 = fig.add_axes([0.1, 0.1, 0.73, 0.8])

norm = mpl.colors.Normalize(
    vmin=np.min(recovered_tensor_model), vmax=np.max(recovered_tensor_model)
)
tensor_mesh.plot_slice(
    tensor_plotting_map * recovered_tensor_model,
    normal="Y",
    ax=ax1,
    ind=int(tensor_mesh.shape_cells[1] / 2),
    grid=True,
    pcolor_opts={"cmap": mpl.cm.plasma, "norm": norm},
)
ax1.set_title("Recovered L2 Model (slice at y = 0 m)")

ax2 = fig.add_axes([0.85, 0.1, 0.03, 0.8])
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.plasma
)
cbar.set_label("$SI$", rotation=270, labelpad=15, size=16)

plt.savefig("magnetics_L2_recovered_model.png", dpi=150, bbox_inches="tight")
print("✓ L2 recovered model plot saved as 'magnetics_L2_recovered_model.png'")
plt.close()

print(f"  L2 model range: [{np.min(recovered_tensor_model):.6f}, {np.max(recovered_tensor_model):.6f}] SI")
print()


# ============================================================================
# ITERATIVELY RE-WEIGHTED LEAST-SQUARES INVERSION
# ============================================================================

print()
print("=" * 80)
print("ITERATIVELY RE-WEIGHTED LEAST-SQUARES INVERSION")
print("=" * 80)
print()


# ============================================================================
# Reassign Uncertainties
# ============================================================================

print("=" * 80)
print("Step 18: Reassigning Uncertainties for IRLS")
print("=" * 80)

normalized_data_misfits = (dobs - dpred) / uncertainties
new_uncertainties = uncertainties.copy()
new_uncertainties[np.abs(normalized_data_misfits) > 2.0] /= 2.5

print(f"  Data with reduced uncertainty: {(np.abs(normalized_data_misfits) > 2.0).sum()}")
print()

new_data_object = data.Data(survey, dobs=dobs, standard_deviation=new_uncertainties)


# ============================================================================
# Design Tree Mesh
# ============================================================================

print("=" * 80)
print("Step 19: Designing Tree Mesh")
print("=" * 80)

dx = 5  # minimum cell width in x
dy = 5  # minimum cell width in y
dz = 5  # minimum cell width in z

x_length = 240.0
y_length = 240.0
z_length = 120.0

nbcx = 2 ** int(np.round(np.log(x_length / dx) / np.log(2.0)))
nbcy = 2 ** int(np.round(np.log(y_length / dy) / np.log(2.0)))
nbcz = 2 ** int(np.round(np.log(z_length / dz) / np.log(2.0)))

hx = [(dx, nbcx)]
hy = [(dy, nbcy)]
hz = [(dz, nbcz)]
tree_mesh = TreeMesh([hx, hy, hz], x0="CCN", diagonal_balance=True)

tree_mesh.origin += np.r_[0.0, 0.0, topo_xyz[:, -1].max()]

tree_mesh.refine_surface(topo_xyz, padding_cells_by_level=[2, 2], finalize=False)

wsb_corner = np.c_[-100, -100, 20]
ent_corner = np.c_[100, 100, 100]
tree_mesh.refine_box(wsb_corner, ent_corner, levels=[-1], finalize=False)

tree_mesh.finalize()

print(f"  Minimum cell size: {dx} m")
print(f"  Number of cells: {tree_mesh.n_cells}")
print()


# ============================================================================
# Define Active Cells (Tree)
# ============================================================================

print("=" * 80)
print("Step 20: Defining Active Cells (Tree)")
print("=" * 80)

ind_tree_active = active_from_xyz(tree_mesh, topo_xyz)
n_tree_active = int(ind_tree_active.sum())

print(f"  Total mesh cells: {tree_mesh.n_cells}")
print(f"  Active cells: {n_tree_active}")
print()


# ============================================================================
# Model Mapping (Tree)
# ============================================================================

print("=" * 80)
print("Step 21: Creating Model Mapping (Tree)")
print("=" * 80)

tree_model_map = maps.IdentityMap(nP=n_tree_active)

print(f"  Model parameters: {tree_model_map.nP}")
print()


# ============================================================================
# Starting and Reference Models (Tree)
# ============================================================================

print("=" * 80)
print("Step 22: Defining Starting and Reference Models (Tree)")
print("=" * 80)

starting_tree_model = 1e-6 * np.ones(n_tree_active)
reference_tree_model = np.zeros_like(starting_tree_model)

print(f"  Starting model: {starting_tree_model[0]:.2e} SI")
print()


# ============================================================================
# Define Forward Simulation (IRLS)
# ============================================================================

print("=" * 80)
print("Step 23: Defining Forward Simulation (IRLS)")
print("=" * 80)

simulation_irls = magnetics.simulation.Simulation3DIntegral(
    survey=survey,
    mesh=tree_mesh,
    model_type="scalar",
    chiMap=tree_model_map,
    active_cells=ind_tree_active,
    engine="choclo",
)

print("  Simulation type: 3D Integral Formulation")
print()


# ============================================================================
# Define Data Misfit (IRLS)
# ============================================================================

print("=" * 80)
print("Step 24: Defining Data Misfit (IRLS)")
print("=" * 80)

dmis_irls = data_misfit.L2DataMisfit(data=new_data_object, simulation=simulation_irls)

print("  Data misfit type: L2")
print()


# ============================================================================
# Define Regularization (IRLS)
# ============================================================================

print("=" * 80)
print("Step 25: Defining Regularization (IRLS)")
print("=" * 80)

reg_irls = regularization.Sparse(
    tree_mesh,
    active_cells=ind_tree_active,
    alpha_s=dh**-2,
    alpha_x=1,
    alpha_y=1,
    alpha_z=1,
    reference_model=reference_tree_model,
    reference_model_in_smooth=False,
    norms=[0, 1, 1, 1],
)

print("  Regularization type: Sparse")
print("  Norms: [0, 1, 1, 1] (compact and blocky)")
print()


# ============================================================================
# Define Optimization (IRLS)
# ============================================================================

print("=" * 80)
print("Step 26: Defining Optimization Algorithm (IRLS)")
print("=" * 80)

opt_irls = optimization.ProjectedGNCG(
    maxIter=100, lower=0.0, maxIterLS=20, maxIterCG=10, tolCG=1e-2
)

print("  Algorithm: Projected Gauss-Newton CG")
print()


# ============================================================================
# Define Inverse Problem (IRLS)
# ============================================================================

print("=" * 80)
print("Step 27: Defining Inverse Problem (IRLS)")
print("=" * 80)

inv_prob_irls = inverse_problem.BaseInvProblem(dmis_irls, reg_irls, opt_irls)

print("  Inverse problem created")
print()


# ============================================================================
# Define Directives (IRLS)
# ============================================================================

print("=" * 80)
print("Step 28: Defining Inversion Directives (IRLS)")
print("=" * 80)

sensitivity_weights_irls = directives.UpdateSensitivityWeights(every_iteration=False)
starting_beta_irls = directives.BetaEstimate_ByEig(beta0_ratio=10)
update_jacobi_irls = directives.UpdatePreconditioner(update_every_iteration=True)
update_irls = directives.UpdateIRLS(
    cooling_factor=2,
    f_min_change=1e-4,
    max_irls_iterations=25,
    chifact_start=1.0,
)

directives_list_irls = [
    update_irls,
    sensitivity_weights_irls,
    starting_beta_irls,
    update_jacobi_irls,
]

print("  Directives configured (including IRLS)")
print()


# ============================================================================
# Run IRLS Inversion
# ============================================================================

print("=" * 80)
print("Step 29: Running IRLS Inversion")
print("=" * 80)
print()

inv_irls = inversion.BaseInversion(inv_prob_irls, directives_list_irls)
recovered_tree_model = inv_irls.run(starting_tree_model)

print()
print("✓ IRLS inversion completed")
print()


# ============================================================================
# Compare All Models
# ============================================================================

print("=" * 80)
print("Step 30: Comparing All Models")
print("=" * 80)

# Recreate True Model on a Tensor Mesh
background_susceptibility = 0.0001
sphere_susceptibility = 0.01

true_model = background_susceptibility * np.ones(n_tensor_active)
ind_sphere = model_builder.get_indices_sphere(
    np.r_[0.0, 0.0, 55.0], 16.0, tensor_mesh.cell_centers[active_tensor_cells]
)
true_model[ind_sphere] = sphere_susceptibility

# Plot all models
mesh_list = [tensor_mesh, tensor_mesh, tree_mesh]
ind_list = [active_tensor_cells, active_tensor_cells, ind_tree_active]
model_list = [true_model, recovered_tensor_model, recovered_tree_model]
title_list = ["True Model", "L2 Tensor Model", "IRLS Tree Model"]
cplot = 3 * [None]
cbar = 3 * [None]
norm = 3 * [None]

fig = plt.figure(figsize=(7, 8))
ax1 = [fig.add_axes([0.1, 0.7 - 0.3 * ii, 0.75, 0.23]) for ii in range(0, 3)]
ax2 = [fig.add_axes([0.88, 0.7 - 0.3 * ii, 0.025, 0.23]) for ii in range(0, 3)]

for ii, mesh in enumerate(mesh_list):
    plotting_map = maps.InjectActiveCells(mesh, ind_list[ii], np.nan)
    max_abs = np.max(np.abs(model_list[ii]))
    norm[ii] = mpl.colors.Normalize(vmin=0.0, vmax=max_abs)

    cplot[ii] = mesh.plot_slice(
        plotting_map * model_list[ii],
        normal="Y",
        ax=ax1[ii],
        ind=int(mesh.shape_cells[1] / 2),
        grid=False,
        pcolor_opts={"cmap": mpl.cm.plasma, "norm": norm[ii]},
    )
    ax1[ii].set_xlim([-150, 150])
    ax1[ii].set_ylim([topo_xyz[:, -1].max() - 100, topo_xyz[:, -1].max()])
    if ii < 2:
        ax1[ii].set_xlabel("")
        ax1[ii].set_xticks([])
    ax1[ii].set_title(title_list[ii])

    cbar[ii] = mpl.colorbar.ColorbarBase(
        ax2[ii], norm=norm[ii], orientation="vertical", cmap=mpl.cm.plasma
    )
    cbar[ii].set_label("$SI$", labelpad=5)

plt.savefig("magnetics_model_comparison.png", dpi=150, bbox_inches="tight")
print("✓ Model comparison plot saved as 'magnetics_model_comparison.png'")
plt.close()
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
print(f"  L2 Tensor Mesh:")
print(f"    - Total cells:     {tensor_mesh.n_cells}")
print(f"    - Active cells:    {n_tensor_active}")
print(f"    - Model range:     [{np.min(recovered_tensor_model):.6f}, {np.max(recovered_tensor_model):.6f}] SI")
print()
print(f"  IRLS Tree Mesh:")
print(f"    - Total cells:     {tree_mesh.n_cells}")
print(f"    - Active cells:    {n_tree_active}")
print(f"    - Model range:     [{np.min(recovered_tree_model):.6f}, {np.max(recovered_tree_model):.6f}] SI")
print()
print("Generated Files:")
print("-" * 80)
print("  1. magnetics_data_and_topo.png       - Observed data and topography")
print("  2. magnetics_L2_data_misfit.png      - L2 inversion data fit")
print("  3. magnetics_L2_recovered_model.png  - L2 recovered model")
print("  4. magnetics_model_comparison.png    - Comparison of all models")
print()
print("Key Findings:")
print("-" * 80)
print("  - L2 inversion recovers smooth susceptibility structures")
print("  - IRLS inversion recovers more compact/blocky structures")
print("  - Both methods successfully locate magnetic anomaly sources")
print("  - Sensitivity weighting reduces near-surface artifacts")
print("=" * 80)
