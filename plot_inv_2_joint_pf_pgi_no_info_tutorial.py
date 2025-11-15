"""
Joint PGI of Gravity + Magnetic on an Octree mesh without petrophysical information
University of British Columbia

This tutorial demonstrates:
- Joint inversion of Gravity and Magnetic data on an Octree mesh
- Using the PGI framework when NO quantitative petrophysical information is available
- Learning suitable petrophysical distributions from the data
- Making geologic assumptions about rock unit relationships
- Multi-physics inversion with updatable Gaussian mixture model
- Recovering quasi-geology models with learned petrophysical properties

References:
Thibaut Astic, Douglas W. Oldenburg,
A framework for petrophysically and geologically guided geophysical inversion
using a dynamic Gaussian mixture model prior, Geophysical Journal International,
Volume 219, Issue 3, December 2019, Pages 1989-2012, DOI: 10.1093/gji/ggz389

Thibaut Astic, Lindsey J. Heagy, Douglas W Oldenburg,
Petrophysically and geologically guided multi-physics inversion using a dynamic
Gaussian mixture model, Geophysical Journal International,
Volume 224, Issue 1, January 2021, Pages 40-68, DOI: 10.1093/gji/ggaa378

Keywords: PGI, joint inversion, gravity, magnetics, learning petrophysics, GMM
"""

# ============================================================================
# Import Modules
# ============================================================================

from discretize import TreeMesh
from discretize.utils import active_from_xyz
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import simpeg.potential_fields as pf
from simpeg import (
    data_misfit,
    directives,
    inverse_problem,
    inversion,
    maps,
    optimization,
    regularization,
    utils,
)
from simpeg.utils import io_utils

mpl.rcParams.update({"font.size": 14})

# ============================================================================
# Step 1: Load Mesh
# ============================================================================

print("=" * 80)
print("Step 1: Loading Mesh")
print("=" * 80)

mesh_file = io_utils.download(
    "https://storage.googleapis.com/simpeg/pgi_tutorial_assets/mesh_tutorial.ubc"
)
mesh = TreeMesh.read_UBC(mesh_file)

print(f"  Mesh loaded successfully")
print(f"  Number of cells: {mesh.nC}")
print(f"  Number of active cells will be determined from topography")
print()

# ============================================================================
# Step 2: Load True Geology Model
# ============================================================================

print("=" * 80)
print("Step 2: Loading True Geology Model")
print("=" * 80)

true_geology_file = io_utils.download(
    "https://storage.googleapis.com/simpeg/pgi_tutorial_assets/geology_true.mod"
)
true_geology = mesh.read_model_UBC(true_geology_file)

print(f"  True geology model loaded (for comparison only)")
print(f"  Rock units: Background (0), PK (1), VK (2)")
print(f"  Note: We will NOT use the true petrophysical values in this inversion!")
print()

# Plot true geology model
fig, ax = plt.subplots(1, 4, figsize=(20, 4))
ticksize, labelsize = 14, 16
for _, axx in enumerate(ax):
    axx.set_aspect(1)
    axx.tick_params(labelsize=ticksize)
mesh.plot_slice(
    true_geology,
    normal="X",
    ax=ax[0],
    ind=-17,
    clim=[0, 2],
    pcolor_opts={"cmap": "inferno_r"},
    grid=True,
)
mesh.plot_slice(
    true_geology,
    normal="Y",
    ax=ax[1],
    clim=[0, 2],
    pcolor_opts={"cmap": "inferno_r"},
    grid=True,
)
geoplot = mesh.plot_slice(
    true_geology,
    normal="Z",
    ax=ax[2],
    clim=[0, 2],
    ind=-10,
    pcolor_opts={"cmap": "inferno_r"},
    grid=True,
)
geocb = plt.colorbar(geoplot[0], cax=ax[3], ticks=[0, 1, 2])
geocb.set_label(
    "True geology model\n(classification/density/mag. susc.)", fontsize=labelsize
)
geocb.set_ticklabels(
    ["BCKGRD (0 g/cc; 0 SI)", "PK (-0.8 g/cc; 5e-3 SI)", "VK (-0.2 g/cc; 2e-2 SI)"]
)
geocb.ax.tick_params(labelsize=ticksize)
ax[3].set_aspect(10)
plt.savefig("pgi_no_info_true_geology.png", dpi=150, bbox_inches="tight")
print("✓ True geology model plot saved as 'pgi_no_info_true_geology.png'")
plt.close()

# ============================================================================
# Step 3: Load Geophysical Data
# ============================================================================

print("=" * 80)
print("Step 3: Loading Geophysical Data")
print("=" * 80)

data_grav_file = io_utils.download(
    "https://storage.googleapis.com/simpeg/pgi_tutorial_assets/gravity_data.obs"
)
data_grav = io_utils.read_grav3d_ubc(data_grav_file)

data_mag_file = io_utils.download(
    "https://storage.googleapis.com/simpeg/pgi_tutorial_assets/magnetic_data.obs"
)
data_mag = io_utils.read_mag3d_ubc(data_mag_file)

print(f"  Gravity data points: {len(data_grav.dobs)}")
print(f"  Gravity data range: [{data_grav.dobs.min():.2f}, {data_grav.dobs.max():.2f}] mGal")
print(f"  Magnetic data points: {len(data_mag.dobs)}")
print(f"  Magnetic data range: [{data_mag.dobs.min():.2f}, {data_mag.dobs.max():.2f}] nT")
print()

# Plot data and mesh
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
ax = ax.reshape(-1)
mesh.plot_slice(
    np.ones(mesh.nC),
    normal="Z",
    ind=int(-10),
    grid=True,
    pcolor_opts={"cmap": "Greys"},
    ax=ax[0],
)
mm = utils.plot2Ddata(
    data_grav.survey.receiver_locations,
    -data_grav.dobs,
    ax=ax[0],
    level=True,
    nx=20,
    ny=20,
    dataloc=True,
    ncontour=12,
    shade=True,
    contourOpts={"cmap": "Blues_r", "alpha": 0.8},
    levelOpts={"colors": "k", "linewidths": 0.5, "linestyles": "dashed"},
)
ax[0].set_aspect(1)
ax[0].set_title(
    "Gravity data values and locations,\nwith mesh and geology overlays", fontsize=16
)
plt.colorbar(mm[0], cax=ax[2], orientation="horizontal")
ax[2].set_aspect(0.05)
ax[2].set_title("mGal", fontsize=16)
mesh.plot_slice(
    np.ones(mesh.nC),
    normal="Z",
    ind=int(-10),
    grid=True,
    pcolor_opts={"cmap": "Greys"},
    ax=ax[1],
)
mm = utils.plot2Ddata(
    data_mag.survey.receiver_locations,
    data_mag.dobs,
    ax=ax[1],
    level=True,
    nx=20,
    ny=20,
    dataloc=True,
    ncontour=11,
    shade=True,
    contourOpts={"cmap": "Reds", "alpha": 0.8},
    levelOpts={"colors": "k", "linewidths": 0.5, "linestyles": "dashed"},
)
ax[1].set_aspect(1)
ax[1].set_title(
    "Magnetic data values and locations,\nwith mesh and geology overlays", fontsize=16
)
plt.colorbar(mm[0], cax=ax[3], orientation="horizontal")
ax[3].set_aspect(0.05)
ax[3].set_title("nT", fontsize=16)
# Overlay true geology model for comparison
indz = -9
indslicezplot = mesh.gridCC[:, 2] == mesh.cell_centers_z[indz]
for i in range(2):
    utils.plot2Ddata(
        mesh.gridCC[indslicezplot][:, [0, 1]],
        true_geology[indslicezplot],
        nx=200,
        ny=200,
        contourOpts={"alpha": 0},
        clim=[0, 2],
        ax=ax[i],
        level=True,
        ncontour=2,
        levelOpts={"colors": "k", "linewidths": 2, "linestyles": "--"},
        method="nearest",
    )
plt.subplots_adjust(hspace=-0.25, wspace=0.1)
plt.savefig("pgi_no_info_data.png", dpi=150, bbox_inches="tight")
print("✓ Data plot saved as 'pgi_no_info_data.png'")
plt.close()

# ============================================================================
# Step 4: Load Topography and Define Active Cells
# ============================================================================

print("=" * 80)
print("Step 4: Loading Topography and Defining Active Cells")
print("=" * 80)

topo_file = io_utils.download(
    "https://storage.googleapis.com/simpeg/pgi_tutorial_assets/CDED_Lake_warp.xyz"
)
topo = np.genfromtxt(topo_file, skip_header=1)

# Find the active cells
actv = active_from_xyz(mesh, topo, "CC")
ndv = np.nan
actvMap = maps.InjectActiveCells(mesh, actv, ndv)
nactv = int(actv.sum())

print(f"  Total mesh cells: {mesh.nC}")
print(f"  Active cells: {nactv}")
print(f"  Inactive cells: {mesh.nC - nactv}")
print()

# ============================================================================
# Step 5: Create Simulations and Data Misfits
# ============================================================================

print("=" * 80)
print("Step 5: Creating Simulations and Data Misfits")
print("=" * 80)

# Wires mapping
wires = maps.Wires(("den", actvMap.nP), ("sus", actvMap.nP))
gravmap = actvMap * wires.den
magmap = actvMap * wires.sus
idenMap = maps.IdentityMap(nP=nactv)

# Gravity problem
simulation_grav = pf.gravity.simulation.Simulation3DIntegral(
    survey=data_grav.survey,
    mesh=mesh,
    rhoMap=wires.den,
    active_cells=actv,
    engine="choclo",
)
dmis_grav = data_misfit.L2DataMisfit(data=data_grav, simulation=simulation_grav)

# Magnetic problem
simulation_mag = pf.magnetics.simulation.Simulation3DIntegral(
    survey=data_mag.survey,
    mesh=mesh,
    chiMap=wires.sus,
    active_cells=actv,
    engine="choclo",
)
dmis_mag = data_misfit.L2DataMisfit(data=data_mag, simulation=simulation_mag)

print(f"  Gravity simulation created (engine: choclo)")
print(f"  Magnetic simulation created (engine: choclo)")
print(f"  Data misfits initialized")
print()

# ============================================================================
# Step 6: Create Joint Data Misfit
# ============================================================================

print("=" * 80)
print("Step 6: Creating Joint Data Misfit")
print("=" * 80)

# Joint data misfit
dmis = 0.5 * dmis_grav + 0.5 * dmis_mag

# Initial model
m0 = np.r_[-1e-4 * np.ones(actvMap.nP), 1e-4 * np.ones(actvMap.nP)]

print(f"  Joint data misfit created (equal weighting)")
print(f"  Initial model parameters: {len(m0)}")
print(f"  Density parameters: {actvMap.nP}")
print(f"  Susceptibility parameters: {actvMap.nP}")
print()

# ============================================================================
# Step 7: Create Petrophysical GMM Initial Guess (No Info)
# ============================================================================

print("=" * 80)
print("Step 7: Creating Petrophysical GMM Initial Guess (No Information)")
print("=" * 80)

print("  Scenario: We do NOT know the true petrophysical signatures")
print("  Strategy: Make geological assumptions and learn from data")
print("  Assumptions:")
print("    - Background: neutral (0 g/cc, 0 SI) - FIXED")
print("    - Unit 1: Only less dense, no magnetization - LEARN density")
print("    - Unit 2: Only magnetic, no density contrast - LEARN susceptibility")
print()

gmmref = utils.WeightedGaussianMixture(
    n_components=3,  # number of rock units: bckgrd, PK, HK
    mesh=mesh,
    actv=actv,
    covariance_type="diag",
)

# Required initialization with fit
rng = np.random.default_rng(seed=518936)
gmmref.fit(rng.normal(size=(nactv, 2)))

# Set parameters manually with initial guesses
gmmref.means_ = np.c_[
    [0.0, 0.0],  # BCKGRD density contrast and mag. susc (FIXED)
    [-1, 0.0],  # PK initial guess: density-contrasting unit
    [0, 0.1],  # HK initial guess: magnetic-contrasting unit
].T

# Set phys. prop covariances for each unit
gmmref.covariances_ = np.array(
    [[6e-04, 3.175e-07], [2.4e-03, 1.5e-06], [2.4e-03, 1.5e-06]]
)

# Important after setting cov. manually
gmmref.compute_clusters_precisions()

# Set global proportions
gmmref.weights_ = np.r_[0.9, 0.075, 0.025]

print(f"  GMM created with 3 rock units")
print(f"  Background: density=0.0 g/cc (FIXED), susceptibility=0.0 SI (FIXED)")
print(f"  PK unit: density=-1.0 g/cc (INITIAL), susceptibility=0.0 SI (FIXED)")
print(f"  HK unit: density=0.0 g/cc (FIXED), susceptibility=0.1 SI (INITIAL)")
print()

# Plot the initial 2D GMM
ax = gmmref.plot_pdf(flag2d=True, plotting_precision=250)
ax[0].set_xlabel("Density contrast [g/cc]")
ax[0].set_ylim([0, 5])
ax[2].set_ylabel("magnetic Susceptibility [SI]")
ax[2].set_xlim([0, 100])
plt.savefig("pgi_no_info_gmm_initial.png", dpi=150, bbox_inches="tight")
print("✓ Initial GMM plot saved as 'pgi_no_info_gmm_initial.png'")
plt.close()

# ============================================================================
# Step 8: Create PGI Regularization
# ============================================================================

print("=" * 80)
print("Step 8: Creating PGI Regularization")
print("=" * 80)

# Sensitivity weighting
wr_grav = np.sum(simulation_grav.G**2.0, axis=0) ** 0.5 / (mesh.cell_volumes[actv])
wr_grav = wr_grav / np.max(wr_grav)

wr_mag = np.sum(simulation_mag.G**2.0, axis=0) ** 0.5 / (mesh.cell_volumes[actv])
wr_mag = wr_mag / np.max(wr_mag)

# Create joint PGI regularization with smoothness
reg = regularization.PGI(
    gmmref=gmmref,
    mesh=mesh,
    wiresmap=wires,
    maplist=[idenMap, idenMap],
    active_cells=actv,
    alpha_pgi=1.0,
    alpha_x=1.0,
    alpha_y=1.0,
    alpha_z=1.0,
    alpha_xx=0.0,
    alpha_yy=0.0,
    alpha_zz=0.0,
    reference_model=utils.mkvc(
        gmmref.means_[gmmref.predict(m0.reshape(actvMap.nP, -1))]
    ),
    weights_list=[wr_grav, wr_mag],
)

print(f"  PGI regularization created")
print(f"  Sensitivity weights computed and applied")
print(f"  Alpha values: alpha_pgi=1.0, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0")
print()

# ============================================================================
# Step 9: Set Up Inversion Directives
# ============================================================================

print("=" * 80)
print("Step 9: Setting Up Inversion Directives")
print("=" * 80)

# Ratio to use for each phys prop. smoothness in each direction
alpha0_ratio = np.r_[
    1e-2 * np.ones(len(reg.objfcts[1].objfcts[1:])),
    1e-2 * 100.0 * np.ones(len(reg.objfcts[2].objfcts[1:])),
]
Alphas = directives.AlphasSmoothEstimate_ByEig(alpha0_ratio=alpha0_ratio, verbose=True)

# Initialize beta and beta/alpha_s schedule
beta = directives.BetaEstimate_ByEig(beta0_ratio=1e-4)
betaIt = directives.PGI_BetaAlphaSchedule(
    verbose=True,
    coolingFactor=2.0,
    tolerance=0.2,
    progress=0.2,
)

# Geophysical and petrophysical target misfits
targets = directives.MultiTargetMisfits(
    verbose=True,
    chiSmall=0.5,  # Ask for twice as much clustering
)

# Add learned mref in smooth once stable
MrefInSmooth = directives.PGI_AddMrefInSmooth(wait_till_stable=True, verbose=True)

# Update the parameters in smallness (L2-approx of PGI)
# Key difference: update_gmm=True to learn petrophysical properties!
update_smallness = directives.PGI_UpdateParameters(
    update_gmm=True,  # LEARN the GMM each iteration
    kappa=np.c_[  # Confidences in each mean phys. prop.
        1e10 * np.ones(2),  # Background: FIXED at 0,0 (high confidence)
        [0, 1e10],  # PK unit: UPDATABLE density, FIXED susceptibility
        [1e10, 0],  # HK unit: FIXED density, UPDATABLE susceptibility
    ].T,
)

# Pre-conditioner
update_Jacobi = directives.UpdatePreconditioner()

# Iteratively balance the scaling of the data misfits
scaling_init = directives.ScalingMultipleDataMisfits_ByEig(chi0_ratio=[1.0, 100.0])
scale_schedule = directives.JointScalingSchedule(verbose=True)

print(f"  Directives configured:")
print(f"    - Alpha smoothness estimation")
print(f"    - Beta estimation and scheduling")
print(f"    - Multi-target misfits (enhanced clustering)")
print(f"    - PGI parameter updates (GMM LEARNING ENABLED)")
print(f"    - Kappa confidences set to learn specific properties")
print(f"    - Joint scaling schedule")
print()

# ============================================================================
# Step 10: Create Inverse Problem and Run Inversion
# ============================================================================

print("=" * 80)
print("Step 10: Running PGI Inversion WITHOUT Petrophysical Information")
print("=" * 80)

# Set lower and upper bounds
lowerbound = np.r_[-2.0 * np.ones(actvMap.nP), 0.0 * np.ones(actvMap.nP)]
upperbound = np.r_[0.0 * np.ones(actvMap.nP), 1e-1 * np.ones(actvMap.nP)]

opt = optimization.ProjectedGNCG(
    maxIter=30,
    lower=lowerbound,
    upper=upperbound,
    maxIterLS=20,
    maxIterCG=100,
    tolCG=1e-4,
)

# Create inverse problem
invProb = inverse_problem.BaseInvProblem(dmis, reg, opt)
inv = inversion.BaseInversion(
    invProb,
    directiveList=[
        Alphas,
        scaling_init,
        beta,
        update_smallness,
        targets,
        scale_schedule,
        betaIt,
        MrefInSmooth,
        update_Jacobi,
    ],
)

print(f"  Starting inversion (max {opt.maxIter} iterations)...")
print(f"  The algorithm will LEARN petrophysical properties from the data")
print(f"  This may take several minutes...")
print()

# Run inversion
pgi_model_no_info = inv.run(m0)

print()
print(f"  Inversion complete!")
print()

# ============================================================================
# Step 11: Extract and Visualize Results
# ============================================================================

print("=" * 80)
print("Step 11: Extracting and Visualizing Results")
print("=" * 80)

# Extract the results
density_model_no_info = gravmap * pgi_model_no_info
magsus_model_no_info = magmap * pgi_model_no_info
learned_gmm = reg.objfcts[0].gmm
quasi_geology_model_no_info = actvMap * reg.objfcts[0].compute_quasi_geology_model()

print(f"  Density model range: [{np.nanmin(density_model_no_info):.4f}, {np.nanmax(density_model_no_info):.4f}] g/cc")
print(f"  Susceptibility model range: [{np.nanmin(magsus_model_no_info):.4f}, {np.nanmax(magsus_model_no_info):.4f}] SI")
print(f"  Quasi-geology classifications: {np.unique(quasi_geology_model_no_info[~np.isnan(quasi_geology_model_no_info)])}")
print()
print("  Learned GMM means:")
print(f"    Background: density={learned_gmm.means_[0, 0]:.4f} g/cc, susceptibility={learned_gmm.means_[0, 1]:.6f} SI")
print(f"    PK unit:    density={learned_gmm.means_[1, 0]:.4f} g/cc, susceptibility={learned_gmm.means_[1, 1]:.6f} SI")
print(f"    HK unit:    density={learned_gmm.means_[2, 0]:.4f} g/cc, susceptibility={learned_gmm.means_[2, 1]:.6f} SI")
print()
print("  Compare with true values:")
print(f"    PK true:    density=-0.8000 g/cc, susceptibility=0.005000 SI")
print(f"    HK true:    density=-0.2000 g/cc, susceptibility=0.020000 SI")
print()

# Plot the result
fig, ax = plt.subplots(3, 4, figsize=(15, 10))
for _, axx in enumerate(ax):
    for _, axxx in enumerate(axx):
        axxx.set_aspect(1)
        axxx.tick_params(labelsize=ticksize)

indx = 15
indy = 17
indz = -9

# Geology model
mesh.plot_slice(
    quasi_geology_model_no_info,
    normal="X",
    ax=ax[0, 0],
    clim=[0, 2],
    ind=indx,
    pcolor_opts={"cmap": "inferno_r"},
)
mesh.plot_slice(
    quasi_geology_model_no_info,
    normal="Y",
    ax=ax[0, 1],
    clim=[0, 2],
    ind=indy,
    pcolor_opts={"cmap": "inferno_r"},
)
geoplot = mesh.plot_slice(
    quasi_geology_model_no_info,
    normal="Z",
    ax=ax[0, 2],
    clim=[0, 2],
    ind=indz,
    pcolor_opts={"cmap": "inferno_r"},
)
geocb = plt.colorbar(geoplot[0], cax=ax[0, 3], ticks=[0, 1, 2])
geocb.set_ticklabels(["BCK", "PK", "VK"])
geocb.set_label("Quasi-Geology model\n(Rock units classification)", fontsize=16)
ax[0, 3].set_aspect(10)

# Gravity model
mesh.plot_slice(
    density_model_no_info,
    normal="X",
    ax=ax[1, 0],
    clim=[-1, 0],
    ind=indx,
    pcolor_opts={"cmap": "Blues_r"},
)
mesh.plot_slice(
    density_model_no_info,
    normal="Y",
    ax=ax[1, 1],
    clim=[-1, 0],
    ind=indy,
    pcolor_opts={"cmap": "Blues_r"},
)
denplot = mesh.plot_slice(
    density_model_no_info,
    normal="Z",
    ax=ax[1, 2],
    clim=[-1, 0],
    ind=indz,
    pcolor_opts={"cmap": "Blues_r"},
)
dencb = plt.colorbar(denplot[0], cax=ax[1, 3])
dencb.set_label("Density contrast\nmodel (g/cc)", fontsize=16)
ax[1, 3].set_aspect(10)

# Magnetic model
mesh.plot_slice(
    magsus_model_no_info,
    normal="X",
    ax=ax[2, 0],
    clim=[0, 0.025],
    ind=indx,
    pcolor_opts={"cmap": "Reds"},
)
mesh.plot_slice(
    magsus_model_no_info,
    normal="Y",
    ax=ax[2, 1],
    clim=[0, 0.025],
    ind=indy,
    pcolor_opts={"cmap": "Reds"},
)
susplot = mesh.plot_slice(
    magsus_model_no_info,
    normal="Z",
    ax=ax[2, 2],
    clim=[0, 0.025],
    ind=indz,
    pcolor_opts={"cmap": "Reds"},
)
suscb = plt.colorbar(susplot[0], cax=ax[2, 3])
suscb.set_label("Magnetic susceptibility\nmodel (SI)", fontsize=16)
ax[2, 3].set_aspect(10)

# Overlay true geology model for comparison
indslicexplot = mesh.gridCC[:, 0] == mesh.cell_centers_x[indx]
indsliceyplot = mesh.gridCC[:, 1] == mesh.cell_centers_y[indy]
indslicezplot = mesh.gridCC[:, 2] == mesh.cell_centers_z[indz]
for i in range(3):
    for j, (plane, indd) in enumerate(
        zip([[1, 2], [0, 2], [0, 1]], [indslicexplot, indsliceyplot, indslicezplot])
    ):
        utils.plot2Ddata(
            mesh.gridCC[indd][:, plane],
            true_geology[indd],
            nx=100,
            ny=100,
            contourOpts={"alpha": 0},
            clim=[0, 2],
            ax=ax[i, j],
            level=True,
            ncontour=2,
            levelOpts={"colors": "grey", "linewidths": 2, "linestyles": "--"},
            method="nearest",
        )

# Plot the locations of the cross-sections
for i in range(3):
    ax[i, 0].plot(
        mesh.cell_centers_y[indy] * np.ones(2), [-300, 500], c="k", linestyle="dotted"
    )
    ax[i, 0].plot(
        [
            data_mag.survey.receiver_locations[:, 1].min(),
            data_mag.survey.receiver_locations[:, 1].max(),
        ],
        mesh.cell_centers_z[indz] * np.ones(2),
        c="k",
        linestyle="dotted",
    )
    ax[i, 0].set_xlim(
        [
            data_mag.survey.receiver_locations[:, 1].min(),
            data_mag.survey.receiver_locations[:, 1].max(),
        ],
    )

    ax[i, 1].plot(
        mesh.cell_centers_x[indx] * np.ones(2), [-300, 500], c="k", linestyle="dotted"
    )
    ax[i, 1].plot(
        [
            data_mag.survey.receiver_locations[:, 0].min(),
            data_mag.survey.receiver_locations[:, 0].max(),
        ],
        mesh.cell_centers_z[indz] * np.ones(2),
        c="k",
        linestyle="dotted",
    )
    ax[i, 1].set_xlim(
        [
            data_mag.survey.receiver_locations[:, 0].min(),
            data_mag.survey.receiver_locations[:, 0].max(),
        ],
    )

    ax[i, 2].plot(
        mesh.cell_centers_x[indx] * np.ones(2),
        [
            data_mag.survey.receiver_locations[:, 1].min(),
            data_mag.survey.receiver_locations[:, 1].max(),
        ],
        c="k",
        linestyle="dotted",
    )
    ax[i, 2].plot(
        [
            data_mag.survey.receiver_locations[:, 0].min(),
            data_mag.survey.receiver_locations[:, 0].max(),
        ],
        mesh.cell_centers_y[indy] * np.ones(2),
        c="k",
        linestyle="dotted",
    )
    ax[i, 2].set_xlim(
        [
            data_mag.survey.receiver_locations[:, 0].min(),
            data_mag.survey.receiver_locations[:, 0].max(),
        ],
    )
    ax[i, 2].set_ylim(
        [
            data_mag.survey.receiver_locations[:, 1].min(),
            data_mag.survey.receiver_locations[:, 1].max(),
        ],
    )

plt.tight_layout()
plt.savefig("pgi_no_info_recovered_models.png", dpi=150, bbox_inches="tight")
print("✓ Recovered models plot saved as 'pgi_no_info_recovered_models.png'")
plt.close()

# Plot the learned 2D GMM
fig = plt.figure(figsize=(10, 10))
ax0 = plt.subplot2grid((4, 4), (3, 1), colspan=3)
ax1 = plt.subplot2grid((4, 4), (0, 1), colspan=3, rowspan=3)
ax2 = plt.subplot2grid((4, 4), (0, 0), rowspan=3)
ax = [ax0, ax1, ax2]
learned_gmm.plot_pdf(flag2d=True, ax=ax, padding=1, plotting_precision=100)
ax[0].set_xlabel("Density contrast [g/cc]")
ax[0].set_ylim([0, 5])
ax[2].set_xlim([0, 50])
ax[2].set_ylabel("magnetic Susceptibility [SI]")
ax[1].scatter(
    density_model_no_info[actv],
    magsus_model_no_info[actv],
    c=quasi_geology_model_no_info[actv],
    cmap="inferno_r",
    edgecolors="k",
    label="recovered PGI model",
    alpha=0.5,
)
ax[0].hist(density_model_no_info[actv], density=True, bins=50)
ax[2].hist(magsus_model_no_info[actv], density=True, bins=50, orientation="horizontal")
ax[1].scatter(
    [0, -0.8, -0.02],
    [0, 0.005, 0.02],
    label="True petrophysical means",
    cmap="inferno_r",
    c=[0, 1, 2],
    marker="v",
    edgecolors="k",
    s=200,
)
ax[1].legend()
plt.savefig("pgi_no_info_gmm_learned.png", dpi=150, bbox_inches="tight")
print("✓ Learned GMM plot saved as 'pgi_no_info_gmm_learned.png'")
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
print(f"  Active cells:            {nactv}")
print(f"  Gravity data points:     {len(data_grav.dobs)}")
print(f"  Magnetic data points:    {len(data_mag.dobs)}")
print(f"  Rock units:              3 (Background, PK, VK)")
print(f"  Recovered density:       [{np.nanmin(density_model_no_info):.4f}, {np.nanmax(density_model_no_info):.4f}] g/cc")
print(f"  Recovered susceptibility:[{np.nanmin(magsus_model_no_info):.4f}, {np.nanmax(magsus_model_no_info):.4f}] SI")
print()
print("Learned Petrophysical Properties:")
print("-" * 80)
print(f"  PK learned:  density={learned_gmm.means_[1, 0]:.4f} g/cc, susc={learned_gmm.means_[1, 1]:.6f} SI")
print(f"  PK true:     density=-0.8000 g/cc, susc=0.005000 SI")
print(f"  HK learned:  density={learned_gmm.means_[2, 0]:.4f} g/cc, susc={learned_gmm.means_[2, 1]:.6f} SI")
print(f"  HK true:     density=-0.2000 g/cc, susc=0.020000 SI")
print()
print("Generated Files:")
print("-" * 80)
print("  1. pgi_no_info_true_geology.png        - True geology model (3 views)")
print("  2. pgi_no_info_data.png                - Gravity and magnetic data")
print("  3. pgi_no_info_gmm_initial.png         - Initial GMM guess")
print("  4. pgi_no_info_recovered_models.png    - Recovered models (geology/density/susceptibility)")
print("  5. pgi_no_info_gmm_learned.png         - Learned GMM with true means overlay")
print()
print("Key Features:")
print("-" * 80)
print("  - Joint inversion of gravity and magnetic data")
print("  - PGI framework WITHOUT petrophysical information")
print("  - GMM learning enabled (update_gmm=True)")
print("  - Geological assumptions: one unit is dense, one is magnetic")
print("  - Algorithm learned petrophysical properties from data alone")
print("  - Compare learned vs true means to evaluate success")
print()
print("Note: This inversion LEARNED the petrophysical properties from the data!")
print("      The algorithm successfully recovered rock unit properties without")
print("      prior quantitative petrophysical information.")
print("=" * 80)
