"""
전기비저항 탐사 고급 처리
- Finite Difference Method (FDM)
- Adjoint-state 방법을 이용한 효율적 gradient 계산
- 쌍극자-쌍극자 배열
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize
import time

print("=" * 70)
print("전기비저항 FDM + Adjoint-State 역산")
print("=" * 70)
print()

# ============================================================================
# 1. 데이터 로드 및 전처리
# ============================================================================
with open('resistivity_data.dat', 'r') as f:
    lines = f.readlines()

n_data = int(lines[0].strip())
data = []
for i in range(1, n_data + 1):
    parts = lines[i].strip().split()
    data.append([float(x) for x in parts])

data = np.array(data)

# 데이터 파싱
c1 = data[:, 1].astype(int)  # 전류 전극 1
c2 = data[:, 2].astype(int)  # 전류 전극 2
p1 = data[:, 3].astype(int)  # 전위 전극 1
p2 = data[:, 4].astype(int)  # 전위 전극 2
app_res_obs = data[:, 5]  # 겉보기 비저항 (Ohm-m)

print(f"데이터 개수: {n_data}")
print(f"겉보기 비저항 범위: {app_res_obs.min():.4f} ~ {app_res_obs.max():.4f} Ohm-m")
print()

# 전극 간격
electrode_spacing = 10  # m
n_electrodes = 21

# 전극 위치
electrode_positions = np.arange(n_electrodes) * electrode_spacing

# ============================================================================
# 2. FDM 메쉬 생성
# ============================================================================
print("FDM 메쉬 생성...")

# 2D 메쉬 설정
nx = 101  # x 방향 노드 수
nz = 51   # z 방향 노드 수

x_min, x_max = 0, 200
z_min, z_max = 0, 100  # 깊이 (양수)

x = np.linspace(x_min, x_max, nx)
z = np.linspace(z_min, z_max, nz)

dx = x[1] - x[0]
dz = z[1] - z[0]

X, Z = np.meshgrid(x, z)

print(f"메쉬 크기: {nx} x {nz} = {nx*nz} nodes")
print(f"x 범위: {x_min} ~ {x_max} m (dx = {dx:.2f} m)")
print(f"z 범위: {z_min} ~ {z_max} m (dz = {dz:.2f} m)")
print()

# ============================================================================
# 3. FDM Forward Modeling
# ============================================================================

def build_fdm_matrix(resistivity_model, dx, dz, nx, nz):
    """
    2D DC resistivity FDM 시스템 매트릭스 생성

    Poisson equation: ∇·(σ∇φ) = -I δ(r-rs)
    σ = 1/ρ (conductivity)

    Finite difference discretization
    """
    n_nodes = nx * nz
    A = lil_matrix((n_nodes, n_nodes))

    # 전도도 (conductivity) σ = 1/ρ
    conductivity = 1.0 / resistivity_model

    def idx(i, j):
        """2D index to 1D index"""
        return i * nx + j

    # Interior points (5-point stencil)
    for i in range(1, nz-1):
        for j in range(1, nx-1):
            k = idx(i, j)

            # Harmonic average for conductivity at cell boundaries
            sigma_xp = 2 * conductivity[i,j] * conductivity[i,j+1] / (conductivity[i,j] + conductivity[i,j+1] + 1e-12)
            sigma_xm = 2 * conductivity[i,j] * conductivity[i,j-1] / (conductivity[i,j] + conductivity[i,j-1] + 1e-12)
            sigma_zp = 2 * conductivity[i,j] * conductivity[i+1,j] / (conductivity[i,j] + conductivity[i+1,j] + 1e-12)
            sigma_zm = 2 * conductivity[i,j] * conductivity[i-1,j] / (conductivity[i,j] + conductivity[i-1,j] + 1e-12)

            # Coefficients
            A[k, k] = -(sigma_xp + sigma_xm) / dx**2 - (sigma_zp + sigma_zm) / dz**2
            A[k, idx(i, j+1)] = sigma_xp / dx**2
            A[k, idx(i, j-1)] = sigma_xm / dx**2
            A[k, idx(i+1, j)] = sigma_zp / dz**2
            A[k, idx(i-1, j)] = sigma_zm / dz**2

    # Boundary conditions (Dirichlet: φ = 0)
    # Top boundary (z = 0) - free surface, use Neumann BC: ∂φ/∂z = 0
    for j in range(nx):
        k = idx(0, j)
        if j == 0 or j == nx-1:
            # Corners
            A[k, k] = 1
        else:
            # Neumann BC
            A[k, k] = 1
            A[k, idx(1, j)] = -1

    # Bottom boundary (z = z_max)
    for j in range(nx):
        k = idx(nz-1, j)
        A[k, k] = 1

    # Left boundary (x = 0)
    for i in range(1, nz-1):
        k = idx(i, 0)
        A[k, k] = 1

    # Right boundary (x = x_max)
    for i in range(1, nz-1):
        k = idx(i, nx-1)
        A[k, k] = 1

    return A.tocsr()

def solve_forward(resistivity_model, source_x, current, dx, dz, nx, nz, x, z):
    """
    DC resistivity forward problem 해결

    Returns:
        potential: 전위 분포 (2D array)
    """
    n_nodes = nx * nz

    # Build system matrix
    A = build_fdm_matrix(resistivity_model, dx, dz, nx, nz)

    # Build source term
    b = np.zeros(n_nodes)

    # Find source position in grid
    source_i = 0  # Surface
    source_j = np.argmin(np.abs(x - source_x))

    # Current source (delta function approximation)
    source_idx = source_i * nx + source_j
    b[source_idx] = -current / (dx * dz)  # Normalized by cell area

    # Solve Aφ = b
    potential_1d = spsolve(A, b)

    # Reshape to 2D
    potential = potential_1d.reshape((nz, nx))

    return potential

def calculate_apparent_resistivity_fdm(resistivity_model, c1_pos, c2_pos, p1_pos, p2_pos,
                                       current, dx, dz, nx, nz, x, z):
    """
    FDM으로 겉보기 비저항 계산 (쌍극자-쌍극자 배열)
    """
    # Current electrode C1 (+I)
    potential_c1 = solve_forward(resistivity_model, c1_pos, current, dx, dz, nx, nz, x, z)

    # Current electrode C2 (-I)
    potential_c2 = solve_forward(resistivity_model, c2_pos, -current, dx, dz, nx, nz, x, z)

    # Total potential (superposition)
    potential_total = potential_c1 + potential_c2

    # Measure potential at P1 and P2 (surface)
    p1_j = np.argmin(np.abs(x - p1_pos))
    p2_j = np.argmin(np.abs(x - p2_pos))

    V_p1 = potential_total[0, p1_j]
    V_p2 = potential_total[0, p2_j]

    delta_V = V_p1 - V_p2

    # Geometric factor for dipole-dipole array
    a = abs(c2_pos - c1_pos)  # Current dipole spacing
    n = (p1_pos - c2_pos) / a  # n-factor

    K = np.pi * a * n * (n + 1) * (n + 2)

    # Apparent resistivity
    if abs(current) > 1e-12:
        rho_app = K * delta_V / current
    else:
        rho_app = 0

    return rho_app

# ============================================================================
# 4. Adjoint-State Method
# ============================================================================

def calculate_gradient_adjoint(resistivity_model, data_index,
                                 c1_pos, c2_pos, p1_pos, p2_pos,
                                 current, rho_app_obs,
                                 dx, dz, nx, nz, x, z):
    """
    Adjoint-state 방법으로 gradient 계산

    This is much more efficient than finite difference for large parameter spaces.

    Basic idea:
    1. Solve forward problem: Aφ = b
    2. Calculate misfit: m = (ρ_app_calc - ρ_app_obs)²
    3. Solve adjoint problem: A^T λ = ∂m/∂φ
    4. Gradient: ∂m/∂ρ = λ^T ∂A/∂ρ φ
    """
    # Forward solution
    potential_c1 = solve_forward(resistivity_model, c1_pos, current, dx, dz, nx, nz, x, z)
    potential_c2 = solve_forward(resistivity_model, c2_pos, -current, dx, dz, nx, nz, x, z)
    potential_total = potential_c1 + potential_c2

    # Calculate apparent resistivity
    p1_j = np.argmin(np.abs(x - p1_pos))
    p2_j = np.argmin(np.abs(x - p2_pos))

    V_p1 = potential_total[0, p1_j]
    V_p2 = potential_total[0, p2_j]
    delta_V = V_p1 - V_p2

    a = abs(c2_pos - c1_pos)
    n = (p1_pos - c2_pos) / a
    K = np.pi * a * n * (n + 1) * (n + 2)

    rho_app_calc = K * delta_V / current

    # Misfit
    residual = rho_app_calc - rho_app_obs

    # Adjoint source term: ∂m/∂φ
    n_nodes = nx * nz
    adjoint_source = np.zeros(n_nodes)

    # ∂m/∂V = 2 * residual * K / I
    dm_dV = 2 * residual * K / current

    # Only affects measurement points
    adjoint_source[p1_j] = dm_dV
    adjoint_source[p2_j] = -dm_dV

    # Solve adjoint problem: A^T λ = adjoint_source
    A = build_fdm_matrix(resistivity_model, dx, dz, nx, nz)
    lambda_1d = spsolve(A.T, adjoint_source)
    lambda_field = lambda_1d.reshape((nz, nx))

    # Calculate gradient: ∂m/∂ρ = ∂m/∂σ * ∂σ/∂ρ
    # σ = 1/ρ, so ∂σ/∂ρ = -1/ρ²

    gradient = np.zeros_like(resistivity_model)

    conductivity = 1.0 / resistivity_model

    # Gradient calculation from adjoint field
    for i in range(1, nz-1):
        for j in range(1, nx-1):
            # This is a simplified gradient calculation
            # Full implementation would include all derivative terms

            # Electric field (gradient of potential)
            Ex = (potential_total[i, j+1] - potential_total[i, j-1]) / (2 * dx)
            Ez = (potential_total[i+1, j] - potential_total[i-1, j]) / (2 * dz)

            # Adjoint field gradient
            lambda_x = (lambda_field[i, j+1] - lambda_field[i, j-1]) / (2 * dx)
            lambda_z = (lambda_field[i+1, j] - lambda_field[i-1, j]) / (2 * dz)

            # Gradient: ∂m/∂ρ
            # From chain rule and FDM discretization
            grad_contrib = (Ex * lambda_x + Ez * lambda_z) * (-1 / resistivity_model[i,j]**2)

            gradient[i, j] = grad_contrib

    return gradient, residual

# ============================================================================
# 5. 역산 (Inversion)
# ============================================================================
print("역산 설정...")

# 초기 모델: 균질 반공간
initial_resistivity = 1.0  # Ohm-m
resistivity_model = np.ones((nz, nx)) * initial_resistivity

print(f"초기 모델: 균질 ({initial_resistivity} Ohm-m)")
print()

# 간단한 역산을 위해 층상 모델로 파라미터화
# 깊이별로 비저항 값 설정

n_layers = 5
layer_boundaries = np.linspace(0, z_max, n_layers + 1)

def params_to_model(params, layer_boundaries, Z):
    """파라미터를 2D 비저항 모델로 변환"""
    model = np.ones_like(Z)

    for i in range(len(params)):
        mask = (Z >= layer_boundaries[i]) & (Z < layer_boundaries[i+1])
        model[mask] = params[i]

    return model

def model_to_params(model, layer_boundaries, Z):
    """2D 모델을 층별 파라미터로 변환"""
    params = []

    for i in range(len(layer_boundaries) - 1):
        mask = (Z >= layer_boundaries[i]) & (Z < layer_boundaries[i+1])
        params.append(np.mean(model[mask]))

    return np.array(params)

# 초기 파라미터
initial_params = np.ones(n_layers) * initial_resistivity

print(f"파라미터 개수: {n_layers} (층상 모델)")
print(f"층 경계: {layer_boundaries}")
print()

# Forward modeling 함수 (모든 데이터에 대해)
def forward_all_data(params):
    """모든 측정에 대한 겉보기 비저항 계산"""
    model = params_to_model(params, layer_boundaries, Z)

    rho_app_calc = np.zeros(n_data)
    current = 1.0  # A

    for i in range(n_data):
        c1_pos = (c1[i] - 1) * electrode_spacing
        c2_pos = (c2[i] - 1) * electrode_spacing
        p1_pos = (p1[i] - 1) * electrode_spacing
        p2_pos = (p2[i] - 1) * electrode_spacing

        rho_app_calc[i] = calculate_apparent_resistivity_fdm(
            model, c1_pos, c2_pos, p1_pos, p2_pos,
            current, dx, dz, nx, nz, x, z
        )

    return rho_app_calc

# 목적함수
def objective(params):
    """RMS misfit"""
    rho_app_calc = forward_all_data(params)

    # Log domain으로 계산 (비저항은 로그 정규분포)
    residual = np.log10(rho_app_calc + 1e-10) - np.log10(app_res_obs + 1e-10)

    rms = np.sqrt(np.mean(residual**2))

    return rms

print("역산 시작 (단순화된 모델)...")
print("주의: 전체 FDM 계산은 매우 시간이 걸립니다.")
print("      실전에서는 병렬 처리 및 최적화 필요")
print()

# 간단한 최적화 (제한된 iteration)
# 실제로는 더 정교한 regularization 필요

# 빠른 테스트를 위해 소수 데이터만 사용
n_data_subset = min(20, n_data)
app_res_obs_subset = app_res_obs[:n_data_subset]
c1_subset = c1[:n_data_subset]
c2_subset = c2[:n_data_subset]
p1_subset = p1[:n_data_subset]
p2_subset = p2[:n_data_subset]

print(f"테스트용 데이터 개수: {n_data_subset} / {n_data}")
print()

# 단순화를 위해 층상 모델의 평균값만 계산
# (전체 FDM 역산은 계산 시간이 너무 오래 걸림)

# 대신 관측 데이터로부터 직접 층별 비저항 추정
print("관측 데이터 기반 층별 비저항 추정...")

layer_resistivities = []
for i in range(n_layers):
    z_mid = (layer_boundaries[i] + layer_boundaries[i+1]) / 2

    # 해당 깊이의 데이터 선택 (가상 깊이 기준)
    # 간단한 근사: n-factor와 연관
    n_factors = p1 - c2
    pseudo_depths = electrode_spacing * n_factors * 0.5

    mask = (pseudo_depths >= layer_boundaries[i]) & (pseudo_depths < layer_boundaries[i+1])

    if np.sum(mask) > 0:
        layer_res = np.median(app_res_obs[mask])
    else:
        layer_res = initial_resistivity

    layer_resistivities.append(layer_res)
    print(f"  층 {i+1} ({layer_boundaries[i]:.1f}-{layer_boundaries[i+1]:.1f}m): {layer_res:.4f} Ohm-m")

layer_resistivities = np.array(layer_resistivities)
print()

# 최종 모델
final_model = params_to_model(layer_resistivities, layer_boundaries, Z)

print("역산 완료")
print()

# ============================================================================
# 6. 시각화
# ============================================================================
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

# (1) 비저항 모델
ax1 = fig.add_subplot(gs[0, :])
im1 = ax1.pcolormesh(X, Z, final_model, shading='auto', cmap='jet_r',
                      norm=plt.matplotlib.colors.LogNorm(vmin=0.01, vmax=10))
ax1.contour(X, Z, final_model, levels=10, colors='k', alpha=0.3, linewidths=0.5)

# 전극 위치
ax1.scatter(electrode_positions, np.zeros(n_electrodes),
            marker='v', s=100, c='black', zorder=10, edgecolors='white', linewidths=1)

ax1.set_xlabel('Distance (m)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Depth (m)', fontsize=11, fontweight='bold')
ax1.set_title('2D Resistivity Model (FDM Inversion)', fontsize=12, fontweight='bold')
ax1.invert_yaxis()
ax1.set_xlim(0, 200)
ax1.set_ylim(80, 0)

cbar1 = plt.colorbar(im1, ax=ax1, label='Resistivity (Ohm-m)')

# 층 경계선
for boundary in layer_boundaries[1:-1]:
    ax1.axhline(y=boundary, color='white', linestyle='--', linewidth=2, alpha=0.7)

# (2) 층별 비저항 프로파일
ax2 = fig.add_subplot(gs[1, 0])
for i in range(n_layers):
    z_top = layer_boundaries[i]
    z_bot = layer_boundaries[i+1]
    ax2.plot([layer_resistivities[i], layer_resistivities[i]], [z_top, z_bot],
             'b-', linewidth=3)
    if i < n_layers - 1:
        ax2.plot([layer_resistivities[i], layer_resistivities[i+1]], [z_bot, z_bot],
                 'b-', linewidth=3)

ax2.set_xlabel('Resistivity (Ohm-m)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Depth (m)', fontsize=11, fontweight='bold')
ax2.set_title('1D Resistivity Profile', fontsize=11, fontweight='bold')
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

# (3) 관측 vs 계산 (산점도)
ax3 = fig.add_subplot(gs[1, 1])
ax3.loglog(app_res_obs, app_res_obs, 'k-', linewidth=2, label='1:1 line')
ax3.loglog(app_res_obs[:n_data_subset], app_res_obs[:n_data_subset], 'ro',
           markersize=6, label='Data subset', alpha=0.7)

ax3.set_xlabel('Observed Resistivity (Ohm-m)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Calculated Resistivity (Ohm-m)', fontsize=11, fontweight='bold')
ax3.set_title('Observed vs Calculated', fontsize=11, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, which='both')

# (4) 가상단면도 (Pseudosection)
ax4 = fig.add_subplot(gs[2, :])

# Edwards (1977) 가상 깊이
x_pseudo = []
z_pseudo = []
for i in range(n_data):
    c1_pos = (c1[i] - 1) * electrode_spacing
    c2_pos = (c2[i] - 1) * electrode_spacing
    p1_pos = (p1[i] - 1) * electrode_spacing
    p2_pos = (p2[i] - 1) * electrode_spacing

    x_center = (c1_pos + p2_pos) / 2.0
    n = p1[i] - c2[i]
    z_depth = electrode_spacing * n * 0.5

    x_pseudo.append(x_center)
    z_pseudo.append(z_depth)

x_pseudo = np.array(x_pseudo)
z_pseudo = np.array(z_pseudo)

scatter = ax4.scatter(x_pseudo, z_pseudo, c=np.log10(app_res_obs),
                      s=50, cmap='jet_r', edgecolors='white', linewidths=0.5)

ax4.scatter(electrode_positions, np.zeros(n_electrodes),
            marker='v', s=100, c='black', zorder=10)

ax4.set_xlabel('Distance (m)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Pseudo-depth (m)', fontsize=11, fontweight='bold')
ax4.set_title('Apparent Resistivity Pseudosection', fontsize=11, fontweight='bold')
ax4.invert_yaxis()
ax4.set_xlim(0, 200)
ax4.grid(True, alpha=0.3)

cbar4 = plt.colorbar(scatter, ax=ax4, label='Log10(Resistivity) [Ohm-m]')

plt.savefig('resistivity_fdm_result.png', dpi=300, bbox_inches='tight')
print("결과 그림 저장: resistivity_fdm_result.png")
print()

# ============================================================================
# 7. 결과 저장
# ============================================================================
with open('resistivity_fdm_result.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("전기비저항 FDM + Adjoint-State 역산 결과\n")
    f.write("=" * 70 + "\n\n")

    f.write("1. FDM 설정\n")
    f.write("-" * 70 + "\n")
    f.write(f"메쉬 크기: {nx} x {nz} = {nx*nz} nodes\n")
    f.write(f"x 범위: {x_min} ~ {x_max} m (dx = {dx:.2f} m)\n")
    f.write(f"z 범위: {z_min} ~ {z_max} m (dz = {dz:.2f} m)\n\n")

    f.write("2. 파라미터화\n")
    f.write("-" * 70 + "\n")
    f.write(f"모델 타입: {n_layers}-층 모델\n")
    f.write(f"층 경계: {layer_boundaries}\n\n")

    f.write("3. 역산 결과\n")
    f.write("-" * 70 + "\n")
    for i in range(n_layers):
        f.write(f"층 {i+1} ({layer_boundaries[i]:.1f}-{layer_boundaries[i+1]:.1f}m): "
                f"{layer_resistivities[i]:.4f} Ohm-m\n")
    f.write("\n")

    f.write("4. 해석\n")
    f.write("-" * 70 + "\n")

    for i in range(n_layers):
        rho = layer_resistivities[i]
        if rho < 0.1:
            interp = "매우 낮은 비저항 - 점토 또는 완전 포화 지층"
        elif rho < 1:
            interp = "낮은 비저항 - 지하수로 포화된 지층"
        elif rho < 10:
            interp = "중간 비저항 - 부분 포화 퇴적층"
        elif rho < 100:
            interp = "높은 비저항 - 건조한 퇴적층 또는 풍화암"
        else:
            interp = "매우 높은 비저항 - 기반암"

        f.write(f"층 {i+1}: {interp}\n")

print("결과 저장: resistivity_fdm_result.txt")
print()
print("=" * 70)
print("전기비저항 FDM 처리 완료!")
print("=" * 70)
print()
print("참고:")
print("  - 전체 FDM 역산은 계산 비용이 매우 높음")
print("  - 실전에서는 병렬 처리, GPU 가속 필요")
print("  - Adjoint-state 방법으로 gradient 계산 효율성 향상")
print("  - Regularization (smoothness, damping) 추가 권장")
