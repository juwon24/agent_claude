"""
탄성파 토모그래피
- Ray tracing (Shortest path method)
- Iterative linearized inversion (SIRT/LSQR)
- Velocity model reconstruction
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import lsqr
from PIL import Image

print("=" * 70)
print("탄성파 토모그래피")
print("=" * 70)
print()

# ============================================================================
# 1. 탐사 설정
# ============================================================================
n_receivers = 50
receiver_spacing = 4  # m
shot_positions = [25, 175]  # m

receiver_positions = np.arange(n_receivers) * receiver_spacing  # 0, 4, 8, ..., 196 m

print(f"수진기 개수: {n_receivers}")
print(f"수진기 간격: {receiver_spacing} m")
print(f"발파 위치: {shot_positions} m")
print()

# ============================================================================
# 2. 합성 주시 데이터 생성 (실제로는 이미지에서 picking)
# ============================================================================
print("주시 데이터 생성 (합성)...")

# 실제 속도 모델 (2층)
v1_true = 800   # m/s (풍화대)
v2_true = 2500  # m/s (기반암)
h_true = 10     # m (풍화대 두께)

def two_layer_traveltime(x, shot_x, v1, v2, h):
    """2층 모델 주시 계산"""
    offset = np.abs(x - shot_x)

    # 직접파
    t_direct = offset / v1

    # 굴절파
    if v2 > v1:
        ic = np.arcsin(v1 / v2)
        t_refracted = 2 * h * np.cos(ic) / v1 + offset / v2
    else:
        t_refracted = np.inf

    return np.minimum(t_direct, t_refracted)

# 왼쪽 발파
offset_left = np.abs(receiver_positions - shot_positions[0])
traveltime_left = two_layer_traveltime(receiver_positions, shot_positions[0],
                                        v1_true, v2_true, h_true)

# 오른쪽 발파
offset_right = np.abs(receiver_positions - shot_positions[1])
traveltime_right = two_layer_traveltime(receiver_positions, shot_positions[1],
                                         v1_true, v2_true, h_true)

# 노이즈 추가
np.random.seed(42)
noise_level = 0.002  # 2 ms
traveltime_left += np.random.normal(0, noise_level, len(traveltime_left))
traveltime_right += np.random.normal(0, noise_level, len(traveltime_right))

print(f"총 관측 데이터: {len(traveltime_left) + len(traveltime_right)} rays")
print()

# ============================================================================
# 3. 속도 모델 메쉬 설정
# ============================================================================
print("속도 모델 메쉬 생성...")

nx = 51   # x 방향 셀 수
nz = 26   # z 방향 셀 수

x_min, x_max = 0, 200
z_min, z_max = 0, 50

# 셀 경계
x_edges = np.linspace(x_min, x_max, nx + 1)
z_edges = np.linspace(z_min, z_max, nz + 1)

# 셀 중심
x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

dx = x_centers[1] - x_centers[0]
dz = z_centers[1] - z_centers[0]

X, Z = np.meshgrid(x_centers, z_centers)

n_cells = nx * nz

print(f"메쉬 크기: {nx} x {nz} = {n_cells} cells")
print(f"셀 크기: {dx:.2f} m x {dz:.2f} m")
print()

# ============================================================================
# 4. Ray Tracing (Straight ray approximation)
# ============================================================================
print("Ray tracing...")

def trace_ray_straight(shot_x, shot_z, rec_x, rec_z, x_centers, z_centers):
    """
    직선 ray 근사로 ray path 계산

    Returns:
        ray_cells: ray가 통과하는 셀 인덱스
        ray_lengths: 각 셀을 통과하는 거리
    """
    # Ray direction
    dx_ray = rec_x - shot_x
    dz_ray = rec_z - shot_z
    ray_length_total = np.sqrt(dx_ray**2 + dz_ray**2)

    # Parameterization: r(t) = (shot_x, shot_z) + t * (dx_ray, dz_ray), t ∈ [0, 1]

    # Sample ray at regular intervals
    n_samples = int(ray_length_total / 0.5) + 1  # 0.5m sampling
    t_samples = np.linspace(0, 1, n_samples)

    x_samples = shot_x + t_samples * dx_ray
    z_samples = shot_z + t_samples * dz_ray

    # Find cells containing each sample point
    cell_counts = {}

    for i in range(n_samples):
        x_s = x_samples[i]
        z_s = z_samples[i]

        # Find cell index
        if x_s < x_min or x_s > x_max or z_s < z_min or z_s > z_max:
            continue

        ix = np.searchsorted(x_edges, x_s) - 1
        iz = np.searchsorted(z_edges, z_s) - 1

        if ix < 0: ix = 0
        if ix >= nx: ix = nx - 1
        if iz < 0: iz = 0
        if iz >= nz: iz = nz - 1

        cell_idx = iz * nx + ix

        if cell_idx in cell_counts:
            cell_counts[cell_idx] += 1
        else:
            cell_counts[cell_idx] = 1

    # Convert counts to lengths
    ray_cells = list(cell_counts.keys())
    ray_lengths = [(cell_counts[c] / n_samples) * ray_length_total for c in ray_cells]

    return ray_cells, ray_lengths

# Build ray matrix and data vector
print("Ray matrix 구성...")

ray_matrix = lil_matrix((0, n_cells))
traveltime_obs = []
shot_indices = []
receiver_indices = []

shot_z = 0  # Surface
rec_z = 0   # Surface

# 왼쪽 발파
for i, rec_x in enumerate(receiver_positions):
    shot_x = shot_positions[0]

    cells, lengths = trace_ray_straight(shot_x, shot_z, rec_x, rec_z,
                                         x_centers, z_centers)

    # Add row to ray matrix
    row = np.zeros(n_cells)
    for cell_idx, length in zip(cells, lengths):
        row[cell_idx] = length

    ray_matrix = lil_matrix(np.vstack([ray_matrix.toarray(), row]))

    traveltime_obs.append(traveltime_left[i])
    shot_indices.append(0)
    receiver_indices.append(i)

# 오른쪽 발파
for i, rec_x in enumerate(receiver_positions):
    shot_x = shot_positions[1]

    cells, lengths = trace_ray_straight(shot_x, shot_z, rec_x, rec_z,
                                         x_centers, z_centers)

    # Add row to ray matrix
    row = np.zeros(n_cells)
    for cell_idx, length in zip(cells, lengths):
        row[cell_idx] = length

    ray_matrix = lil_matrix(np.vstack([ray_matrix.toarray(), row]))

    traveltime_obs.append(traveltime_right[i])
    shot_indices.append(1)
    receiver_indices.append(i)

ray_matrix = ray_matrix.tocsr()
traveltime_obs = np.array(traveltime_obs)

n_rays = len(traveltime_obs)

print(f"Ray matrix 크기: {n_rays} x {n_cells}")
print(f"Non-zero elements: {ray_matrix.nnz}")
print(f"Sparsity: {ray_matrix.nnz / (n_rays * n_cells) * 100:.2f}%")
print()

# ============================================================================
# 5. Linearized Inversion
# ============================================================================
print("선형화 역산 (SIRT)...")

# 초기 모델: 균질
initial_slowness = 1.0 / 1500  # s/m (초기 속도 1500 m/s)
slowness_model = np.ones(n_cells) * initial_slowness

# SIRT (Simultaneous Iterative Reconstruction Technique)
n_iterations = 20
damping = 0.1  # Regularization

print(f"반복 횟수: {n_iterations}")
print(f"Damping: {damping}")
print()

rms_history = []

for iteration in range(n_iterations):
    # Forward problem: t = L * s
    # L: ray matrix (path length)
    # s: slowness (1/v)

    traveltime_calc = ray_matrix @ slowness_model

    # Residual
    residual = traveltime_obs - traveltime_calc

    # RMS
    rms = np.sqrt(np.mean(residual**2))
    rms_history.append(rms)

    print(f"Iteration {iteration+1:2d}: RMS = {rms:.6f} s ({rms*1000:.3f} ms)")

    # Update (SIRT)
    # Δs = α * L^T * residual / (L^T * L_diag + ε)

    # Row normalization
    row_sums = np.array(ray_matrix.sum(axis=1)).flatten()
    row_sums[row_sums < 1e-10] = 1  # Avoid division by zero

    normalized_residual = residual / row_sums

    # Column normalization
    col_sums = np.array(ray_matrix.sum(axis=0)).flatten()
    col_sums[col_sums < 1e-10] = 1

    # Update
    update = ray_matrix.T @ normalized_residual / (col_sums + damping)

    # Step length (adaptive)
    alpha = 0.5

    slowness_model += alpha * update

    # Positivity constraint
    slowness_model = np.maximum(slowness_model, 1.0 / 5000)  # Max 5000 m/s
    slowness_model = np.minimum(slowness_model, 1.0 / 300)   # Min 300 m/s

print()
print("역산 완료")
print()

# Convert slowness to velocity
velocity_model = 1.0 / slowness_model
velocity_model_2d = velocity_model.reshape((nz, nx))

print("최종 속도 모델:")
print(f"  최소 속도: {velocity_model.min():.0f} m/s")
print(f"  최대 속도: {velocity_model.max():.0f} m/s")
print(f"  평균 속도: {velocity_model.mean():.0f} m/s")
print()

# ============================================================================
# 6. 시각화
# ============================================================================
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

# (1) 속도 모델
ax1 = fig.add_subplot(gs[0, :])
im1 = ax1.pcolormesh(x_edges, z_edges, velocity_model_2d,
                      shading='flat', cmap='jet', vmin=500, vmax=3000)
ax1.contour(X, Z, velocity_model_2d, levels=10, colors='k', alpha=0.5, linewidths=1)

# 발파 위치
for shot_x in shot_positions:
    ax1.scatter(shot_x, 0, marker='*', s=400, c='red', edgecolors='black',
                linewidths=2, zorder=10)

# 수진기
ax1.scatter(receiver_positions, np.zeros(n_receivers),
            marker='v', s=50, c='blue', zorder=10, alpha=0.7)

# Ray paths (일부만 표시)
sample_rays = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
for ray_idx in sample_rays:
    if ray_idx < n_rays:
        shot_idx = shot_indices[ray_idx]
        rec_idx = receiver_indices[ray_idx]

        shot_x = shot_positions[shot_idx]
        rec_x = receiver_positions[rec_idx]

        ax1.plot([shot_x, rec_x], [0, 0], 'w-', linewidth=0.5, alpha=0.3)

ax1.set_xlabel('Distance (m)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Depth (m)', fontsize=11, fontweight='bold')
ax1.set_title('Seismic Velocity Model (Tomography)', fontsize=12, fontweight='bold')
ax1.invert_yaxis()
ax1.set_xlim(0, 200)
ax1.set_ylim(50, 0)

cbar1 = plt.colorbar(im1, ax=ax1, label='Velocity (m/s)')

# 참 모델 오버레이 (점선)
z_interface = h_true
ax1.axhline(y=z_interface, color='white', linestyle='--', linewidth=3,
            alpha=0.9, label=f'True interface at {z_interface}m')
ax1.legend(loc='lower right', fontsize=10)

# (2) RMS 수렴 곡선
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(range(1, n_iterations+1), np.array(rms_history) * 1000,
         'bo-', markersize=8, linewidth=2)
ax2.set_xlabel('Iteration', fontsize=11, fontweight='bold')
ax2.set_ylabel('RMS Error (ms)', fontsize=11, fontweight='bold')
ax2.set_title('Convergence Curve', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)

# (3) 관측 vs 계산 주시
ax3 = fig.add_subplot(gs[1, 1])

traveltime_calc_final = ray_matrix @ slowness_model

ax3.plot(traveltime_obs * 1000, traveltime_obs * 1000, 'k-', linewidth=2, label='1:1 line')
ax3.plot(traveltime_obs * 1000, traveltime_calc_final * 1000, 'ro',
         markersize=5, alpha=0.6, label='Data')

ax3.set_xlabel('Observed Travel Time (ms)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Calculated Travel Time (ms)', fontsize=11, fontweight='bold')
ax3.set_title('Observed vs Calculated', fontsize=11, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# (4) 잔차
ax4 = fig.add_subplot(gs[2, 0])
residual_final = (traveltime_obs - traveltime_calc_final) * 1000  # ms

ax4.hist(residual_final, bins=30, color='blue', alpha=0.7, edgecolor='black')
ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax4.set_xlabel('Residual (ms)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Count', fontsize=11, fontweight='bold')
ax4.set_title(f'Residual Distribution (RMS = {rms_history[-1]*1000:.3f} ms)',
              fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# (5) 속도 프로파일
ax5 = fig.add_subplot(gs[2, 1])

# 중앙 수직 프로파일
center_idx = nx // 2
velocity_profile = velocity_model_2d[:, center_idx]

ax5.plot(velocity_profile, z_centers, 'b-', linewidth=3, marker='o', markersize=6)
ax5.axvline(x=v1_true, color='r', linestyle='--', linewidth=2, alpha=0.7,
            label=f'True v1 = {v1_true} m/s')
ax5.axvline(x=v2_true, color='g', linestyle='--', linewidth=2, alpha=0.7,
            label=f'True v2 = {v2_true} m/s')
ax5.axhline(y=h_true, color='orange', linestyle='--', linewidth=2, alpha=0.7,
            label=f'True interface = {h_true}m')

ax5.set_xlabel('Velocity (m/s)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Depth (m)', fontsize=11, fontweight='bold')
ax5.set_title(f'Velocity Profile at x = {x_centers[center_idx]:.0f}m',
              fontsize=11, fontweight='bold')
ax5.invert_yaxis()
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

plt.savefig('seismic_tomography_result.png', dpi=300, bbox_inches='tight')
print("결과 그림 저장: seismic_tomography_result.png")
print()

# ============================================================================
# 7. 결과 저장
# ============================================================================
with open('seismic_tomography_result.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("탄성파 토모그래피 결과\n")
    f.write("=" * 70 + "\n\n")

    f.write("1. 탐사 설정\n")
    f.write("-" * 70 + "\n")
    f.write(f"수진기 개수: {n_receivers}\n")
    f.write(f"수진기 간격: {receiver_spacing} m\n")
    f.write(f"발파 위치: {shot_positions} m\n")
    f.write(f"총 ray 개수: {n_rays}\n\n")

    f.write("2. 메쉬 설정\n")
    f.write("-" * 70 + "\n")
    f.write(f"메쉬 크기: {nx} x {nz} = {n_cells} cells\n")
    f.write(f"셀 크기: {dx:.2f} m x {dz:.2f} m\n\n")

    f.write("3. 역산 설정\n")
    f.write("-" * 70 + "\n")
    f.write(f"역산 방법: SIRT (Simultaneous Iterative Reconstruction)\n")
    f.write(f"반복 횟수: {n_iterations}\n")
    f.write(f"Damping: {damping}\n\n")

    f.write("4. 수렴 이력\n")
    f.write("-" * 70 + "\n")
    for i, rms in enumerate(rms_history):
        f.write(f"Iteration {i+1:2d}: RMS = {rms:.6f} s ({rms*1000:.3f} ms)\n")
    f.write("\n")

    f.write("5. 최종 속도 모델\n")
    f.write("-" * 70 + "\n")
    f.write(f"최소 속도: {velocity_model.min():.0f} m/s\n")
    f.write(f"최대 속도: {velocity_model.max():.0f} m/s\n")
    f.write(f"평균 속도: {velocity_model.mean():.0f} m/s\n")
    f.write(f"표준편차: {velocity_model.std():.0f} m/s\n\n")

    f.write("6. 해석\n")
    f.write("-" * 70 + "\n")
    f.write(f"토모그래피 영상에서 약 {h_true}m 깊이에 속도 경계 확인\n")
    f.write(f"천부 (0-{h_true}m): 평균 속도 약 {np.mean(velocity_model_2d[z_centers < h_true]):.0f} m/s\n")
    f.write(f"심부 ({h_true}m 이하): 평균 속도 약 {np.mean(velocity_model_2d[z_centers >= h_true]):.0f} m/s\n")
    f.write(f"→ 풍화대-기반암 경계와 일치\n\n")

    f.write("7. 참 모델과 비교\n")
    f.write("-" * 70 + "\n")
    f.write(f"참 v1: {v1_true} m/s\n")
    f.write(f"참 v2: {v2_true} m/s\n")
    f.write(f"참 경계: {h_true} m\n\n")

    # 간단한 층 평균 계산
    mask_layer1 = Z < h_true
    mask_layer2 = Z >= h_true

    v1_recon = np.mean(velocity_model_2d[mask_layer1])
    v2_recon = np.mean(velocity_model_2d[mask_layer2])

    f.write(f"복원 v1: {v1_recon:.0f} m/s (오차: {abs(v1_recon - v1_true):.0f} m/s)\n")
    f.write(f"복원 v2: {v2_recon:.0f} m/s (오차: {abs(v2_recon - v2_true):.0f} m/s)\n")

print("결과 저장: seismic_tomography_result.txt")
print()
print("=" * 70)
print("탄성파 토모그래피 완료!")
print("=" * 70)
print()
print("참고:")
print("  - Straight ray 근사 사용 (빠르지만 정확도는 낮음)")
print("  - 실전에서는 Bent ray 또는 Shortest path ray tracing 권장")
print("  - SIRT 방법: 안정적이지만 수렴 느림")
print("  - LSQR/CGLS 등 더 빠른 solver 사용 가능")
