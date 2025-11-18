"""
중력 탐사 고급 자료 처리
- 보조중력계를 이용한 일변화 보정
- 기준점 대비 상대중력 계산
- 구형 이상체 역산
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d

# 중력 상수
G = 6.674e-11  # m^3/(kg*s^2)

# 데이터 로드
data = np.loadtxt('gravity_data.txt')
station_num = data[:, 0].astype(int)
position = data[:, 1]  # m
gravity_obs = data[:, 2]  # mgal
elevation = data[:, 3]  # m
base_station_indicator = data[:, 4].astype(int)  # 1 if base station measurement exists
base_gravity = data[:, 5]  # 보조중력계 측정값 (mgal)

print("=" * 70)
print("고급 중력 탐사 자료 처리")
print("=" * 70)
print(f"측정점 개수: {len(position)}")
print(f"측정 거리: {position[0]}m ~ {position[-1]}m")
print()

# ============================================================================
# 1. 일변화 보정 (Tidal Drift Correction)
# ============================================================================
print("1. 일변화 보정 (Tidal Drift Correction)")
print("-" * 70)

# 보조중력계 데이터 분석
print(f"보조중력계 측정값 범위: {base_gravity.min():.6f} ~ {base_gravity.max():.6f} mgal")
print(f"보조중력계 변화량: {base_gravity.max() - base_gravity.min():.6f} mgal")

# 시간에 따른 선형 드리프트 가정
# 각 측정점이 순차적으로 측정되었다고 가정
time_index = np.arange(len(station_num))

# 보조중력계의 드리프트 추세 계산
drift_trend = base_gravity - base_gravity[0]

print(f"최대 드리프트: {drift_trend.max():.6f} mgal")
print(f"최소 드리프트: {drift_trend.min():.6f} mgal")

# 일변화 보정 적용
gravity_drift_corrected = gravity_obs - drift_trend

print(f"보정 전 중력 범위: {gravity_obs.min():.6f} ~ {gravity_obs.max():.6f} mgal")
print(f"보정 후 중력 범위: {gravity_drift_corrected.min():.6f} ~ {gravity_drift_corrected.max():.6f} mgal")
print()

# ============================================================================
# 2. 기준점 대비 상대중력 계산
# ============================================================================
print("2. 기준점 대비 상대중력 계산")
print("-" * 70)

# 첫 번째 측정점을 기준점으로 설정
reference_station = 0
reference_gravity = gravity_drift_corrected[reference_station]
reference_position = position[reference_station]
reference_elevation = elevation[reference_station]

print(f"기준점: Station {station_num[reference_station]} (위치: {reference_position}m)")
print(f"기준 중력: {reference_gravity:.6f} mgal")
print(f"기준 고도: {reference_elevation:.6f} m")

# 상대중력 계산 (기준점 대비 차이)
relative_gravity = gravity_drift_corrected - reference_gravity

print(f"상대중력 범위: {relative_gravity.min():.6f} ~ {relative_gravity.max():.6f} mgal")
print()

# ============================================================================
# 3. 고도 보정
# ============================================================================
print("3. 고도 보정")
print("-" * 70)

# 기준점 대비 상대 고도
relative_elevation = elevation - reference_elevation

# Free-air correction: -0.3086 mgal/m
free_air_correction = -0.3086 * relative_elevation

# Bouguer correction: +0.04191 * ρ * h mgal
# 배경 암석 밀도 (사암)
background_density = 2000  # kg/m^3
bouguer_correction = 0.04191 * background_density * relative_elevation / 1000

# 지형 보정 (간단한 무한 판 근사)
terrain_correction = 0  # 간단화를 위해 0으로 가정

# 완전 부게 이상 (Complete Bouguer Anomaly)
bouguer_anomaly = relative_gravity + free_air_correction - bouguer_correction

print(f"상대 고도 범위: {relative_elevation.min():.4f} ~ {relative_elevation.max():.4f} m")
print(f"Free-air 보정: {free_air_correction.min():.4f} ~ {free_air_correction.max():.4f} mgal")
print(f"Bouguer 보정: {bouguer_correction.min():.4f} ~ {bouguer_correction.max():.4f} mgal")
print(f"부게 이상: {bouguer_anomaly.min():.4f} ~ {bouguer_anomaly.max():.4f} mgal")
print()

# ============================================================================
# 4. 구형 이상체 모델 역산
# ============================================================================
print("4. 구형 이상체 모델 역산")
print("-" * 70)

def sphere_gravity_anomaly(x_obs, x_center, z_center, radius, density_contrast):
    """
    구형 이상체에 의한 중력 이상 계산

    Parameters:
    -----------
    x_obs : array
        관측점 x 좌표 (m)
    x_center : float
        구의 중심 x 좌표 (m)
    z_center : float
        구의 중심 깊이 (m, 양수는 지하)
    radius : float
        구의 반지름 (m)
    density_contrast : float
        밀도 대비 (kg/m^3)

    Returns:
    --------
    g_anomaly : array
        중력 이상 (mgal)
    """
    g_anomaly = np.zeros_like(x_obs)

    # 구의 질량
    volume = (4.0/3.0) * np.pi * radius**3
    mass = volume * density_contrast

    for i, x in enumerate(x_obs):
        # 관측점과 구 중심 사이의 거리
        r = np.sqrt((x - x_center)**2 + z_center**2)

        # 구에 의한 연직 중력 성분
        # g_z = G * M * z / r^3
        if r > 0:
            g_z = G * mass * z_center / (r**3)
            # mGal 단위로 변환 (1 mGal = 10^-5 m/s^2)
            g_anomaly[i] = g_z * 1e5

    return g_anomaly

def multi_sphere_gravity(x_obs, params):
    """
    다중 구형 이상체 모델
    params = [x1, z1, r1, rho1, x2, z2, r2, rho2, ...]
    """
    n_spheres = len(params) // 4
    g_total = np.zeros_like(x_obs)

    for i in range(n_spheres):
        x_c = params[i*4 + 0]
        z_c = params[i*4 + 1]
        r = params[i*4 + 2]
        rho = params[i*4 + 3]

        g_total += sphere_gravity_anomaly(x_obs, x_c, z_c, r, rho)

    return g_total

def objective_function(params, x_obs, g_obs):
    """역산 목적함수 (RMS 오차)"""
    g_calc = multi_sphere_gravity(x_obs, params)
    residual = g_calc - g_obs
    rms = np.sqrt(np.mean(residual**2))
    return rms

# 초기 모델: 중앙부에 하나의 구형 이상체
# 파라미터: [x_center, z_center, radius, density_contrast]
print("초기 모델 설정:")
print("  - 단일 구형 이상체")
print("  - 위치: 중앙부 (100m)")
print("  - 깊이: 10m")
print("  - 반지름: 20m")
print("  - 밀도 대비: -200 kg/m³")
print()

# 부게 이상의 최대값 위치를 찾아서 이상체 중심 추정
max_anomaly_idx = np.argmax(np.abs(bouguer_anomaly))
initial_x = position[max_anomaly_idx]

initial_params = [
    initial_x,  # x_center
    10.0,       # z_center (depth)
    20.0,       # radius
    -200.0      # density_contrast
]

# 파라미터 경계 설정
bounds = [
    (0, 200),      # x_center
    (5, 50),       # z_center
    (5, 50),       # radius
    (-1000, -50)   # density_contrast (풍화대는 밀도가 낮음)
]

print("역산 시작...")

# Differential Evolution 알고리즘 사용 (전역 최적화)
result = differential_evolution(
    objective_function,
    bounds,
    args=(position, bouguer_anomaly),
    strategy='best1bin',
    maxiter=1000,
    popsize=15,
    tol=0.01,
    mutation=(0.5, 1),
    recombination=0.7,
    seed=42,
    disp=True
)

# 역산 결과
best_params = result.x
x_center_opt = best_params[0]
z_center_opt = best_params[1]
radius_opt = best_params[2]
density_contrast_opt = best_params[3]

print()
print("역산 결과:")
print(f"  구의 중심 위치 (x): {x_center_opt:.2f} m")
print(f"  구의 중심 깊이 (z): {z_center_opt:.2f} m")
print(f"  구의 반지름 (r):    {radius_opt:.2f} m")
print(f"  밀도 대비 (Δρ):     {density_contrast_opt:.2f} kg/m³")
print(f"  풍화대 밀도:        {background_density + density_contrast_opt:.2f} kg/m³")
print(f"  RMS 오차:          {result.fun:.6f} mgal")
print()

# 풍화대 두께 추정 (구의 상단)
weathering_top = z_center_opt - radius_opt
weathering_bottom = z_center_opt + radius_opt
weathering_thickness_equivalent = 2 * radius_opt

print("풍화대 추정:")
print(f"  상단 깊이: {max(0, weathering_top):.2f} m")
print(f"  하단 깊이: {weathering_bottom:.2f} m")
print(f"  등가 두께: {weathering_thickness_equivalent:.2f} m")
print()

# 계산된 중력 이상
g_calculated = multi_sphere_gravity(position, best_params)
residual = bouguer_anomaly - g_calculated

# ============================================================================
# 5. 시각화
# ============================================================================
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

# (1) 원시 데이터 및 일변화 보정
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(position, gravity_obs, 'o-', label='Raw Gravity', markersize=5, linewidth=1.5)
ax1.plot(position, base_gravity, 's-', label='Base Station', markersize=4, linewidth=1, alpha=0.7)
ax1.plot(position, gravity_drift_corrected, '^-', label='Drift Corrected', markersize=5, linewidth=1.5)
ax1.axhline(y=reference_gravity, color='r', linestyle='--', linewidth=1,
            alpha=0.5, label=f'Reference ({reference_gravity:.2f} mgal)')
ax1.set_xlabel('Distance (m)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Gravity (mgal)', fontsize=11, fontweight='bold')
ax1.set_title('Gravity Data and Drift Correction', fontsize=12, fontweight='bold')
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)

# (2) 일변화 드리프트
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(time_index, drift_trend, 'ro-', markersize=5, linewidth=2)
ax2.fill_between(time_index, 0, drift_trend, alpha=0.3, color='red')
ax2.set_xlabel('Measurement Sequence', fontsize=11, fontweight='bold')
ax2.set_ylabel('Drift (mgal)', fontsize=11, fontweight='bold')
ax2.set_title('Tidal Drift from Base Station', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)

# (3) 상대중력 및 부게 이상
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(position, relative_gravity, 'bs-', markersize=5, linewidth=1.5, label='Relative Gravity')
ax3.plot(position, bouguer_anomaly, 'go-', markersize=5, linewidth=1.5, label='Bouguer Anomaly')
ax3.set_xlabel('Distance (m)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Gravity Anomaly (mgal)', fontsize=11, fontweight='bold')
ax3.set_title('Relative Gravity and Bouguer Anomaly', fontsize=11, fontweight='bold')
ax3.legend(loc='best', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)

# (4) 관측 vs 계산 부게 이상
ax4 = fig.add_subplot(gs[2, :])
ax4.plot(position, bouguer_anomaly, 'ko-', markersize=7, linewidth=2,
         label='Observed Bouguer Anomaly', zorder=5)
ax4.plot(position, g_calculated, 'r-', linewidth=3,
         label=f'Calculated (Sphere Model)', zorder=4)
ax4.fill_between(position, bouguer_anomaly, g_calculated,
                  alpha=0.3, color='gray', label=f'Residual (RMS={result.fun:.4f} mgal)')
ax4.set_xlabel('Distance (m)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Bouguer Anomaly (mgal)', fontsize=11, fontweight='bold')
ax4.set_title('Observed vs Calculated Bouguer Anomaly (Spherical Body Model)',
              fontsize=12, fontweight='bold')
ax4.legend(loc='best', fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)

# (5) 잔차
ax5 = fig.add_subplot(gs[3, 0])
ax5.plot(position, residual, 'ms-', markersize=5, linewidth=1.5)
ax5.fill_between(position, 0, residual, alpha=0.3, color='magenta')
ax5.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax5.set_xlabel('Distance (m)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Residual (mgal)', fontsize=11, fontweight='bold')
ax5.set_title(f'Residual (RMS = {result.fun:.4f} mgal)', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)

# (6) 구형 이상체 모델 단면
ax6 = fig.add_subplot(gs[3, 1])

# 배경
ax6.fill_between([0, 200], [0, 0], [-100, -100],
                  color='lightblue', alpha=0.4, label=f'Background ({background_density} kg/m³)')

# 구형 이상체
circle = plt.Circle((x_center_opt, -z_center_opt), radius_opt,
                     color='orange', alpha=0.7,
                     edgecolor='black', linewidth=2,
                     label=f'Sphere (Δρ={density_contrast_opt:.0f} kg/m³)')
ax6.add_patch(circle)

# 구의 중심 표시
ax6.plot(x_center_opt, -z_center_opt, 'r*', markersize=20,
         label=f'Center ({x_center_opt:.1f}m, {z_center_opt:.1f}m depth)')

# 측정점 표시
ax6.scatter(position, np.zeros_like(position), marker='v', s=50,
            c='black', zorder=10, alpha=0.6, label='Stations')

ax6.set_xlabel('Distance (m)', fontsize=11, fontweight='bold')
ax6.set_ylabel('Depth (m)', fontsize=11, fontweight='bold')
ax6.set_title('Spherical Anomaly Model', fontsize=11, fontweight='bold')
ax6.set_xlim(0, 200)
ax6.set_ylim(-80, 10)
ax6.legend(loc='lower right', fontsize=9)
ax6.grid(True, alpha=0.3)
ax6.axhline(y=0, color='k', linewidth=2)

# 등가 풍화대 경계선
ax6.axhline(y=-weathering_top, color='r', linewidth=1.5, linestyle='--',
            alpha=0.7, label=f'Top: {weathering_top:.1f}m')
ax6.axhline(y=-weathering_bottom, color='r', linewidth=1.5, linestyle='--',
            alpha=0.7, label=f'Bottom: {weathering_bottom:.1f}m')

plt.savefig('gravity_result_advanced.png', dpi=300, bbox_inches='tight')
print("결과 그림 저장: gravity_result_advanced.png")
print()

# ============================================================================
# 6. 결과 저장
# ============================================================================
with open('gravity_result_advanced.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("고급 중력 탐사 자료 처리 결과\n")
    f.write("=" * 70 + "\n\n")

    f.write("1. 일변화 보정\n")
    f.write("-" * 70 + "\n")
    f.write(f"최대 드리프트: {drift_trend.max():.6f} mgal\n")
    f.write(f"최소 드리프트: {drift_trend.min():.6f} mgal\n")
    f.write(f"총 변화량: {drift_trend.max() - drift_trend.min():.6f} mgal\n\n")

    f.write("2. 기준점 정보\n")
    f.write("-" * 70 + "\n")
    f.write(f"기준점: Station {station_num[reference_station]}\n")
    f.write(f"위치: {reference_position:.2f} m\n")
    f.write(f"고도: {reference_elevation:.4f} m\n")
    f.write(f"기준 중력: {reference_gravity:.6f} mgal\n\n")

    f.write("3. 부게 이상\n")
    f.write("-" * 70 + "\n")
    f.write(f"최소: {bouguer_anomaly.min():.4f} mgal\n")
    f.write(f"최대: {bouguer_anomaly.max():.4f} mgal\n")
    f.write(f"범위: {bouguer_anomaly.max() - bouguer_anomaly.min():.4f} mgal\n\n")

    f.write("4. 구형 이상체 역산 결과\n")
    f.write("-" * 70 + "\n")
    f.write(f"구의 중심 위치 (x): {x_center_opt:.2f} m\n")
    f.write(f"구의 중심 깊이 (z): {z_center_opt:.2f} m\n")
    f.write(f"구의 반지름 (r): {radius_opt:.2f} m\n")
    f.write(f"밀도 대비 (Δρ): {density_contrast_opt:.2f} kg/m³\n")
    f.write(f"풍화대 밀도: {background_density + density_contrast_opt:.2f} kg/m³\n")
    f.write(f"배경 암석 밀도: {background_density} kg/m³\n")
    f.write(f"RMS 오차: {result.fun:.6f} mgal\n\n")

    f.write("5. 풍화대 추정\n")
    f.write("-" * 70 + "\n")
    f.write(f"상단 깊이: {max(0, weathering_top):.2f} m\n")
    f.write(f"하단 깊이: {weathering_bottom:.2f} m\n")
    f.write(f"등가 두께: {weathering_thickness_equivalent:.2f} m\n\n")

    f.write("6. 관측 데이터\n")
    f.write("-" * 70 + "\n")
    f.write("Station  Position(m)  Bouguer(mgal)  Calculated(mgal)  Residual(mgal)\n")
    for i in range(len(position)):
        f.write(f"{station_num[i]:7d}  {position[i]:11.2f}  {bouguer_anomaly[i]:13.6f}  "
                f"{g_calculated[i]:16.6f}  {residual[i]:14.6f}\n")

print("결과 저장: gravity_result_advanced.txt")
print()
print("=" * 70)
print("고급 중력 탐사 처리 완료!")
print("=" * 70)
