"""
중력 탐사 자료 처리 및 밀도 모델 추정
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.interpolate import interp1d

# 중력 상수
G = 6.674e-11  # m^3/(kg*s^2)

# 데이터 로드
data = np.loadtxt('gravity_data.txt')
station_num = data[:, 0]
position = data[:, 1]  # m
gravity_obs = data[:, 2]  # mgal
elevation = data[:, 3]  # m
base_gravity = data[:, 5]  # 보조중력계 (mgal)

print("=" * 60)
print("중력 탐사 자료 처리")
print("=" * 60)
print(f"측정점 개수: {len(position)}")
print(f"측정 거리: {position[0]}m ~ {position[-1]}m")
print(f"측정 중력 범위: {gravity_obs.min():.2f} ~ {gravity_obs.max():.2f} mgal")
print()

# 1. 일변화 보정 (Drift correction)
# 보조중력계의 시간에 따른 변화를 이용하여 보정
base_drift = base_gravity - base_gravity[0]
gravity_drift_corrected = gravity_obs - base_drift

print("1. 일변화 보정")
print(f"   보조중력계 드리프트: {base_drift.min():.4f} ~ {base_drift.max():.4f} mgal")
print()

# 2. 고도 보정
# Free-air correction: -0.3086 mgal/m
# Bouguer correction: 0.04191 * density (kg/m^3) * h (m) mgal
elevation_relative = elevation - elevation[0]
free_air_correction = -0.3086 * elevation_relative  # mgal

# 배경 암석 밀도 (사암: 2000 kg/m^3)
background_density = 2000  # kg/m^3
bouguer_correction = 0.04191 * background_density * elevation_relative / 1000  # mgal

# 부게 이상 계산
bouguer_anomaly = gravity_drift_corrected + free_air_correction - bouguer_correction
# 기준점 보정 (첫 번째 측정점을 0으로)
bouguer_anomaly = bouguer_anomaly - bouguer_anomaly[0]

print("2. 고도 보정")
print(f"   고도 변화: {elevation_relative.min():.2f} ~ {elevation_relative.max():.2f} m")
print(f"   Free-air 보정: {free_air_correction.min():.4f} ~ {free_air_correction.max():.4f} mgal")
print(f"   Bouguer 보정: {bouguer_correction.min():.4f} ~ {bouguer_correction.max():.4f} mgal")
print(f"   부게 이상: {bouguer_anomaly.min():.4f} ~ {bouguer_anomaly.max():.4f} mgal")
print()

# 3. 2D 밀도 모델 설정 (2층 모델)
# 풍화대와 기반암으로 구성된 모델
# 파라미터: [풍화대 두께(m), 풍화대 밀도(kg/m^3)]

def forward_gravity_2layer(params, x_obs):
    """
    2층 모델에 대한 중력 이상 순방향 모델링
    params: [layer1_thickness, layer1_density, layer2_thickness, layer2_density, ...]
    """
    # 단순화된 무한 수평 판 모델 (Bouguer slab)
    # g = 2 * pi * G * rho * h

    # 풍화대 모델링 (중앙부에 저밀도 이상체 가정)
    weathering_center = 100  # m
    weathering_width = 60  # m
    weathering_depth = params[0]  # m
    weathering_density_contrast = params[1]  # kg/m^3 (밀도 대비)

    g_anomaly = np.zeros_like(x_obs)

    # 각 관측점에 대해 중력 이상 계산
    for i, x in enumerate(x_obs):
        # 무한 수평 판 근사
        # 풍화대의 영향을 가우시안 함수로 근사
        distance = abs(x - weathering_center)
        if distance < weathering_width:
            # 무한 판 공식: g = 2 * pi * G * rho * h
            g_slab = 2 * np.pi * G * weathering_density_contrast * weathering_depth
            # 가우시안 가중치
            weight = np.exp(-(distance / weathering_width) ** 2)
            g_anomaly[i] = g_slab * weight * 1e5  # m/s^2 to mgal

    return g_anomaly

def residual_func(params, x_obs, g_obs):
    """잔차 함수"""
    g_calc = forward_gravity_2layer(params, x_obs)
    return g_calc - g_obs

# 초기 추정값
# [풍화대 두께(m), 밀도 대비(kg/m^3)]
initial_params = [20, -500]  # 풍화대는 배경보다 밀도가 낮음
bounds = ([5, -1000], [50, -100])  # 파라미터 범위

print("3. 밀도 모델 역산")
print(f"   초기 모델: 풍화대 두께 = {initial_params[0]} m, 밀도 대비 = {initial_params[1]} kg/m³")

# 비선형 최소제곱법으로 역산
result = least_squares(
    residual_func,
    initial_params,
    args=(position, bouguer_anomaly),
    bounds=bounds,
    verbose=0
)

# 역산 결과
best_params = result.x
weathering_thickness = best_params[0]
weathering_density_contrast = best_params[1]
weathering_density = background_density + weathering_density_contrast

print(f"   역산 결과:")
print(f"   - 풍화대 두께: {weathering_thickness:.2f} m")
print(f"   - 풍화대 밀도 대비: {weathering_density_contrast:.2f} kg/m³")
print(f"   - 풍화대 밀도: {weathering_density:.2f} kg/m³")
print(f"   - RMS 오차: {np.sqrt(np.mean(result.fun**2)):.4f} mgal")
print()

# 순방향 모델링 결과
g_calculated = forward_gravity_2layer(best_params, position)

# 시각화
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# (1) 원시 데이터 및 보정
ax1 = axes[0]
ax1.plot(position, gravity_obs, 'o-', label='관측 중력', markersize=4)
ax1.plot(position, gravity_drift_corrected, 's-', label='일변화 보정', markersize=4)
ax1.set_xlabel('거리 (m)')
ax1.set_ylabel('중력 (mgal)')
ax1.set_title('중력 관측 데이터 및 일변화 보정')
ax1.legend()
ax1.grid(True, alpha=0.3)

# (2) 부게 이상 및 모델 적합
ax2 = axes[1]
ax2.plot(position, bouguer_anomaly, 'ko-', label='관측 부게 이상', markersize=6)
ax2.plot(position, g_calculated, 'r-', linewidth=2, label='계산 부게 이상')
ax2.fill_between(position, 0, bouguer_anomaly, alpha=0.2, color='blue')
ax2.set_xlabel('거리 (m)')
ax2.set_ylabel('부게 이상 (mgal)')
ax2.set_title('부게 이상 및 모델 적합')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# (3) 밀도 모델 단면
ax3 = axes[2]
# 배경
ax3.fill_between([0, 200], [0, 0], [-100, -100],
                  color='lightblue', alpha=0.5, label=f'사암 ({background_density} kg/m³)')
# 풍화대
weathering_x = [100 - 30, 100 + 30]
weathering_y = [0, 0]
weathering_bottom = [-weathering_thickness, -weathering_thickness]
ax3.fill_between(weathering_x, weathering_y, weathering_bottom,
                  color='orange', alpha=0.7, label=f'풍화대 ({weathering_density:.0f} kg/m³)')
ax3.set_xlabel('거리 (m)')
ax3.set_ylabel('깊이 (m)')
ax3.set_title('추정된 밀도 모델')
ax3.set_xlim(0, 200)
ax3.set_ylim(-80, 10)
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='k', linewidth=2)

plt.tight_layout()
plt.savefig('gravity_result.png', dpi=300, bbox_inches='tight')
print("결과 그림 저장: gravity_result.png")
print()

# 결과를 파일로 저장
with open('gravity_result.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("중력 탐사 자료 처리 결과\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"측정점 개수: {len(position)}\n")
    f.write(f"측정 거리: {position[0]}m ~ {position[-1]}m\n\n")
    f.write("역산 결과:\n")
    f.write(f"  - 풍화대 두께: {weathering_thickness:.2f} m\n")
    f.write(f"  - 풍화대 밀도: {weathering_density:.2f} kg/m³\n")
    f.write(f"  - 배경 암석 밀도: {background_density} kg/m³\n")
    f.write(f"  - 밀도 대비: {weathering_density_contrast:.2f} kg/m³\n")
    f.write(f"  - RMS 오차: {np.sqrt(np.mean(result.fun**2)):.4f} mgal\n\n")
    f.write("부게 이상 데이터:\n")
    f.write("위치(m)  부게이상(mgal)  계산값(mgal)  잔차(mgal)\n")
    for i in range(len(position)):
        f.write(f"{position[i]:6.1f}  {bouguer_anomaly[i]:14.4f}  {g_calculated[i]:13.4f}  "
                f"{bouguer_anomaly[i] - g_calculated[i]:11.4f}\n")

print("결과 저장: gravity_result.txt")
print()
print("=" * 60)
print("중력 탐사 처리 완료!")
print("=" * 60)
