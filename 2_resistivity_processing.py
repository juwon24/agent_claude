"""
전기비저항 탐사 자료 처리 및 비저항 모델 추정
쌍극자-쌍극자 배열 (Dipole-Dipole Array)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.interpolate import griddata

# 데이터 로드
with open('resistivity_data.dat', 'r') as f:
    lines = f.readlines()

n_data = int(lines[0].strip())
data = []
for i in range(1, n_data + 1):
    parts = lines[i].strip().split()
    data.append([float(x) for x in parts])

data = np.array(data)
print("=" * 60)
print("전기비저항 탐사 자료 처리")
print("=" * 60)
print(f"데이터 개수: {n_data}")
print()

# 데이터 파싱
point_num = data[:, 0]
c1 = data[:, 1].astype(int)  # 전류 전극 1
c2 = data[:, 2].astype(int)  # 전류 전극 2
p1 = data[:, 3].astype(int)  # 전위 전극 1
p2 = data[:, 4].astype(int)  # 전위 전극 2
app_res = data[:, 5]  # 겉보기 비저항 (Ohm-m)

# 전극 간격
electrode_spacing = 10  # m

# 쌍극자-쌍극자 배열 분석
# a: 전극 간격 (C1-C2 또는 P1-P2)
# n: 쌍극자 간 간격 배수
a = electrode_spacing
n_factor = p1 - c2  # n factor

# 가상깊이 계산 (Edwards, 1977)
# 쌍극자-쌍극자 배열의 조사 깊이
# 중앙 위치: (C1 + P2) / 2
# 깊이: a * n / 2

x_pos = []
z_pos = []
for i in range(len(data)):
    # 전극 위치 계산 (전극 번호 - 1) * spacing
    c1_pos = (c1[i] - 1) * electrode_spacing
    c2_pos = (c2[i] - 1) * electrode_spacing
    p1_pos = (p1[i] - 1) * electrode_spacing
    p2_pos = (p2[i] - 1) * electrode_spacing

    # 중앙 위치
    x_center = (c1_pos + p2_pos) / 2.0

    # n factor
    n = p1[i] - c2[i]

    # 가상 깊이 (Edwards 1977)
    z_depth = a * n * 0.5

    x_pos.append(x_center)
    z_pos.append(z_depth)

x_pos = np.array(x_pos)
z_pos = np.array(z_pos)

print("1. 겉보기 비저항 분석")
print(f"   비저항 범위: {app_res.min():.4f} ~ {app_res.max():.4f} Ohm-m")
print(f"   평균 비저항: {app_res.mean():.4f} Ohm-m")
print(f"   조사 깊이: 0 ~ {z_pos.max():.1f} m")
print()

# 2D 단면도 작성을 위한 그리드 생성
x_grid = np.linspace(0, 200, 100)
z_grid = np.linspace(0, z_pos.max() + 10, 50)
X_grid, Z_grid = np.meshgrid(x_grid, z_grid)

# 보간 (kriging 또는 IDW 대신 griddata 사용)
points = np.column_stack([x_pos, z_pos])
res_grid = griddata(points, app_res, (X_grid, Z_grid), method='cubic', fill_value=app_res.mean())

print("2. 2D 단면 생성")
print(f"   그리드 크기: {len(x_grid)} x {len(z_grid)}")
print()

# 간단한 역산 (층상 모델)
# 깊이별 평균 비저항 계산
depth_bins = [0, 10, 20, 30, 50]
layer_resistivity = []
layer_depth = []

for i in range(len(depth_bins) - 1):
    mask = (z_pos >= depth_bins[i]) & (z_pos < depth_bins[i+1])
    if np.sum(mask) > 0:
        avg_res = np.mean(app_res[mask])
        layer_resistivity.append(avg_res)
        layer_depth.append((depth_bins[i] + depth_bins[i+1]) / 2)

print("3. 층상 모델 역산")
print("   깊이 구간별 평균 비저항:")
for i in range(len(layer_resistivity)):
    print(f"   {depth_bins[i]:3d} - {depth_bins[i+1]:3d} m: {layer_resistivity[i]:.4f} Ohm-m")
print()

# 풍화대 추정
# 높은 비저항 영역 (풍화대 - 포화되지 않은 암석)
# 낮은 비저항 영역 (지하수 포화 암석)

# 비저항 이상 위치 추정 (중앙부에 저비저항 이상?)
weathering_zone_res = layer_resistivity[0] if len(layer_resistivity) > 0 else app_res.mean()
bedrock_res = layer_resistivity[-1] if len(layer_resistivity) > 1 else app_res.mean()

print("4. 지질학적 해석")
print(f"   표층 비저항: {weathering_zone_res:.4f} Ohm-m")
print(f"   심부 비저항: {bedrock_res:.4f} Ohm-m")

# 비저항 값에 따른 지질 해석
if weathering_zone_res < 1.0:
    print("   표층 해석: 점토질 풍화대 또는 포화된 퇴적층")
elif weathering_zone_res < 10:
    print("   표층 해석: 습윤 퇴적층 또는 부분 풍화대")
else:
    print("   표층 해석: 건조 풍화대 또는 사질층")

if bedrock_res < 100:
    print("   심부 해석: 포화된 기반암 또는 점토층")
else:
    print("   심부 해석: 건조한 기반암")
print()

# 시각화
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# (1) 겉보기 비저항 가상단면도 (Pseudosection)
ax1 = axes[0]

# 로그 스케일 사용
res_log = np.log10(res_grid)
vmin, vmax = np.log10(app_res.min()), np.log10(app_res.max())

# 컨투어 플롯
cf = ax1.contourf(X_grid, -Z_grid, res_log, levels=20, cmap='jet_r', vmin=vmin, vmax=vmax)
ax1.scatter(x_pos, -z_pos, c=np.log10(app_res), s=30, edgecolors='white',
            linewidths=0.5, cmap='jet_r', vmin=vmin, vmax=vmax)

# 전극 위치 표시
electrode_positions = np.arange(0, 21) * electrode_spacing
ax1.scatter(electrode_positions, np.zeros_like(electrode_positions),
            marker='v', s=100, c='black', zorder=10, label='Electrodes')

ax1.set_xlabel('Distance (m)', fontsize=11)
ax1.set_ylabel('Depth (m)', fontsize=11)
ax1.set_title('Apparent Resistivity Pseudosection (Dipole-Dipole Array)', fontsize=12, fontweight='bold')
ax1.set_xlim(0, 200)
ax1.set_ylim(-60, 5)
ax1.grid(True, alpha=0.3)

# 컬러바
cbar = plt.colorbar(cf, ax=ax1, label='Resistivity [Ohm-m]')
# 로그 눈금을 실제 값으로 변환
tick_locs = np.linspace(vmin, vmax, 6)
cbar.set_ticks(tick_locs)
cbar.set_ticklabels([f'{10**t:.2f}' for t in tick_locs])

# (2) 층상 모델
ax2 = axes[1]

# 전체 배경 (기반암)
ax2.fill_between([0, 200], [0, 0], [-60, -60], color='lightblue', alpha=0.5)

# 층별로 그리기
colors_layer = plt.cm.jet_r(np.linspace(0, 1, len(layer_resistivity)))
for i in range(len(layer_resistivity)):
    top = -depth_bins[i]
    bottom = -depth_bins[i+1]
    color_idx = int((layer_resistivity[i] - min(layer_resistivity)) /
                   (max(layer_resistivity) - min(layer_resistivity) + 1e-10) * (len(colors_layer) - 1))
    ax2.fill_between([0, 200], [top, top], [bottom, bottom],
                      color=colors_layer[color_idx], alpha=0.7,
                      label=f'{depth_bins[i]}-{depth_bins[i+1]}m: {layer_resistivity[i]:.2f} Ohm-m')

ax2.set_xlabel('Distance (m)', fontsize=11)
ax2.set_ylabel('Depth (m)', fontsize=11)
ax2.set_title('Layered Resistivity Model', fontsize=12, fontweight='bold')
ax2.set_xlim(0, 200)
ax2.set_ylim(-60, 5)
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linewidth=2)

plt.tight_layout()
plt.savefig('resistivity_result.png', dpi=300, bbox_inches='tight')
print("결과 그림 저장: resistivity_result.png")
print()

# 결과 저장
with open('resistivity_result.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("전기비저항 탐사 자료 처리 결과\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"탐사 배열: 쌍극자-쌍극자\n")
    f.write(f"전극 개수: 21개\n")
    f.write(f"전극 간격: {electrode_spacing} m\n")
    f.write(f"데이터 개수: {n_data}\n\n")
    f.write("겉보기 비저항 통계:\n")
    f.write(f"  최소: {app_res.min():.4f} Ohm-m\n")
    f.write(f"  최대: {app_res.max():.4f} Ohm-m\n")
    f.write(f"  평균: {app_res.mean():.4f} Ohm-m\n\n")
    f.write("층상 모델 역산 결과:\n")
    for i in range(len(layer_resistivity)):
        f.write(f"  {depth_bins[i]:3d} - {depth_bins[i+1]:3d} m: {layer_resistivity[i]:.4f} Ohm-m\n")
    f.write("\n")
    f.write("지질학적 해석:\n")
    f.write(f"  표층 비저항: {weathering_zone_res:.4f} Ohm-m\n")
    f.write(f"  심부 비저항: {bedrock_res:.4f} Ohm-m\n")

print("결과 저장: resistivity_result.txt")
print()
print("=" * 60)
print("전기비저항 탐사 처리 완료!")
print("=" * 60)
