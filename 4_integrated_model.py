"""
세 가지 탐사 방법의 통합 지하 지질구조 모델
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

print("=" * 60)
print("통합 지하 지질구조 모델")
print("=" * 60)
print()

# 각 탐사 방법의 결과
# 1. 중력 탐사
gravity_weathering_thickness = 7.21  # m
gravity_weathering_density = 1841.35  # kg/m³
gravity_bedrock_density = 2000  # kg/m³

# 2. 전기비저항 탐사
resistivity_layer1 = 3.30  # Ohm-m (0-10m)
resistivity_layer2 = 0.32  # Ohm-m (10-20m)
resistivity_layer3 = 0.05  # Ohm-m (20-30m)

# 3. 탄성파 탐사
seismic_weathering_thickness = 13.62  # m
seismic_v1 = 1023  # m/s
seismic_v2 = 2564  # m/s

print("개별 탐사 결과:")
print()
print("1. 중력 탐사")
print(f"   풍화대 두께: {gravity_weathering_thickness:.2f} m")
print(f"   풍화대 밀도: {gravity_weathering_density:.0f} kg/m³")
print(f"   기반암 밀도: {gravity_bedrock_density:.0f} kg/m³")
print()
print("2. 전기비저항 탐사")
print(f"   0-10m:  {resistivity_layer1:.2f} Ohm-m")
print(f"   10-20m: {resistivity_layer2:.2f} Ohm-m")
print(f"   20-30m: {resistivity_layer3:.2f} Ohm-m")
print()
print("3. 탄성파 탐사")
print(f"   풍화대 두께: {seismic_weathering_thickness:.2f} m")
print(f"   풍화대 속도: {seismic_v1:.0f} m/s")
print(f"   기반암 속도: {seismic_v2:.0f} m/s")
print()

# 통합 모델 제시
print("=" * 60)
print("통합 지질구조 모델")
print("=" * 60)
print()

# 풍화대 두께: 중력과 탄성파의 평균 사용
integrated_weathering_thickness = (gravity_weathering_thickness + seismic_weathering_thickness) / 2

print("제안 모델:")
print()
print(f"층 1: 표층 풍화대 (0 ~ {integrated_weathering_thickness:.1f} m)")
print(f"  - 밀도: {gravity_weathering_density:.0f} kg/m³")
print(f"  - 탄성파 속도: {seismic_v1:.0f} m/s")
print(f"  - 전기비저항: {resistivity_layer1:.2f} Ohm-m")
print(f"  - 해석: 부분 풍화된 사암층, 공극률이 높고 부분적으로 지하수 포화")
print()
print(f"층 2: 기반암 ({integrated_weathering_thickness:.1f} m 이하)")
print(f"  - 밀도: {gravity_bedrock_density:.0f} kg/m³")
print(f"  - 탄성파 속도: {seismic_v2:.0f} m/s")
print(f"  - 전기비저항: {resistivity_layer2:.2f} Ohm-m (천부), {resistivity_layer3:.2f} Ohm-m (심부)")
print(f"  - 해석: 지하수로 포화된 사암층, 깊이에 따라 공극률 감소")
print()

# 물리적 특성 상관관계 분석
print("=" * 60)
print("물리적 특성 상관관계 분석")
print("=" * 60)
print()

# 공극률 추정 (밀도로부터)
# ρ_bulk = ρ_matrix * (1 - φ) + ρ_water * φ
rho_matrix = 2650  # kg/m³ (사암 매트릭스)
rho_water = 1000   # kg/m³

porosity_weathering = (rho_matrix - gravity_weathering_density) / (rho_matrix - rho_water)
porosity_bedrock = (rho_matrix - gravity_bedrock_density) / (rho_matrix - rho_water)

print("공극률 추정 (중력 데이터로부터):")
print(f"  풍화대: {porosity_weathering * 100:.1f}%")
print(f"  기반암: {porosity_bedrock * 100:.1f}%")
print()

# 포화도 추정 (전기비저항으로부터)
# Archie's law: ρ = a * φ^(-m) * S^(-n) * ρ_w
# 간단한 추정: 낮은 비저항 = 높은 포화도

print("지하수 포화도 해석 (전기비저항 데이터로부터):")
print(f"  표층 (0-10m): 부분 포화 (비저항 {resistivity_layer1:.2f} Ohm-m)")
print(f"  중층 (10-20m): 높은 포화 (비저항 {resistivity_layer2:.2f} Ohm-m)")
print(f"  심층 (20-30m): 완전 포화 (비저항 {resistivity_layer3:.2f} Ohm-m)")
print()

# 암석 강도 추정 (탄성파 속도로부터)
# 일반적으로 속도가 높을수록 강도가 높음
print("암석 강도 해석 (탄성파 데이터로부터):")
print(f"  풍화대: 낮은 강도 (v = {seismic_v1:.0f} m/s)")
print(f"  기반암: 중간 강도 (v = {seismic_v2:.0f} m/s)")
print()

# 시각화
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

# (1) 통합 지질 단면도
ax1 = fig.add_subplot(gs[0, :])

# 풍화대
weathering_patch = Rectangle((0, 0), 200, -integrated_weathering_thickness,
                              facecolor='yellow', edgecolor='black', linewidth=2,
                              alpha=0.7, label=f'Weathered Zone (0-{integrated_weathering_thickness:.1f}m)')
ax1.add_patch(weathering_patch)

# 기반암
bedrock_patch = Rectangle((0, -integrated_weathering_thickness), 200, -80 + integrated_weathering_thickness,
                           facecolor='brown', edgecolor='black', linewidth=2,
                           alpha=0.7, label=f'Bedrock (>{integrated_weathering_thickness:.1f}m)')
ax1.add_patch(bedrock_patch)

# 주석 추가
ax1.text(100, -integrated_weathering_thickness/2,
         f'Weathered Sandstone\nDensity: {gravity_weathering_density:.0f} kg/m³\n'
         f'Velocity: {seismic_v1:.0f} m/s\nResistivity: {resistivity_layer1:.2f} Ohm-m',
         ha='center', va='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax1.text(100, -integrated_weathering_thickness - 20,
         f'Saturated Sandstone\nDensity: {gravity_bedrock_density:.0f} kg/m³\n'
         f'Velocity: {seismic_v2:.0f} m/s\nResistivity: {resistivity_layer2:.2f} Ohm-m',
         ha='center', va='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax1.axhline(y=0, color='black', linewidth=3, label='Ground Surface')
ax1.axhline(y=-integrated_weathering_thickness, color='red', linewidth=2, linestyle='--',
            label=f'Interface at {integrated_weathering_thickness:.1f} m')

ax1.set_xlabel('Distance (m)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
ax1.set_title('Integrated Geological Model', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 200)
ax1.set_ylim(-80, 10)
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)

# (2) 밀도 프로파일
ax2 = fig.add_subplot(gs[1, 0])
depths = [0, -integrated_weathering_thickness, -integrated_weathering_thickness, -80]
densities = [gravity_weathering_density, gravity_weathering_density,
             gravity_bedrock_density, gravity_bedrock_density]
ax2.plot(densities, depths, 'b-', linewidth=3, marker='o', markersize=10)
ax2.fill_betweenx(depths, densities, alpha=0.3)
ax2.set_xlabel('Density (kg/m³)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Depth (m)', fontsize=11, fontweight='bold')
ax2.set_title('Density Profile (Gravity)', fontsize=12, fontweight='bold')
ax2.axhline(y=0, color='k', linewidth=2)
ax2.axhline(y=-integrated_weathering_thickness, color='r', linewidth=1, linestyle='--', alpha=0.7)
ax2.grid(True, alpha=0.3)

# (3) 비저항 프로파일
ax3 = fig.add_subplot(gs[1, 1])
depths_res = [0, -10, -10, -20, -20, -30]
resistivities = [resistivity_layer1, resistivity_layer1,
                 resistivity_layer2, resistivity_layer2,
                 resistivity_layer3, resistivity_layer3]
ax3.semilogx(resistivities, depths_res, 'g-', linewidth=3, marker='s', markersize=10)
ax3.fill_betweenx(depths_res, resistivities, alpha=0.3, color='green')
ax3.set_xlabel('Resistivity (Ohm-m)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Depth (m)', fontsize=11, fontweight='bold')
ax3.set_title('Resistivity Profile', fontsize=12, fontweight='bold')
ax3.axhline(y=0, color='k', linewidth=2)
ax3.axhline(y=-integrated_weathering_thickness, color='r', linewidth=1, linestyle='--', alpha=0.7)
ax3.grid(True, alpha=0.3, which='both')

# (4) 속도 프로파일
ax4 = fig.add_subplot(gs[1, 2])
depths_vel = [0, -integrated_weathering_thickness, -integrated_weathering_thickness, -80]
velocities = [seismic_v1, seismic_v1, seismic_v2, seismic_v2]
ax4.plot(velocities, depths_vel, 'r-', linewidth=3, marker='^', markersize=10)
ax4.fill_betweenx(depths_vel, velocities, alpha=0.3, color='red')
ax4.set_xlabel('Velocity (m/s)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Depth (m)', fontsize=11, fontweight='bold')
ax4.set_title('Seismic Velocity Profile', fontsize=12, fontweight='bold')
ax4.axhline(y=0, color='k', linewidth=2)
ax4.axhline(y=-integrated_weathering_thickness, color='r', linewidth=1, linestyle='--', alpha=0.7)
ax4.grid(True, alpha=0.3)

# (5) 물리 특성 비교표
ax5 = fig.add_subplot(gs[2, :])
ax5.axis('off')

# 표 데이터
table_data = [
    ['Property', 'Weathered Zone\n(0-{:.1f}m)'.format(integrated_weathering_thickness),
     'Bedrock\n(>{:.1f}m)'.format(integrated_weathering_thickness), 'Interpretation'],
    ['Density\n(kg/m³)', f'{gravity_weathering_density:.0f}', f'{gravity_bedrock_density:.0f}',
     'Lower density in weathered zone\ndue to higher porosity'],
    ['Velocity\n(m/s)', f'{seismic_v1:.0f}', f'{seismic_v2:.0f}',
     'Lower velocity indicates\nlower strength'],
    ['Resistivity\n(Ohm-m)', f'{resistivity_layer1:.2f}', f'{resistivity_layer3:.2f}',
     'Very low resistivity indicates\nhigh water saturation'],
    ['Porosity\n(%)', f'{porosity_weathering * 100:.1f}', f'{porosity_bedrock * 100:.1f}',
     'Higher porosity in\nweathered zone'],
    ['Rock Type', 'Weathered Sandstone', 'Saturated Sandstone',
     'Consistent with sedimentary\nbasin geology'],
]

table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.15, 0.25, 0.25, 0.35])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 3)

# 헤더 스타일
for i in range(4):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 행 색상 교대
for i in range(1, len(table_data)):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E8F5E9')
        else:
            table[(i, j)].set_facecolor('white')

ax5.set_title('Physical Properties Comparison', fontsize=14, fontweight='bold', pad=20)

plt.savefig('integrated_model.png', dpi=300, bbox_inches='tight')
print("결과 그림 저장: integrated_model.png")
print()

# 결과 저장
with open('integrated_model.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("통합 지하 지질구조 모델\n")
    f.write("=" * 70 + "\n\n")

    f.write("1. 개별 탐사 결과 요약\n")
    f.write("-" * 70 + "\n\n")
    f.write("중력 탐사:\n")
    f.write(f"  - 풍화대 두께: {gravity_weathering_thickness:.2f} m\n")
    f.write(f"  - 풍화대 밀도: {gravity_weathering_density:.0f} kg/m³\n")
    f.write(f"  - 기반암 밀도: {gravity_bedrock_density:.0f} kg/m³\n\n")

    f.write("전기비저항 탐사:\n")
    f.write(f"  - 표층 (0-10m): {resistivity_layer1:.2f} Ohm-m\n")
    f.write(f"  - 중층 (10-20m): {resistivity_layer2:.2f} Ohm-m\n")
    f.write(f"  - 심층 (20-30m): {resistivity_layer3:.2f} Ohm-m\n\n")

    f.write("탄성파 탐사:\n")
    f.write(f"  - 풍화대 두께: {seismic_weathering_thickness:.2f} m\n")
    f.write(f"  - 풍화대 속도: {seismic_v1:.0f} m/s\n")
    f.write(f"  - 기반암 속도: {seismic_v2:.0f} m/s\n\n")

    f.write("=" * 70 + "\n")
    f.write("2. 통합 지질구조 모델\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"층 1: 표층 풍화대 (0 ~ {integrated_weathering_thickness:.1f} m)\n")
    f.write(f"  물리적 특성:\n")
    f.write(f"    - 밀도: {gravity_weathering_density:.0f} kg/m³\n")
    f.write(f"    - 탄성파 속도: {seismic_v1:.0f} m/s\n")
    f.write(f"    - 전기비저항: {resistivity_layer1:.2f} Ohm-m\n")
    f.write(f"    - 추정 공극률: {porosity_weathering * 100:.1f}%\n")
    f.write(f"  지질학적 해석:\n")
    f.write(f"    - 암석: 부분 풍화된 사암\n")
    f.write(f"    - 특성: 공극률이 높고 부분적으로 지하수 포화\n")
    f.write(f"    - 강도: 낮음 (풍화 작용으로 약화됨)\n\n")

    f.write(f"층 2: 기반암 ({integrated_weathering_thickness:.1f} m 이하)\n")
    f.write(f"  물리적 특성:\n")
    f.write(f"    - 밀도: {gravity_bedrock_density:.0f} kg/m³\n")
    f.write(f"    - 탄성파 속도: {seismic_v2:.0f} m/s\n")
    f.write(f"    - 전기비저항: {resistivity_layer2:.2f} - {resistivity_layer3:.2f} Ohm-m\n")
    f.write(f"    - 추정 공극률: {porosity_bedrock * 100:.1f}%\n")
    f.write(f"  지질학적 해석:\n")
    f.write(f"    - 암석: 지하수로 포화된 사암\n")
    f.write(f"    - 특성: 깊이에 따라 공극률 감소, 완전 포화\n")
    f.write(f"    - 강도: 중간 (미풍화 상태)\n\n")

    f.write("=" * 70 + "\n")
    f.write("3. 통합 해석 및 결론\n")
    f.write("=" * 70 + "\n\n")

    f.write("지질구조 특성:\n")
    f.write("  1) 2층 구조: 풍화대와 기반암으로 명확히 구분됨\n")
    f.write(f"  2) 풍화대 두께: 약 {integrated_weathering_thickness:.1f} m\n")
    f.write("  3) 주 암석: 사암 (퇴적암)\n")
    f.write("  4) 지하수: 기반암 내 높은 포화도\n\n")

    f.write("물리 탐사 상관관계:\n")
    f.write("  - 중력: 밀도 감소 → 공극률 증가\n")
    f.write("  - 전기비저항: 매우 낮은 값 → 지하수 포화\n")
    f.write("  - 탄성파: 속도 증가 → 암석 강도 증가\n")
    f.write("  → 세 탐사 결과가 일관된 지질 모델을 지지함\n\n")

    f.write("지질공학적 의미:\n")
    f.write("  - 표층 풍화대는 공사 시 안정성 주의 필요\n")
    f.write("  - 지하수 부존: 기반암 내 풍부한 지하수 존재\n")
    f.write("  - 기초 설계: 풍화대를 고려한 설계 필요\n")

print("결과 저장: integrated_model.txt")
print()
print("=" * 60)
print("통합 모델 생성 완료!")
print("=" * 60)
