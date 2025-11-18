"""
탄성파 굴절법 해석 및 속도 모델 추정
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

print("=" * 60)
print("탄성파 굴절법 해석")
print("=" * 60)

# 탐사 정보
n_receivers = 50
receiver_spacing = 4  # m
shot_positions = [25, 175]  # m
total_distance = 200  # m
total_time = 0.2  # s

# 수진기 위치
receiver_positions = np.arange(n_receivers) * receiver_spacing  # 0, 4, 8, ..., 196 m

print(f"수진기 개수: {n_receivers}")
print(f"수진기 간격: {receiver_spacing} m")
print(f"발파 위치: {shot_positions} m")
print()

# 이미지 로드
img = Image.open('seismic_data.png')
img_array = np.array(img)

print(f"이미지 크기: {img_array.shape}")
print()

# 이미지에서 초도달 시간 자동 추출
# 이미지는 시간-거리 도메인에서 표시됨
# x축: 거리 (트레이스 번호), y축: 시간

# 이미지가 RGB인 경우 그레이스케일로 변환
if len(img_array.shape) == 3:
    img_gray = np.mean(img_array[:, :, :3], axis=2)
else:
    img_gray = img_array

# 초도달 시간 추출 (각 트레이스에서 첫 번째 강한 신호)
# 간단한 방법: 각 수직 라인에서 threshold를 넘는 첫 번째 픽셀 찾기

height, width = img_gray.shape
first_arrivals = []

# 이미지를 두 부분으로 나누어 처리 (왼쪽 발파, 오른쪽 발파)
# 왼쪽 절반: 왼쪽 발파 (25m)
# 오른쪽 절반: 오른쪽 발파 (175m)

half_width = width // 2

def extract_first_arrivals(img_section, n_traces):
    """이미지 섹션에서 초도달 시간 추출"""
    height, width = img_section.shape
    arrivals = []

    for i in range(n_traces):
        # 각 트레이스의 x 위치
        x = int(i * width / n_traces + width / (2 * n_traces))

        if x >= width:
            x = width - 1

        # 해당 트레이스의 신호
        trace = img_section[:, x]

        # threshold (평균보다 낮은 값 - 검은색이 신호)
        threshold = np.mean(trace) - 0.5 * np.std(trace)

        # 위에서부터 검색하여 threshold 이하인 첫 번째 픽셀 찾기
        first_arrival_idx = np.where(trace < threshold)[0]

        if len(first_arrival_idx) > 0:
            # 시간으로 변환 (y축 픽셀 → 시간)
            time = first_arrival_idx[0] / height * total_time
            arrivals.append(time)
        else:
            # 초도달을 찾지 못한 경우
            arrivals.append(np.nan)

    return np.array(arrivals)

# 왼쪽 발파 데이터 (shot at 25m)
print("1. 왼쪽 발파 (25m) 초도달 시간 추출")
left_img = img_gray[:, :half_width]
first_arrivals_left = extract_first_arrivals(left_img, n_receivers)

# 오른쪽 발파 데이터 (shot at 175m)
print("2. 오른쪽 발파 (175m) 초도달 시간 추출")
right_img = img_gray[:, half_width:]
first_arrivals_right = extract_first_arrivals(right_img, n_receivers)

# 발파점으로부터의 거리 계산
offset_left = np.abs(receiver_positions - shot_positions[0])
offset_right = np.abs(receiver_positions - shot_positions[1])

# 유효한 데이터만 선택
valid_left = ~np.isnan(first_arrivals_left)
valid_right = ~np.isnan(first_arrivals_right)

print(f"   왼쪽 발파 유효 데이터: {np.sum(valid_left)}/{n_receivers}")
print(f"   오른쪽 발파 유효 데이터: {np.sum(valid_right)}/{n_receivers}")
print()

# 굴절법 해석
# 2층 모델 가정: 풍화대 + 기반암
# v1: 풍화대 속도, v2: 기반암 속도, h: 풍화대 두께

def two_layer_traveltime(x, v1, v2, h):
    """
    2층 모델의 주시곡선
    x: offset
    v1: 1층 속도
    v2: 2층 속도
    h: 1층 두께
    """
    # 직접파 (direct wave)
    t_direct = x / v1

    # 굴절파 (refracted wave)
    # 임계각
    ic = np.arcsin(v1 / v2)

    # 굴절파 주시
    t_refracted = 2 * h * np.cos(ic) / v1 + x / v2

    # 둘 중 빠른 것 (초도달)
    return np.minimum(t_direct, t_refracted)

# 왼쪽 발파 데이터로 모델 피팅
# 초기 추정: v1 = 1000 m/s (풍화대), v2 = 3000 m/s (기반암), h = 10 m

# 수동 해석: 시간-거리 그래프에서 기울기 읽기
# 근거리 기울기 → 1/v1
# 원거리 기울기 → 1/v2

# 간단한 선형 피팅으로 속도 추정
# 근거리 데이터 (직접파)
near_offset_mask_left = offset_left < 60
if np.sum(near_offset_mask_left & valid_left) > 3:
    near_data_x = offset_left[near_offset_mask_left & valid_left]
    near_data_t = first_arrivals_left[near_offset_mask_left & valid_left]

    # 선형 피팅: t = x/v1
    slope1, _ = np.polyfit(near_data_x, near_data_t, 1)
    v1_est = 1.0 / slope1
else:
    v1_est = 1000

# 원거리 데이터 (굴절파)
far_offset_mask_left = offset_left > 80
if np.sum(far_offset_mask_left & valid_left) > 3:
    far_data_x = offset_left[far_offset_mask_left & valid_left]
    far_data_t = first_arrivals_left[far_offset_mask_left & valid_left]

    # 선형 피팅: t = x/v2 + intercept
    slope2, intercept2 = np.polyfit(far_data_x, far_data_t, 1)
    v2_est = 1.0 / slope2

    # 교차점에서 깊이 계산
    # intercept = 2h * cos(ic) / v1
    # ic = arcsin(v1/v2)
    ic = np.arcsin(v1_est / v2_est)
    h_est = intercept2 * v1_est / (2 * np.cos(ic))
else:
    v2_est = 3000
    h_est = 10

print("3. 굴절법 역산 결과")
print(f"   1층 속도 (v1): {v1_est:.0f} m/s")
print(f"   2층 속도 (v2): {v2_est:.0f} m/s")
print(f"   1층 두께 (h):  {h_est:.2f} m")
print()

# 속도에 따른 암석 종류 추정
print("4. 지질학적 해석")
if v1_est < 1000:
    rock_type1 = "토양/풍화대 (매우 느림)"
elif v1_est < 2000:
    rock_type1 = "풍화된 암석"
elif v1_est < 3000:
    rock_type1 = "부분 풍화 암석"
else:
    rock_type1 = "신선한 암석"

if v2_est < 2000:
    rock_type2 = "풍화된 퇴적암"
elif v2_est < 4000:
    rock_type2 = "퇴적암 (사암/셰일)"
elif v2_est < 6000:
    rock_type2 = "고결된 퇴적암/변성암"
else:
    rock_type2 = "화강암/변성암"

print(f"   1층 해석: {rock_type1} (v = {v1_est:.0f} m/s)")
print(f"   2층 해석: {rock_type2} (v = {v2_est:.0f} m/s)")
print()

# 시각화
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# (1) 원본 탄성파 이미지
ax1 = fig.add_subplot(gs[0, :])
ax1.imshow(img_array, aspect='auto', extent=[0, total_distance, total_time, 0])
ax1.set_xlabel('Distance (m)', fontsize=11)
ax1.set_ylabel('Time (s)', fontsize=11)
ax1.set_title('Seismic Data (Left shot: 25m, Right shot: 175m)', fontsize=12, fontweight='bold')
ax1.axvline(x=shot_positions[0], color='red', linewidth=2, linestyle='--', label='Shot 1 (25m)')
ax1.axvline(x=shot_positions[1], color='orange', linewidth=2, linestyle='--', label='Shot 2 (175m)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# (2) 왼쪽 발파 주시곡선
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(offset_left[valid_left], first_arrivals_left[valid_left], 'ko', markersize=5, label='Picked arrivals')

# 이론 곡선
x_model = np.linspace(0, max(offset_left), 100)
t_model = two_layer_traveltime(x_model, v1_est, v2_est, h_est)
ax2.plot(x_model, t_model, 'r-', linewidth=2, label=f'Model (v1={v1_est:.0f}, v2={v2_est:.0f}, h={h_est:.1f}m)')

# 직접파와 굴절파 개별 표시
t_direct = x_model / v1_est
ax2.plot(x_model, t_direct, 'b--', linewidth=1, alpha=0.7, label=f'Direct wave (v={v1_est:.0f} m/s)')

ic = np.arcsin(v1_est / v2_est)
t_refracted = 2 * h_est * np.cos(ic) / v1_est + x_model / v2_est
ax2.plot(x_model, t_refracted, 'g--', linewidth=1, alpha=0.7, label=f'Refracted wave (v={v2_est:.0f} m/s)')

ax2.set_xlabel('Offset (m)', fontsize=11)
ax2.set_ylabel('Time (s)', fontsize=11)
ax2.set_title('Travel Time Curve - Left Shot (25m)', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, max(offset_left))
ax2.set_ylim(0, max(first_arrivals_left[valid_left]) * 1.1)

# (3) 오른쪽 발파 주시곡선
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(offset_right[valid_right], first_arrivals_right[valid_right], 'ko', markersize=5, label='Picked arrivals')

# 이론 곡선
x_model_r = np.linspace(0, max(offset_right), 100)
t_model_r = two_layer_traveltime(x_model_r, v1_est, v2_est, h_est)
ax3.plot(x_model_r, t_model_r, 'r-', linewidth=2, label=f'Model (v1={v1_est:.0f}, v2={v2_est:.0f}, h={h_est:.1f}m)')

ax3.set_xlabel('Offset (m)', fontsize=11)
ax3.set_ylabel('Time (s)', fontsize=11)
ax3.set_title('Travel Time Curve - Right Shot (175m)', fontsize=11, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, max(offset_right))
ax3.set_ylim(0, max(first_arrivals_right[valid_right]) * 1.1)

# (4) 속도 모델
ax4 = fig.add_subplot(gs[2, :])

# 2층 모델 그리기
ax4.fill_between([0, 200], [0, 0], [-h_est, -h_est],
                  color='yellow', alpha=0.6, label=f'Layer 1: {rock_type1} (v={v1_est:.0f} m/s)')
ax4.fill_between([0, 200], [-h_est, -h_est], [-80, -80],
                  color='brown', alpha=0.6, label=f'Layer 2: {rock_type2} (v={v2_est:.0f} m/s)')

# 발파점 표시
ax4.scatter(shot_positions, [0, 0], marker='*', s=300, c='red', edgecolors='black',
            linewidths=1, zorder=10, label='Shot points')

# 수진기 표시
ax4.scatter(receiver_positions, np.zeros_like(receiver_positions),
            marker='v', s=30, c='blue', zorder=10, alpha=0.5, label='Receivers')

ax4.set_xlabel('Distance (m)', fontsize=11)
ax4.set_ylabel('Depth (m)', fontsize=11)
ax4.set_title('Seismic Velocity Model (2-Layer)', fontsize=12, fontweight='bold')
ax4.set_xlim(0, 200)
ax4.set_ylim(-80, 10)
ax4.legend(loc='lower right', fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='k', linewidth=2)
ax4.axhline(y=-h_est, color='k', linewidth=1, linestyle='--', alpha=0.5)

plt.savefig('seismic_result.png', dpi=300, bbox_inches='tight')
print("결과 그림 저장: seismic_result.png")
print()

# 결과 저장
with open('seismic_result.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("탄성파 굴절법 해석 결과\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"탐사 설정:\n")
    f.write(f"  수진기 개수: {n_receivers}\n")
    f.write(f"  수진기 간격: {receiver_spacing} m\n")
    f.write(f"  발파 위치: {shot_positions} m\n\n")
    f.write("역산 결과 (2층 모델):\n")
    f.write(f"  1층 속도 (v1): {v1_est:.0f} m/s\n")
    f.write(f"  2층 속도 (v2): {v2_est:.0f} m/s\n")
    f.write(f"  1층 두께 (h):  {h_est:.2f} m\n\n")
    f.write("지질학적 해석:\n")
    f.write(f"  1층: {rock_type1}\n")
    f.write(f"  2층: {rock_type2}\n\n")
    f.write("초도달 시간 데이터:\n")
    f.write("왼쪽 발파 (25m):\n")
    f.write("  거리(m)  시간(s)\n")
    for i in range(len(offset_left)):
        if valid_left[i]:
            f.write(f"  {offset_left[i]:6.1f}  {first_arrivals_left[i]:.6f}\n")

print("결과 저장: seismic_result.txt")
print()
print("=" * 60)
print("탄성파 굴절법 해석 완료!")
print("=" * 60)
