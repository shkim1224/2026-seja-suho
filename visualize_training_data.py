"""
LSTM 학습 데이터 구조 시각화
- 샘플 1: [0h ~ 23h] → 26h 발전량 예측
- 샘플 2: [1h ~ 24h] → 27h 발전량 예측
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# --- 48시간 분량 데이터 생성 (시각화용) ---
hours = 48
irradiance = []
temperature = []
power = []

for i in range(hours):
    hour = i % 24
    if 6 <= hour <= 18:
        sun_angle = np.sin(np.pi * (hour - 6) / 12)
        cloud_factor = np.random.uniform(0.6, 1.0)
        irr = 900 * sun_angle * cloud_factor + np.random.normal(0, 15)
        irr = max(0, irr)
    else:
        irr = 0.0

    base_temp = 2.0
    daily_var = 6 * np.sin(np.pi * (hour - 6) / 18) if 6 <= hour <= 24 else -2
    temp = base_temp + daily_var + np.random.normal(0, 1.0)

    if irr > 0:
        eff = 0.18 - 0.004 * max(0, temp - 25)
        pwr = irr * 500 * eff / 1000 + np.random.normal(0, 0.5)
        pwr = max(0, pwr)
    else:
        pwr = 0.0

    irradiance.append(irr)
    temperature.append(temp)
    power.append(pwr)

irradiance = np.array(irradiance)
temperature = np.array(temperature)
power = np.array(power)
time_labels = [f"{h % 24:02d}시" for h in range(hours)]

# ============================================================
# 그림 그리기
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(18, 14))
fig.suptitle('LSTM 학습 데이터 구조: 2개 샘플 비교', fontsize=18, fontweight='bold', y=0.98)

seq_len = 24  # 입력 시퀀스 길이
target_offset = 3  # 3시간 후 예측

# --- 샘플 정의 ---
samples = [
    {"start": 0, "color_in": "#2196F3", "color_out": "#FF5722", "label": "샘플 1"},
    {"start": 1, "color_in": "#4CAF50", "color_out": "#E91E63", "label": "샘플 2"},
]

features = [
    {"data": irradiance, "name": "일사량 (W/m²)", "unit": "W/m²", "base_color": "#333333"},
    {"data": temperature, "name": "온도 (°C)", "unit": "°C", "base_color": "#333333"},
    {"data": power,       "name": "발전량 (kW)", "unit": "kW", "base_color": "#333333"},
]

for ax_idx, feat in enumerate(features):
    ax = axes[ax_idx]
    data = feat["data"]

    # 전체 데이터 회색 배경선
    ax.plot(range(hours), data, color='#CCCCCC', linewidth=1.5, zorder=1)

    for s in samples:
        start = s["start"]
        end = start + seq_len  # 입력 끝
        target_idx = end + target_offset - 1  # 3시간 후 인덱스

        # 입력 구간 (X) 강조
        x_range = range(start, end)
        ax.plot(x_range, data[start:end], color=s["color_in"], linewidth=2.5, zorder=3)
        # 입력 구간 배경 음영
        ax.axvspan(start, end - 1, alpha=0.08, color=s["color_in"], zorder=0)

        # 입력 시작/끝 마커
        ax.scatter([start], [data[start]], color=s["color_in"], s=60, zorder=5, marker='o')
        ax.scatter([end - 1], [data[end - 1]], color=s["color_in"], s=60, zorder=5, marker='o')

        # 타겟 (y) 지점 표시
        if target_idx < hours:
            ax.scatter([target_idx], [data[target_idx]], color=s["color_out"],
                       s=200, zorder=6, marker='*', edgecolors='black', linewidths=0.8)

            # 입력 끝 → 타겟 점선 화살표
            ax.annotate('', xy=(target_idx, data[target_idx]),
                        xytext=(end - 1, data[end - 1]),
                        arrowprops=dict(arrowstyle='->', color=s["color_out"],
                                        lw=1.5, linestyle='dashed'))

        # 맨 위 그래프에만 샘플 라벨 표시
        if ax_idx == 0:
            mid = start + seq_len // 2
            y_pos = max(data[start:end]) + 60
            ax.text(mid, y_pos, f'← {s["label"]} 입력 (X) →',
                    ha='center', fontsize=10, fontweight='bold', color=s["color_in"],
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=s["color_in"], alpha=0.9))

            if target_idx < hours:
                ax.text(target_idx, data[target_idx] + 50,
                        f'{s["label"]}\n타겟(y)',
                        ha='center', fontsize=9, fontweight='bold', color=s["color_out"],
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=s["color_out"], alpha=0.9))

    ax.set_ylabel(feat["name"], fontsize=12, fontweight='bold')
    ax.set_xticks(range(0, hours, 1))
    ax.set_xticklabels([time_labels[i] for i in range(hours)], rotation=90, fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 35)

axes[2].set_xlabel('시간', fontsize=12, fontweight='bold')

# --- 범례 ---
legend_elements = [
    mpatches.Patch(facecolor='#2196F3', alpha=0.3, edgecolor='#2196F3',
                   label='샘플1 입력(X): 0h~23h (24시간)'),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#FF5722',
               markersize=15, markeredgecolor='black', label='샘플1 타겟(y): 26h 발전량'),
    mpatches.Patch(facecolor='#4CAF50', alpha=0.3, edgecolor='#4CAF50',
                   label='샘플2 입력(X): 1h~24h (24시간)'),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#E91E63',
               markersize=15, markeredgecolor='black', label='샘플2 타겟(y): 27h 발전량'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2,
           fontsize=11, frameon=True, fancybox=True, shadow=True,
           bbox_to_anchor=(0.5, 0.01))

# --- 하단 설명 박스 ---
desc_text = (
    "[ 학습 데이터 구조 ]\n"
    "입력(X): 연속 24시간의 (일사량, 온도, 발전량) → shape: (24, 3)\n"
    "타겟(y): 입력 마지막 시점에서 +3시간 후의 발전량 (스칼라)\n"
    "슬라이딩 윈도우: 1시간씩 이동하며 샘플 생성"
)
fig.text(0.5, 0.06, desc_text, ha='center', fontsize=11,
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF9C4', edgecolor='#FFC107', alpha=0.95),
         family='monospace')

plt.tight_layout(rect=[0, 0.13, 1, 0.96])
plt.savefig('training_data_structure.png', dpi=150, bbox_inches='tight')
print("저장 완료 → training_data_structure.png")
plt.show()
