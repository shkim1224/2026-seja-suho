"""
과거 1개월(30일) 운전데이터 CSV 생성 및 그래프 시각화
- 일사량 (W/m²), 온도 (°C), 발전량 (kW)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# ============================================================
# 1개월 운전데이터 생성
# ============================================================
days = 30
hours = days * 24
timestamps = pd.date_range(start='2026-02-01', periods=hours, freq='h')

irradiance = []
temperature = []
power = []

for i in range(hours):
    hour = i % 24
    day = i // 24

    # 일사량: 6시~18시 사인 패턴 + 구름 영향
    if 6 <= hour <= 18:
        sun_angle = np.sin(np.pi * (hour - 6) / 12)
        cloud_factor = np.random.uniform(0.5, 1.0)
        irr = 900 * sun_angle * cloud_factor + np.random.normal(0, 20)
        irr = max(0, irr)
    else:
        irr = 0.0

    # 온도: 새벽 최저, 오후 최고 (2월 기준)
    base_temp = 2.0
    daily_variation = 6 * np.sin(np.pi * (hour - 6) / 18) if 6 <= hour <= 24 else -2
    temp = base_temp + daily_variation + np.random.normal(0, 1.5)

    # 발전량: 일사량 × 패널면적 × 효율
    if irr > 0:
        efficiency = 0.18 - 0.004 * max(0, temp - 25)
        panel_area = 500  # m²
        pwr = irr * panel_area * efficiency / 1000  # kW
        pwr = max(0, pwr + np.random.normal(0, 1))
    else:
        pwr = 0.0

    irradiance.append(round(irr, 2))
    temperature.append(round(temp, 2))
    power.append(round(pwr, 2))

df = pd.DataFrame({
    'timestamp': timestamps,
    'irradiance_Wm2': irradiance,
    'temperature_C': temperature,
    'power_kW': power
})

# CSV 저장
csv_path = 'historical_data_30days.csv'
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"CSV 저장 완료 → {csv_path} ({len(df)}행)")
print(f"\n--- 데이터 미리보기 (처음 10행) ---")
print(df.head(10).to_string(index=False))
print(f"\n--- 기초 통계량 ---")
print(df.describe().round(2).to_string())

# ============================================================
# 그래프 시각화
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
fig.suptitle('소규모 태양광발전단지 1개월 운전데이터 (2026.02.01 ~ 02.28)',
             fontsize=16, fontweight='bold')

# 일사량
ax1 = axes[0]
ax1.fill_between(df['timestamp'], df['irradiance_Wm2'], alpha=0.4, color='#FF9800')
ax1.plot(df['timestamp'], df['irradiance_Wm2'], color='#E65100', linewidth=0.5)
ax1.set_ylabel('일사량 (W/m²)', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 1000)
ax1.grid(True, alpha=0.3)
ax1.legend(['일사량'], loc='upper right', fontsize=10)

# 온도
ax2 = axes[1]
ax2.plot(df['timestamp'], df['temperature_C'], color='#2196F3', linewidth=0.8)
ax2.fill_between(df['timestamp'], df['temperature_C'], alpha=0.2, color='#2196F3')
ax2.set_ylabel('온도 (°C)', fontsize=12, fontweight='bold')
ax2.axhline(y=0, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
ax2.grid(True, alpha=0.3)
ax2.legend(['온도'], loc='upper right', fontsize=10)

# 발전량
ax3 = axes[2]
ax3.fill_between(df['timestamp'], df['power_kW'], alpha=0.4, color='#4CAF50')
ax3.plot(df['timestamp'], df['power_kW'], color='#1B5E20', linewidth=0.5)
ax3.set_ylabel('발전량 (kW)', fontsize=12, fontweight='bold')
ax3.set_xlabel('날짜', fontsize=12, fontweight='bold')
ax3.set_ylim(0)
ax3.grid(True, alpha=0.3)
ax3.legend(['발전량'], loc='upper right', fontsize=10)

# x축 날짜 포맷
import matplotlib.dates as mdates
ax3.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('historical_data_30days.png', dpi=150, bbox_inches='tight')
print(f"\n그래프 저장 완료 → historical_data_30days.png")
plt.show()
