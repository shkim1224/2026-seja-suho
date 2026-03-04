"""
소규모 태양광발전단지 3시간 후 발전량 예측 프로그램
- LSTM 기반 예측 알고리즘
- 과거 1개월 운전데이터(랜덤 생성) 학습
- 현 시점 + 기상청 3시간 예측값으로 발전량 예측
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1단계: 과거 1개월 운전데이터 랜덤 생성
# ============================================================
def generate_historical_data(days=30):
    """시간별 일사량, 온도, 발전량 데이터를 합리적으로 생성"""
    np.random.seed(42)
    hours = days * 24
    timestamps = pd.date_range(start='2026-02-01', periods=hours, freq='h')

    irradiance = []  # 일사량 (W/m²)
    temperature = []  # 온도 (°C)
    power = []  # 발전량 (kW)

    for i in range(hours):
        hour = i % 24
        day = i // 24

        # 일사량: 일출~일몰(6시~18시) 사인 패턴 + 노이즈
        if 6 <= hour <= 18:
            sun_angle = np.sin(np.pi * (hour - 6) / 12)
            cloud_factor = np.random.uniform(0.5, 1.0)  # 구름 영향
            irr = 900 * sun_angle * cloud_factor + np.random.normal(0, 20)
            irr = max(0, irr)
        else:
            irr = 0.0

        # 온도: 일변화 패턴 (새벽 최저, 오후 2시 최고) + 계절 반영
        base_temp = 2.0  # 2월 평균 기온
        daily_variation = 6 * np.sin(np.pi * (hour - 6) / 18) if 6 <= hour <= 24 else -2
        temp = base_temp + daily_variation + np.random.normal(0, 1.5)

        # 발전량: 일사량과 온도의 함수 (온도 높으면 효율 약간 감소)
        if irr > 0:
            efficiency = 0.18 - 0.004 * max(0, temp - 25)  # 패널 효율
            panel_area = 500  # m² (소규모 단지)
            pwr = irr * panel_area * efficiency / 1000  # kW 변환
            pwr = max(0, pwr + np.random.normal(0, 1))
        else:
            pwr = 0.0

        irradiance.append(irr)
        temperature.append(temp)
        power.append(pwr)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'irradiance': irradiance,
        'temperature': temperature,
        'power': power
    })
    print(f"[1단계] 과거 {days}일 운전데이터 생성 완료 ({len(df)}시간)")
    return df


# ============================================================
# 2단계: 데이터 전처리
# ============================================================
def preprocess_data(df, seq_length=24):
    """정규화 및 시퀀스 데이터 구성"""
    features = df[['irradiance', 'temperature', 'power']].values

    # MinMaxScaler 정규화
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    # 슬라이딩 윈도우로 시퀀스 생성
    # 입력: seq_length 시간의 (일사량, 온도, 발전량)
    # 출력: 3시간 후 발전량
    X, y = [], []
    for i in range(len(scaled) - seq_length - 3):
        X.append(scaled[i:i + seq_length])  # 과거 seq_length 시간 데이터
        y.append(scaled[i + seq_length + 2, 2])  # 3시간 후 발전량 (index 2)

    X = np.array(X)
    y = np.array(y)

    # 학습/검증 분할 (80:20)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"[2단계] 전처리 완료 - 학습: {len(X_train)}개, 검증: {len(X_val)}개")
    return X_train, X_val, y_train, y_val, scaler


# ============================================================
# 3단계: LSTM 모델 구축 및 학습
# ============================================================
def build_and_train(X_train, X_val, y_train, y_val):
    """LSTM 모델 설계, 학습, 평가"""
    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]),
             return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
    print(f"[3단계] 모델 학습 완료 - 검증 Loss: {val_loss:.4f}, MAE: {val_mae:.4f}")
    return model, history


# ============================================================
# 4단계: 실시간 입력 시뮬레이션
# ============================================================
def simulate_realtime_input():
    """현 시점 데이터 + 기상청 3시간 예측값 랜덤 생성"""
    np.random.seed(None)  # 매번 다른 값

    # 현재 시각을 오후 1시(13시)로 가정 - 발전 활발한 시간
    hour = 13
    sun_angle = np.sin(np.pi * (hour - 6) / 12)

    # 현 시점 데이터
    current_irr = 900 * sun_angle * np.random.uniform(0.6, 1.0)
    current_temp = 5.0 + 6 * np.sin(np.pi * (hour - 6) / 18) + np.random.normal(0, 1)
    efficiency = 0.18 - 0.004 * max(0, current_temp - 25)
    current_power = current_irr * 500 * efficiency / 1000 + np.random.normal(0, 1)
    current_power = max(0, current_power)

    print(f"\n[4단계] 현 시점 데이터 (13시 기준)")
    print(f"  일사량: {current_irr:.1f} W/m², 온도: {current_temp:.1f}°C, 발전량: {current_power:.1f} kW")

    # 기상청 예측값 (+1h, +2h, +3h)
    forecast = []
    for h_offset in range(1, 4):
        future_hour = hour + h_offset
        if future_hour <= 18:
            f_sun = np.sin(np.pi * (future_hour - 6) / 12)
            f_irr = 900 * f_sun * np.random.uniform(0.5, 1.0)
        else:
            f_irr = 0.0
        f_temp = 5.0 + 6 * np.sin(np.pi * (future_hour - 6) / 18) + np.random.normal(0, 1.5)
        forecast.append([f_irr, f_temp])
        print(f"  +{h_offset}h 예측 → 일사량: {f_irr:.1f} W/m², 온도: {f_temp:.1f}°C")

    return current_irr, current_temp, current_power, forecast


# ============================================================
# 5단계: 3시간 후 발전량 예측
# ============================================================
def predict_future_power(model, scaler, df, current_irr, current_temp, current_power, forecast, seq_length=24):
    """학습된 LSTM으로 3시간 후 발전량 예측"""

    # 과거 seq_length-1 시간 데이터 + 현재 시점 데이터로 입력 구성
    # 과거 데이터에서 마지막 (seq_length-4) 시간 가져오기
    recent = df[['irradiance', 'temperature', 'power']].values[-(seq_length - 4):]

    # 현재 시점 데이터
    current_row = np.array([[current_irr, current_temp, current_power]])

    # 기상청 예측값으로 +1h, +2h, +3h 발전량 추정 (단순 추정)
    forecast_rows = []
    for f_irr, f_temp in forecast:
        eff = 0.18 - 0.004 * max(0, f_temp - 25)
        est_power = max(0, f_irr * 500 * eff / 1000)
        forecast_rows.append([f_irr, f_temp, est_power])
    forecast_rows = np.array(forecast_rows)

    # 입력 시퀀스 조합
    input_seq = np.vstack([recent, current_row, forecast_rows])
    # 정확히 seq_length 길이로 맞춤
    input_seq = input_seq[-seq_length:]

    # 정규화
    input_scaled = scaler.transform(input_seq)
    input_scaled = input_scaled.reshape(1, seq_length, 3)

    # 모델 추론
    pred_scaled = model.predict(input_scaled, verbose=0)[0][0]

    # 역정규화 (발전량 컬럼만 복원)
    dummy = np.zeros((1, 3))
    dummy[0, 2] = pred_scaled
    pred_actual = scaler.inverse_transform(dummy)[0, 2]
    pred_actual = max(0, pred_actual)

    print(f"\n[5단계] 3시간 후 예측 발전량: {pred_actual:.2f} kW")
    return pred_actual


# ============================================================
# 6단계: 결과 출력 및 시각화
# ============================================================
def visualize_results(df, history, model, scaler, pred_power, seq_length=24):
    """학습 과정 및 예측 결과 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('태양광 발전량 LSTM 예측 시스템', fontsize=16, fontweight='bold')

    # (1) 과거 운전데이터 (최근 7일)
    ax1 = axes[0, 0]
    recent_7d = df.tail(168)  # 7일 = 168시간
    ax1.plot(recent_7d['timestamp'], recent_7d['irradiance'], label='일사량 (W/m²)', color='orange')
    ax1.set_ylabel('일사량 (W/m²)')
    ax1.set_title('최근 7일 일사량')
    ax1.legend(loc='upper right')
    ax1.tick_params(axis='x', rotation=45)

    # (2) 학습 Loss 추이
    ax2 = axes[0, 1]
    ax2.plot(history.history['loss'], label='학습 Loss')
    ax2.plot(history.history['val_loss'], label='검증 Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (MSE)')
    ax2.set_title('모델 학습 추이')
    ax2.legend()

    # (3) 검증 데이터 실제 vs 예측 비교
    ax3 = axes[1, 0]
    features = df[['irradiance', 'temperature', 'power']].values
    scaled = scaler.transform(features)
    X_all, y_all = [], []
    for i in range(len(scaled) - seq_length - 3):
        X_all.append(scaled[i:i + seq_length])
        y_all.append(scaled[i + seq_length + 2, 2])
    X_all = np.array(X_all)
    y_all = np.array(y_all)

    split = int(len(X_all) * 0.8)
    X_val = X_all[split:]
    y_val_actual = y_all[split:]

    y_val_pred = model.predict(X_val, verbose=0).flatten()

    # 역정규화
    dummy_actual = np.zeros((len(y_val_actual), 3))
    dummy_actual[:, 2] = y_val_actual
    actual_power = scaler.inverse_transform(dummy_actual)[:, 2]

    dummy_pred = np.zeros((len(y_val_pred), 3))
    dummy_pred[:, 2] = y_val_pred
    predicted_power = scaler.inverse_transform(dummy_pred)[:, 2]

    ax3.plot(actual_power[-100:], label='실제 발전량', color='blue', alpha=0.7)
    ax3.plot(predicted_power[-100:], label='예측 발전량', color='red', alpha=0.7, linestyle='--')
    ax3.set_xlabel('시간 인덱스')
    ax3.set_ylabel('발전량 (kW)')
    ax3.set_title('검증 데이터: 실제 vs 예측')
    ax3.legend()

    # (4) 3시간 후 예측 결과 표시
    ax4 = axes[1, 1]
    labels = ['현재 발전량', '3시간 후 예측']
    # 현재 발전량은 최근 데이터에서 가져옴
    current_pwr = df['power'].iloc[-1]
    values = [current_pwr, pred_power]
    colors = ['#2196F3', '#FF5722']
    bars = ax4.bar(labels, values, color=colors, width=0.5, edgecolor='black')
    ax4.set_ylabel('발전량 (kW)')
    ax4.set_title('3시간 후 발전량 예측 결과')
    for bar, val in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{val:.1f} kW', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('solar_prediction_result.png', dpi=150, bbox_inches='tight')
    print(f"\n[6단계] 결과 시각화 저장 완료 → solar_prediction_result.png")
    plt.show()


# ============================================================
# 메인 실행
# ============================================================
def main():
    SEQ_LENGTH = 24  # 과거 24시간 데이터를 입력으로 사용

    # 1단계: 데이터 생성
    df = generate_historical_data(days=30)

    # 2단계: 전처리
    X_train, X_val, y_train, y_val, scaler = preprocess_data(df, SEQ_LENGTH)

    # 3단계: 모델 학습
    model, history = build_and_train(X_train, X_val, y_train, y_val)

    # 4단계: 실시간 입력 시뮬레이션
    current_irr, current_temp, current_power, forecast = simulate_realtime_input()

    # 5단계: 3시간 후 발전량 예측
    pred_power = predict_future_power(
        model, scaler, df, current_irr, current_temp, current_power, forecast, SEQ_LENGTH
    )

    # 6단계: 결과 시각화
    visualize_results(df, history, model, scaler, pred_power, SEQ_LENGTH)


if __name__ == '__main__':
    main()
