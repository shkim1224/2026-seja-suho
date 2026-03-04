소규모 태양광발전단지에서 과거 한달 동안의 운전데이터(일사량, 온도 및 발전량)이 확보된 상태에서 현 시점의 일사량, 온도 및 발전량을 기반으로 기상청으로부터 지금으로부터 1시간 간격으로 3시간후의 예측된 일사량, 온도값을 사용하여 3시간후의 발전량을 예측하는 알고리즘을 구현하고자 함. 그런데 예측알고리즘은 lstm을 사용하고 실제 한달 동안의 운전데이터의 확보가 어려운 관계로 시간에 따른 합리적인 일사량, 온도 및 발전량을 랜덤으로 발생시키고 현 시점의 일사량, 온도 및 발전량과 기상청으로부터의 3시간 예측값도 랜덤으로 발생시켜서 3시간 후의 발전량을 예측하려고 함. 코드는 파이선으로 작업하려고 함

---

## 프로젝트 실행 블록도

```mermaid
flowchart TD
    subgraph DATA_GEN["1단계: 데이터 생성"]
        A["과거 1개월 운전데이터<br/>랜덤 생성"] --> A1["시간별 일사량<br/>(0~1000 W/m²)"]
        A --> A2["시간별 온도<br/>(일변화 패턴 반영)"]
        A --> A3["시간별 발전량<br/>(일사량·온도 기반 산출)"]
    end

    subgraph PREPROCESS["2단계: 데이터 전처리"]
        B1["데이터 정규화<br/>(MinMaxScaler)"]
        B2["시퀀스 데이터 구성<br/>(슬라이딩 윈도우)"]
        B3["학습/검증 데이터 분할<br/>(Train / Validation)"]
    end

    subgraph MODEL["3단계: LSTM 모델 구축 및 학습"]
        C1["LSTM 모델 설계<br/>(입력: 일사량, 온도, 발전량)"]
        C2["모델 학습<br/>(Epoch, Batch Size 설정)"]
        C3["학습 성능 평가<br/>(Loss, MAE 등)"]
    end

    subgraph REALTIME["4단계: 실시간 입력 시뮬레이션"]
        D1["현 시점 데이터 생성<br/>(일사량, 온도, 발전량)"]
        D2["기상청 예측값 생성<br/>(+1h, +2h, +3h)"]
        D2 --> D2a["예측 일사량<br/>(1h/2h/3h 후)"]
        D2 --> D2b["예측 온도<br/>(1h/2h/3h 후)"]
    end

    subgraph PREDICT["5단계: 3시간 후 발전량 예측"]
        E1["입력 데이터 구성<br/>(현재값 + 기상예측값)"]
        E2["데이터 정규화<br/>(학습 시 사용한 Scaler 적용)"]
        E3["LSTM 모델 추론"]
        E4["역정규화<br/>(실제 발전량 스케일 복원)"]
    end

    subgraph OUTPUT["6단계: 결과 출력"]
        F1["3시간 후 예측 발전량 출력"]
        F2["실제값 vs 예측값 비교<br/>(시각화)"]
    end

    DATA_GEN --> PREPROCESS
    PREPROCESS --> MODEL
    MODEL --> PREDICT
    REALTIME --> PREDICT
    PREDICT --> OUTPUT

    style DATA_GEN fill:#e8f5e9,stroke:#4caf50,stroke-width:2px
    style PREPROCESS fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    style MODEL fill:#fff3e0,stroke:#ff9800,stroke-width:2px
    style REALTIME fill:#fce4ec,stroke:#e91e63,stroke-width:2px
    style PREDICT fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
    style OUTPUT fill:#e0f2f1,stroke:#009688,stroke-width:2px
```