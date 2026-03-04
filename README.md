# 시스템 블록도
```mermaid
graph TD
    A[시작] --> B[데이터 입력]
    B --> C{조건 확인}
    C -->|Yes| D[처리]
    C -->|No| E[오류 처리]
    D --> F[결과 출력]
    E --> F
    F --> G[종료]
```
