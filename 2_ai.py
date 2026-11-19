import os
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("🧠 [2단계] 실제 2026년 기상청 AWS 시간별 데이터 기반 AI 3시간 후 체감온도 예측 모델 학습 시작...")

# 1. 전처리된 실제 기상 데이터 로드
df = pd.read_csv('data/processed_weather.csv')
df['일시'] = pd.to_datetime(df['일시'])
df = df.sort_values(['지점명', '일시']).reset_index(drop=True)

# 2. 시계열 피처(과거 1시간·2시간 변화량) 및 3시간 후 실제 관측 정답(target) 생성
df['지점코드'] = df['지점명'].astype('category').cat.codes
df['1시간전_기온'] = df.groupby('지점명')['기온(°C)'].shift(1)
df['2시간전_기온'] = df.groupby('지점명')['기온(°C)'].shift(2)
df['기온상승속도'] = df['기온(°C)'] - df['1시간전_기온']

# 정답(Target): 실제 3시간 후 관측된 체감온도
df['실제_3시간후_체감온도'] = df.groupby('지점명')['체감온도(°C)'].shift(-3)
df_model = df.dropna().copy()

# 3. 시계열 검증 분리 (7월~8월 중순 데이터로 학습 -> 8월 하순 실제 폭염 데이터로 검증)
split_date = pd.Timestamp('2026-08-18')
train_df = df_model[df_model['일시'] < split_date]
test_df = df_model[df_model['일시'] >= split_date].copy()

features = ['지점코드', '월', '시간', '기온(°C)', '체감온도(°C)', '습도(%)', '풍속(m/s)', '풍향(deg)', '1시간전_기온', '2시간전_기온', '기온상승속도']
X_train, y_train = train_df[features], train_df['실제_3시간후_체감온도']
X_test, y_test = test_df[features], test_df['실제_3시간후_체감온도']

# 4. AI 모델 학습 및 실제 미래 구간 예측
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.06, max_depth=5, random_state=42)
model.fit(X_train, y_train)

test_df['AI_예측_3시간후_체감온도'] = np.round(model.predict(X_test), 2)
mae = mean_absolute_error(y_test, test_df['AI_예측_3시간후_체감온도'])
r2 = r2_score(y_test, test_df['AI_예측_3시간후_체감온도'])

print(f"✅ [실제 데이터 검증 완료] 학습 데이터: {len(train_df):,}건 | 검증 데이터: {len(test_df):,}건")
print(f"🎯 AI 3시간 후 체감온도 예측 정확도(R²): {r2:.4f} | 평균 절대 오차(MAE): ±{mae:.2f}°C")

# 피크 시간대(13~17시) 고위험 폭염 경보(체감온도 35°C 이상) 선별
test_df['고위험_타겟'] = (test_df['AI_예측_3시간후_체감온도'] >= 35.0).astype(int)
test_df.to_csv('data/ai_real_predictions.csv', index=False, encoding='utf-8-sig')
print(f"🚨 검증 기간 중 AI가 사전 포착한 3시간 후 체감온도 35°C 이상 초고위험 시간대: 총 {test_df['고위험_타겟'].sum():,}건")

# 5. 결과 시각화 (2단 그래프)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# [왼쪽] 대구 주요 도심 관측소(대구서구 or 최다폭염지점) 3일간 실제 관측 vs AI 3시간 전 예측 시계열 비교
top_station = test_df.groupby('지점명')['실제_3시간후_체감온도'].mean().idxmax()
sample_st = test_df[test_df['지점명'] == top_station].head(72)  # 72시간(3일) 추이

axes[0].plot(sample_st['일시'], sample_st['실제_3시간후_체감온도'], label='실제 관측 체감온도 (°C)', color='#d9534f', linewidth=2)
axes[0].plot(sample_st['일시'], sample_st['AI_예측_3시간후_체감온도'], label='AI 3시간 전 예측 (°C)', color='#1E90FF', linestyle='--', linewidth=2)
axes[0].axhline(35.0, color='darkred', linestyle=':', label='폭염 경보 기준 (35°C)')
axes[0].set_title(f'[{top_station}] 실제 관측 vs AI 3시간 전 예측 추이 (72시간)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('체감온도 (°C)')
axes[0].tick_params(axis='x', rotation=30)
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.5)

# [오른쪽] 전체 검증 데이터(8월 하순) 실제 관측 vs AI 예측 산점도
axes[1].scatter(y_test, test_df['AI_예측_3시간후_체감온도'], alpha=0.35, color='#2e8b57', s=12)
min_v, max_v = min(y_test.min(), test_df['AI_예측_3시간후_체감온도'].min()), max(y_test.max(), test_df['AI_예측_3시간후_체감온도'].max())
axes[1].plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='완벽 예측 기준선 (y=x)')
axes[1].set_title(f'실제 기상청 검증 데이터 AI 예측 성능 (R² = {r2:.3f}, MAE ±{mae:.2f}°C)', fontsize=11, fontweight='bold')
axes[1].set_xlabel('실제 3시간 후 관측 체감온도 (°C)')
axes[1].set_ylabel('AI 예측 체감온도 (°C)')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('outputs/2_real_ai_prediction.png', dpi=150, bbox_inches='tight')
print("✅ [2단계 완료] 실제 AI 예측 검증 그래프 저장 완료: outputs/2_real_ai_prediction.png")