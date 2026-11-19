import os
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('outputs', exist_ok=True)
print("📊 [1단계] 실제 2026년 대구 기상·전력·무더위쉼터 공공데이터 통합 분석 시작...")

# ---------------------------------------------------------
# 1. 실제 기상청 AWS 데이터 (weather.csv) 분석
# ---------------------------------------------------------
weather = pd.read_csv('data/weather.csv', encoding='cp949')
weather['일시'] = pd.to_datetime(weather['일시'])
weather['월'] = weather['일시'].dt.month
weather['시간'] = weather['일시'].dt.hour

# 결측치 보간
for col in ['기온(°C)', '습도(%)', '풍속(m/s)', '풍향(deg)']:
    weather[col] = pd.to_numeric(weather[col], errors='coerce')
weather = weather.sort_values(['지점명', '일시']).reset_index(drop=True); cols = ['기온(°C)', '습도(%)', '풍속(m/s)', '풍향(deg)']; weather[cols] = weather.groupby('지점명')[cols].ffill(); weather[cols] = weather.groupby('지점명')[cols].bfill(); weather[cols] = weather[cols].fillna(weather[cols].median())

# 기상청 여름철 공식 체감온도(습구온도 기반) 계산
T = weather['기온(°C)']
RH = weather['습도(%)'].clip(1, 100)
Tw = (
    T * np.arctan(0.151977 * np.sqrt(RH + 8.313659))
    + np.arctan(T + RH)
    - np.arctan(RH - 1.67633)
    + 0.00391838 * (RH ** 1.5) * np.arctan(0.023101 * RH)
    - 4.686035
)
weather['체감온도(°C)'] = np.round(-0.2442 + 0.55399 * Tw + 0.45535 * T - 0.0022 * (Tw ** 2) + 0.00278 * Tw * T + 3.0, 2)

# 전처리된 실제 기상 데이터 저장
weather.to_csv('data/processed_weather.csv', index=False, encoding='utf-8-sig')

# 관측소별 폭염 통계 집계
station_stats = weather.groupby('지점명').agg(
    평균기온=('기온(°C)', 'mean'),
    최고기온=('기온(°C)', 'max'),
    평균체감온도=('체감온도(°C)', 'mean'),
    폭염시간_33도이상=('기온(°C)', lambda x: (x >= 33.0).sum())
).sort_values('평균기온', ascending=False)

print("\n🌡️ [실제 관측] 2026년 7~8월 대구 AWS 지점별 폭염 현황:")
print(station_stats.round(2))

# ---------------------------------------------------------
# 2. 실제 한전 전력사용량 데이터 (power_08.xlsx) 분석
# ---------------------------------------------------------
power_raw = pd.read_excel('data/power_08.xlsx', header=2)
daegu_power = power_raw[
    (power_raw['시도'].astype(str).str.contains('대구')) &
    (~power_raw['시군구'].astype(str).str.contains('합계|소계')) &
    (~power_raw['계약종별'].astype(str).str.contains('합계|소계'))
].copy()

month_cols = ['4월', '5월', '7월', '8월']
for c in month_cols:
    daegu_power[c] = pd.to_numeric(daegu_power[c], errors='coerce').fillna(0)

# 봄철 기저부하(4~5월 평균) 대비 한여름 피크(7~8월 평균) 냉방 전력 급증량 산출
daegu_power['봄철_평균_kWh'] = (daegu_power['4월'] + daegu_power['5월']) / 2
daegu_power['여름_평균_kWh'] = (daegu_power['7월'] + daegu_power['8월']) / 2
daegu_power['냉방급증_kWh'] = np.maximum(0, daegu_power['여름_평균_kWh'] - daegu_power['봄철_평균_kWh'])
# 온실가스 배출계수 (0.459 kgCO2/kWh) 적용 -> 톤(tCO2) 환산
daegu_power['냉방탄소급증_tCO2'] = (daegu_power['냉방급증_kWh'] * 0.459) / 1000

gu_power = daegu_power.groupby('시군구').agg(
    봄철_MWh=('봄철_평균_kWh', lambda x: x.sum() / 1000),
    여름_MWh=('여름_평균_kWh', lambda x: x.sum() / 1000),
    냉방탄소급증_tCO2=('냉방탄소급증_tCO2', 'sum')
).sort_values('냉방탄소급증_tCO2', ascending=False)

gu_power['전력증가율(%)'] = ((gu_power['여름_MWh'] - gu_power['봄철_MWh']) / gu_power['봄철_MWh']) * 100
gu_power.to_csv('data/processed_power.csv', encoding='utf-8-sig')

print("\n⚡ [실제 관측] 2026년 대구 구·군별 여름철(7~8월) 냉방 전력 및 탄소 배출 급증량(월평균):")
print(gu_power.round(1))

# ---------------------------------------------------------
# 3. 실제 대구 서구 무더위쉼터 (shelter.csv) 분석
# ---------------------------------------------------------
shelter = pd.read_csv('data/shelter.csv', encoding='cp949')
shelter['행정동'] = shelter['도로명주소'].astype(str).str.extract(r'([가-힣]+동|[가-힣]+로)')
print(f"\n🏠 [실제 관측] 대구 서구 무더위쉼터 총 {len(shelter)}개소 | 총 수용인원: {shelter['수용인원(명)'].sum():,}명 | 평균 연면적: {shelter['연면적(제곱미터)'].mean():.1f}㎡")

# ---------------------------------------------------------
# 4. 실제 데이터 통합 현황 시각화 (3단 그래프)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# [그래프 1] 대구 관측소별 7~8월 폭염(33°C 이상) 누적 시간
colors_1 = ['#d9534f' if v >= station_stats['폭염시간_33도이상'].median() else '#f0ad4e' for v in station_stats['폭염시간_33도이상']]
bars1 = axes[0].bar(station_stats.index, station_stats['폭염시간_33도이상'], color=colors_1, edgecolor='black')
axes[0].set_title('대구 AWS 관측소별 폭염(≥33°C) 누적 시간\n(2026.07~08 실제 관측)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('누적 폭염 시간 (시간)')
axes[0].tick_params(axis='x', rotation=35)
axes[0].grid(axis='y', linestyle='--', alpha=0.5)
for b in bars1:
    axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 2, f"{int(b.get_height())}h", ha='center', fontsize=9, fontweight='bold')

# [그래프 2] 대구 구·군별 여름철 냉방 기인 추가 탄소 배출량
bars2 = axes[1].bar(gu_power.index, gu_power['냉방탄소급증_tCO2'], color='#2b5c8f', edgecolor='black')
axes[1].set_title('대구 구·군별 여름철(7~8월) 냉방 탄소 급증량\n(봄철 4~5월 대비 월평균 증가분)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('월평균 냉방 탄소 증가량 (tCO₂)')
axes[1].tick_params(axis='x', rotation=35)
axes[1].grid(axis='y', linestyle='--', alpha=0.5)
for b in bars2:
    axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 150, f"{b.get_height():,.0f}", ha='center', fontsize=8.5, fontweight='bold')

# [그래프 3] 대구 서구 무더위쉼터 시설 유형별 수용인원 분포
shelter['시설유형'] = np.where(shelter['쉼터명'].str.contains('경로당'), '경로당',
                    np.where(shelter['쉼터명'].str.contains('행정복지센터|동사무소'), '행정복지센터', '복지관·기타'))
type_sum = shelter.groupby('시설유형')['수용인원(명)'].sum().sort_values(ascending=False)
axes[2].pie(type_sum.values, labels=type_sum.index, autopct='%1.1f%%', startangle=140, colors=['#ff9999', '#66b3ff', '#99ff99'])
axes[2].set_title(f'대구 서구 무더위쉼터 유형별 수용인원 비중\n(총 {len(shelter)}개소, {shelter["수용인원(명)"].sum():,}명 수용)', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/1_real_data_analysis.png', dpi=150, bbox_inches='tight')
print("\n✅ [1단계 완료] 실제 데이터 현황 그래프 저장 완료: outputs/1_real_data_analysis.png")
