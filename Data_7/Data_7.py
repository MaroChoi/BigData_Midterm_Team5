# --- 메인 작업 ---

# 1. 데이터 로드
df = pd.read_csv('/path/to/dirty_cafe_sales.csv')  # 파일 경로 수정 필요

# 2. 원본 데이터에서 결측치 처리
df = missing_value_handler(df, numerical_cols=['Total Spent'])

# 3. 필요한 컬럼만 선택
df_selected = df[['Transaction Date', 'Total Spent']].copy()

# 4. 파생변수 생성
df_selected['Transaction Date'] = pd.to_datetime(df_selected['Transaction Date'])  # 문자열을 datetime으로 변환
df_selected['Weekday'] = df_selected['Transaction Date'].dt.dayofweek              # 요일 추출 (0=월요일, 6=일요일)
df_selected['Hour'] = df_selected['Transaction Date'].dt.hour                      # 방문 시간 추출 (0~23시)
df_selected['high_spender'] = df_selected.apply(create_high_spender, axis=1)        # 고액 소비 여부 타겟 생성

# 5. 이상치 제거 (Total Spent만)
df_selected = outlier_handler(df_selected, numerical_cols=['Total Spent'])

# 6. 정규화 (Total Spent만)
scaler = StandardScaler()
df_selected['Total Spent'] = scaler.fit_transform(df_selected[['Total Spent']])

# 7. One-Hot Encoding (Weekday)
df_selected = safe_one_hot_encoding(df_selected, columns=['Weekday'])