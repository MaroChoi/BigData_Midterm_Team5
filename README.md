# BigData_Midterm_Team5
BigData_Midterm_Team5
# 어떤 함수가 어떤 데이터셋을 전처리하는지, 어떻게 함수를 호출하는지 간단히 설명한 문서

사용 전략 : 기존 파이프 라인을 사용하여 전처리한 후, 각 문제에 맞는 전처리 함수를 호출하여 전처리 진행
inspect_unique_values(df)	각 컬럼의 고유값을 확인하여 데이터 탐색(데이터 분류) (EDA) 지원
missing_value_handler_v2(df, ...)	결측치 및 중복 데이터 처리, 필요시 특정 컬럼 삭제
drop_unknown_or_nan_rows(df)	'Unknown' 또는 NaN 값이 포함된 행 제거
remove_outliers_iqr(df, numerical_cols)	수치형 컬럼의 이상치를 IQR 방식으로 제거
encode_ordinal_numeric(df, cols)	순서가 있는 숫자형 컬럼 인코딩 (Label Encoding)
encode_nominal_numeric(df, cols)	순서가 없는 숫자형 컬럼 인코딩 (One-hot Encoding)
encode_ordinal_string(df, cols)	순서가 있는 문자형 컬럼 인코딩 (Label Encoding)
encode_nominal_string(df, cols)	순서가 없는 문자형 컬럼 인코딩 (One-hot Encoding)
normalization_handler(df, numerical_cols, scaler_type)	수치형 컬럼 정규화 (MinMaxScaler 또는 StandardScaler 선택)
run_eda(df, numerical_cols, categorical_cols)	히트맵, 박스플롯, 분포플롯, 상관관계 히트맵 등을 시각화
some_function(input_file)	상기 모든 과정을 종합하여 최종 전처리된 CSV 파일 생성
* 순서: 결측치 처리 -> 사용할 컬럼만 추출한 후 새로운 df에 넣기 -> 이상치 처리(수치형만 가능) -> 파생 변수 생성 및 추가 -> 해당 범주형 인코딩 -> 정규화


# 사용 모델:
RandomForestClassifier, LogisticRegression, LinearRegression
RandomForestClassifier: 정규화 불필요
LogisticRegression, LinearRegression: 정규화 필요

# 결측치 처리 방법:
결측치 처리 방법 
컬럼별 결측치 비율 40% 초과 시 해당 컬럼 삭제
나머지 결측치는 다음 기준으로 대체:
수치형 컬럼: 중앙값(median) 대체
순서형/명목형 범주형 컬럼: 최빈값(mode) 대체
기타 컬럼: 'Unknown'으로 대체

# 이상치 처리 방법:
IQR(Interquartile Range) 방식을 사용   수치형 데이터만 처리 가능 함
*파생변수, 정규화 및 어떠한 작업을 하기 전에 이상치 처리를 하는게 전처리에 적합함(outlier 데이터의 영향을 없애기 위해)

# 범주형 인코딩 방법:
숫자형: 순서가 있는 숫자형 컬럼 인코딩 (Label Encoding)
숫자형: 순서가 없는 숫자형 컬럼 인코딩 (One-hot Encoding) -> 이 방법은 true/fale인 bool 결과값으로 나올 때가 있는데 그걸 다시 숫자로 바꿔주는 코드 추가함
명목형: 순서가 있는 문자형 컬럼 인코딩 (Label Encoding)
명목형: 순서가 없는 문자형 컬럼 인코딩 (One-hot Encoding) -> 이 방법은 true/fale인 bool 결과값으로 나올 때가 있는데 그걸 다시 숫자로 바꿔주는 코드 추가함

# 사용 정규화 방법: 
MinMaxScaler 또는 StandardScaler 선택 가능 (scaler_type 설정)
웬만하면 StandardScaler 사용

ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

# 1번 문제 Readme
사용 모델 : Logistic Regression
모델에 따른 정규화 : StandardScaler 사용
칼럼 선택:
    selected_columns = ['age', 'workclass', 'education', 'education.num','marital.status', 'occupation', 'relationship','race', 'sex', 'hours.per.week', 'native.country''income']
파생변수 생성: 
    df_selected['work_hours_per_year'] = df_selected['hours.per.week'] * 52

refractoring: 당장은 만족하지만, 시간이 더 있다면, work_hours_per_year' 말고도 logicstic regression의 성능을 높이기 위하여 새로운 파생변수도 더 만들 수 있을 것 같습니다. 예를 들어, 나이와 혈압의 비율을 구하는 등의 작업을 할 수 있을 것 같습니다.

# 2번 문제 Readme
사용 모델 : Logistic regression
모델에 따른 정규화 : StandardScaler 사용
칼럼 선택： 
    selected_columns = ['LIMIT_BAL','AGE','MARRIAGE','SEX','EDUCATION','default.payment.next.month']
파생변수 생성:
    # 한도 대비 나이 비율
    df_selected['LIMIT_PER_AGE'] = df_selected['LIMIT_BAL'] / (df_selected['AGE'] + 1)
    # 나이 그룹
    df_selected['AGE_GROUP'] = pd.cut(df_selected['AGE'],bins=[0, 29, 39, 120],labels=['20s', '30s', '40+'])# 3. 결혼 여부 이진화
    df_selected['IS_MARRIED'] = (df_selected['MARRIAGE'] == 1).astype(int)
    #고학력 여부 이진화
    df_selected['IS_HIGH_EDU'] = df_selected['EDUCATION'].isin([1,2]).astype(int)
target 컬럼:
    target_col = 'default.payment.next.month'

refractoring: 시간이 있었다면
    1. 전처리 함수 some_function()에 시각화 기능 포함시키기  (전처리 및 데이터 시각화를 하나의 통합 함수로 만들어 작업 효율 향상)
    2. Logistic Regression 모델 학습 이후 평가를 통해 Accuracy를 높이기 위해 추가적인 파생변수 생성, Feature Selection 등을 진행했을 것

# 3번 문제 Readme
사용 모델 : Classification
모델에 따른 정규화 : standard
칼럼 선택 :
selected_columns = [
    'id', 'name', 'host_id', 'host_name','price','minimum_nights','number_of_reviews','reviews_per_month',
    'calculated_host_listings_count','availability_365','last_review','neighbourhood_group','neighbourhood'
]
파생변수 생성:
# 예약 가능 여부 (타깃)
df_selected['reservation_possible'] = (df_selected['availability_365'] > 0).astype(int)

# 연간 잠재 수익 = price × availability_365
df_selected['annual_potential_revenue'] = df_selected['price'] * df_selected['availability_365']

#지역 인기도
freq = df_selected['neighbourhood'].value_counts()
df_selected['neighbourhood_freq'] = df_selected['neighbourhood'].map(freq)

refractoring : 전처리 전 체계적으로 계획을 세우고 진행하는게 매우 중요하다는 것을 깨닫게 되었다.
1. 빠르게 어떻게든 데이터를 전처리해야한다는 생각에 데이터 분석 목표를 명확하게 생각 하지 못했다. 데이터 분석 목표를 정하는 것은 방향성을 결정하는 길인데, 이를 명확하게 못한게 가장 아쉽니다. 다시 분석한다면 이 숙소가 향후 예약될 확률(1/0)을 예측하는 것으로 목표를 산정해서 다시한번 데이터 전처리를 진행하고 싶다.
2. 'name', 'host_name' 등은 필요없는 칼럼인데 굳이 포함했다. 범주형 칼럼을 자동으로 encoding하게 파이프라인이 구성되어있는데, 쓸데 없는 칼럼으로 인해 무한루프가 걸리는 현상이 발생했다. 이 또한 처음 전처리 전략을 수립하는 단계에서 조금 더 신중했으면 좋았다는 생각이 들었다.
3. "neighbourhood_group", 'neighbourhood' 등은 범주형 인코딩이 실제로 필요없는 칼럼인데, 이를 고려하지 않고 one-hot 인코딩 진행하여 무한루프에 빠지게 되었다.


# 4번 문제 Readme
사용 모델 : Logistic regression
모델에 따른 정규화 : StandardScaler
칼럼 선택 : selected_columns = ['Age', 'SMS_received', 'AppointmentDay', 'No-show']
파생변수 생성 : df_selected['Probability_of_Noshow'] = df_selected['Age'] / (df_selected['SMS_received'] + 1)

refractoring : 날짜형 데이터를 사용하여 파생변수를 만들었을 것이다.

# 5번 문제 Readme
사용 모델 : 
모델에 따른 정규화 : MinMaxScaler
칼럼 선택: selected_columns = ['overall', 'potential', 'value_eur']
파생변수 생성: df_selected['value_per_rating'] = df_selected['value_eur'] / df_selected['overall']

refractoring: 목표에 더 적합한 파생변수를 만들었을 것이다.

# 6번 문제 Readme
사용 모델 : Linear Regression
모델에 따른 정규화 : MinMaxScaler
칼럼 선택 : selected_columns = ['Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'Price', 'Customer ID', 'Country']
파생변수 생성 : df_selected['TotalAmount'] = df_selected['Quantity'] * df_selected['Price']

refractoring: 시간이 없어서 그룹화의 부분에 큰 신경을 쓰지는 못했다. 좀 더 시간을 썼다면, 그룹화의 기준을 더 세분화 하는데 집중했을 것 같다.

# 7번 문제 Readme
사용 모델 : RandomForest
모델에 따른 정규화 : minmax / RandomForest는 정규화가 엄청 필요하지는 않지만 정규화 작업이 필요하다고 판단이 되어서 처리했습니다.
칼럼 선택:
    selected_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
파생변수 생성:
    df_selected['BP_CHOL_RATIO'] = df_selected['trestbps'] / (df_selected['chol'] + 1)

refractoring: 시간이 더 있었다면
    1. 빠르게 목적에 맞는 핵심 파생변수를 만들고, 추가로 필요한 변수를 탐색하여 성능 향상에 도움이 되는지 확인해보고 싶습니다.
    2. 전처리 과정이 실제 데이터에 제대로 적용되었는지 최종적으로 점검하고, 이상치 제거, 범주형 인코딩, 정규화까지 모두 완벽하게 처리했는지 검토하고 싶었습니다.
    3. 코드의 반복을 줄이고 전처리 파이프라인 some_function()과 시각화 파이프라인run_eda()을 사용하여 과정을 더 진행해보고 싶었습니다