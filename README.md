# BigData_Midterm_Team5
BigData_Midterm_Team5
# 어떤 함수가 어떤 데이터셋을 전처리하는지, 어떻게 함수를 호출하는지 간단히 설명한 문서

사용 전략 : 기존 파이프 라인을 사용하여 전처리한 후, 각 문제에 맞는 전처리 함수를 호출하여 전처리 진행
inspect_unique_values(df)	각 컬럼의 고유값을 확인하여 데이터 탐색 (EDA) 지원
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

# 1번 문제 Readme
사용 모델 : 
모델에 따른 정규화 :
결측치 처리 방법




# 2번 문제 Readme
사용 모델 : 
모델에 따른 정규화 :
결측치 처리 방법





# 3번 문제 Readme
사용 모델 : 
모델에 따른 정규화 :
결측치 처리 방법




# 4번 문제 Readme
사용 모델 : 
모델에 따른 정규화 :
결측치 처리 방법




# 5번 문제 Readme
사용 모델 : 
모델에 따른 정규화 :
결측치 처리 방법