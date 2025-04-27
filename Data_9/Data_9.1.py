
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score, mean_squared_error

# 1단계: 컬럼별 고유값 출력 함수
def inspect_unique_values(df, max_display=10):
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        print(f"\n🔎 {col} (고유값 {len(unique_vals)}개):")
        print(unique_vals[:max_display])
        if len(unique_vals) > max_display:
            print("... (이하 생략)")

# 2단계: 결측치 및 중복 처리 함수
def missing_value_handler_v2(df, numerical_cols, ordinal_numeric_cols, nominal_numeric_cols, ordinal_string_cols, nominal_string_cols):
    df = df.copy()
    df = df.drop_duplicates()
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > 0.4].index.tolist()
    df.drop(columns=cols_to_drop, inplace=True)
    df = df.dropna(thresh=int(df.shape[1]*0.95))

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if col in numerical_cols:
                df[col] = df[col].fillna(df[col].median())
            elif col in ordinal_numeric_cols:
                df[col] = df[col].fillna(df[col].median())
            elif col in ordinal_string_cols:
                df[col] = df[col].fillna(df[col].mode()[0])
            elif col in nominal_numeric_cols:
                df[col] = df[col].fillna(df[col].mode()[0])
            elif col in nominal_string_cols:
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna('Unknown')
    return df

# 추가: Unknown 또는 NaN이 있는 행 삭제 함수
def drop_unknown_or_nan_rows(df, unknown_value='Unknown'):
    df = df.copy()
    condition = df.isnull().any(axis=1) | df.isin([unknown_value]).any(axis=1)
    df_cleaned = df[~condition].reset_index(drop=True)
    return df_cleaned

# 3단계: 수치형 이상치 제거 함수 (IQR 방식)
def remove_outliers_iqr(df, numerical_cols):
    df = df.copy()
    outlier_indices = set()
    for col in numerical_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower) | (df[col] > upper)].index
            outlier_indices.update(outliers)
    df = df.drop(index=outlier_indices)
    return df

# 4단계: 범주형 인코딩 함수들
# 4-1-1 숫자형인데 순서 있는 데이터
def encode_ordinal_numeric(df, cols):
    df_encoded = df.copy()
    label_encoder = LabelEncoder()
    for col in cols:
        df_encoded[col] = label_encoder.fit_transform(df[col])
    return df_encoded
# 4-1-2 숫자형인데 순서 없는 데이터
def encode_nominal_numeric(df, cols):
    df_encoded = pd.get_dummies(df, columns=cols)
    bool_cols = df_encoded.select_dtypes(include=['bool']).columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
    return df_encoded
# 4-2-1 문자형인데 순서 있는 데이터
def encode_ordinal_string(df, cols):
    df_encoded = df.copy()
    label_encoder = LabelEncoder()
    for col in cols:
        df_encoded[col] = label_encoder.fit_transform(df[col])
    return df_encoded
# 4-2-2 문자형인데 순서 없는 데이터
def encode_nominal_string(df, cols):
    df_encoded = pd.get_dummies(df, columns=cols)
    bool_cols = df_encoded.select_dtypes(include=['bool']).columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
    return df_encoded

# 5단계: 정규화 함수 (MinMaxScaler / StandardScaler 선택 가능)
def normalization_handler(df, numerical_cols, scaler_type='minmax'):
    df = df.copy()
    scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df


def some_function(input_file):
    # 사용자 지정 부분 (원하는 컬럼들)
    selected_columns = ['원하는 컬럼1', '원하는 컬럼2', '원하는 컬럼3']  # 수정 필요
    numerical_cols = ['수치형 컬럼1', '수치형 컬럼2']  # 수정 필요
    ordinal_numeric_cols = []
    nominal_numeric_cols = []
    ordinal_string_cols = []
    nominal_string_cols = []

    # 파일 불러오기
    df = pd.read_csv(input_file)

    # 결측치 및 중복 처리
    df = missing_value_handler_v2(df, numerical_cols, ordinal_numeric_cols, nominal_numeric_cols, ordinal_string_cols, nominal_string_cols)

    # 필요한 컬럼만 선택
    df_selected = df[selected_columns]

    # 이상치 제거
    df_selected = remove_outliers_iqr(df_selected, numerical_cols=[col for col in numerical_cols if col in df_selected.columns])

    # Unknown/Nan 행 삭제
    df_selected = drop_unknown_or_nan_rows(df_selected)

    # ✨ 파생변수 추가 (예시)
    df_selected['새로운_파생변수'] = df_selected['원하는 컬럼1'] / (df_selected['원하는 컬럼2'] + 1)

    # ➡️ 새로 만든 파생변수를 수치형 컬럼 리스트에 추가
    numerical_cols.append('새로운_파생변수')

    # 범주형 인코딩
    df_encoded = df_selected.copy()
    if ordinal_numeric_cols:
        df_encoded = encode_ordinal_numeric(df_encoded, ordinal_numeric_cols)
    if nominal_numeric_cols:
        df_encoded = encode_nominal_numeric(df_encoded, nominal_numeric_cols)
    if ordinal_string_cols:
        df_encoded = encode_ordinal_string(df_encoded, ordinal_string_cols)
    if nominal_string_cols:
        df_encoded = encode_nominal_string(df_encoded, nominal_string_cols)

    # 정규화
    df_encoded = normalization_handler(df_encoded, numerical_cols=numerical_cols, scaler_type='minmax')

    # 저장
    output_file = 'preprocessed_' + input_file
    df_encoded.to_csv(output_file, index=False)
    return output_file

df = pd.read_csv("/Users/kyeom/Desktop/BigData_Midterm_Team5/Data_9/cwurData.csv")
print(df.head())

# 1단계: 고유값 확인
inspect_unique_values(df)

# 1단계: 고유값 확인
inspect_unique_values(df)

# 컬럼 직접 구분
numerical_cols = ['score', 'world_rank', 'patents']
ordinal_numeric_cols = []
nominal_numeric_cols = []
ordinal_string_cols = []
nominal_string_cols = []


# 2단계: 결측치 처리
df = missing_value_handler_v2(df, numerical_cols, ordinal_numeric_cols, nominal_numeric_cols, ordinal_string_cols, nominal_string_cols)

# 원하는 컬럼만 선택해서 새로운 DataFrame 만들기
selected_columns = ['score', 'world_rank', 'patents']
df_selected = df[selected_columns]

# 추가: unkown+nan 제거
df_selected = drop_unknown_or_nan_rows(df_selected)

# 파생변수 생성 / gpt에 물어봐서 추가
# 5) 특허 생산성 지표
df_selected['patent_prod']     = df_selected['patents'] / df_selected['world_rank']

# 6) 순위 역수 변환
df_selected['rank_inverse']    = 1 / df_selected['world_rank']

# 8) 종합 점수 지표
df_selected['composite_score'] = (
    df_selected['score']*0.6 +
    df_selected['patent_prod']*0.3 +
    df_selected['rank_inverse']*0.1
)

df_selected['good_univ'] = (df_selected['composite_score'] >= 0.75).astype(int)

# 3단계: 수치형 컬럼만 이상치 제거 (IQR)
df_selected = remove_outliers_iqr(df_selected, [col for col in numerical_cols if col in df_selected.columns])

# 5개 그룹 재분리
numerical_cols_selected = [col for col in numerical_cols if col in df_selected.columns]
ordinal_numeric_cols_selected = [col for col in ordinal_numeric_cols if col in df_selected.columns]
nominal_numeric_cols_selected = [col for col in nominal_numeric_cols if col in df_selected.columns]
ordinal_string_cols_selected = [col for col in ordinal_string_cols if col in df_selected.columns]
nominal_string_cols_selected = [col for col in nominal_string_cols if col in df_selected.columns]

# 새로 만든 파생변수 추가
numerical_cols_selected.append('composite_score')
numerical_cols_selected.append('good_univ')

# 4단계: 범주형 인코딩
df_encoded = df_selected.copy()
if ordinal_numeric_cols_selected:
    df_encoded = encode_ordinal_numeric(df_encoded, ordinal_numeric_cols_selected)
if nominal_numeric_cols_selected:
    df_encoded = encode_nominal_numeric(df_encoded, nominal_numeric_cols_selected)
if ordinal_string_cols_selected:
    df_encoded = encode_ordinal_string(df_encoded, ordinal_string_cols_selected)
if nominal_string_cols_selected:
    df_encoded = encode_nominal_string(df_encoded, nominal_string_cols_selected)

# 5단계: 정규화 적용 (수치형 컬럼 기준, MinMaxScaler 또는 StandardScaler(logistic regression, linear regression) 선택)
scaler_type = 'minmax'  # 'minmax' 또는 'standard' 중 선택 가능
df_encoded = normalization_handler(df_encoded, numerical_cols_selected, scaler_type=scaler_type)

# 결과 확인
print("\n✅ 최종 데이터프레임:")
print(df_encoded.head())

# ✨ 최종 데이터 저장
output_path = 'final_preprocessed_data_9.1.csv'
df_encoded.to_csv(output_path, index=False)
print(f"\n✅ 최종 데이터 저장 완료: {output_path}")