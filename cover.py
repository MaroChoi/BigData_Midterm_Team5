# =============================
# ✨ 0. 라이브러리 호출
# =============================
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt  # [✨ 추가] EDA 시각화용
import seaborn as sns             # [✨ 추가] EDA 시각화용

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score, mean_squared_error

# =============================
# ✨ 1. EDA 함수 정의 (✨ 새로 추가된 섹션)
# =============================

# 결측치 분포를 히트맵으로 시각화
def plot_missing_heatmap(df):
    plt.figure(figsize=(10,6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title('Missing Value Heatmap')
    plt.show()

# 수치형 컬럼의 이상치를 Boxplot으로 시각화
def plot_boxplots(df, numerical_cols):
    for col in numerical_cols:
        if col in df.columns:
            plt.figure(figsize=(6,4))
            sns.boxplot(x=df[col])
            plt.title(f'Boxplot of {col}')
            plt.show()

# 수치형 컬럼의 데이터 분포를 히스토그램 + KDE로 시각화
def plot_numeric_distributions(df, numerical_cols):
    for col in numerical_cols:
        if col in df.columns:
            plt.figure(figsize=(6,4))
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.show()

# 수치형 컬럼 간 상관관계를 Heatmap으로 시각화
def plot_correlation_heatmap(df, numerical_cols):
    if len(numerical_cols) > 1:
        plt.figure(figsize=(8,6))
        corr = df[numerical_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap')
        plt.show()

# 전체 EDA 플롯 실행
def run_eda(df, numerical_cols):
    print("\n📊 [1/4] 결측치 히트맵")
    plot_missing_heatmap(df)

    print("\n📊 [2/4] 수치형 컬럼 Boxplot (이상치 시각화)")
    plot_boxplots(df, numerical_cols)

    print("\n📊 [3/4] 수치형 변수 분포 (Histplot)")
    plot_numeric_distributions(df, numerical_cols)

    print("\n📊 [4/4] 수치형 변수 간 상관관계 (Heatmap)")
    plot_correlation_heatmap(df, numerical_cols)

    print("\n✅ EDA 시각화 완료.")

# =============================
# ✨ 2. 전처리 함수 정의
# =============================

# 각 컬럼별 고유값(Unique Values) 출력
def inspect_unique_values(df, max_display=10):
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        print(f"\n🔎 {col} (고유값 {len(unique_vals)}개):")
        print(unique_vals[:max_display])
        if len(unique_vals) > max_display:
            print("... (이하 생략)")

# 결측치 및 중복 데이터 처리 함수
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

# Unknown 또는 NaN을 포함한 행 제거
def drop_unknown_or_nan_rows(df, unknown_value='Unknown'):
    df = df.copy()
    condition = df.isnull().any(axis=1) | df.isin([unknown_value]).any(axis=1)
    df_cleaned = df[~condition].reset_index(drop=True)
    return df_cleaned

# 수치형 변수의 이상치(IQR 기준) 제거
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

# 숫자형인데 순서 있는 컬럼(Label Encoding)
def encode_ordinal_numeric(df, cols):
    df_encoded = df.copy()
    label_encoder = LabelEncoder()
    for col in cols:
        df_encoded[col] = label_encoder.fit_transform(df[col])
    return df_encoded

# 숫자형인데 순서 없는 컬럼(One-Hot Encoding)
def encode_nominal_numeric(df, cols):
    df_encoded = pd.get_dummies(df, columns=cols)
    bool_cols = df_encoded.select_dtypes(include=['bool']).columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
    return df_encoded

# 문자형인데 순서 있는 컬럼(Label Encoding)
def encode_ordinal_string(df, cols):
    df_encoded = df.copy()
    label_encoder = LabelEncoder()
    for col in cols:
        df_encoded[col] = label_encoder.fit_transform(df[col])
    return df_encoded

# 문자형인데 순서 없는 컬럼(One-Hot Encoding)
def encode_nominal_string(df, cols):
    df_encoded = pd.get_dummies(df, columns=cols)
    bool_cols = df_encoded.select_dtypes(include=['bool']).columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
    return df_encoded

# 수치형 변수 정규화(MinMaxScaler 또는 StandardScaler 적용)
def normalization_handler(df, numerical_cols, scaler_type='minmax'):
    df = df.copy()
    scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df


# 🔥 전체 파이프라인 실행 예시 (아래 코드 추가)

# 파일 불러오기
input_file_path = '원하는_파일경로/원하는_파일명.csv'  # 예시: 'c:/폴더명/파일명.csv'
df = pd.read_csv(input_file_path)

# 1단계: 고유값 확인
inspect_unique_values(df)

# 컬럼 직접 구분
numerical_cols = ['수치형 컬럼 이름1', '수치형 컬럼 이름2']  # 예시: 'Age', 'Income'
ordinal_numeric_cols = ['순서 있는 숫자형 컬럼 이름1']    # 예시: 'Education_Level'
nominal_numeric_cols = ['순서 없는 숫자형 컬럼 이름1']   # 예시: 'Zipcode'
ordinal_string_cols = ['순서 있는 문자형 컬럼 이름1']    # 예시: 'Customer_Rank'
nominal_string_cols = ['순서 없는 문자형 컬럼 이름1']    # 예시: 'City'

# 2단계: 결측치 및 중복 처리
df = missing_value_handler_v2(df, numerical_cols, ordinal_numeric_cols, nominal_numeric_cols, ordinal_string_cols, nominal_string_cols)

# 원하는 컬럼만 선택해서 새로운 DataFrame 만들기
selected_columns = ['원하는 컬럼1', '원하는 컬럼2', '원하는 컬럼3']  # 예시: 'Age', 'Income', 'Purchase_Amount'
df_selected = df[selected_columns]

# 추가: Unknown 또는 NaN 제거
df_selected = drop_unknown_or_nan_rows(df_selected)

# ✨ 파생변수 생성 예시
# (원하는 방식으로 파생변수를 추가하세요. 아래는 예시입니다.)
df_selected['새로운_파생변수'] = df_selected['원하는 컬럼1'] / (df_selected['원하는 컬럼2'] + 1)

# 3단계: 수치형 컬럼만 이상치 제거 (IQR 방식)
df_selected = remove_outliers_iqr(df_selected, [col for col in numerical_cols if col in df_selected.columns])

# 4단계: EDA 실행
# (이 시점에서 EDA를 수행하는 것이 가장 자연스럽고 정확함)
eda_numerical_cols = [col for col in numerical_cols if col in df_selected.columns] + ['새로운_파생변수']
run_eda(df_selected, numerical_cols=eda_numerical_cols)

# 5단계: 5개 그룹 재분리
numerical_cols_selected = [col for col in numerical_cols if col in df_selected.columns]
ordinal_numeric_cols_selected = [col for col in ordinal_numeric_cols if col in df_selected.columns]
nominal_numeric_cols_selected = [col for col in nominal_numeric_cols if col in df_selected.columns]
ordinal_string_cols_selected = [col for col in ordinal_string_cols if col in df_selected.columns]
nominal_string_cols_selected = [col for col in nominal_string_cols if col in df_selected.columns]

# 새로 만든 파생변수 추가
numerical_cols_selected.append('새로운_파생변수')

# 6단계: 범주형 인코딩
df_encoded = df_selected.copy()
if ordinal_numeric_cols_selected:
    df_encoded = encode_ordinal_numeric(df_encoded, ordinal_numeric_cols_selected)
if nominal_numeric_cols_selected:
    df_encoded = encode_nominal_numeric(df_encoded, nominal_numeric_cols_selected)
if ordinal_string_cols_selected:
    df_encoded = encode_ordinal_string(df_encoded, ordinal_string_cols_selected)
if nominal_string_cols_selected:
    df_encoded = encode_nominal_string(df_encoded, nominal_string_cols_selected)

# 7단계: 정규화 적용 (수치형 컬럼 기준, MinMaxScaler 또는 StandardScaler 선택 가능)
scaler_type = 'minmax'  # 'minmax' 또는 'standard' 중 선택 가능
df_encoded = normalization_handler(df_encoded, numerical_cols_selected, scaler_type=scaler_type)

# 결과 확인
print("\n✅ 최종 데이터프레임:")
print(df_encoded.head())

# ✨ 최종 데이터 저장
output_path = '원하는_저장경로/최종파일명.csv'  # 예시: 'c:/폴더명/최종파일.csv'
df_encoded.to_csv(output_path, index=False)
print(f"\n✅ 최종 데이터 저장 완료: {output_path}")
