# 📦 전체 파이프라인 함수 모음 (Full Functions)

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt  # EDA 시각화용
import seaborn as sns            # EDA 시각화용

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

# 4단계: 범주형 인코딩 함수
def encode_ordinal_numeric(df, cols):
    df_encoded = df.copy()
    label_encoder = LabelEncoder()
    for col in cols:
        df_encoded[col] = label_encoder.fit_transform(df[col])
    return df_encoded

def encode_nominal_numeric(df, cols):
    df_encoded = pd.get_dummies(df, columns=cols)
    bool_cols = df_encoded.select_dtypes(include=['bool']).columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
    return df_encoded

def encode_ordinal_string(df, cols):
    df_encoded = df.copy()
    label_encoder = LabelEncoder()
    for col in cols:
        df_encoded[col] = label_encoder.fit_transform(df[col])
    return df_encoded

def encode_nominal_string(df, cols):
    df_encoded = pd.get_dummies(df, columns=cols)
    bool_cols = df_encoded.select_dtypes(include=['bool']).columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
    return df_encoded

# 5단계: 정규화 함수
def normalization_handler(df, numerical_cols, scaler_type='minmax'):
    df = df.copy()
    scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

# 6단계: EDA 시각화 함수
def plot_categorical_distributions(df, categorical_cols):
    for col in categorical_cols:
        if col in df.columns:
            plt.figure(figsize=(6,4))
            sns.countplot(x=col, data=df)
            plt.title(f'Count Plot of {col}')
            plt.xticks(rotation=45)
            plt.show()

def plot_missing_heatmap(df):
    plt.figure(figsize=(10,6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title('Missing Value Heatmap')
    plt.show()

def plot_boxplots(df, numerical_cols):
    for col in numerical_cols:
        if col in df.columns:
            plt.figure(figsize=(6,4))
            sns.boxplot(x=df[col])
            plt.title(f'Boxplot of {col}')
            plt.show()

def plot_numeric_distributions(df, numerical_cols):
    for col in numerical_cols:
        if col in df.columns:
            plt.figure(figsize=(6,4))
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.show()

def plot_correlation_heatmap(df, numerical_cols):
    if len(numerical_cols) > 1:
        plt.figure(figsize=(8,6))
        corr = df[numerical_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap')
        plt.show()

def run_eda(df, numerical_cols, categorical_cols=None):
    print("\n📊 [1/5] 결측치 히트맵")
    plot_missing_heatmap(df)

    print("\n📊 [2/5] 수치형 컬럼 Boxplot")
    plot_boxplots(df, numerical_cols)

    print("\n📊 [3/5] 수치형 변수 분포 (Histplot)")
    plot_numeric_distributions(df, numerical_cols)

    if categorical_cols:
        print("\n📊 [4/5] 범주형 변수 Count Plot")
        plot_categorical_distributions(df, categorical_cols)

    print("\n📊 [5/5] 수치형 변수 간 상관관계 (Heatmap)")
    plot_correlation_heatmap(df, numerical_cols)

    print("\n✅ EDA 시각화 완료.")

# 파일 불러오기
input_file_path = '원하는_파일경로/원하는_파일명.csv'
df = pd.read_csv(input_file_path)

# 1단계: 고유값 확인
inspect_unique_values(df)

# 2단계: 컬럼 직접 구분
numerical_cols = ['수치형 컬럼 이름1', '수치형 컬럼 이름2']
ordinal_numeric_cols = ['순서 있는 숫자형 컬럼 이름1']
nominal_numeric_cols = ['순서 없는 숫자형 컬럼 이름1']
ordinal_string_cols = ['순서 있는 문자형 컬럼 이름1']
nominal_string_cols = ['순서 없는 문자형 컬럼 이름1']

# 3단계: 결측치 및 중복 처리
print("\n📊 [추가] 결측치 제거 전 히트맵")
plot_missing_heatmap(df)

df = missing_value_handler_v2(df, numerical_cols, ordinal_numeric_cols, nominal_numeric_cols, ordinal_string_cols, nominal_string_cols)

# 원하는 컬럼만 선택
selected_columns = ['원하는 컬럼1', '원하는 컬럼2', '원하는 컬럼3']
df_selected = df[selected_columns]

# 추가: Unknown/Nan 제거
df_selected = drop_unknown_or_nan_rows(df_selected)

# ✨ 파생변수 생성 예시
df_selected['새로운_파생변수'] = df_selected['원하는 컬럼1'] / (df_selected['원하는 컬럼2'] + 1)

# 4단계: 수치형 컬럼만 이상치 제거
df_selected = remove_outliers_iqr(df_selected, [col for col in numerical_cols if col in df_selected.columns])

# 5단계: EDA 실행
eda_numerical_cols = [col for col in numerical_cols if col in df_selected.columns] + ['새로운_파생변수']
run_eda(df_selected, numerical_cols=eda_numerical_cols)

# 6단계: 5개 그룹 재분리
numerical_cols_selected = [col for col in numerical_cols if col in df_selected.columns]
ordinal_numeric_cols_selected = [col for col in ordinal_numeric_cols if col in df_selected.columns]
nominal_numeric_cols_selected = [col for col in nominal_numeric_cols if col in df_selected.columns]
ordinal_string_cols_selected = [col for col in ordinal_string_cols if col in df_selected.columns]
nominal_string_cols_selected = [col for col in nominal_string_cols if col in df_selected.columns]

# 새로 만든 파생변수 추가
numerical_cols_selected.append('새로운_파생변수')

# 7단계: 범주형 인코딩
df_encoded = df_selected.copy()
if ordinal_numeric_cols_selected:
    df_encoded = encode_ordinal_numeric(df_encoded, ordinal_numeric_cols_selected)
if nominal_numeric_cols_selected:
    df_encoded = encode_nominal_numeric(df_encoded, nominal_numeric_cols_selected)
if ordinal_string_cols_selected:
    df_encoded = encode_ordinal_string(df_encoded, ordinal_string_cols_selected)
if nominal_string_cols_selected:
    df_encoded = encode_nominal_string(df_encoded, nominal_string_cols_selected)

# 8단계: 정규화
scaler_type = 'minmax'  # 또는 'standard'
df_encoded = normalization_handler(df_encoded, numerical_cols_selected, scaler_type=scaler_type)

# 결과 확인
print("\n✅ 최종 데이터프레임:")
print(df_encoded.head())

# ✨ 최종 데이터 저장
output_path = '원하는_저장경로/최종파일명.csv'
df_encoded.to_csv(output_path, index=False)
print(f"\n✅ 최종 데이터 저장 완료: {output_path}")
