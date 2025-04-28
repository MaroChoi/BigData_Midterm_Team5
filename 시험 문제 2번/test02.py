# 📦 전체 파이프라인 함수 모음 (Full Functions)
import os
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score, mean_squared_error


import matplotlib.pyplot as plt  # EDA 시각화용
import seaborn as sns            # EDA 시각화용

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
    import os

    # 사용자 지정 부분 (원하는 컬럼들)
    numerical_cols = ['LIMIT_BAL', 'AGE','BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3','BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6','PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3','PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    ordinal_numeric_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    nominal_numeric_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
    ordinal_string_cols = []
    nominal_string_cols = []

    # 파일 불러오기
    df = pd.read_csv(input_file)

    # 결측치 및 중복 처리
    df = missing_value_handler_v2(df, numerical_cols, ordinal_numeric_cols, nominal_numeric_cols, ordinal_string_cols, nominal_string_cols)

    # 필요한 컬럼만 선택
    selected_columns = ['LIMIT_BAL','AGE','MARRIAGE','SEX','EDUCATION','default.payment.next.month']
    df_selected = df[selected_columns]

    # 이상치 제거
    df_selected = remove_outliers_iqr(df_selected, numerical_cols=[col for col in numerical_cols if col in df_selected.columns])

    # Unknown/Nan 행 삭제
    df_selected = drop_unknown_or_nan_rows(df_selected)

    # ✨ 파생변수 추가
    # 한도 대비 나이 비율
    df_selected['LIMIT_PER_AGE'] = df_selected['LIMIT_BAL'] / (df_selected['AGE'] + 1)
    # 나이 그룹
    df_selected['AGE_GROUP'] = pd.cut(df_selected['AGE'],bins=[0, 29, 39, 120],labels=['20s', '30s', '40+'])# 3. 결혼 여부 이진화
    df_selected['IS_MARRIED'] = (df_selected['MARRIAGE'] == 1).astype(int)
    #고학력 여부 이진화
    df_selected['IS_HIGH_EDU'] = df_selected['EDUCATION'].isin([1,2]).astype(int)

    # 5개 그룹 재분리 (※ 여기 중요)
    numerical_cols_selected = [col for col in numerical_cols if col in df_selected.columns]
    ordinal_numeric_cols_selected = [col for col in ordinal_numeric_cols if col in df_selected.columns]
    nominal_numeric_cols_selected = [col for col in nominal_numeric_cols if col in df_selected.columns]
    ordinal_string_cols_selected = [col for col in ordinal_string_cols if col in df_selected.columns]
    nominal_string_cols_selected = [col for col in nominal_string_cols if col in df_selected.columns]

    # 새로 만든 파생변수 추가
    # 새로 만든 파생변수 추가
    numerical_cols_selected.append('LIMIT_PER_AGE')
    ordinal_string_cols_selected.append('AGE_GROUP')
    nominal_numeric_cols_selected.append('IS_MARRIED')
    nominal_numeric_cols_selected.append('IS_HIGH_EDU')

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

    # 정규화 (※ 여기 수정!!)
    df_encoded = normalization_handler(df_encoded, numerical_cols=numerical_cols_selected, scaler_type='minmax')

    # ✨ (필요하면 여기서 target 추가 가능)
    # Target 컬럼
    target_col = 'default.payment.next.month'
    # Feature, Target 나누기
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 최종 저장
    save_folder = os.path.expanduser('~/Downloads')  # 맥북 기본 Downloads 폴더
    save_filename = 'final_preprocessed_data.csv'
    output_path = os.path.join(save_folder, save_filename)
    df_encoded.to_csv(output_path, index=False)

    # 결과 확인
    print("\n✅ 최종 데이터프레임:")
    print(df_encoded.head())
    print(f"\n✅ 최종 데이터 저장 완료! 저장 위치: {output_path}")

    return output_path

# 🔥 전체 파이프라인 실행 예시 (아래 코드 추가)
df = pd.read_csv('/Users/imsu-in/Downloads/myproject/midtermtest/BigData_Midterm_Team5/BigData_Midterm_Team5-8/시험 문제 2번/2_Card.csv')
# 컬럼 직접 구분
numerical_cols = ['LIMIT_BAL', 'AGE','BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3','BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6','PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3','PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
ordinal_numeric_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
nominal_numeric_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
ordinal_string_cols = []
nominal_string_cols = []
run_eda(df, numerical_cols, categorical_cols=None)



# 최종 값
input_file = '/Users/imsu-in/Downloads/myproject/midtermtest/BigData_Midterm_Team5/BigData_Midterm_Team5-8/시험 문제 2번/2_Card.csv'
output_file = some_function(input_file)
