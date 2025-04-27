# --- 필수 라이브러리 불러오기 ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
import re

# --- 1. 필수 함수 선언 (전처리, 인코딩, 정규화, EDA) ---

# 1-1. 고유값 확인 함수
def inspect_unique_values(df, max_display=10):
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        print(f"\n🔎 {col} (고유값 {len(unique_vals)}개):")
        print(unique_vals[:max_display])
        if len(unique_vals) > max_display:
            print("... (이하 생략)")

# 1-2. 결측치 및 중복 처리 함수
def missing_value_handler_v2(df, numerical_cols, ordinal_numeric_cols, nominal_numeric_cols, ordinal_string_cols, nominal_string_cols):
    df = df.copy()
    df = df.drop_duplicates()
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > 0.4].index.tolist()
    df.drop(columns=cols_to_drop, inplace=True)
    df = df.dropna(thresh=int(df.shape[1]*0.95))
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if col in numerical_cols or col in ordinal_numeric_cols:
                df[col] = df[col].fillna(df[col].median())
            elif col in ordinal_string_cols or col in nominal_numeric_cols or col in nominal_string_cols:
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna('Unknown')
    return df

# 1-3. 이상치 제거 함수 (IQR 방식)
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
    print(f"✅ 이상치 제거: {len(outlier_indices)}개 데이터 삭제됨")
    df = df.drop(index=outlier_indices)
    return df

# 1-4. 인코딩 함수들
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

# 1-5. 정규화 함수
def normalization_handler(df, numerical_cols, scaler_type='minmax'):
    df = df.copy()
    scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

# 1-6. EDA 블록
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

def plot_categorical_distributions(df, categorical_cols):
    for col in categorical_cols:
        if col in df.columns:
            plt.figure(figsize=(6,4))
            sns.countplot(x=col, data=df)
            plt.title(f'Count Plot of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.show()

def plot_correlation_heatmap(df, numerical_cols):
    if len(numerical_cols) > 1:
        plt.figure(figsize=(8,6))
        corr = df[numerical_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap')
        plt.show()

def run_eda(df, numerical_cols, categorical_cols):
    print("\n📊 [1/5] 결측치 히트맵")
    plot_missing_heatmap(df)

    print("\n📊 [2/5] 수치형 컬럼 Boxplot (이상치 시각화)")
    plot_boxplots(df, numerical_cols)

    print("\n📊 [3/5] 수치형 변수 분포 (Histplot)")
    plot_numeric_distributions(df, numerical_cols)

    print("\n📊 [4/5] 범주형 변수 분포 (Countplot)")
    plot_categorical_distributions(df, categorical_cols)

    print("\n📊 [5/5] 수치형 변수 간 상관관계 (Heatmap)")
    plot_correlation_heatmap(df, numerical_cols)

    print("\n✅ EDA 시각화 완료.")

# ------------------------------------------------------------
# --- 2. 메인 실행 블록 (파일 불러오기 -> 전처리 -> 저장)
# ------------------------------------------------------------

# 2-1. 데이터 로드 (나중에 파일 넣을 때 수정)
df = pd.read_csv('파일_경로를_여기에.csv')

# 2-2. 컬럼 분류 (나중에 데이터에 맞게 수정)
numerical_cols = []  # 예: ['Age', 'Income']
ordinal_numeric_cols = []  # 예: ['Education_Level']
nominal_numeric_cols = []  # 예: ['Job_Type']
ordinal_string_cols = []  # 예: ['Satisfaction_Level']
nominal_string_cols = []  # 예: ['Gender', 'Country']

# 2-3. 고유값 확인 (나중에 df 로드 후 사용)
inspect_unique_values(df)

# 2-4. 결측치 및 중복 처리
df = missing_value_handler_v2(df, numerical_cols, ordinal_numeric_cols, nominal_numeric_cols, ordinal_string_cols, nominal_string_cols)

# 2-5. 사용할 컬럼만 선택
selected_columns = [col for col in (numerical_cols + ordinal_numeric_cols + nominal_numeric_cols + ordinal_string_cols + nominal_string_cols)]
df_selected = df[selected_columns].copy()

# 2-6. EDA 시각화 (선택적 실행)
categorical_cols = ordinal_numeric_cols + nominal_numeric_cols + ordinal_string_cols + nominal_string_cols
run_eda(df_selected, numerical_cols, categorical_cols)

# 2-7. 파생변수 추가 (여기서 직접 추가)
df_selected['new_feature'] = df_selected['Feature1'] / (df_selected['Feature2'] + 1)

# 2-8. 이상치 제거
df_selected = remove_outliers_iqr(df_selected, numerical_cols)

# 2-9. 범주형 인코딩
df_encoded = df_selected.copy()
if ordinal_numeric_cols:
    df_encoded = encode_ordinal_numeric(df_encoded, ordinal_numeric_cols)
if nominal_numeric_cols:
    df_encoded = encode_nominal_numeric(df_encoded, nominal_numeric_cols)
if ordinal_string_cols:
    df_encoded = encode_ordinal_string(df_encoded, ordinal_string_cols)
if nominal_string_cols:
    df_encoded = encode_nominal_string(df_encoded, nominal_string_cols)

# 2-10. Feature Leakage 제거 (타겟 관련 컬럼 제거 필요)
X = df_encoded.drop(['Target_Yes', 'Target_No'], axis=1)
y = df_encoded['Target_Yes']

# 2-11. 수치형 정규화
scaler_type = 'minmax'  # 또는 'standard'
X = normalization_handler(X, numerical_cols, scaler_type=scaler_type)

# 2-12. 최종 데이터 저장
output_path = 'preprocessed_data.csv'
final_df_for_save = pd.concat([X, y], axis=1)
final_df_for_save.to_csv(output_path, index=False)

# 2-13. 모델 학습 (Logistic Regression)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n🎯 Logistic Regression 최종 성능 평가:")
print("정확도 (Accuracy):", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
