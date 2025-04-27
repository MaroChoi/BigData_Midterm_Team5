# =============================
# ✨ 0. 라이브러리 호출
# =============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# =============================
# ✨ 1. EDA 함수 정의
# =============================
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

# =============================
# ✨ 2. 전처리 함수 정의
# =============================
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

def normalization_handler(df, numerical_cols, scaler_type='minmax'):
    df = df.copy()
    scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

# =============================
# ✨ 3. 전체 파이프라인 실행
# =============================

# 파일 불러오기
input_file_path = 'c:/BigData_Midterm_Team5/BigData_Midterm_Team5/Data_9/cwurData.csv'
df = pd.read_csv(input_file_path)

# 타겟 생성
df_processed = df.copy()
score_cutoff = df_processed['score'].quantile(0.7)
df_processed['target'] = (df_processed['score'] >= score_cutoff).astype(int)

# Feature 선정
selected_features = [
    'quality_of_education', 'alumni_employment', 'quality_of_faculty',
    'publications', 'influence', 'citations', 'broad_impact', 'patents'
]
numerical_cols = selected_features
categorical_cols = []  # 이번 데이터에선 따로 없음

# 결측치 처리
df_processed = missing_value_handler_v2(
    df_processed,
    numerical_cols,
    [],
    [],
    [],
    []
)

# 이상치 제거
df_processed = remove_outliers_iqr(df_processed, numerical_cols=selected_features)

# 정규화
df_processed = normalization_handler(df_processed, numerical_cols=selected_features, scaler_type='minmax')

# Feature-Target 분리
X = df_processed[selected_features]
y = df_processed['target']

# =============================
# ✨ 4. 전처리 완료 후 EDA 실행
# =============================
run_eda(df_processed, numerical_cols=selected_features, categorical_cols=categorical_cols)

# =============================
# ✨ 5. 모델 학습 및 성능 평가
# =============================

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 예측
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# 성능 평가
print("\n✅ [Train 데이터 성능]")
print(f"Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Precision: {precision_score(y_train, y_pred_train):.4f}")
print(f"Recall: {recall_score(y_train, y_pred_train):.4f}")
print(f"F1 Score: {f1_score(y_train, y_pred_train):.4f}")
print("Confusion Matrix (Train):")
print(confusion_matrix(y_train, y_pred_train))

print("\n========================================\n")

print("✅ [Test 데이터 성능]")
print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_test):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_test):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_test):.4f}")
print("Confusion Matrix (Test):")
print(confusion_matrix(y_test, y_pred_test))

print("\n\n📊 [상세 Classification Report (Test)]")
print(classification_report(y_test, y_pred_test))
