# 📦 전체 파이프라인 + 모델 학습 + 평가 (Random Forest / Logistic / Linear Regression) + 선택 컬럼 저장은 별도 함수로 분리 + 타겟 생성 기능 추가 + 평가시 target 자동 지정 + 범주형 인코딩 방식 숫자형/문자형 구분 후 함수 분리

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score, mean_squared_error

# 1. 결측치 및 중복 처리
def missing_value_handler(df, numerical_cols, ordinal_cols, nominal_cols):
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
            elif col in ordinal_cols:
                df[col] = df[col].fillna(df[col].median())
            elif col in nominal_cols:
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna('Unknown')
    return df

# 2. 이상치 처리 (IQR)
def outlier_handler(df, numerical_cols, method='IQR'):
    df = df.copy()
    for col in numerical_cols:
        if method == 'IQR':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]
        elif method == 'zscore':
            mean = df[col].mean()
            std = df[col].std()
            df = df[(np.abs((df[col] - mean) / std) <= 3)]
    return df

# 3. 범주형 인코딩 함수 분리
# 3-1-1 숫자형인데 순서 있는 데이터
def encode_ordinal_numeric(df, cols):
    df_encoded = df.copy()
    label_encoder = LabelEncoder()
    for col in cols:
        df_encoded[col] = label_encoder.fit_transform(df[col])
    return df_encoded

# 3-1-2 숫자형인데 순서 없는 데이터
def encode_nominal_numeric(df, cols):
    df_encoded = pd.get_dummies(df, columns=cols)
    return df_encoded
  
# 3-2-1 문자형인데 순서 있는 데이터
def encode_ordinal_string(df, cols):
    df_encoded = df.copy()
    label_encoder = LabelEncoder()
    for col in cols:
        df_encoded[col] = label_encoder.fit_transform(df[col])
    return df_encoded

# 3-2-2 문자형인데 순서 없는 데이터
def encode_nominal_string(df, cols):
    df_encoded = pd.get_dummies(df, columns=cols)
    return df_encoded

# 4. 정규화 처리
def normalization_handler(df, numerical_cols, scaler_type='minmax'):
    df = df.copy()
    scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

# 5. 선택된 컬럼 저장
def save_selected_columns(df, selected_cols, label_col, save_path='saved_selected_data.csv'):
    df_saved = df[selected_cols + [label_col]]
    df_saved.to_csv(save_path, index=False, encoding='utf-8-sig')

# 6. 타겟 생성 함수
def create_target(df, target_name, rule_func):
    df[target_name] = df.apply(rule_func, axis=1)
    return df

# 7. 평가용 모델 함수 각각 분리
def evaluate_randomforest(df, feature_cols, target_col, test_size=0.2):
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        'model': model,
        'accuracy': accuracy_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }

def evaluate_logistic(df, feature_cols, target_col, test_size=0.2):
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        'model': model,
        'accuracy': accuracy_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }

def evaluate_linear(df, feature_cols, target_col, test_size=0.2):
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        'model': model,
        'r2_score': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }