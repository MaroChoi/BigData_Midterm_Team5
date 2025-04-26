# 📦 전체 파이프라인 + 데이터 타입 구분 스크립트 추가 (1단계: 수치형/범주형, 2단계: 범주형 세부 구분)

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

# 2단계: 결측치 및 중복 처리
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

df = pd.read_csv('/Users/imsu-in/Downloads/myproject/midtermtest/BigData_Midterm_Team5/BigData_Midterm_Team5-2/Data_9/cwurData.csv')

# 출력된 고유값을 그래도 gpt에 물어봐서 수치형, 범주형(숫자형(순서 상관 여부), 명목형(순서 상관 여부))확인
inspect_unique_values(df)



