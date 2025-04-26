# 📦 전체 파이프라인 + 데이터 타입 구분 스크립트 추가 (1단계: 수치형/범주형, 2단계: 범주형 세부 구분)

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score, mean_squared_error

# 1단계: 수치형/범주형만 빠르게 분류

def classify_basic(df):
    numerical_cols = []
    categorical_cols = []
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)
    return numerical_cols, categorical_cols

# 2단계: 범주형 중 순서형/명목형 추가 분류

def classify_categorical(df, categorical_cols, order_keywords=['낮음', '중간', '높음', '1등급', '2등급', '3등급']):
    ordinal_cols = []
    nominal_cols = []
    for col in categorical_cols:
        values = df[col].dropna().astype(str).unique()
        if any(any(keyword in v for keyword in order_keywords) for v in values):
            ordinal_cols.append(col)
        else:
            nominal_cols.append(col)
    return ordinal_cols, nominal_cols

df = pd.read_csv('/Users/imsu-in/Downloads/myproject/midtermtest/BigData_Midterm_Team5/BigData_Midterm_Team5-2/Data_7/dirty_cafe_sales.csv')

numerical_cols, categorical_cols = classify_basic(df)
#ordinal_cols, nominal_cols = classify_categorical(df, categorical_cols)
print("수치형:", numerical_cols)
print("범주형", categorical_cols)
#print("순서형 범주형:", ordinal_cols)
#print("명목형 범주형:", nominal_cols)
