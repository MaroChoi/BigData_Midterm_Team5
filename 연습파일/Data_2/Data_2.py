# 1단계: 데이터 로딩 및 구조 파악
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 1단계: 데이터 로밍 및 구조파악
import pandas as pd
df = pd.read_csv('C:\BigData_Midterm_Team5\BigData_Midterm_Team5\Data_2\국민건강보험공단_건강검진정보_2023.CSV', encoding='euc-kr')
# df.info()
# df.head()

# 2단계: 결측치 시각화
import seaborn as sns
import matplotlib.pyplot as plt

plt.rc('font', family='NanumBarunGothic')
sns.heatmap(df.isnull(), cbar=False)

# 3단계: 결측치 처리
# 결측치 비율 확인 및 40% 이상 결측치 제거
# 1% 이상 결측치 비율 보기
missing_ratio = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
missing_ratio = missing_ratio[missing_ratio >= 1].round(1)
print(missing_ratio.astype(str) + "%")

# 40% 이상 결측치 제거
missing_ratio = df.isnull().mean()
drop_cols = missing_ratio[missing_ratio > 0.4].index
df = df.drop(columns=drop_cols)

print(df)