# 1단계: 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# 2단계: 데이터 로딩
df = pd.read_csv('C:/BigData_Midterm_Team5/BigData_Midterm_Team5/Data_10/survey.csv')

# 3단계: 결측치 시각화 및 중복 확인
sns.heatmap(df.isnull(), cbar=False)
plt.title('Missing Values Heatmap')
plt.show()

print("중복 개수:", df.duplicated().sum())
sns.countplot(x=df.duplicated())
plt.title('Duplicate Data Count')
plt.xlabel('Duplicate')
plt.ylabel('Count')
plt.show()

# 4단계: 파생변수 생성
df_model = df.copy()

# 연령대 그룹화
def age_group(age):
    if age < 25:
        return 'young'
    elif age < 40:
        return 'adult'
    else:
        return 'senior'
df_model['age_group'] = df_model['Age'].apply(age_group)

# 정신건강이 업무에 미치는 영향 점수화
interfere_map = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3}
df_model['interfere_level'] = df_model['work_interfere'].map(interfere_map)

# 정신건강 관련 지원 점수화
support_features = ['coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview']
support_map = {'No': 0, 'Some of them': 1, 'Yes': 2, 'Maybe': 1, "Don't know": 0}
for col in support_features:
    df_model[col + '_score'] = df_model[col].map(support_map)
df_model['support_score'] = df_model[[col + '_score' for col in support_features]].sum(axis=1)

# 복지항목 점수화
benefit_features = ['benefits', 'care_options', 'wellness_program']
benefit_map = {'No': 0, 'Yes': 1, 'Not sure': 0}
for col in benefit_features:
    df_model[col + '_score'] = df_model[col].map(benefit_map)
df_model['benefits_score'] = df_model[[col + '_score' for col in benefit_features]].sum(axis=1)

# 지역 그룹화
north_america = ['United States', 'Canada']
europe = ['United Kingdom', 'Germany', 'France', 'Netherlands', 'Ireland', 'Switzerland',
          'Italy', 'Sweden', 'Austria', 'Belgium']
df_model['region_group'] = df_model['Country'].apply(
    lambda x: 'North America' if x in north_america else ('Europe' if x in europe else 'Other')
)

# 5단계: 결측치 처리 전략 적용
# 범주형 변수 → 'Unknown'으로 대체
df_model[['age_group', 'region_group']] = df_model[['age_group', 'region_group']].fillna('Unknown')

# 수치형 변수 → 중앙값으로 대체
num_cols = ['interfere_level', 'support_score', 'benefits_score']
for col in num_cols:
    df_model[col] = df_model[col].fillna(df_model[col].median())

# 타겟 변수 결측 제거
df_model = df_model.dropna(subset=['treatment'])

# 6단계: 인코딩 (One-Hot Encoding)
df_encoded = pd.get_dummies(df_model, columns=['age_group', 'region_group'], drop_first=True)

# 타겟 인코딩 (Yes → 1, No → 0)
df_encoded['treatment'] = df_encoded['treatment'].map({'Yes': 1, 'No': 0})

# 7단계: 입력 및 타겟 변수 설정
feature_cols = ['interfere_level', 'support_score', 'benefits_score'] + \
               [col for col in df_encoded.columns if 'age_group_' in col or 'region_group_' in col]
X = df_encoded[feature_cols]
y = df_encoded['treatment']

# 8단계: 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9단계: 모델 훈련
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 10단계: 예측 및 평가
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("✅ 혼동행렬:\n", conf_matrix)
print("\n✅ 분류 리포트:\n", report)

