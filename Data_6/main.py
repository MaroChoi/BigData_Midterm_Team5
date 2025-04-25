# 1단계: 데이터 로딩 및 구조 파악
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.read_csv('C:/BigData_Midterm_Team5/BigData_Midterm_Team5/Data_6/studentsperformance.csv')
df.head()
df.info()

# 2단계: 데이터 전처리
# 결측치 확인 및 처리
print(df.isnull().sum())
# 결측치 없음

# 결측치 시각화
sns.heatmap(df.isnull(), cbar=False)
plt.title('Missing Values Heatmap')
plt.show()

# 중복 데이터 확인 및 처리
print(df.duplicated().sum())
# 중복 데이터 없음

# 중복 데이터 시각화
sns.countplot
x=df.duplicated()
plt.title('Duplicate Data Count')
plt.xlabel('Duplicate')
plt.ylabel('Count')
plt.show()

# 3단계 파생변수 생성

# 성적 총합, 평균, 고득점자 여부, 수학 과목 합격 여부 파생변수 생성
df['total_score'] = df[['math score', 'reading score', 'writing score']].sum(axis=1)
df['average_score'] = df['total_score'] / 3
df['high_achiever'] = (df['average_score'] >= 90).astype(int)
df['pass_math'] = (df['math score'] >= 40).astype(int)

# 성적 총합, 평균, 고득점자 여부, 수학 과목 합격 여부 시각화
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.hist(df['total_score'], bins=20, color='skyblue', edgecolor='black')
plt.title('Total Score Distribution')
plt.xlabel('Total Score')
plt.ylabel('Frequency')
plt.show()

# 4단계: 범주형 변수 인코딩
df['gender'] = LabelEncoder().fit_transform(df['gender'])
onehot_cols = [
        'race/ethnicity',
        'parental level of education',
        'lunch',
        'test preparation course'
    ]
df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)

# 4. 분석용 원본: 시각화 및 요약 통계용
df_original = pd.read_csv('C:/BigData_Midterm_Team5/BigData_Midterm_Team5/Data_6/studentsperformance.csv')
df_original['average_score'] = df['average_score']
df_original['high_achiever'] = df['high_achiever']

    # 5. 부모 교육 수준 정렬 순서 지정 (시각화용)
education_order = [
        "master's degree",
        "bachelor's degree",
        "associate's degree",
        "some college",
        "some high school",
        "high school"
    ]
df_original['parental level of education'] = pd.Categorical(
df_original['parental level of education'],
categories=education_order,
ordered=True
    )

    # 6. 요약 통계 계산
avg_by_edu = df_original.groupby('parental level of education')['average_score'].mean()
high_achievers_by_edu = df_original.groupby('parental level of education')['high_achiever'].mean()

    # 7. 박스플롯 시각화 (정렬 적용)
plt.figure(figsize=(10, 6))
sns.boxplot(
        data=df_original,
        x='parental level of education',
        y='average_score',
        order=education_order
    )
plt.xticks(rotation=45)
plt.title("Average Score by Parental Level of Education")
plt.xlabel("Parental Level of Education")
plt.ylabel("Average Score")
plt.tight_layout()
plt.savefig("average_score_by_parental_education_sorted.png")
plt.show()
    
        # 8. 고득점자 비율 시각화 (정렬 적용)
plt.figure(figsize=(10, 6))
sns.barplot(
        data=df_original,
        x='parental level of education',
        y='high_achiever',
        order=education_order
    )
plt.xticks(rotation=45)
plt.title("High Achiever Ratio by Parental Level of Education")
plt.xlabel("Parental Level of Education")
plt.ylabel("High Achiever Ratio")
plt.tight_layout()
plt.savefig("high_achiever_ratio_by_parental_education_sorted.png")

plt.show()
# 9. CSV 파일로 저장
df_original.to_csv("students_performance_final_sorted.csv", index=False)
# 10. 결과 확인
print("📊 평균 점수 (교육 수준별):\n", avg_by_edu
      )
print("\n🌟 고득점자 비율 (90점 이상):\n", high_achievers_by_edu
      )
# 11. 데이터 시각화
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.hist(df['total_score'], bins=20, color='skyblue', edgecolor='black')
plt.title('Total Score Distribution')
plt.xlabel('Total Score')
plt.ylabel('Frequency')
plt.subplot(2, 2, 2)
plt.hist(df['average_score'], bins=20, color='lightgreen', edgecolor='black')
plt.title('Average Score Distribution')
plt.xlabel('Average Score')
plt.ylabel('Frequency')
plt.subplot(2, 2, 3)
plt.hist(df['high_achiever'], bins=2, color='salmon', edgecolor='black')
plt.title('High Achiever Distribution')
plt.xlabel('High Achiever')
plt.ylabel('Frequency')
plt.subplot(2, 2, 4)
plt.hist(df['pass_math'], bins=2, color='lightcoral', edgecolor='black')
plt.title('Pass Math Distribution')
plt.xlabel('Pass Math')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("student_performance_histograms.png")
plt.show()

# 12. 상관관계 분석
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.show()