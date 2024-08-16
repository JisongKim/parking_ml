import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 데이터 불러오기
df = pd.read_csv('C:/Users/jison/Desktop/IDTTI/parking_ml/data/parking.csv')
# df2 = pd.read_csv('C:/Users/jison/Desktop/IDTTI/parking_ml/data/parking2.csv')

# 두 데이터프레임 합치기
# df = pd.concat([df1, df2], ignore_index=True)

# 날짜 변수 처리
df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['date'].dt.hour
df['minute'] = df['date'].dt.minute
df['weekday'] = df['date'].dt.strftime('%A')

# 결측치 제거
df = df.dropna()

# 독립 변수 및 종속 변수 설정
X = df.drop(columns=['current_park_space', 'avail_park_space', 'total_space', 'id', 'address', 'date'])
y = df['avail_park_space']

# 학습 데이터와 테스트 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# OneHotEncoder를 사용한 범주형 변수 처리
categorical_features = ['parking_code', 'weekday']
numerical_features = ['hour', 'minute']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', 'passthrough', numerical_features)
    ])

# 파이프라인 구성
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=100, max_depth=None, n_estimators=200))
])

# 모델 학습
model.fit(X_train, y_train)

# 테스트 데이터 예측
pred = model.predict(X_test)

# 모델 평가
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("MSE: ", mse)
print("R2: ", r2)

# 최적의 모델 및 전처리기 저장
joblib.dump(model, 'best_parking_model.pkl')


