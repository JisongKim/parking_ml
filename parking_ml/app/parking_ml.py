import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_squared_log_error
import joblib
import time

# 데이터 불러오기
df1 = pd.read_csv('C:/Users/jison/Desktop/IDTTI/parking_ml/data/parking2.csv')
df2 = pd.read_csv('C:/Users/jison/Desktop/IDTTI/parking_ml/data/parking3.csv')

# 두 데이터프레임 합치기
df = pd.concat([df1, df2], ignore_index=True)

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
    ('regressor', RandomForestRegressor(random_state=100,max_depth=None, verbose=1, n_estimators=200))
])

# 하이퍼파라미터 그리드 설정
# param_grid = {
#     'regressor__n_estimators': [100, 200, 300],
#     'regressor__max_depth': [None, 10, 20],
#     'regressor__min_samples_split': [2, 5, 10],
#     'regressor__min_samples_leaf': [1, 2, 4]
# }

# GridSearchCV 설정
# grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)

start_time = time.time()

# 모델 학습 (최적의 하이퍼파라미터 찾기)
model.fit(X_train, y_train)

end_time = time.time()

# 최적의 모델로 예측
# best_model = grid_search.best_estimator_
pred = model.predict(X_test)

# 모델 평가
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
msle = mean_squared_log_error(y_test, pred)

# print("Best Parameters: ", grid_search.best_params_)
print("MSE: ", mse)
print("R2: ", r2)
print("MAE: ", mae)
print("MSLE: ", msle)

#학습 시간 출력
elapsed_time = end_time - start_time
print(f"Training time: {elapsed_time:.2f} seconds")

# 최적의 모델 및 전처리기 저장
joblib.dump(model, 'best_parking_model.pkl')
