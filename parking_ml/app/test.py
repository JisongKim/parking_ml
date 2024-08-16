import pandas as pd
import joblib

# 모델 불러오기
model = joblib.load('best_parking_model.pkl')

# 예측할 데이터 프레임 생성
data = pd.DataFrame({
    'parking_code': [171721, 1366590, 171900, 172051, 173005],
    'hour': [16, 16, 18, 19, 13],
    'minute': [27, 33, 46, 46, 1],
    'weekday': ['Tuesday', 'Monday', 'Monday', 'Monday', 'Monday']
})

# 예측 수행
predictions = model.predict(data)

# 예측값을 반올림하여 정수로 변환
rounded_predictions = [round(pred) for pred in predictions]

# 결과를 parking_code와 함께 DataFrame으로 저장
result = pd.DataFrame({
    'parking_code': data['parking_code'],
    'predicted_avail_park_space': rounded_predictions
})

# 결과 출력
print(result)

