from flask import Flask, request, jsonify
import pandas as pd
import joblib
import logging
import mysql.connector

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 저장된 모델 로드
model = joblib.load('best_parking_model.pkl')

# 주차장 코드 리스트 (예시로 사용자가 직접 정의, 실제로는 DB나 다른 데이터 소스로부터 가져올 수 있음)
# parking_codes = [173831, 173867, 1037932, 1040225]  # 여기에 모든 주차장 코드를 나열

db_config = {
    'user': 'root',        # DB 사용자명
    'password': 'paxp',    # DB 비밀번호
    'host': 'localhost',            # DB 호스트 (예: 'localhost')
    'database': 'parking_management',    # 사용할 데이터베이스명
}

def fetch_parking_codes():
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        cursor.execute("SELECT code FROM parking_codes")  # 테이블 이름을 실제 이름으로 변경하세요.
        codes = [int(row[0]) for row in cursor.fetchall()]
        cursor.close()
        connection.close()
        return codes
    except mysql.connector.Error as err:
        logger.error(f"Database error: {err}")
        return []

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # 쿼리 파라미터에서 도착 시간과 요일을 받아옴
        hour = int(request.args.get('hour'))
        minute = int(request.args.get('minute'))
        weekday = request.args.get('weekday').strip()

        parking_codes = fetch_parking_codes()
        if not parking_codes:
            return jsonify({"error": "Failed to fetch parking codes."}), 500

        # 모든 주차장 코드를 반복하며 예측
        predictions = []
        for parking_code in parking_codes:
            input_data = pd.DataFrame([{
                'parking_code': parking_code,
                'hour': hour,
                'minute': minute,
                'weekday': weekday
            }])

            # 예측 수행
            pred = model.predict(input_data)
            rounded_pred = round(pred[0])

            # 결과를 리스트에 추가
            predictions.append({
                'parking_code': parking_code,
                'predicted_avail_park_space': rounded_pred
            })

        # 결과를 JSON 형태로 반환
        result = {'predictions': predictions}

        # 로그 기록
        logger.info(f"Received request for hour: {hour}, minute: {minute}, weekday: {weekday}")
        logger.info(f"Predictions: {result}")

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        return jsonify({"error": "An error occurred while processing the request."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
