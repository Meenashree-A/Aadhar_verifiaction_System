from flask import Flask, request, jsonify
import json
import mysql.connector
from datetime import datetime
from flask_cors import CORS  

app = Flask(__name__)
CORS(app)

# Database connection function
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="aadhaar_db"
    )

@app.route('/scan', methods=['POST'])
def scan_qr():
    try:
        print("\n🔍 Raw Request Data:", request.data)  # Debugging
        data = request.get_json()

        if not data or "qr_data" not in data:
            return jsonify({"message": "❌ Invalid QR Code"}), 400

        qr_data = data["qr_data"]

        if isinstance(qr_data, str):
            qr_data = qr_data.strip()  # Trim spaces & newlines

        print("📜 Scanned QR Data:", qr_data)  # Debugging

        # Parse JSON from QR Code
        try:
            aadhaar_info = json.loads(qr_data) if isinstance(qr_data, str) else qr_data
        except json.JSONDecodeError:
            return jsonify({"message": "❌ Invalid QR Format"}), 400

        # Extract details
        name = aadhaar_info.get("name")
        aadhaar_number = aadhaar_info.get("aadhaar_number", "").replace("-", "").strip()
        dob = aadhaar_info.get("dob")
        gender = aadhaar_info.get("gender")
        address = aadhaar_info.get("address")

        # Validate required fields
        if not all([name, aadhaar_number, dob, gender, address]):
            return jsonify({"message": "❌ Missing fields in QR data"}), 400

        # Convert DOB format to match MySQL
        try:
            dob = datetime.strptime(dob, "%d-%m-%Y").strftime("%Y-%m-%d")
        except ValueError:
            return jsonify({"message": "❌ Invalid DOB Format. Expected format: DD-MM-YYYY"}), 400

        # Database verification
        try:
            db = get_db_connection()
            cursor = db.cursor(dictionary=True)

            query = """
                SELECT * FROM users 
                WHERE name=%s AND aadhaar_number=%s AND dob=%s AND gender=%s AND address=%s
            """
            values = (name, aadhaar_number, dob, gender, address)
            cursor.execute(query, values)
            user = cursor.fetchone()

            cursor.close()
            db.close()

            if user:
                return jsonify({"message": "✅ Aadhaar Verification Successful!", "status": "success"})
            else:
                return jsonify({"message": "❌ Aadhaar Verification Failed!", "status": "failed"}), 400

        except mysql.connector.Error as db_error:
            print("⚠️ Database Error:", db_error)
            return jsonify({"message": "❌ Database Error", "error": str(db_error)}), 500

    except Exception as e:
        print("⚠️ Error:", e)  # Debugging
        return jsonify({"message": "❌ Server Error", "error": str(e)}), 500

# Run Flask server on a safe port (5000)
if __name__ == '__main__':
    print("🚀 Starting Flask server on port 5000...")
    app.run(debug=True, port=5000)
