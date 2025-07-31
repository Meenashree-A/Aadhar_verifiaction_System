from flask import Flask, request, render_template, jsonify, render_template_string, url_for, redirect
import cv2
from deepface import DeepFace
import os
import numpy as np
import base64
import json
import mysql.connector
from datetime import datetime
from flask_cors import CORS
import boto3  # Import boto3
from twilio.rest import Client

app = Flask(__name__, template_folder='templates')
CORS(app)
app.secret_key = 'your_very_secure_secret_key_here'

haar_cascade_path = r"E:\OneDrive\Desktop\aadhar_pro\haarcascade_frontalface_default.xml"

if not os.path.exists(haar_cascade_path):
    print(f"Error: Haar cascade file not found at {haar_cascade_path}")
    face_cascade = None
else:
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
    if face_cascade.empty():
        print("Error: Haar cascade failed to load.")
        face_cascade = None

# AWS SNS Configuration
# sns_client = boto3.client(
#     'sns',
#     region_name='ap-south-1'  # Replace with your AWS region
# )

def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root",
            database="aadhaar_db"
        )
        return conn
    except mysql.connector.Error as e:
        print(f"Database connection error: {e}")
        return None

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if face_cascade is not None:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)
        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)
        return faces
    else:
        return []

def preprocess_image_without_dlib(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = detect_faces(image)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        pad_w, pad_h = int(0.1 * w), int(0.1 * h)
        x, y = max(0, x - pad_w), max(0, y - pad_h)
        w, h = min(image.shape[1] - x, w + 2 * pad_w), min(image.shape[0] - y, h + 2 * pad_h)
        cropped_face = image[y:y + h, x:x + w]
        resized_face = cv2.resize(cropped_face, (224, 224))
        return resized_face
    else:
        return None

@app.route('/')
def qrcode_scanner():
    try:
        with open('aadhar_base.html', 'r', encoding='utf-8') as f:
            return render_template_string(f.read())
    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError: {e}")
        return "Error: Unable to decode the HTML file.", 500
    except FileNotFoundError:
        return "Error: HTML File not found", 404

@app.route('/faceverification')
def faceverification_page():
    try:
        with open('faceverification.html', 'r', encoding='utf-8') as f:
            return render_template_string(f.read())
    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError: {e}")
        return "Error: Unable to decode the HTML file.", 500
    except FileNotFoundError:
        return "Error: HTML File not found", 404

@app.route('/detect_face', methods=['POST'])
def detect_face():
    try:
        image_data = request.json['image']
        header, encoded = image_data.split(',', 1)
        image_bytes = base64.b64decode(encoded)
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            print("Error: cv2.imdecode failed to decode image data.")
            return jsonify({'faceDetected': False})

        if image.size == 0:
            print("Error: Decoded image is empty.")
            return jsonify({'faceDetected': False})

        cv2.imwrite("captured_debug.jpg", image)
        print(f"Captured image shape: {image.shape}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(image)
        print(f"Number of faces detected: {len(faces)}")

        if len(faces) > 0:
            return jsonify({'faceDetected': True})
        else:
            return jsonify({'faceDetected': False})
    except Exception as e:
        print(f"Error in detect_face: {e}")
        return jsonify({'faceDetected': False})

@app.route('/verify_face', methods=['POST'])
def verify_face():
    try:
        aadhaar_number = request.form['aadhaar_number']
        live_image_data = request.form['live_image_data']
        header, encoded = live_image_data.split(',', 1)
        live_image_bytes = base64.b64decode(encoded)
        live_image_np = np.frombuffer(live_image_bytes, np.uint8)
        live_image = cv2.imdecode(live_image_np, cv2.IMREAD_COLOR)

        conn = get_db_connection()
        if conn is None:
            return render_template("verification_result.html", verification="failed", message="Database connection error.")

        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT photo_path FROM face_verification WHERE aadhaar_number = %s", (aadhaar_number,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if result is None or result['photo_path'] is None:
            return render_template("verification_result.html", verification="failed", message="Aadhaar number not found or no photo path in database.")

        db_image_path = result['photo_path']

        if not os.path.exists(db_image_path):
            return render_template("verification_result.html", verification="failed", message=f"Image path {db_image_path} does not exist.")

        db_image = cv2.imread(db_image_path)

        if db_image is None:
            return render_template("verification_result.html", verification="failed", message=f"Could not read image from {db_image_path}.")

        processed_live = preprocess_image_without_dlib(live_image)
        processed_db = preprocess_image_without_dlib(db_image)

        if processed_live is None or processed_db is None:
            return render_template("verification_result.html", verification="failed", message="Face preprocessing failed.")

        try:
            result = DeepFace.verify(
                img1_path=processed_db,
                img2_path=processed_live,
                model_name="ArcFace",
                distance_metric="cosine",
                enforce_detection=False
            )
            distance = result["distance"]
            print(f"Deepface distance: {distance}")
            threshold = 0.92

            if distance <= threshold:
                return render_template("verification_result.html", verification="success", message="Face verification successful.", aadhaar_number=aadhaar_number)
            else:
                return render_template("verification_result.html", verification="failed", message="Faces do not match.")

        except Exception as e:
            return render_template("verification_result.html", verification="failed", message=f"Verification error: {e}")

    except Exception as e:
        return render_template("verification_result.html", verification="failed", message=f"Verification error: {e}")

@app.route('/scan', methods=['POST'])
def scan_qr():
    try:
        data = request.get_json()

        if not data or "qr_data" not in data:
            return jsonify({"message": "❌ Invalid QR Code"}), 400

        qr_data = data["qr_data"]

        if isinstance(qr_data, str):
            qr_data = qr_data.strip()

        try:
            aadhaar_info = json.loads(qr_data) if isinstance(qr_data, str) else qr_data
            print(f"Aadhaar Info: {aadhaar_info}")
        except json.JSONDecodeError:
            return jsonify({"message": "❌ Invalid QR Format"}), 400

        name = aadhaar_info.get("name")
        aadhaar_number = aadhaar_info.get("aadhaar_number", "").replace("-", "").strip()
        dob = aadhaar_info.get("dob")
        gender = aadhaar_info.get("gender")
        address = aadhaar_info.get("address")
        print(f"Date before conversion: {dob}")

        try:
            dob = datetime.strptime(dob, "%d-%m-%Y").strftime("%Y-%m-%d")
            print(f"Date after conversion: {dob}")
        except ValueError:
            return jsonify({"message": "❌ Invalid DOB Format. Expected format: DD-MM-YYYY"}), 400

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
        print("⚠️ Error:", e)
        return jsonify({"message": "❌ Server Error", "error": str(e)}), 500

def get_risk_assessment(aadhaar_number, db_config):
    """Retrieves complete risk assessment data from the database."""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        query = """
            SELECT aadhaar_number, risk_category, calculated_score, income, employment_years, 
                   existing_loans, credit_cards, phone_number 
            FROM risk_assessments 
            WHERE aadhaar_number = %s
        """
        cursor.execute(query, (aadhaar_number,))
        result = cursor.fetchone()
        conn.close()
        return result
    except mysql.connector.Error as e:
        print(f"Database error: {e}")
        return None


def send_twilio_sms(phone_number, message):
    """Sends an SMS message using Twilio."""
    # Your Twilio account SID and authentication token
    twilio_account_sid = 'AC85b485ac4b3c240b654ed385f5263bbc'  # Replace with your Twilio Account SID
    twilio_auth_token = 'd8c230835976be885c62818d3b07041f'  # Replace with your Twilio Auth Token
    twilio_phone_number = '+18709372998'  # Replace with your Twilio phone number
    
    try:
        # Create a Twilio client
        client = Client(twilio_account_sid, twilio_auth_token)
        
        # Send the SMS
        message = client.messages.create(
            body=message,
            from_=twilio_phone_number,
            to=phone_number
        )
        
        print(f"SMS sent successfully: {message.sid}")
        return True
    except Exception as e:
        print(f"Error sending SMS with Twilio: {e}")
        return False

def approve_loan(aadhaar_number, db_config):
    """Checks the risk assessment and approves the loan, displaying detailed information."""
    risk_data = get_risk_assessment(aadhaar_number, db_config)

    if risk_data:
        calculated_score = risk_data['calculated_score']
        risk_category = risk_data['risk_category']
        income = risk_data['income']
        employment_years = risk_data['employment_years']
        existing_loans = risk_data['existing_loans']
        credit_cards = risk_data['credit_cards']

        # Determining loan approval status
        if calculated_score >= 75:
            loan_status = "Approved"
            message = "Loan approved based on risk assessment."
        else:
            loan_status = "Rejected"
            message = "Loan rejected due to risk assessment."

        # Returning detailed information
        return {
            "aadhaar_number": aadhaar_number,
            "loan_status": loan_status,
            "message": message,
            "calculated_score": calculated_score,
            "risk_category": risk_category,
            "income": income,
            "employment_years": employment_years,
            "existing_loans": existing_loans,
            "credit_cards": credit_cards
        }
    else:
        return {"error": "Risk assessment not found."}

@app.route('/loan_approval/<aadhaar_number>')
def loan_approval(aadhaar_number):
    approval_result = approve_loan(aadhaar_number, db_config)

    if "error" not in approval_result:
        risk_data = get_risk_assessment(aadhaar_number, db_config)
        if risk_data and risk_data["phone_number"]:
            phone_number = risk_data["phone_number"]

            if approval_result["loan_status"] == "Approved":
                message_body = "Your loan has been approved!"
            else:
                message_body = "Your loan application was rejected."

    send_twilio_sms(phone_number, message_body)

    return render_template('loan_approval_result.html', approval_result=approval_result)

@app.route('/loan_approval_api/<aadhaar_number>')
def loan_approval_api(aadhaar_number):
    approval_result = approve_loan(aadhaar_number, db_config)
    return jsonify(approval_result)

@app.route('/verification_result', methods=['GET', 'POST'])
def verification_result():
    if request.method == 'POST':
        aadhaar_number = request.form.get('aadhaar_number')
        verification = request.form.get('verification')
        message = request.form.get('message')

        if verification == 'success':
            return redirect(url_for('loan_approval', aadhaar_number=aadhaar_number))
        else:
            return render_template('verification_result.html', verification=verification, message=message)
    else:
        aadhaar_number = request.args.get('aadhaar_number')
        verification = request.args.get('verification')
        message = request.args.get('message')
        return render_template('verification_result.html', verification=verification, message=message, aadhaar_number = aadhaar_number)

if __name__ == '__main__':
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': 'root',
        'database': 'aadhaar_db'
    }
    app.run(debug=True, port=8000)