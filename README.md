# Aadhar_verifiaction_System
An Aadhaar Verification System using Python and AI that reads the QR code embedded in Aadhaar cards to extract and verify identity details, and optionally matches user identity via face recognition.

Features
QR Code extraction from Aadhaar card

Cross-verification of extracted data with uploaded user input

Optional face recognition for matching user identity

Validates key Aadhaar fields (Name, DOB, Gender, Aadhaar Number)

Tampering detection by comparing QR data with visual input (optional)

Use Cases
This system can be used for:

Government digital KYC

FinTech and Banking onboarding

Online identity validation in exams or job applications

College or hostel admission verification

Tech Stack
Technology	Purpose
Python	Core logic
OpenCV	Image processing
pyzbar	QR code scanning from image
xml.etree	Parsing embedded XML in QR
face_recognition	Face match (optional)
Streamlit (optional)	Frontend UI

How to Run
Clone the Repository

bash
Copy
Edit
git clone https://github.com/your-username/aadhaar-verifier.git
cd aadhaar-verifier
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the Application

If using Streamlit:

bash
Copy
Edit
streamlit run app.py
If using only terminal:

bash
Copy
Edit
python main.py
