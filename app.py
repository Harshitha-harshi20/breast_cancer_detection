import os
from flask import Flask, render_template, request, send_file, redirect, url_for, session
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from datetime import datetime
from PIL import Image, ImageFile
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle, Paragraph, Frame
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.utils import ImageReader
import werkzeug
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

# ----------------------------
# Load Environment Variables
# ----------------------------
load_dotenv()
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "supersecretkey")

# ----------------------------
# Flask App Setup
# ----------------------------
app = Flask(__name__)
app.secret_key = SECRET_KEY

# ----------------------------
# Load Model
# ----------------------------
model_path = "models/breast_cancer_resnet50_cnn_finetuned.keras"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model not found: {model_path}")
model = tf.keras.models.load_model(model_path)
print(f"✅ Model loaded from: {model_path}")

# ----------------------------
# Pillow Fix for Truncated Images
# ----------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ----------------------------
# Utility Functions
# ----------------------------
def ensure_png(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        png_path = os.path.splitext(image_path)[0] + ".png"
        img.save(png_path, "PNG")
        return png_path
    except Exception as e:
        print(f"⚠️ Error processing image {image_path}: {e}")
        return None

def detect_stage(confidence, result):
    if result == "Benign":
        return "Stage 0"
    elif confidence < 0.60:
        return "Stage I"
    elif confidence < 0.75:
        return "Stage II"
    elif confidence < 0.90:
        return "Stage III"
    else:
        return "Stage IV"

def send_email(recipient_email, subject, body, attachment_path):
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        print("⚠️ Email credentials missing!")
        return False
    try:
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = EMAIL_SENDER
        msg['To'] = recipient_email
        msg.set_content(body)
        with open(attachment_path, 'rb') as f:
            msg.add_attachment(f.read(), maintype='application', subtype='pdf',
                               filename=os.path.basename(attachment_path))
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        print(f"✅ Email sent to {recipient_email}")
        return True
    except Exception as e:
        print(f"⚠️ Email sending failed: {e}")
        return False

# ----------------------------
# Routes
# ----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file or file.filename == '':
        return redirect(request.url)

    filename = werkzeug.utils.secure_filename(file.filename)
    filepath = os.path.join("static", filename)
    file.save(filepath)

    scan_image = ensure_png(filepath)
    if not scan_image:
        return "❌ Invalid image uploaded. Please try again."

    # Preprocess image
    img = image.load_img(scan_image, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Model prediction
    prediction = model.predict(img_array)
    if prediction.shape[-1] == 1:
        confidence = float(prediction[0][0])
        result = "Malignant" if confidence > 0.5 else "Benign"
    else:
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx])
        result = "Malignant" if class_idx == 1 else "Benign"

    stage = detect_stage(confidence, result)

    report = {
        "Name": request.form.get("name"),
        "Age": request.form.get("age"),
        "Address": request.form.get("address"),
        "Email": request.form.get("email"),
        "Result": result,
        "Stage": stage,
        "Confidence": f"{confidence*100:.2f}%",
        "Consultation": "Consult your oncologist immediately." if result=="Malignant" else "Routine screening advised.",
        "Precautions": "Follow treatment and maintain healthy lifestyle." if result=="Malignant" else "Maintain healthy diet and exercise.",
        "GeneratedOn": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    session['report'] = report
    session['result'] = result
    session['scan_image'] = scan_image

    # ----------------------------
    # Generate PDF Report
    # ----------------------------
    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join("reports", f"{timestamp}_Breast_Cancer_Report.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    styles = getSampleStyleSheet()
    normal_style = styles['Normal']

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "Breast Cancer Diagnostic Center")
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(50, height - 70, "Bengaluru | Email: support@bcdc.in | Phone: +91 9876543210")
    c.line(50, height - 95, width - 50, height - 95)

    # Patient Table
    patient_data = [
        ["Field", "Details"],
        ["Name", report['Name']],
        ["Age", str(report['Age'])],
        ["Address", report['Address']],
        ["Email", report['Email']],
        ["Patient ID", request.form.get("patient_id", "N/A")],
        ["Ref. Doctor", request.form.get("ref_doctor", "N/A")]
    ]
    table = Table(patient_data, colWidths=[120, 360])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
    ]))
    table.wrapOn(c, width, height)
    table.drawOn(c, 50, height - 280)

    # Scan Image below table
    try:
        img = Image.open(scan_image)
        aspect = img.height / img.width
        img_width = 300
        img_height = img_width * aspect
        image_y = height - 280 - img_height - 10
        c.drawImage(ImageReader(img), 50, image_y, width=img_width, height=img_height)
    except Exception as e:
        print(f"⚠️ Could not add scan image: {e}")
        img_height = 0
        image_y = height - 280 - 10

    # Clinical Table
    clinical_data = [
        ["Field", "Details"],
        ["Imaging Modality", request.form.get("modality", "N/A")],
        ["Predicted Result", report['Result']],
        ["Confidence", report['Confidence']],
        ["Consultation Advice", report['Consultation']],
        ["Precautions / Notes", report['Precautions']]
    ]
    table2 = Table(clinical_data, colWidths=[130, 350])
    table2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
    ]))
    clinical_table_y = image_y - 150
    table2.wrapOn(c, width, height)
    table2.drawOn(c, 50, clinical_table_y)

    # Stage Highlight Box
    stage_color = colors.green
    if report['Stage'] in ["II", "Stage II"]:
        stage_color = colors.orange
    elif report['Stage'] in ["III", "IV", "Stage III", "Stage IV"]:
        stage_color = colors.red
    stage_box_y = clinical_table_y - 40
    c.setFillColor(stage_color)
    c.rect(50, stage_box_y, 150, 20, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(55, stage_box_y + 5, f"Cancer Stage: {report['Stage']}")
    c.setFillColor(colors.black)

    # Impression Section at bottom
    impression_text = f"""
Findings suggest {report['Stage']} {report['Result']}.
{report['Consultation']}
{report['Precautions']}
"""
    impression_style = styles['Normal']
    impression_style.fontSize = 10
    impression_style.leading = 14
    frame = Frame(50, 50, width - 100, stage_box_y - 80, showBoundary=0)
    p = Paragraph(impression_text, impression_style)
    frame.addFromList([p], c)

    # Footer with generated date
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(50, 30, f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Save PDF
    c.save()
    session['pdf_path'] = pdf_path

    # Optionally send email
    email_status = False
    if EMAIL_SENDER and EMAIL_PASSWORD and report["Email"]:
        subject = "Your Breast Cancer Report"
        body = f"Dear {report['Name']},\n\nPrediction Result: {report['Result']}\nCancer Stage: {report['Stage']}\nConfidence: {report['Confidence']}\n\nPrecautions: {report['Precautions']}\n\nPlease find the attached report.\n\nBest regards,\nBreast Cancer Detection Team"
        email_status = send_email(report["Email"], subject, body, pdf_path)

    return render_template('result.html', report=report, result=result, stage=stage, email_status=email_status)

@app.route('/download')
def download():
    pdf_path = session.get('pdf_path')
    if not pdf_path or not os.path.exists(pdf_path):
        return redirect(url_for('index'))
    return send_file(pdf_path, as_attachment=True)

# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)


