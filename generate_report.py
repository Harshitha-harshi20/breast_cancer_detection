from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf_report(
    filename,
    patient_name,
    age,
    address,
    patient_id,
    ref_doctor,
    imaging_modality,
    predicted_result,
    confidence,
    stage,
    consultation_advice,
    precautions
):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    styles = getSampleStyleSheet()
    normal_style = styles['Normal']

    # --- Header ---
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Breast Cancer Diagnostic Center")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 70, "Bengaluru | Email: support@bcdc.in | Phone: +91 9876543210")
    c.line(50, height - 95, width - 50, height - 95)

    # --- Patient Info Table ---
    patient_data = [
        ["Name", patient_name],
        ["Age", str(age)],
        ["Address", address],
        ["Patient ID", patient_id],
        ["Ref. Doctor", ref_doctor]
    ]
    table = Table(patient_data, colWidths=[100, 350])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('ALIGN', (0,0), (-1,-1), 'LEFT')
    ]))
    table.wrapOn(c, width, height)
    table.drawOn(c, 50, height - 250)

    # --- Clinical Info Table ---
    clinical_data = [
        ["Imaging Modality", imaging_modality],
        ["Predicted Result", predicted_result],
        ["Confidence", f"{confidence:.2f}%"],
        ["Detected Stage", stage],
        ["Consultation Advice", consultation_advice],
        ["Precautions / Notes", precautions]
    ]
    table2 = Table(clinical_data, colWidths=[130, 320])
    
    # Stage color coding
    stage_color = colors.green
    if stage == "II":
        stage_color = colors.orange
    elif stage in ["III", "IV"]:
        stage_color = colors.red
    
    table2_style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
    ])
    table2.setStyle(table2_style)

    table2.wrapOn(c, width, height)
    table2.drawOn(c, 50, height - 450)

    # Draw colored stage box manually over stage cell
    c.setFillColor(stage_color)
    c.rect(180, height - 373, 60, 15, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(185, height - 370, stage)
    c.setFillColor(colors.black)

    # --- Impression Section ---
    y = height - 470
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Impression")
    y -= 20
    c.setFont("Helvetica", 10)
    impression_text = f"Findings suggest Stage {stage} {predicted_result}. {consultation_advice} {precautions}"
    p = Paragraph(impression_text, normal_style)
    p.wrapOn(c, width - 100, 100)
    p.drawOn(c, 50, y - 50)

    # --- Doctor Signature ---
    y_sig = 100
    c.line(50, y_sig, 200, y_sig)
    c.drawString(50, y_sig - 15, "Radiologist Signature")
    c.line(width - 250, y_sig, width - 50, y_sig)
    c.drawString(width - 250, y_sig - 15, "End of Report")

    # --- Footer ---
    from datetime import datetime
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(50, 30, f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c.save()
    print(f"PDF report saved as {filename}")

# --- Example usage ---
generate_pdf_report(
    filename="Breast_Cancer_Report_Professional.pdf",
    patient_name="Sonu Harshitha",
    age=28,
    address="Bengaluru, India",
    patient_id="BC12345",
    ref_doctor="Dr. Mehta",
    imaging_modality="Mammography",
    predicted_result="Malignant",
    confidence=92.5,
    stage="II",
    consultation_advice="Consult oncologist for further evaluation.",
    precautions="Maintain regular follow-ups and report any new symptoms."
)
