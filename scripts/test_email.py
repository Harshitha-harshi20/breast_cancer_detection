import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()

EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

recipient = "myclinic760@gmail.com"  # Replace with your own test email

msg = EmailMessage()
msg['Subject'] = "Test Email"
msg['From'] = EMAIL_SENDER
msg['To'] = recipient
msg.set_content("This is a test email from your Flask app.")

try:
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
    print("✅ Test email sent successfully!")
except Exception as e:
    print(f"⚠️ Failed to send test email: {e}")
