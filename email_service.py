"""
email_service.py — Async Gmail SMTP email sender using aiosmtplib.
Uses explicit connection management for better performance.
"""
import os
import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

SMTP_HOST     = "smtp.gmail.com"
SMTP_PORT     = 587
SMTP_USER     = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
FROM_NAME     = "AcneGuard AI"


def _build_otp_email(to_email: str, otp: str) -> MIMEMultipart:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Your AcneGuard Password Reset OTP"
    msg["From"]    = f"{FROM_NAME} <{SMTP_USER}>"
    msg["To"]      = to_email

    html = f"""
    <!DOCTYPE html>
    <html>
    <body style="margin:0;padding:0;font-family:'Segoe UI',Arial,sans-serif;background:#f8fafc">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr><td align="center" style="padding:40px 20px">
          <table width="480" cellpadding="0" cellspacing="0"
                 style="background:#fff;border-radius:16px;border:1px solid #e2e8f0;overflow:hidden">

            <tr><td style="background:#0d9488;padding:32px;text-align:center">
              <h1 style="color:#fff;margin:0;font-size:24px;font-weight:800">🛡️ AcneGuard</h1>
              <p style="color:#99f6e4;margin:8px 0 0;font-size:14px">AI-Powered Skin Analysis</p>
            </td></tr>

            <tr><td style="padding:36px 40px">
              <h2 style="color:#0f172a;font-size:20px;margin:0 0 12px">Password Reset OTP</h2>
              <p style="color:#475569;font-size:15px;line-height:1.6;margin:0 0 28px">
                Use the OTP below to reset your password.
                It expires in <strong>10 minutes</strong>.
              </p>

              <div style="background:#f0fdf4;border:2px dashed #0d9488;border-radius:12px;
                          padding:24px;text-align:center;margin:0 0 28px">
                <p style="margin:0 0 8px;color:#64748b;font-size:13px;text-transform:uppercase;
                           letter-spacing:1px">Your OTP</p>
                <p style="margin:0;font-size:44px;font-weight:900;letter-spacing:12px;
                           color:#0d9488;font-family:'Courier New',monospace">{otp}</p>
              </div>

              <p style="color:#94a3b8;font-size:13px;line-height:1.6;margin:0">
                If you didn't request this, you can safely ignore this email.
              </p>
            </td></tr>

            <tr><td style="background:#f8fafc;border-top:1px solid #e2e8f0;
                           padding:20px 40px;text-align:center">
              <p style="color:#94a3b8;font-size:12px;margin:0">
                © 2025 AcneGuard AI · For informational purposes only
              </p>
            </td></tr>
          </table>
        </td></tr>
      </table>
    </body>
    </html>
    """

    plain = f"Your AcneGuard password reset OTP is: {otp}\nExpires in 10 minutes."
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html, "html"))
    return msg


async def send_otp_email(to_email: str, otp: str) -> None:
    """Send OTP via Gmail SMTP using explicit connection management."""
    if not SMTP_USER or not SMTP_PASSWORD:
        raise RuntimeError("Email not configured. Set SMTP_USER and SMTP_PASSWORD in .env")

    message = _build_otp_email(to_email, otp)

    smtp = aiosmtplib.SMTP(
        hostname=SMTP_HOST,
        port=SMTP_PORT,
        start_tls=True,
    )

    await smtp.connect()
    await smtp.login(SMTP_USER, SMTP_PASSWORD)
    await smtp.send_message(message)
    await smtp.quit()