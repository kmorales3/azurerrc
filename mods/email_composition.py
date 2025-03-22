import os
from email.mime.multipart import MIMEMultipart
import tempfile
from email.mime.text import MIMEText
import ssl
import smtplib
import pandas as pd
import base64
import io
from email.mime.base import MIMEBase
from email import encoders
from mods.pass_processing import process_train_pass_data

def create_mail_attachment(html_body, subject, corridor):
    if corridor == "the Northeast Corridor":
        distribution_list = os.environ.get("NE_CORRIDOR")
    elif corridor == "the Southeast Corridor":
        distribution_list = os.environ.get("SE_CORRIDOR")
    elif corridor == "the Southwest Corridor" or corridor == "the SW, Out of Network,":
        distribution_list = os.environ.get("SW_CORRIDOR")
    elif corridor == "the Northwest Corridor":
        distribution_list = os.environ.get("NW_CORRIDOR")
    elif corridor == "the SE, Out of Network,":
        distribution_list = os.environ.get("SE_OON")
    elif corridor == "the NE, Out of Network,":
        distribution_list = os.environ.get("NE_OON")
    else:
        distribution_list = ""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["To"] = distribution_list
    msg["Cc"] = os.environ.get("ATTACHMENT_CC")
    msg.attach(MIMEText(html_body, "html"))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".eml") \
            as temp_file:
        temp_file.write(msg.as_bytes())
        temp_file_path = temp_file.name

    return temp_file_path, f"{corridor}.eml"


def generate_email_body_text(corridor, train_passes):
    """Generates a DataFrame containing email body text."""
    subject_line = "Hello,<br><br>Please see the below open door(s) identified by RoboRailCop. "
    if len(train_passes) < 2:
        subject_line += f"This train is destined for {corridor} with # open container(s).<br><br>"
    else:
        subject_line += f"These trains are destined for {corridor} with # open container(s).<br><br>"
    
    subject_line += "Thank you,<br>RoboRailCop Team<br><br>"
    
    email_df = pd.DataFrame({"Email Body": [subject_line]})
    return email_df

def send_email(msg, logger):
    """Handles the actual sending of the email via SMTP."""
    
    from_address = os.environ.get("OUTLOOK_FROM_EMAIL")
    group_mailbox = os.environ.get("OUTLOOK_GROUP_MAILBOX")
    to_addresses = os.environ.get("OUTLOOK_TO_EMAIL").split(",")

    try:
        outlook_pswd = os.environ.get("OUTLOOK_PSWD")
        context = ssl.create_default_context()
        server = smtplib.SMTP("smtp-mail.outlook.com", 587)
        server.starttls(context=context)
        server.login(from_address, outlook_pswd)
        text = msg.as_string()
        server.sendmail(group_mailbox, to_addresses, text)
        server.quit()
        logger.info("Email sent successfully")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        
def compose_email(body_html, attachments, logger):
    """Composes the email message with HTML body and attachments."""
    
    from_address = os.environ.get("OUTLOOK_FROM_EMAIL")
    group_mailbox = os.environ.get("OUTLOOK_GROUP_MAILBOX")
    to_addresses = os.environ.get("OUTLOOK_TO_EMAIL").split(",")
    subj_prefix = os.environ.get("EMAIL_SUBJ_PRFX")

    msg = MIMEMultipart("related")
    msg["From"] = group_mailbox
    msg["To"] = ", ".join(to_addresses)
    msg["Subject"] = f"{subj_prefix}: Open Intermodal Container/Trailer Door Detected"

    # Create the HTML part
    html_part = MIMEText(body_html, "html")  # Convert to HTML
    msg.attach(html_part)

    # Attach any attachments
    for attachment, filename in attachments:
        logger.info(f"Processing attachment: {filename}")
        logger.info(f"Attachment type: {type(attachment)}")

        part = MIMEBase("application", "octet-stream")

        if filename.endswith(".eml"):
            # Read the content of the temporary file as bytes
            with open(attachment, "rb") as attachment_file:
                attachment_data = attachment_file.read()
            part.set_payload(attachment_data)
        else:
            # Directly use the image bytes
            part.set_payload(attachment)

        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{filename}"')
        msg.attach(part)

    return msg

def format_email_body(processed_pass_data):
    """Formats the email body and `.eml` attachments with proper structure."""
    attachments = []
    main_body_df = pd.DataFrame()  # Holds only the main email text data

    for corridor, train_passes in processed_pass_data.items():
        # 🔥 Extract the email greeting text (NOT a dataframe)
        email_body_text = generate_email_body_text(corridor, train_passes).iloc[0, 0]  # Extract as plain text

        # 🔥 Process train details (for the main email)
        train_pass_df = process_train_pass_data(processed_pass_data)  # ✅ Do NOT transpose this

        # 🔥 Prepare `.eml` attachment content (Start with greeting)
        attachment_html = f"<p>{email_body_text}</p>"  # Render greeting properly
        
        for train_pass in train_passes:
            train_details_html = pd.DataFrame([train_pass.pass_data]).T.to_html(index=True, header=False)  # ✅ Transposed

            # 🔥 Process each detection separately (Transposed, NOT in a dataframe for email body)
            detection_html_blocks = []
            for detection in train_pass.pass_detections:
                detection_df = pd.DataFrame([detection["detection_data"]]).T.to_html(index=True, header=False)  # ✅ Transposed

                # 🔥 Embed images directly BELOW the detection entry
                images_html = ""
                if "crops" in detection and "crops" in detection["crops"]:
                    for crop_image in detection["crops"]["crops"]:
                        crop_buffer = io.BytesIO()
                        crop_image.save(crop_buffer, format="JPEG")
                        crop_bytes = crop_buffer.getvalue()
                        crop_encoded = base64.b64encode(crop_bytes).decode()
                        images_html += f'<img src="data:image/jpeg;base64,{crop_encoded}" width="2200"><br>'

                detection_html_blocks.append(f"{detection_df}{images_html}")

            # 🔥 Apply styling to remove table borders for `.eml` attachments
            clean_html = (
                '<style>'
                '.dataframe {border: none; text-align: left; white-space: nowrap;}'
                '.dataframe td, .dataframe th {border: none !important; text-align: left; padding: 0 10px;}'
                '</style>'
            )

            # 🔥 Merge all train & detection info for the `.eml`
            full_attachment_html = (
                f"{clean_html}{attachment_html}<br>"
                f"<strong>Train Pass:</strong><br>{train_details_html}<br>"
                f"<strong>Detections:</strong><br>{''.join(detection_html_blocks)}"
            )

            # 🔥 Attach `.eml` files with properly formatted data
            subject = "RoboRailCop: Open Intermodal Container/Trailer Door Detected"
            attachments.append(create_mail_attachment(full_attachment_html, subject, corridor))

        # 🔥 The main email should NOT contain greeting or images
        final_email_df = train_pass_df  # ✅ Do NOT transpose this
        main_body_df = pd.concat([main_body_df, final_email_df], ignore_index=True)

    return main_body_df, attachments  # 🔥 `.eml` files have transposed data, main email does NOT!