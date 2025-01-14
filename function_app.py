# Standard Library Imports
import io
import json
import logging
import os
import re
import smtplib
import ssl
import tempfile
import time
from datetime import datetime, timedelta, timezone
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Third-Party Imports
import pandas as pd
import pytz
import requests
from PIL import Image
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from azure.keyvault.secrets import SecretClient
from opencensus.ext.azure.log_exporter import AzureLogHandler

# Azure Functions Imports
import azure.functions as func
import itertools

app = func.FunctionApp()

# Constants
MAX_EMAIL_SIZE = 20 * 1024 * 1024  # 20MB
CENTRAL_TZ = pytz.timezone("US/Central")
MOUNTAIN_TZ = pytz.timezone("US/Mountain")
UTC_TZ = pytz.utc
TIMEZONES = {
    "Mountain": MOUNTAIN_TZ,
    "Central": CENTRAL_TZ,
    "UTC": UTC_TZ,
}


@app.timer_trigger(schedule="0 */1 * * * *", arg_name="myTimer",
                   run_on_startup=False, use_monitor=False)
def rrc_trigger(myTimer: func.TimerRequest) -> None:

    class Detection:

        def __init__(self):
            self.detection_data = {
                "Created at": "",
                "Car ID": "",
                "Intermodal Container ID": "",
                "Camera ID": "",
            }

            self.image = None

        def create_detection(self, creation_time, car_id, camera, image):
            self.detection_data["Created at"] = creation_time
            self.detection_data["Car ID"] = car_id
            self.detection_data["Intermodal Container ID"] = "Not Visible"
            self.detection_data["Camera ID"] = camera

            self.image = image

        def get_image(self):
            return self.image

    class TrainPass:
        def __init__(self):
            self.pass_data = {
                "Train Symbol": "",
                "Train Arrival Date/Time": "",
                "Train Destination": "",
                "Destination Corridor": "",
                "Train Sequence Number": "",
                "Detector Site": "",
                "Track Number": "",
                "Mile Post": "",
            }

            self.pass_detections = []

        def add_detection(self, detection):
            self.pass_detections.append(detection.__dict__)

        # Function to create TrainPass objects from car_grouped_symbols
    def create_train_pass_objects(car_grouped_symbols):
        train_pass_objects = []

        for corridor, corridor_groups in car_grouped_symbols.items():
            for train_symbol, symbol_groups in corridor_groups:
                sorted_symbol_groups = sorted(symbol_groups, key=lambda x:
                                              x[1][0][3])
                corridor_symbol_pass = TrainPass()
                corridor_symbol_pass.pass_data["Train Symbol"] = train_symbol
                train_detail = sorted_symbol_groups[0][1][0]
                corridor_symbol_pass.pass_data["Train Arrival Date/Time"] = \
                    train_detail[3]
                corridor_symbol_pass.pass_data["Train Destination"] = (
                    f"{train_detail[11]} - {train_detail[12]}"
                )
                corridor_symbol_pass.pass_data["Destination Corridor"] = \
                    train_detail[14]
                corridor_symbol_pass.pass_data["Train Sequence Number"] = \
                    train_detail[5]
                (
                    corridor_symbol_pass.pass_data["Detector Site"],
                    corridor_symbol_pass.pass_data["Track Number"],
                ) = transform_site_string(train_detail[6])
                corridor_symbol_pass.pass_data["Mile Post"] = "681.1"

                for car_id, cars in symbol_groups:
                    for car in cars:
                        car_detection = Detection()
                        car_detection.create_detection(
                            car[9],
                            f"{car[0]}-{car[1]}",
                            car[7],
                            car[13],
                        )
                        corridor_symbol_pass.add_detection(car_detection)

                train_pass_objects.append(corridor_symbol_pass)

            # Group the TrainPass objects by "Destination Corridor"
        grouped_by_corridor = {}
        for train_pass in train_pass_objects:
            destination_corridor = train_pass.pass_data["Destination Corridor"]
            if destination_corridor not in grouped_by_corridor:
                grouped_by_corridor[destination_corridor] = []
            grouped_by_corridor[destination_corridor].append(train_pass)

        return grouped_by_corridor

    def transform_site_string(input_str):
        # Add a space before every capital letter except the first one
        spaced_str = re.sub(r"(?<!^)(?=[A-Z])", " ", input_str)

        # Extract numeric characters at the end of the string
        match = re.search(r"\d+$", spaced_str)
        number = match.group() if match else ""

        # Remove numeric characters at the end of the string
        result_str = re.sub(r"\d+$", "", spaced_str)

        return [result_str, number]

    def group_by_corridor(car_list):

        northeast_corridor = {
            "AL",
            "AR",
            "IA",
            "IL",
            "KY",
            "MN",
            "MO",
            "MS",
            "TN",
            "WI",
        }
        southeast_corridor = {"AZ", "KS", "LS", "NM", "OK", "TX"}
        southwest_corridor = {"CA", "NV"}
        northwest_corridor = {
            "CO",
            "ID",
            "MT",
            "ND",
            "NE",
            "OR",
            "SD",
            "UT",
            "WA",
            "WY",
            "BC",
        }
        ga = {"GA"}
        ny = {"NY"}

        for car_detection in car_list:
            dest_state = car_detection[12]
            if dest_state in northeast_corridor:
                car_detection.append("Northeast Corridor")
            elif dest_state in southeast_corridor:
                car_detection.append("Southeast Corridor")
            elif dest_state in southwest_corridor:
                car_detection.append("Southwest Corridor")
            elif dest_state in northwest_corridor:
                car_detection.append("Northwest Corridor")
            elif dest_state in ga:
                car_detection.append("GA")
            elif dest_state in ny:
                car_detection.append("NY")
            else:
                car_detection.append("Unknown Corridor")

            # Sort the car_list by the last index (corridor)
        sorted_car_list = sorted(car_list, key=lambda x: x[-1])

        # Group the sorted car_list by the last index (corridor)
        grouped_car_list = itertools.groupby(sorted_car_list,
                                             key=lambda x: x[-1])

        # Convert the groupby object to a dictionary
        grouped_corridors = {key: list(group) for key, group in
                             grouped_car_list}

        return grouped_corridors

    def group_by_train_symbol(grouped_corridors):
        symbol_grouped_corridors = {}

        for corridor, car_list in grouped_corridors.items():
            symbol_sorted_car_list = sorted(car_list, key=lambda x: str(x[10]))
            symbol_grouped_cars = itertools.groupby(symbol_sorted_car_list,
                                                    key=lambda x: x[10])

            # Convert the groupby object to a list of tuples
            grouped_corridor = [(key, list(group)) for key, group in
                                symbol_grouped_cars]
            symbol_grouped_corridors[corridor] = grouped_corridor

        return symbol_grouped_corridors

    def group_by_car_id(symbol_grouped_corridors):
        car_grouped_symbols = {}

        for corridor, grouped_data in symbol_grouped_corridors.items():
            car_grouped_data = []
            for train_symbol, cars in grouped_data:
                # Sort cars by the combined string of indices 0 and 1
                sorted_cars = sorted(cars, key=lambda x: f"{x[0]}_{x[1]}")
                # Group by the combined string of indices 0 and 1
                combined_grouped = itertools.groupby(
                    sorted_cars, key=lambda x: f"{x[0]}_{x[1]}"
                )
                # Convert the groupby object to a list of tuples
                grouped_cars = [(key, list(group)) for key, group in
                                combined_grouped]

                # Keep only one car_id chosen by the top camera_id (index 8)
                filtered_grouped_cars = []
                for key, group in grouped_cars:
                    top_car = max(group, key=lambda x: x[8])
                    filtered_grouped_cars.append((key, [top_car]))

                car_grouped_data.append((train_symbol, filtered_grouped_cars))
            car_grouped_symbols[corridor] = car_grouped_data

        return car_grouped_symbols

    def execute_notebook(databricks_instance, token, notebook_path, car_id,
                         car_number):
        # API endpoint for running a notebook
        url = f"{databricks_instance}/api/2.0/jobs/runs/submit"

        # Define the payload for the API request
        payload = {
            "run_name": "Car Query Run",
            "existing_cluster_id": "0422-200925-pints678",
            "notebook_task": {
                "notebook_path": notebook_path,
                "base_parameters": {"car_id": car_id, "car_number":
                                    car_number},
            },
        }

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        # Run the notebook
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code}, {response.text}")

        run_id = response.json().get("run_id")
        if not run_id:
            raise Exception("Failed to get run_id from the response")

        # Check job status
        while True:
            status_url = f"{databricks_instance}/api/2.0/jobs/runs/get?run_id={run_id}"
            status_response = requests.get(status_url, headers=headers)
            run_status = status_response.json()

            life_cycle_state = run_status["state"]["life_cycle_state"]

            if life_cycle_state in ["TERMINATED", "SKIPPED", "FAILED"]:
                # Job has completed
                break

            # Sleep for a while before checking again
            time.sleep(5)  # Wait for 5 seconds before the next check

        # Retrieve results from the output
        output_url = f"{databricks_instance}/api/2.0/jobs/runs/get-output?run_id={run_id}"
        output_response = requests.get(output_url, headers=headers)
        return output_response.json()

    def get_blob_service_client():
        account_url = os.environ.get("AZURE_STRG_ACCT_URL")
        if not account_url:
            raise ValueError("AZURE_STRG_ACCT_URL environment \
                variable is missing.")

        credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(
            account_url=account_url, credential=credential
        )
        return blob_service_client

    def get_recent_passes(container_client, seconds):
        recent_passes = []
        time_threshold = datetime.now(timezone.utc) - \
            timedelta(seconds=seconds)

        # List all blobs in the container
        blobs = container_client.list_blobs()
        for trn_pass in blobs:
            if trn_pass["creation_time"] >= time_threshold:
                recent_passes.append(trn_pass)

        if not recent_passes:
            print("No recent passes found.")

        return recent_passes

    def download_passes(container_client, passes):

        pass_list = []

        if not passes:
            print("No passes available to download.")
            return []

        # Download the most recent pass
        for trn_pass in passes:
            blob_client = container_client.get_blob_client(trn_pass)
            download_stream = blob_client.download_blob()
            pass_image = download_stream.readall()
            print(f"Downloaded: {trn_pass.name}")
            pass_list.append(
                [
                    pass_image,
                    trn_pass.name,
                    trn_pass.creation_time,
                ]
            )

        return pass_list

    def load_tracking_dict():
        key_vault_name = os.environ["KEY_VAULT_NAME"]
        secret_name = "tracking-dict"
        kv_uri = f"https://{key_vault_name}.vault.azure.net"

        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=kv_uri, credential=credential)

        secret = client.get_secret(secret_name)
        tracking_dict = json.loads(secret.value)
        return tracking_dict

    def update_tracking_dict(tracking_dict, processed_keys=set()):
        for key in processed_keys:
            if key in tracking_dict:
                tracking_dict[key]["executions_since_addition"] += 1
                if tracking_dict[key]["executions_since_addition"] > 3:
                    del tracking_dict[key]
            else:
                tracking_dict[key] = {"executions_since_addition": 1}
        for key in list(tracking_dict.keys()):
            if key not in processed_keys:
                tracking_dict[key]["executions_since_addition"] += 1
                if tracking_dict[key]["executions_since_addition"] > 3:
                    del tracking_dict[key]

    def save_tracking_dict(new_dict):
        key_vault_name = os.environ["KEY_VAULT_NAME"]
        secret_name = "tracking-dict"
        kv_uri = f"https://{key_vault_name}.vault.azure.net"

        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=kv_uri, credential=credential)

        new_secret_value = json.dumps(new_dict)
        client.set_secret(secret_name, new_secret_value)

    def create_mail_attachment(html_body, subject, corridor):
        if corridor == "Northeast Corridor":
            distribution_list = os.environ.get("NE_CORRIDOR")
        elif corridor == "Southeast Corridor":
            distribution_list = os.environ.get("SE_CORRIDOR")
        elif corridor == "Southwest Corridor":
            distribution_list = os.environ.get("SW_CORRIDOR")
        elif corridor == "Northwest Corridor":
            distribution_list = os.environ.get("NW_CORRIDOR")
        elif corridor == "GA":
            distribution_list = os.environ.get("GA_TRAINS")
        elif corridor == "NY":
            distribution_list = os.environ.get("NY_TRAINS")
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

    def format_email_body(processed_pass_data, logger):
        attachments = []
        max_email_size = 20 * 1024 * 1024  # 20MB

        # Collect all images and their sizes
        images = []

        # Collect data for the main body DataFrame
        main_body_data = []

        for corridor, train_passes in processed_pass_data.items():
            if len(train_passes) < 2:
                attachment_body = f"Hello,<br><br>Please see the below open \
                    door(s) identified by RoboRailCop. This train is destined \
                        for the {corridor} with # open container(s).\
                            <br><br>Thank you,<br>RoboRailCop Team<br><br>"
            else:
                attachment_body = f"Hello,<br><br>Please see the below open \
                    door(s) identified by RoboRailCop. These trains are \
                        destined for the {corridor} with # open container(s).\
                            <br><br>Thank you,<br>RoboRailCop Team<br><br>"

            train_symbols = []

            for train_pass in train_passes:
                train_symbol = train_pass.pass_data["Train Symbol"]
                if train_symbol not in train_symbols and \
                        isinstance(train_symbol, str):
                    train_symbols.append(train_symbol)  # Collect train symbols

                train_destination = train_pass.pass_data["Train Destination"]

                trn_arrival_dt_mt = format_dt_times(
                    train_pass.pass_data["Train Arrival Date/Time"], "Mountain"
                )

                # Collect detection information
                for detection in train_pass.pass_detections:
                    detection_created_at_mt = format_dt_times(
                        detection["detection_data"]["Created at"], "Mountain"
                    )
                    detection_car_id = detection["detection_data"]["Car ID"]

                    # Define the Central Time Zone
                    central_tz = pytz.timezone("US/Central")

                    # Get the current time in UTC
                    now_utc = datetime.now(pytz.utc)

                    # Convert the current time to Central Time
                    now_central = now_utc.astimezone(central_tz)

                    # Add the key information to the main body data
                    main_body_data.append(
                        {
                            "Email Date/Time (CT)": now_central.strftime(
                                "%Y/%m/%d %H:%M:%S"
                            ),
                            "Train Symbol": train_symbol,
                            "Arrival Date/Time (CT)": format_dt_times(
                                train_pass.pass_data
                                ["Train Arrival Date/Time"],
                                "Central",
                            ),
                            "Destination": train_destination,
                            "Detection Created At (CT)": format_dt_times(
                                detection["detection_data"]["Created at"],
                                "Central"
                            ),
                            "Detection Car ID": detection_car_id,
                        }
                    )

                    detection["detection_data"]["Created at"] = \
                        detection_created_at_mt

                train_pass.pass_data["Train Arrival Date/Time"] = \
                    trn_arrival_dt_mt

                # Drop the "Destination Corridor" key and value
                pass_data = train_pass.pass_data.copy()
                pass_data.pop("Destination Corridor", None)

                # Convert pass_data to a transposed DataFrame with no index
                pass_data_df = pd.DataFrame(pass_data, index=[0]).T

                # Prepare the attachment body
                attachment_body += "Train Pass:<br>"
                attachment_body += pass_data_df.to_html(header=False)
                attachment_body += "<br>Detections:<br>"
                for detection in train_pass.pass_detections:
                    detections_df = pd.DataFrame(detection["detection_data"],
                                                 index=[0]).T
                    attachment_body += detections_df.to_html(header=False)
                    attachment_body += "<br>"

                    # Process image
                    image_data = detection["image"]

                    # Ensure image_data is not None and is of type bytes
                    if not isinstance(image_data, bytes):
                        raise ValueError("image_data must be of type bytes")

                    # Open the image from the byte stream
                    try:
                        image = Image.open(io.BytesIO(image_data))
                    except IOError:
                        raise ValueError("Invalid image data")

                    # Save the image to a buffer
                    buffered = io.BytesIO()
                    image.save(buffered, format="JPEG")
                    image_size = buffered.tell()

                    # Collect image data
                    images.append(
                        (
                            image,
                            f"{corridor}_{detection['detection_data']['Car ID']}",
                            image_size,
                        )
                    )

                attachment_body += "<br><br>"

            # Add CSS to remove borders, ensure text stays on one line,
            # left-align text, and add padding
            attachment_body = (
                """
            <style>
                .dataframe {
                    border: 0;
                    text-align: left;
                    white-space: nowrap;
                }
                .dataframe td, .dataframe th {
                    border: 0 !important;
                    white-space: nowrap;
                    text-align: left;
                    padding: 0 10px;
                }
            </style>
            """
                + attachment_body
            )

            subject = "RoboRailCop: Open Intermodal Container/Trailer \
                Door Detected: " + (
                ", ".join(train_symbols)
            )

            attachments.append(create_mail_attachment(attachment_body,
                                                      subject, corridor))

        # Create the main body DataFrame
        main_body_df = pd.DataFrame(main_body_data)

        # Convert the main body DataFrame to HTML
        body = main_body_df.to_html(index=False)

        # Calculate the size of the email body
        body_size = len(body.encode("utf-8"))

        # Calculate the total size of the original images
        total_image_size = sum(image_size for _, _, image_size in images)

        # Check if resizing is necessary
        if body_size + total_image_size > max_email_size:
            # Calculate the remaining space for images
            remaining_space = max_email_size - body_size

            # Calculate the average space available per image
            avg_space_per_image = remaining_space // len(images) if \
                images else 0

            # Resize images if necessary
            for image, dest_car_id, image_size in images:
                if image_size > avg_space_per_image:
                    scaling_factor = (avg_space_per_image / image_size) ** 0.5
                    new_width = int(image.width * scaling_factor)
                    new_height = int(image.height * scaling_factor)
                    image = image.resize((new_width, new_height),
                                         Image.Resampling.LANCZOS)

                    # Save the resized image to a buffer
                    buffered = io.BytesIO()
                    image.save(buffered, format="JPEG")
                    image_size = buffered.tell()
                else:
                    # Save the original image to a buffer
                    buffered = io.BytesIO()
                    image.save(buffered, format="JPEG")

                # Add the image to the list of attachments
                attachments.append((buffered.getvalue(), f"{dest_car_id}.jpg"))
        else:
            # No resizing needed, add original images to attachments
            for image, dest_car_id, _ in images:
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                attachments.append((buffered.getvalue(), f"{dest_car_id}.jpg"))

        return body, attachments

    def send_email(body, logger, attachments=[]):
        from_address = os.environ.get("OUTLOOK_FROM_EMAIL")
        group_mailbox = os.environ.get("OUTLOOK_GROUP_MAILBOX")
        to_addresses = os.environ.get("OUTLOOK_TO_EMAIL").split(",")

        subj_prefix = os.environ.get("EMAIL_SUBJ_PRFX")

        msg = MIMEMultipart("related")
        msg["From"] = group_mailbox
        msg["To"] = ", ".join(to_addresses)
        msg["Subject"] = f"{subj_prefix}: Open Intermodal Container/Trailer \
            Door Detected"

        # Create the HTML part
        html_part = MIMEText(body, "html")
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
            part.add_header(
                "Content-Disposition",
                f'attachment; filename="{filename}"',
            )
            msg.attach(part)

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

    def format_dt_times(utc_datetime, result_timezone="Mountain"):
        # Ensure the input is a timezone-aware datetime object in UTC
        if (
            not isinstance(utc_datetime, datetime)
            or utc_datetime.tzinfo is None
            or utc_datetime.tzinfo.utcoffset(utc_datetime) is None
        ):
            raise ValueError("The input must be a timezone-aware datetime \
                object in UTC.")

        # Define the timezones
        timezones = {
            "Mountain": pytz.timezone("US/Mountain"),
            "Central": pytz.timezone("US/Central"),
            "UTC": pytz.utc,
        }

        # Get the result timezone
        result_tz = timezones.get(result_timezone, pytz.utc)

        # Convert the time to the result timezone
        local_time = utc_datetime.astimezone(result_tz)

        # Return the date and time as a list in 24-hour format
        if result_timezone == "Mountain":
            return local_time.strftime("%Y-%m-%d %H:%M:%S %Z")

        else:
            return local_time.strftime("%Y-%m-%d %H:%M:%S")

    if myTimer.past_due:
        logging.info('The timer is past due!')

    if "TRACKING_DICT" not in os.environ:
        os.environ["TRACKING_DICT"] = json.dumps({})

    procd_train_data = {}

    # Set up logging to Azure or merging Application Insights
    logger = logging.getLogger(__name__)
    logger.addHandler(
        AzureLogHandler(
            connection_string=os.environ.
            get("APPLICATIONINSIGHTS_CONNECTION_STRING")
        )
    )

    blob_service_client = get_blob_service_client()
    container_name = os.environ.get("CONT_NAME")
    container_client = blob_service_client. \
        get_container_client(container_name)

    # Get recent blobs modified within the last 15 minutes and 10 seconds
    recent_passes = get_recent_passes(
        container_client, seconds=int(os.environ.get("SECONDS_TO_LK_BACK"))
    )

    tracking_dict = load_tracking_dict()

    if recent_passes:
        pass_files = download_passes(container_client, recent_passes)

        if pass_files:

            car_list = []

            for image, file_name, detection_upload_dt_time in pass_files:

                processed_symbol_car_keys = set()

                car_detail_list = re.split(r"[-_.]", file_name)
                car_detail_list[1] = car_detail_list[1].lstrip("0")

                # Combine indices 3 (UTC date string) and 4 (UTC time
                # string) to create a timezone-aware datetime object in UTC
                date_str = car_detail_list[3]  # 'YYYYMMDD'
                time_str = car_detail_list[4]  # 'HHMMSS'
                combined_str = f"{date_str} {time_str}"
                naive_datetime = datetime.strptime(combined_str,
                                                    "%Y%m%d %H%M%S")
                arrival_utc_datetime = pytz.utc.localize(naive_datetime)
                car_detail_list[3] = arrival_utc_datetime
                car_detail_list.pop(4)
                car_detail_list.append(detection_upload_dt_time)

                # TODO uncomment this before deployment to Azure or merging
                databricks_instance = os.environ.get("DATABRICKS_INSTANCE")
                databricks_token = os.environ.get("DATABRICKS_TOKEN")
                db_notebook_path = os.environ.get("DATABRICKS_NOTEBOOK_PTH")

                car_inits = car_detail_list[0]
                car_num = car_detail_list[1]
                car_id = f"{car_inits}-{car_num}"

                if car_id in procd_train_data:
                    train_data = procd_train_data[car_id]
                else:
                    train_data = execute_notebook(
                        databricks_instance,
                        databricks_token,
                        db_notebook_path,
                        car_inits,
                        car_num,
                    )
                    procd_train_data[car_id] = train_data

                try:
                    train_data = train_data["notebook_output"]["result"] \
                        .split("'")
                    car_detail_list.append(train_data[1])
                    car_detail_list.append(train_data[3])
                    car_detail_list.append(train_data[5])
                    car_detail_list.append(image)
                    symbol_car_key = f"{train_data[1]}-{car_id}"

                except KeyError:
                    car_detail_list.append(train_data["metadata"]["job_id"])
                    car_detail_list.append("NA")
                    car_detail_list.append("NA")
                    car_detail_list.append(image)
                    symbol_car_key = f"SymbolNotFound-{car_id}"

                if symbol_car_key not in processed_symbol_car_keys:
                    processed_symbol_car_keys.add(symbol_car_key)

                if symbol_car_key not in tracking_dict:
                    car_list.append(car_detail_list)

            corridor_grouped_cars = group_by_corridor(car_list)
            corr_symbol_car_grps = group_by_train_symbol(corridor_grouped_cars)
            car_grouped_symbols = group_by_car_id(corr_symbol_car_grps)

            processed_pass_data = create_train_pass_objects(car_grouped_symbols)

        if processed_pass_data:
            body, attachments = format_email_body(processed_pass_data, logger)
            send_email(body, logger, attachments)
            update_tracking_dict(tracking_dict, processed_symbol_car_keys)
            save_tracking_dict(tracking_dict)
            
        else:
            update_tracking_dict(tracking_dict)
            save_tracking_dict(tracking_dict)

    logging.info('Python timer trigger function executed.')
