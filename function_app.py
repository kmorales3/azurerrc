# function_app.py

import logging
import os
import pytz
import azure.functions as func
from ultralytics import YOLO
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from datetime import datetime, timezone, timedelta
import requests
import time
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
import random
from PIL import Image, ImageEnhance, ImageOps
import re
import itertools
import json

LOCAL_RUN = False
LOCAL_PASS_DIR = r"C:\Users\c883206\Downloads\testing"

app = func.FunctionApp()

@app.timer_trigger(schedule=os.environ.get("CRON_TIMER", "0 */15 * * * *"), arg_name="myTimer",
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

            self.car_image = None
            self.crops = []

        def create_detection(self, creation_time, car_id, camera, image, crops):
            self.detection_data["Created at"] = creation_time
            self.detection_data["Car ID"] = car_id
            self.detection_data["Intermodal Container ID"] = "Not Visible"
            self.detection_data["Camera ID"] = camera

            self.car_image = image
            self.crops = crops

        def get_image(self):
            return self.car_image, self.crops

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

    STORAGE_CONN_STR = os.environ.get("AZR_APP_STRG_CONN_STRNG")
    CONTAINER_NAME = os.environ.get("APP_TRKNG_CONT")
    TRACKING_FILE = "tracking.json"

    def load_tracking_dict():
        """Loads tracking data from Blob Storage and converts types back to their original states."""
        blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONN_STR)
        blob_client = blob_service_client.get_blob_client(CONTAINER_NAME, TRACKING_FILE)

        try:
            tracking_data = json.loads(blob_client.download_blob().readall())
            return convert_from_serialization(tracking_data)  # 🔥 Convert everything back!
        except Exception:
            return {}  # Return empty dict if file doesn't exist yet

    def update_tracking_dict(tracking_dict, new_entries):
        """
        Updates the tracking dictionary:
        - Increments `executions_since_addition` for all existing keys
        - Adds new keys with counter = 1
        - Only keeps one blob per unique camera-axle combination
        - Deletes entries after 4 executions
        """

        # 🔥 Increment all existing entries first
        for key in list(tracking_dict.keys()):
            tracking_dict[key]["executions_since_addition"] += 1

        for key, blob_list in new_entries.items():  # blob_list is a list of blob entries
            if key not in tracking_dict:
                tracking_dict[key] = {
                    "executions_since_addition": 1,
                    "blobs": []
                }

            # Combine camera and axle into a unique key
            existing_cam_axle_keys = {
                f"{b[7]}-{b[2]}" for b in tracking_dict[key]["blobs"] if len(b) > 7 and len(b) > 2
            }

            for blob in blob_list:  # 🔥 iterate through the actual blobs
                cam_name = blob[7] if len(blob) > 7 else None
                axle_value = blob[2] if len(blob) > 2 else None

                if cam_name and axle_value:
                    cam_axle_key = f"{cam_name}-{axle_value}"
                    if cam_axle_key not in existing_cam_axle_keys:
                        tracking_dict[key]["blobs"].append(blob)
                        existing_cam_axle_keys.add(cam_axle_key)  # ✅ update this too so we don’t duplicate

        # 🔥 Cleanup
        tracking_dict = {
            k: v for k, v in tracking_dict.items()
            if v["executions_since_addition"] < 4
        }

        return tracking_dict

    def save_tracking_dict(tracking_dict):
        """Saves the tracking dictionary to Blob Storage."""
        
        blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONN_STR)
        blob_client = blob_service_client.get_blob_client(CONTAINER_NAME, TRACKING_FILE)
        
        # 🔥 Convert datetime & bytes before saving
        sanitized_dict = convert_for_serialization(tracking_dict)

        blob_client.upload_blob(json.dumps(sanitized_dict, indent=2), overwrite=True)
        print("✅ Tracking dictionary updated in Blob Storage.")
        
    def convert_for_serialization(tracking_dict):
        """Converts specific fields in the tracking dictionary for JSON serialization."""
        for key in tracking_dict:
            blob_lists = tracking_dict[key]["blobs"]
            for blob in blob_lists:
                # Convert datetime fields
                for i in [3, 9]:
                    if i < len(blob) and isinstance(blob[i], datetime):
                        blob[i] = blob[i].isoformat()

                # Convert image to Base64 string
                if len(blob) > 13 and isinstance(blob[13], bytes):
                    blob[13] = base64.b64encode(blob[13]).decode("utf-8")

        return tracking_dict

    def convert_from_serialization(tracking_dict):
        """Restores datetime and bytes fields in the tracking dictionary."""
        for key in tracking_dict:
            blob_lists = tracking_dict[key]["blobs"]
            for blob in blob_lists:
                # Convert datetime fields
                for i in [3, 9]:
                    if i < len(blob) and isinstance(blob[i], str):
                        try:
                            blob[i] = datetime.fromisoformat(blob[i])
                        except ValueError:
                            pass  # Leave as string if not a datetime

                # Convert Base64 string back to bytes
                if len(blob) > 13 and isinstance(blob[13], str):
                    try:
                        blob[13] = base64.b64decode(blob[13])
                    except Exception:
                        pass  # Leave as string if it fails

        return tracking_dict


    def format_dt_times(utc_datetime, result_timezone="Mountain"):
        # Ensure the input is a timezone-aware datetime object in UTC
        if (
            not isinstance(utc_datetime, datetime)
            or utc_datetime.tzinfo is None
            or utc_datetime.tzinfo.utcoffset(utc_datetime) is None
        ):
            raise ValueError("The input must be a timezone-aware datetime object in UTC.")

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

    def parse_car_details_from_filename(file_name, detection_upload_dt_time):
        """
        Parses a car image file name into its component details.

        Args:
            file_name (str): The name of the car image file.
            detection_upload_dt_time (datetime): The UTC datetime the file was uploaded (last modified).

        Returns:
            list: Parsed car details with UTC arrival time and upload timestamp appended.
        """
        car_detail_list = re.split(r"[-_.]", file_name)
        car_detail_list[1] = car_detail_list[1].lstrip("0")

        # Parse timestamp from filename
        date_str, time_str = car_detail_list[3], car_detail_list[4]
        combined_str = f"{date_str} {time_str}"
        naive_datetime = datetime.strptime(combined_str, "%Y%m%d %H%M%S")
        arrival_utc_datetime = pytz.utc.localize(naive_datetime)
        car_detail_list[3] = arrival_utc_datetime
        car_detail_list.pop(4)  # Remove time string, now replaced by datetime object

        car_detail_list.append(detection_upload_dt_time)
        return car_detail_list


    def transform_site_string(input_str):
        # Add a space before every capital letter except the first one
        spaced_str = re.sub(r"(?<!^)(?=[A-Z])", " ", input_str)

        # Extract numeric characters at the end of the string
        match = re.search(r"\d+$", spaced_str)
        number = match.group() if match else ""

        # Remove numeric characters at the end of the string
        result_str = re.sub(r"\d+$", "", spaced_str)

        return [result_str, number]

    # Function to create TrainPass objects from car_grouped_symbols
    def create_train_pass_objects(car_grouped_symbols):
        train_pass_objects = []  # ✅ Persist across all cars (no resetting)

        for corridor, corridor_groups in car_grouped_symbols.items():
            for train_symbol, cars in corridor_groups.items():  # ✅ Process per train symbol
                train_detail = list(cars.values())[0][0]  # ✅ Get the first car's details for train metadata
                train_pass = TrainPass()  # ✅ Create one train pass per train symbol
                train_pass.pass_data["Train Symbol"] = train_symbol
                train_pass.pass_data["Train Arrival Date/Time"] = train_detail[3]
                train_pass.pass_data["Train Destination"] = f"{train_detail[11]} - {train_detail[12]}"
                train_pass.pass_data["Destination Corridor"] = train_detail[14]
                train_pass.pass_data["Train Sequence Number"] = train_detail[5]
                
                (
                    train_pass.pass_data["Detector Site"],
                    train_pass.pass_data["Track Number"],
                ) = transform_site_string(train_detail[6])
                train_pass.pass_data["Mile Post"] = "681.1"

                for car, car_detail in cars.items():
                    train_detail = car_detail[0]  # ✅ Now properly assigned inside loop
                    # Process detections
                    car_detection = Detection()
                    car_detection.create_detection(
                        car_detail[0][9],
                        f"{car_detail[0][0]}-{car_detail[0][1]}",
                        car_detail[0][7],
                        car_detail[0][13],
                        car_detail[0][15],
                    )
                    train_pass.add_detection(car_detection)

                train_pass_objects.append(train_pass)  # ✅ Append ONE train pass per train symbol

        # ✅ Group the TrainPass objects by "Destination Corridor"
        grouped_by_corridor = {}
        for train_pass in train_pass_objects:
            destination_corridor = train_pass.pass_data["Destination Corridor"]
            if destination_corridor not in grouped_by_corridor:
                grouped_by_corridor[destination_corridor] = []
            grouped_by_corridor[destination_corridor].append(train_pass)

        return grouped_by_corridor  # ✅ Now train symbols are properly grouped

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
        SE_OON = {"GA", "FL", "NC", "LA", "SC", "VA"}
        NE_OON = {"NY", "OH", "MI", "NH", "NJ", "RI", "CT", "DE", "ME", "MD", "MA", "PA", "VT", "WV"}
        SW_OON = {"NL", "GJ"}

        for car_detection in car_list:
            dest_state = car_detection[12]
            if dest_state in northeast_corridor:
                car_detection.append("the Northeast Corridor")
            elif dest_state in southeast_corridor:
                car_detection.append("the Southeast Corridor")
            elif dest_state in southwest_corridor:
                car_detection.append("the Southwest Corridor")
            elif dest_state in northwest_corridor:
                car_detection.append("the Northwest Corridor")
            elif dest_state in SE_OON:
                car_detection.append("the SE, Out of Network,")
            elif dest_state in NE_OON:
                car_detection.append("the NE, Out of Network,")
            elif dest_state in SW_OON:
                car_detection.append("the SW, Out of Network,")
            else:
                car_detection.append("an Unknown Corridor")

            # Sort the car_list by the last index (corridor)
        sorted_car_list = sorted(car_list, key=lambda x: x[-1])

        # Group the sorted car_list by the last index (corridor)
        grouped_car_list = itertools.groupby(sorted_car_list,
                                                key=lambda x: x[-1])

        # Convert the groupby object to a dictionary
        grouped_corridors = {key: list(group) for key, group in
                                grouped_car_list}

        return grouped_corridors

    def extract_camera_value(entry):
        """
        Extract the camera value from an entry based on the B#C# pattern.

        Parameters:
            entry (list): A car detail entry containing the camera string.

        Returns:
            int: A numerical representation of the camera for comparison.
        """
        try:
            camera_string = entry[7]  # Assuming the camera string is at index 1 (adjust if needed)
            b, c = map(int, camera_string.split('C'))  # Split on 'C' and convert to integers
            return b * 100 + c  # Combine B and C values to create a sortable number (e.g., B4C2 -> 402)
        except (ValueError, IndexError):
            return 0  # Return a default value if extraction fails
        
    def process_train_pass_data(processed_pass_data):
        """Processes train pass details and returns a structured DataFrame."""
        main_body_data = []

        for corridor, train_passes in processed_pass_data.items():
            for train_pass in train_passes:
                train_symbol = train_pass.pass_data["Train Symbol"]
                train_destination = train_pass.pass_data["Train Destination"]

                trn_arrival_dt_mt = format_dt_times(
                    train_pass.pass_data["Train Arrival Date/Time"], "Mountain"
                )

                # Process detections
                for detection in train_pass.pass_detections:
                    detection_created_at_mt = format_dt_times(
                        detection["detection_data"]["Created at"], "Mountain"
                    )
                    detection_car_id = detection["detection_data"]["Car ID"]

                    now_central = datetime.now(pytz.timezone("US/Central"))

                    main_body_data.append({
                        "Email Date/Time (CT)": now_central.strftime("%Y/%m/%d %H:%M:%S"),
                        "Train Symbol": train_symbol,
                        "Arrival Date/Time (CT)": format_dt_times(
                            train_pass.pass_data["Train Arrival Date/Time"], "Central"
                        ),
                        "Destination": train_destination,
                        "Detection Created At (CT)": format_dt_times(
                            detection["detection_data"]["Created at"], "Central"
                        ),
                        "Detection Car ID": detection_car_id,
                        "Open Doors": len(detection['crops']['crops']),
                    })

                    # Update detection timestamps
                    detection["detection_data"]["Created at"] = detection_created_at_mt

                # Update train arrival time
                train_pass.pass_data["Train Arrival Date/Time"] = trn_arrival_dt_mt

        return pd.DataFrame(main_body_data)

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

                car_grouped_data.append((train_symbol, grouped_cars))
            car_grouped_symbols[corridor] = car_grouped_data

        return car_grouped_symbols

    def process_pass_files(pass_files, local_run):
        """
        Processes raw image files from Azure or local and extracts car details from filenames.

        Args:
            pass_files (list): List of tuples containing (image_bytes, filename, timestamp).
            local_run (bool): Whether the function is running in local mode.

        Returns:
            List of tuples: Each tuple contains (car_detail_list, image_bytes, detection_upload_dt_time).
        """
        parsed_files = []

        for image, file_name, detection_upload_dt_time in pass_files:
            if file_name.endswith('.jpg'):
                car_detail_list = parse_car_details_from_filename(file_name, detection_upload_dt_time)
                parsed_files.append((car_detail_list, image, detection_upload_dt_time))

        return parsed_files

        
    def enrich_with_train_data(car_detail_list, procd_train_data):
        """
        Enriches car_detail_list with train information from Databricks or fallback values,
        and generates the symbol_car_key.

        Args:
            car_detail_list (list): Parsed car details from filename.
            procd_train_data (dict): Dictionary to cache train lookups.

        Returns:
            tuple: (enriched_car_detail_list, symbol_car_key)
        """

        car_inits, car_num = car_detail_list[0], car_detail_list[1]
        car_id = f"{car_inits}-{car_num}"

        if car_id in procd_train_data:
            train_data = procd_train_data[car_id]
        else:
            databricks_instance = os.environ.get("DATABRICKS_INSTANCE")
            databricks_token = os.environ.get("DATABRICKS_TOKEN")
            db_notebook_path = os.environ.get("DATABRICKS_NOTEBOOK_PTH")

            train_data = execute_notebook(
                databricks_instance, databricks_token, db_notebook_path, car_inits, car_num
            )
            procd_train_data[car_id] = train_data

        try:
            train_data = json.loads(train_data["notebook_output"]["result"].strip("'"))
            car_detail_list.extend([
                train_data['trn_id'].strip(),
                train_data['dest_city_frefrm'].strip(),
                train_data['dest_st'].strip(),
            ])
            symbol_car_key = f"{train_data['trn_id'].strip()}-{car_id}"
        except KeyError:
            car_detail_list.extend(["SymbolNotFound", "NA", "NA"])
            symbol_car_key = f"SymbolNotFound-{car_id}"

        return car_detail_list, symbol_car_key


    def initialize_logger():
        """Initializes and returns a logger instance."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Avoid duplicate handlers if reinitializing
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            if not os.getenv("LOCAL_RUN") == "True":
                try:
                    from opencensus.ext.azure.log_exporter import AzureLogHandler

                    conn_str = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
                    if conn_str:
                        logger.addHandler(AzureLogHandler(connection_string=conn_str))
                except Exception as e:
                    logger.warning(f"⚠️ Could not attach AzureLogHandler: {e}")

        return logger

    def augment_image(image):
        """
        Apply a series of random augmentations to simulate different image conditions.

        Parameters:
            image (PIL Image): Original image.

        Returns:
            List[PIL Image]: A list of augmented versions of the original image.
        """
        augmented_images = [image]  # Always include the original

        # Contrast & Brightness Variations
        contrast = ImageEnhance.Contrast(image)
        brightness = ImageEnhance.Brightness(image)
        sharpness = ImageEnhance.Sharpness(image)

        augmented_images.append(contrast.enhance(random.uniform(0.8, 1.2)))  # Slight contrast shift
        augmented_images.append(brightness.enhance(random.uniform(0.8, 1.2)))  # Slight brightness shift
        augmented_images.append(sharpness.enhance(random.uniform(0.8, 1.5)))  # Slight sharpening

        # Gamma Correction
        gamma = random.uniform(0.8, 1.2)
        gamma_adjusted = ImageOps.autocontrast(image.point(lambda p: ((p / 255.0) ** gamma) * 255))
        augmented_images.append(gamma_adjusted)

        # Slight Rotation (±5°)
        augmented_images.append(image.rotate(random.uniform(-5, 5), resample=Image.Resampling.BICUBIC))

        # Slight Noise Injection (simulated by adding small random pixel shifts)
        noisy_image = image.convert("L")  # Convert to grayscale for subtle noise
        pixels = noisy_image.load()
        for _ in range(random.randint(500, 1500)):  # Number of pixels to modify
            x, y = random.randint(0, image.width - 1), random.randint(0, image.height - 1)
            pixels[x, y] = min(255, max(0, pixels[x, y] + random.randint(-10, 10)))  # Shift pixel intensity slightly
        augmented_images.append(noisy_image.convert("RGB"))  # Convert back to RGB

        return augmented_images

    def run_inference(image, model, is_left_camera, confidence_threshold,
                        reference_width, target_height=1024):
        """
        Run inference on an in-memory image using the YOLO model after 
        resizing it to a target height while preserving aspect ratio.

        Logs inference details to a CSV file.

        Parameters:
            image (PIL Image): In-memory image loaded from Azure.
            model (YOLO): YOLO model instance.
            confidence_threshold (float): Minimum confidence to consider a detection.
            target_height (int): Target height for resizing the image.
            is_left_camera (bool): Whether the image was taken from an L# camera (requires flipping).

        Returns:
            list: List of detections, each containing x, width, confidence.
        """
                
        original_width, original_height = image.size
        width_scale_factor = reference_width / original_width

        # Flip image horizontally if it's from an L# camera
        flipped = is_left_camera
        if is_left_camera:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # Resize image while maintaining aspect ratio
        scale_factor = target_height / original_height
        target_width = int(original_width * scale_factor)
        resized_width, resized_height = target_width, target_height
        if scale_factor != 1.0:
            image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

        logging.info(f"Preprocessed Image - Original: ({original_width}, {original_height}), "
                    f"Resized: ({resized_width}, {resized_height}), Flipped: {flipped}")

        augmented_images = augment_image(image)  # Augment the image for better detection
        all_detections = []

        for aug_index, aug_image in enumerate(augmented_images):
            results = model(aug_image)
            results = results if isinstance(results, list) else [results]

            for result in results:
                if result.boxes:
                    for det in result.boxes.data.cpu().numpy():
                        x_min, y_min, x_max, y_max, confidence, class_id = det
                        x = int((x_min + x_max) / 2 * width_scale_factor)  # Adjust x to reference width
                        width = int((x_max - x_min) * width_scale_factor)  # Adjust width to reference width
                        confidence = float(confidence)
                        accepted = confidence >= confidence_threshold

                        if accepted:
                            all_detections.append({'x': x, 'width': width, 'confidence': confidence,
                                                    'is_left_camera': is_left_camera, 'reference_width': resized_width})
                else:
                    logging.info(f"No detections in augmentation {aug_index}.")

        logging.info(f"Final Detections: {len(all_detections)} accepted after confidence filtering.")
        
        return all_detections

    def load_model_from_blob():
        """
        Downloads the YOLO model from blob storage and loads it into memory.
        Returns a YOLO model instance.
        """
        storage_conn_str = os.environ.get("AZR_APP_STRG_CONN_STRNG")
        model_blob_path = os.environ.get("MODEL_BLOB_PATH")
        container_name = os.environ.get("APP_MDL_CONT", "models")  # Adjust if needed

        blob_service_client = BlobServiceClient.from_connection_string(storage_conn_str)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=model_blob_path)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_model_file:
            blob_data = blob_client.download_blob().readall()
            temp_model_file.write(blob_data)
            model_path = temp_model_file.name

        return YOLO(model_path)

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

            # 🔥 Prepare `.eml` attachment content (Start with greeting)
            attachment_html = f"<p>{email_body_text}</p>"  # Render greeting properly

            # 🔥 Aggregate all train passes for the corridor
            train_passes_html = ""  # This will hold all train passes and their detections
            for train_pass in train_passes:
                # Remove the "Destination Corridor" key
                train_pass.pass_data.pop("Destination Corridor", None)

                # Process train pass details
                train_details_html = pd.DataFrame([train_pass.pass_data]).T.to_html(index=True, header=False)  # ✅ Transposed

                # Process detections for the train pass
                detection_html_blocks = []
                for detection in train_pass.pass_detections:

                    detection_df = pd.DataFrame([detection["detection_data"]]).T.to_html(index=True, header=False)  # ✅ Transposed

                    # Embed images directly BELOW the detection entry
                    images_html = ""
                    if "crops" in detection and "crops" in detection["crops"]:
                        for crop_image in detection["crops"]["crops"]:
                            crop_buffer = io.BytesIO()
                            crop_image.save(crop_buffer, format="JPEG")
                            crop_bytes = crop_buffer.getvalue()
                            crop_encoded = base64.b64encode(crop_bytes).decode()
                            images_html += f'<img src="data:image/jpeg;base64,{crop_encoded}" width="1280"><br>'

                    detection_html_blocks.append(f"{detection_df}{images_html}")

                # Combine train pass details and its detections
                train_passes_html += (
                    f"<strong>Train Pass Details:</strong><br>{train_details_html}<br>"
                    f"<strong>Detections:</strong><br>{''.join(detection_html_blocks)}<br><hr>"
                )

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
                f"{train_passes_html}"  # Add all train passes and their detections
            )

            # 🔥 Attach a single `.eml` file for the entire corridor
            subject = "RoboRailCop: Open Intermodal Container/Trailer Door Detected"
            attachments.append(create_mail_attachment(full_attachment_html, subject, corridor))

            # 🔥 Add train pass data to the main email body
            train_pass_df = process_train_pass_data({corridor: train_passes})  # ✅ Process the current corridor's data
            main_body_df = pd.concat([main_body_df, train_pass_df], ignore_index=True)

        return main_body_df, attachments  # 🔥 `.eml` files have consolidated data for each corridor!

    def crop_and_normalize(detection, target_width=1280, target_height=768):
        """
        Crop a region around the detection center while ensuring the crop stays within image boundaries.
        Resizes the image **before cropping** to ensure correct detection alignment.
        
        Parameters:
            detection (dict): Detection containing image_blob, x, width, confidence, is_left_camera, reference_width.
            target_width (int): Target width for the crop (default: 2200).
            target_height (int): Target height for the crop (default: 1024).

        Returns:
            PIL Image: Cropped and resized image.
        """

        image_blob = detection['image_blob']
        center_x = detection['x']
        is_left_camera = detection['is_left_camera']
        reference_width = detection['reference_width']

        # Open image
        image = Image.open(io.BytesIO(image_blob))

        # Get original dimensions
        original_width, original_height = image.size

        # 🔥 Resize image **before cropping** to ensure we get the full view
        scale_factor = target_height / original_height  # Scale based on height
        new_width = int(original_width * scale_factor)
        resized_image = image.resize((new_width, target_height), Image.Resampling.LANCZOS)

        # Update dimensions after resizing
        resized_width, resized_height = resized_image.size

        # Reverse scaling for cropping (map x-coordinates to resized image)
        center_x = int(center_x * resized_width / reference_width)

        # 🔥 Fix: Instead of flipping, invert `x` for left cameras
        if is_left_camera:
            center_x = resized_width - center_x  # Mirror X coordinate

        # Calculate cropping box (centered on detection)
        left = max(0, min(center_x - target_width // 2, resized_width - target_width))
        right = left + target_width
        top, bottom = 0, target_height  # Image is already resized to 1024px height

        cropped_image = resized_image.crop((left, top, right, bottom))

        return cropped_image

    def group_detections(detections, proximity_threshold):
        """
        Group detections into regions based on their x-coordinate proximity.
        
        Parameters:
            detections (list): List of detection dictionaries, each containing 'x' (center x-coordinate).
            proximity_threshold (int): Maximum pixel distance to consider detections part of the same region.
        
        Returns:
            list of lists: Each inner list contains detections that are part of the same region.
        """
        grouped_regions = []
        current_region = []

        for i, det in enumerate(detections):
            if not current_region:
                current_region.append(det)
            else:
                last_det = current_region[-1]
                if abs(det['x'] - last_det['x']) <= proximity_threshold:
                    current_region.append(det)
                else:
                    grouped_regions.append(current_region)
                    current_region = [det]

        if current_region:
            grouped_regions.append(current_region)

        return grouped_regions

    def process_car_images(grouped_cars, model, confidence_threshold, proximity_threshold=5000):
        """
        Process car images, run inference on in-memory images (from Azure), group detections into regions,
        pare down car_details to the highest camera entry, and append crops to that entry.
        
        Parameters:
            grouped_cars (dict): Original nested structure of grouped_cars (corridor -> train_symbol -> car_id -> detections).
            model (YOLO model): Pretrained YOLO model for inference.
            confidence_threshold (float): Minimum confidence to consider a detection.
            proximity_threshold (int): Maximum pixel distance between x-coordinates to group detections into regions.
        
        Returns:
            dict: The original grouped_cars structure with only the highest camera entry per car and crops appended.
        """
        # Create copies of structures to store valid entries
        updated_grouped_cars = {}

        for corridor, train_symbols in grouped_cars.items():
            updated_train_symbols = {}

            for train_symbol, cars in train_symbols:
                updated_cars = {}

                for car_id, car_details in cars:
                    print(f"Processing car {car_id}...")  # Debug message
                    all_detections = []
                    max_width = 0
                    
                    # FIRST PASS: Find the max width in one loop
                    for detection in car_details:
                        image = Image.open(io.BytesIO(detection[13]))  # Load image
                        max_width = max(max_width, image.width)
                        
                    # Run inference on each image blob and collect detections
                    for detection in car_details:
                        # Extract camera position from the filename
                        is_left_camera = "L" in detection[2]  # Assuming detection[2] contains B#C# info
                        image = Image.open(io.BytesIO(detection[13]))  # Load the image from the blob
                        detections = run_inference(image, model,
                                                    is_left_camera,
                                                    confidence_threshold,
                                                    max_width)

                        for det in detections:
                            det['image_blob'] = detection[13]  # Add the image blob to each detection for reference
                        all_detections.extend(detections)

                    # Skip cars with no detections
                    if not all_detections:
                        print(f"No detections for car {car_id}. Skipping.")
                        continue

                    # Group detections into regions and select the highest-confidence crops
                    all_detections.sort(key=lambda det: det['x'])
                    regions = group_detections(all_detections, proximity_threshold)
                    crops = []
                    for region in regions:
                        best_detection = max(region, key=lambda det: det['confidence'])
                        crop = crop_and_normalize(best_detection)
                        if crop:
                            crops.append(crop)

                    # Find the highest camera entry and keep only that one
                    highest_camera_entry = max(car_details, key=lambda entry: extract_camera_value(entry))
                    highest_camera_entry.append({'crops': crops})
                    updated_cars[car_id] = [highest_camera_entry]

                # Only add train_symbol if it contains valid cars
                if updated_cars:
                    updated_train_symbols[train_symbol] = updated_cars

            # Only add corridor if it contains valid train symbols
            if updated_train_symbols:
                updated_grouped_cars[corridor] = updated_train_symbols

        return updated_grouped_cars

    def prepare_full_size_attachments(detections, max_total_size=20 * 1024 * 1024):
        """
        Resizes full-size car images as needed and prepares them as attachments.

        Args:
            detections (list): List of detections (each with 'car_image').
            max_total_size (int): Total size allowed for all attachments in bytes.

        Returns:
            List[Tuple[bytes, str]]: List of (image_bytes, filename) tuples for attachment.
        """
        image_entries = []
        total_size = 0

        for corridor, corridor_symbols in detections.items():
            for symbol, symbol_detections in corridor_symbols.items():
                for car, det_list in symbol_detections.items():
                    for det in det_list:
                        if det[13]:
                            image = det[13]  # This is the raw image bytes
                            size = len(image)

                            # Convert bytes to PIL.Image for resizing if needed
                            pil_image = Image.open(io.BytesIO(image))

                            image_entries.append({
                                "image": pil_image,  # Store the PIL.Image object for resizing
                                "bytes": image,  # Store the original bytes
                                "size": size,
                                "filename": f"{det[0]}-{det[1]}.jpg"
                            })
                            total_size += size

        # Resize images if needed
        if total_size > max_total_size:
            avg_size = max_total_size // len(image_entries)
            for entry in image_entries:
                if entry["size"] > avg_size:
                    scale = (avg_size / entry["size"]) ** 0.5
                    new_w = int(entry["image"].width * scale)
                    new_h = int(entry["image"].height * scale)
                    resized = entry["image"].resize((new_w, new_h), Image.Resampling.LANCZOS)

                    # Convert resized image back to bytes
                    buf = io.BytesIO()
                    resized.save(buf, format="JPEG")
                    entry["bytes"] = buf.getvalue()
                    entry["size"] = len(entry["bytes"])

        # Return tuples in the correct order (attachment, filename)
        return [(entry["bytes"], entry["filename"]) for entry in image_entries]

    def execute_notebook(databricks_instance, token, notebook_path, car_id,
                            car_number):
        print(f'Executing notebook: {notebook_path} with car_id: {car_id}-{car_number}')
        
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
        account_url = os.environ.get("MIDS_IMG_STRG_URL")
        if not account_url:
            raise ValueError("MIDS_IMG_STRG_URL environment \
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
        
    processed_symbol_car_keys = {}
    procd_train_data = {}
    car_list = []
    tracking_dict = load_tracking_dict()
    logger = initialize_logger()

    if not LOCAL_RUN:
        blob_service_client = get_blob_service_client()
        container_name = os.environ.get("MIDS_IMG_CONT_NAME")
        container_client = blob_service_client.get_container_client(container_name)

        recent_passes = get_recent_passes(
            container_client, seconds=int(os.environ.get("SECONDS_TO_LK_BACK"))
        )
    else:
        recent_passes = [
            os.path.join(LOCAL_PASS_DIR, file) for file in os.listdir(LOCAL_PASS_DIR)
        ]
        
    # 🔥 Step 1: Handle new passes if present
    if recent_passes:
        if LOCAL_RUN:
            pass_files = [
                (
                    open(file, "rb").read(),
                    os.path.basename(file),
                    datetime.utcfromtimestamp(os.path.getmtime(file)).replace(tzinfo=pytz.UTC),
                )
                for file in recent_passes
            ]
        else:
            pass_files = download_passes(container_client, recent_passes)

        if pass_files:
            for image, file_name, detection_upload_dt_time in pass_files:
                if file_name.endswith(".jpg"):
                    car_detail_list = parse_car_details_from_filename(file_name, detection_upload_dt_time)
                    car_detail_list, symbol_car_key = enrich_with_train_data(car_detail_list, procd_train_data)
                    car_detail_list.append(image)

                    if not LOCAL_RUN:
                        if symbol_car_key not in processed_symbol_car_keys:
                            processed_symbol_car_keys[symbol_car_key] = [car_detail_list]
                        else:
                            processed_symbol_car_keys[symbol_car_key].append(car_detail_list)

                    if LOCAL_RUN:
                        car_list.append(car_detail_list)

    # 🔥 Step 2: Always check for ready-to-process tracking entries (even without new files)
    if not LOCAL_RUN:
        for key, data in tracking_dict.items():
            if data["executions_since_addition"] >= 3:
                if data.get("blobs"):
                    car_list.extend(data["blobs"])
                else:
                    print(f"🚨 Warning: {key} reached 4 executions but has no collected blobs!")

    # 🔥 Step 3: Process if we have anything to work with
    if car_list:
        corridor_grouped = group_by_corridor(car_list)
        symbol_grouped = group_by_train_symbol(corridor_grouped)
        car_grouped = group_by_car_id(symbol_grouped)

        model = load_model_from_blob()
        detection_dict = process_car_images(car_grouped, model, 0.2, os.environ.get("DET_GRP_PRX", 5000))
        processed_data = create_train_pass_objects(detection_dict)

        if processed_data:
            full_sized_img_lst = prepare_full_size_attachments(detection_dict, 20*1024*1024)
            body, attachments = format_email_body(processed_data)
            all_attachments = attachments + full_sized_img_lst
            email_msg = compose_email(body.to_html(index=False), all_attachments, logger)
            send_email(email_msg, logger)

    # 🔥 Step 4: Update tracking dict even if no new passes were found
    if not LOCAL_RUN:
        tracking_dict = update_tracking_dict(tracking_dict, processed_symbol_car_keys)
        save_tracking_dict(tracking_dict)

    logging.info("Python timer trigger function executed")