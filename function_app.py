# function_app.py

import logging
import os
import pytz
from datetime import datetime
import azure.functions as func
from ultralytics import YOLO

from mods.blob_ops import get_blob_service_client, download_passes, get_recent_passes
from mods.tracking import load_tracking_dict, update_tracking_dict, save_tracking_dict
from mods.pass_processing import process_pass_files, parse_car_details_from_filename, enrich_with_train_data
from mods.email_composition import format_email_body, compose_email, send_email
from mods.pass_processing import (
    group_by_corridor,
    group_by_train_symbol,
    group_by_car_id,
    create_train_pass_objects
)
from mods.image_processing import process_car_images

MODEL_PATH = r"/Users/kevinmorales/Documents/Work Stuff/azurerrc/azurerrc/azurerrc/last.pt"
LOCAL_RUN = False
LOCAL_PASS_DIR = r'/Users/kevinmorales/Downloads/2025-03-07 1514_1517'

app = func.FunctionApp()


@app.timer_trigger(schedule="0 */1 * * * *", arg_name="myTimer",
                   run_on_startup=False, use_monitor=False)
def rrc_trigger(myTimer: func.TimerRequest) -> None:

    if myTimer.past_due:
        logging.info("The timer is past due!")

    logger = logging.getLogger(__name__)
    if not LOCAL_RUN:
        from opencensus.ext.azure.log_exporter import AzureLogHandler
        logger.addHandler(
            AzureLogHandler(
                connection_string=os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
            )
        )

    procd_train_data = {}

    if not LOCAL_RUN:
        blob_service_client = get_blob_service_client()
        container_name = os.environ.get("CONT_NAME")
        container_client = blob_service_client.get_container_client(container_name)

        recent_passes = get_recent_passes(
            container_client, seconds=int(os.environ.get("SECONDS_TO_LK_BACK"))
        )
        tracking_dict = load_tracking_dict()
    else:
        recent_passes = [
            os.path.join(LOCAL_PASS_DIR, file) for file in os.listdir(LOCAL_PASS_DIR)
        ]

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
            car_list = []
            processed_symbol_car_keys = {}

            for image, file_name, detection_upload_dt_time in pass_files:
                if file_name.endswith(".jpg"):
                    car_detail_list = parse_car_details_from_filename(file_name, detection_upload_dt_time)
                    car_detail_list, symbol_car_key = enrich_with_train_data(car_detail_list, procd_train_data)
                    car_detail_list.append(image)

                    if not LOCAL_RUN:
                        if symbol_car_key not in processed_symbol_car_keys:
                            processed_symbol_car_keys[symbol_car_key] = car_detail_list
                        else:
                            processed_symbol_car_keys[symbol_car_key].append(car_detail_list)

                    if LOCAL_RUN:
                        car_list.append(car_detail_list)
                        
            if not LOCAL_RUN:
                for key, data in tracking_dict.items():
                    if data["executions_since_addition"] >= 3:
                        if data.get("blobs"):
                            car_list.extend(data["blobs"])
                        else:
                            print(f"🚨 Warning: {key} reached 4 executions but has no collected blobs!")

            if car_list:
                corridor_grouped = group_by_corridor(car_list)
                symbol_grouped = group_by_train_symbol(corridor_grouped)
                car_grouped = group_by_car_id(symbol_grouped)

                model = YOLO(MODEL_PATH)
                detection_dict = process_car_images(car_grouped, model, 0.2)
                processed_data = create_train_pass_objects(detection_dict)

                if processed_data:
                    body, attachments = format_email_body(processed_data)
                    email_msg = compose_email(body.to_html(index=False), attachments, logger)
                    send_email(email_msg, logger)

            if not LOCAL_RUN:
                update_tracking_dict(tracking_dict, processed_symbol_car_keys)
                save_tracking_dict(tracking_dict)

    logging.info("Python timer trigger function execution")
