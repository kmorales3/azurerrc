import re
import itertools
from datetime import datetime
import pytz
import os
import json
from mods.databricks import execute_notebook
from mods.data_classes import TrainPass, Detection
from mods.time_utils import format_dt_times
from mods.blob_ops import download_passes
import pandas as pd

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


def prepare_pass_files(recent_passes, local_mode, container_client=None, local_dir=None):
    
    """
    Prepares a list of pass files with image bytes, filenames, and timestamps.

    Args:
        recent_passes (list): List of recent pass blobs or local file paths.
        local_mode (bool): Whether the function is running locally.
        container_client (BlobContainerClient, optional): Required for non-local runs.
        local_dir (str, optional): Directory for local files.

    Returns:
        list: A list of tuples (image_bytes, filename, creation_time).
    """
    if local_mode:
        return [
            (
                open(file, "rb").read(),
                os.path.basename(file),
                datetime.utcfromtimestamp(os.path.getmtime(file)).replace(tzinfo=pytz.UTC),
            )
            for file in recent_passes
        ]
    else:
        return download_passes(container_client, recent_passes)
    
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
            train_data['data']['trn_id'].strip(),
            train_data['data']['dest_city_frefrm'].strip(),
            train_data['data']['dest_st'].strip(),
        ])
        symbol_car_key = f"{train_data['data']['trn_id'].strip()}-{car_id}"
    except KeyError:
        car_detail_list.extend(["SymbolNotFound", "NA", "NA"])
        symbol_car_key = f"SymbolNotFound-{car_id}"

    return car_detail_list, symbol_car_key
