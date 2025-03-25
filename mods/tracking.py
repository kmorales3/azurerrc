from datetime import datetime
import json
import base64
import os
from azure.storage.blob import BlobServiceClient

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
    - Only keeps one blob per camera (index 7)
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

        existing_cams = {b[7] for b in tracking_dict[key]["blobs"] if len(b) > 7}

        for blob in blob_list:  # 🔥 iterate through the actual blobs
            cam_name = blob[7] if len(blob) > 7 else None

            if cam_name and cam_name not in existing_cams:
                tracking_dict[key]["blobs"].append(blob)
                existing_cams.add(cam_name)  # ✅ update this too so we don’t duplicate

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
