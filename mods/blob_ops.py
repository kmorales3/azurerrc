import os
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from datetime import datetime, timezone, timedelta


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
    
def get_container_client():
    """Initializes and returns the Azure Blob container client."""
    blob_service_client = BlobServiceClient.from_connection_string(
        os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    )
    container_name = os.getenv("MIDS_IMG_CONT_NAME")
    return blob_service_client.get_container_client(container_name)