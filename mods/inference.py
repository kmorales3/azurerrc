import os
import csv
import logging
import random
from PIL import Image, ImageEnhance, ImageOps
from azure.storage.blob import BlobServiceClient
import tempfile
from ultralytics import YOLO  # Adjust the import based on the actual module name

LOG_FILE = "inference_log.csv"

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
    
    # Ensure log file has headers if it doesn't exist
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "Original Width", "Original Height", "Resized Width", "Resized Height",
                "Flipped", "Augmentation Index", "Detection X", "Detection Width",
                "Detection Confidence", "Accepted"
            ])
            
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

                    # Log detection info
                    with open(LOG_FILE, mode="a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            original_width, original_height, resized_width, resized_height,
                            flipped, x, width, confidence, accepted
                        ])

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