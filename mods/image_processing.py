import io
import base64
from PIL import Image
from mods.pass_processing import extract_camera_value
from mods.inference import run_inference

def process_images_for_email(detections, max_email_size=35 * 1024 * 1024):
    """Processes images and returns a list of Base64-encoded images and attachments."""
    images = []
    total_image_size = 0
    
    for detection in detections:
        if "crops" in detection and "crops" in detection["crops"] and detection["crops"]["crops"]:
            for crop_image in detection["crops"]["crops"]:
                try:
                    crop_buffer = io.BytesIO()
                    crop_image.save(crop_buffer, format="JPEG")
                    crop_bytes = crop_buffer.getvalue()
                    crop_encoded = base64.b64encode(crop_bytes).decode()

                    images.append({
                        "Encoded Image": f'<img src="data:image/jpeg;base64,{crop_encoded}" width="2200">',
                        "Size": len(crop_bytes)
                    })
                    total_image_size += len(crop_bytes)
                except IOError:
                    raise ValueError("Invalid crop image data")

    # Resize images if needed
    if total_image_size > max_email_size:
        remaining_space = max_email_size - total_image_size
        avg_space_per_image = remaining_space // len(images) if images else 0

        resized_images = []
        for img in images:
            if img["Size"] > avg_space_per_image:
                img_obj = Image.open(io.BytesIO(base64.b64decode(img["Encoded Image"].split(",")[1])))
                scaling_factor = (avg_space_per_image / img["Size"]) ** 0.5
                new_width, new_height = int(img_obj.width * scaling_factor), int(img_obj.height * scaling_factor)
                img_obj = img_obj.resize((new_width, new_height), Image.Resampling.LANCZOS)

                buffered = io.BytesIO()
                img_obj.save(buffered, format="JPEG")
                img_bytes = buffered.getvalue()

                resized_images.append({
                    "Encoded Image": f'<img src="data:image/jpeg;base64,{base64.b64encode(img_bytes).decode()}" width="2200">',
                    "Size": len(img_bytes)
                })
            else:
                resized_images.append(img)

        return resized_images
    else:
        return images

def crop_and_normalize(detection, target_width=1280, target_height=768):
    """
    Crop a 2200x1024 region around the detection center while ensuring the crop stays within image boundaries.
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

def process_car_images(grouped_cars, model, confidence_threshold, crop_size=(1280, 768), proximity_threshold=5000):
    """
    Process car images, run inference on in-memory images (from Azure), group detections into regions,
    pare down car_details to the highest camera entry, and append crops to that entry.
    
    Parameters:
        grouped_cars (dict): Original nested structure of grouped_cars (corridor -> train_symbol -> car_id -> detections).
        model (YOLO model): Pretrained YOLO model for inference.
        crop_size (tuple): Size of the crop (width, height).
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
