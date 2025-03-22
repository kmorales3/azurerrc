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