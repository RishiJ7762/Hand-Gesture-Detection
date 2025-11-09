import sys
import os
import uuid
import time
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QMessageBox, QLabel, QLineEdit
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from ultralytics import YOLO

# Dataset Path
IMAGES_PATH = os.path.join('images', 'collectedimages')
os.makedirs(IMAGES_PATH, exist_ok=True)
labels = ['hello', 'iloveyou', 'no', 'thanks', 'yes']
num_images = 20

class CollectGestureThread(QThread):
    finished = pyqtSignal(str)

    def __init__(self, label=None):
        super().__init__()
        self.label = label

    def run(self):
        cap = cv2.VideoCapture(0)
        if self.label:
            self.collect_images(cap, self.label)
        else:
            for label in labels:
                self.collect_images(cap, label)
        cap.release()
        cv2.destroyAllWindows()
        self.finished.emit("Gesture images collected successfully!")

    def collect_images(self, cap, label):
        time.sleep(1)
        for img_num in range(num_images):
            ret, frame = cap.read()
            if not ret:
                break
            img_name = os.path.join(IMAGES_PATH, f'{label}_{uuid.uuid1()}.jpg')
            cv2.imwrite(img_name, frame)
            cv2.imshow('Image Collection', frame)
            time.sleep(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

class TrainModelThread(QThread):
    finished = pyqtSignal(str)

    def run(self):
        dataset_yaml = """
        train: images/train
        val: images/val
        nc: 5
        names: ['hello', 'iloveyou', 'no', 'thanks', 'yes']
        """
        with open("dataset.yaml", "w") as f:
            f.write(dataset_yaml)
        model = YOLO("yolov8m.pt")
        model.train(data="dataset.yaml", epochs=50, imgsz=640, device="cuda")
        self.finished.emit("Model training complete!")

class GestureDetectionThread(QThread):
    finished = pyqtSignal(str)

    def run(self):
        cap = cv2.VideoCapture(0)
        model = YOLO("C:/Users/Acer/runs/detect/train/weights/best.pt").to("cuda")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame)
            annotated_frame = results[0].plot()
            cv2.imshow("Hand Gesture Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        self.finished.emit("Gesture detection stopped!")

class HandGestureApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Hand Gesture Detection System")
        self.setGeometry(100, 100, 500, 500)
        layout = QVBoxLayout()

        self.label = QLabel("Hand Gesture Detection System", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(self.label)

        self.label_input = QLineEdit(self)
        self.label_input.setPlaceholderText("Enter label name")
        layout.addWidget(self.label_input)

        self.btn_collect = QPushButton("Add New Gesture", self)
        self.btn_collect.clicked.connect(self.collect_gesture)
        layout.addWidget(self.btn_collect)

        self.btn_collect_all = QPushButton("Collect All Predefined Gestures", self)
        self.btn_collect_all.clicked.connect(self.collect_all_gestures)
        layout.addWidget(self.btn_collect_all)

        self.btn_import = QPushButton("Import Gesture Database", self)
        self.btn_import.clicked.connect(self.import_dataset)
        layout.addWidget(self.btn_import)

        self.btn_train = QPushButton("Train YOLO Model", self)
        self.btn_train.clicked.connect(self.train_model)
        layout.addWidget(self.btn_train)

        self.btn_detect = QPushButton("Start Gesture Detection", self)
        self.btn_detect.clicked.connect(self.start_detection)
        layout.addWidget(self.btn_detect)

        self.setLayout(layout)

    def collect_gesture(self):
        label_name = self.label_input.text().strip()
        if not label_name:
            QMessageBox.warning(self, "Error", "Please enter a label name.")
            return
        self.collect_thread = CollectGestureThread(label_name)
        self.collect_thread.finished.connect(lambda msg: QMessageBox.information(self, "Success", msg))
        self.collect_thread.start()

    def collect_all_gestures(self):
        self.collect_thread = CollectGestureThread()
        self.collect_thread.finished.connect(lambda msg: QMessageBox.information(self, "Success", msg))
        self.collect_thread.start()

    def import_dataset(self):
        dataset_path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if dataset_path:
            QMessageBox.information(self, "Success", f"Dataset imported from {dataset_path}")

    def train_model(self):
        self.train_thread = TrainModelThread()
        self.train_thread.finished.connect(lambda msg: QMessageBox.information(self, "Success", msg))
        self.train_thread.start()

    def start_detection(self):
        self.detect_thread = GestureDetectionThread()
        self.detect_thread.finished.connect(lambda msg: QMessageBox.information(self, "Success", msg))
        self.detect_thread.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandGestureApp()
    window.show()
    sys.exit(app.exec_())
