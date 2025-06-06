import typing
import cv2
import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QCloseEvent
from PIL import Image
import numpy as np

import torch
from torchvision.transforms import ToTensor, Resize

from ui import Ui_Dialog

__model = None
__source = None

def set_model(model):
    global __model
    __model = model

def set_source(source):
    global __source
    __source = source

@torch.no_grad()
def inference(frame: Image.Image) -> torch.Tensor:

    frame = ToTensor()(frame).unsqueeze(0)
    output = __model(frame, *__source).clamp(min=0, max=1)
    return output

def start_track(cap, controller):

    _, frame = cap.read()
    if frame is None:
        return
    
    # Showing 512x512
    frame_512 = Image.fromarray(frame)
    frame_512 = frame_512.resize((512, 512))
    frame_512 = np.array(frame_512)
    height, width, channel = frame_512.shape
    bytesPerline = 3 * width
    qimg = QImage(frame_512, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
    controller.ui.label.setPixmap(QPixmap.fromImage(qimg))

    # Inference with 256x256
    frame_256 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_256 = Image.fromarray(frame_256)
    frame_256 = frame_256.resize((256, 256))
    frame_256 = inference(frame_256)
    frame_256: np.ndarray = Resize((512, 512))(frame_256).detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    frame_256 = (frame_256 * 256).astype(np.uint8)
    frame_256 = cv2.cvtColor(frame_256, cv2.COLOR_RGB2BGR)
    height, width, channel = frame_256.shape
    bytesPerline = 3 * width
    qimg = QImage(frame_256, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
    controller.ui.label_2.setPixmap(QPixmap.fromImage(qimg))

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self, video_path: str, generation_path: str):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.qthread = None
        self.video_path = video_path
        self.generation_path = generation_path
        self.stop_video = True

        if self.video_path is None:
            self.ui.pushButton.clicked.connect(self.async_track)
        else:
            self.ui.pushButton.clicked.connect(self.show_video)
    
    def show_video(self):

        if self.stop_video:
            self.stop_video = False
        else:
            self.stop_video = True
            return

        # Change button
        self.ui.pushButton.setText('Stop')

        # Read video
        vidcap = cv2.VideoCapture(self.video_path)
        success, frame = vidcap.read()

        # Read generated
        if self.generation_path is not None:
            gencap = cv2.VideoCapture(self.generation_path)
            success_gen, frame_gen = gencap.read()

        while not self.stop_video and success and (self.generation_path is None or success_gen):

            # Showing original
            frame_org = Image.fromarray(frame)
            frame_org = frame_org.resize((512, 512))
            frame_org = np.array(frame_org)
            height, width, channel = frame_org.shape
            bytesPerline = 3 * width
            qimg = QImage(frame_org, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
            self.ui.label.setPixmap(QPixmap.fromImage(qimg))

            # Showing generated
            if self.generation_path is not None:
                frame_gen = Image.fromarray(frame_gen)
                frame_gen = frame_gen.resize((512, 512))
                frame_gen = np.array(frame_gen)
            else:
                frame_gen = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_gen = Image.fromarray(frame_gen)
                frame_gen = frame_gen.resize((256, 256))
                frame_gen = inference(frame_gen)
                frame_gen: np.ndarray = Resize((512, 512))(frame_gen).detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
                frame_gen = (frame_gen * 256).astype(np.uint8)
                frame_gen = cv2.cvtColor(frame_gen, cv2.COLOR_RGB2BGR)
            height, width, channel = frame_gen.shape
            bytesPerline = 3 * width
            qimg = QImage(frame_gen, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
            self.ui.label_2.setPixmap(QPixmap.fromImage(qimg))

            success, frame = vidcap.read()
            if self.generation_path is not None:
                success_gen, frame_gen = gencap.read()

            cv2.waitKey(10)
        # Change button
        self.ui.pushButton.setText('Start')
    
    def async_track(self):

        if self.qthread:
            self.qthread.is_stop = True
            self.qthread.wait()
            self.qthread = None
            print('Stoping')
            self.ui.pushButton.setText('Start')
        else:
            self.qthread = ThreadTask(self)
            self.qthread.is_stop = False
            self.qthread.start()
            print('Starting')
            self.ui.pushButton.setText('Stop')
    
    def closeEvent(self, a0: QCloseEvent) -> None:
        super().closeEvent(a0)
        if self.qthread:
            self.qthread.is_stop = True
            self.qthread.wait()
            print('Closing')

class ThreadTask(QThread):

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.is_stop = True
        self.cap = None

    def run(self):

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.cap = cv2.VideoCapture(0)

        while True and not self.is_stop:
            start_track(self.cap, self.controller)
            cv2.waitKey(20)

        if self.cap is not None:
            self.cap.release()
            self.cap = None