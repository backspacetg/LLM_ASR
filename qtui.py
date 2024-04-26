import sys
import librosa
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsRectItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QColor

class WaveformWidget(QGraphicsView):
    def __init__(self, parent=None):
        super(WaveformWidget, self).__init__(parent)
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setStyleSheet("background-color: black;")

    def plot_waveform(self, audio_file):
        # Load audio file
        y, sr = librosa.load(audio_file)

        # Calculate energy for each frame (25ms)
        frame_length = int(sr * 0.025)
        energy = np.array([sum(abs(y[i:i+frame_length])**2) for i in range(0, len(y), frame_length)])

        # Normalize energy to fit within view
        energy /= max(energy)

        # Plot waveform
        self.scene.clear()
        width = self.width() / len(energy)
        for i, e in enumerate(energy):
            rect = QGraphicsRectItem(i * width, 0, width, e * self.height())
            rect.setBrush(QColor("white"))
            self.scene.addItem(rect)

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Audio Waveform Visualization")
        self.setGeometry(100, 100, 800, 400)

        self.waveform_widget = WaveformWidget()
        self.setCentralWidget(self.waveform_widget)

        self.waveform_widget.plot_waveform("E:\\wenet_test\\test.wav")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
