from UI.plots import create_plots, update_plots
from Live.session_state import SessionState
from Live.loaded_model import LoadedModel
from Live.session_runner import step
from Training.model_type import Model
from cpp_py_interface import Current_State
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QTabWidget,
    QLabel,
    QPushButton,
    QComboBox,
    QSpinBox,
)
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Adaptive Motor Dynamics")
        self.resize(1400, 900)

        self.central = QWidget()

        self.root_layout = QHBoxLayout()
        self.central.setLayout(self.root_layout)

        self.sidebar = QWidget()
        self.sidebar_layout = QVBoxLayout()

        self.sidebar.setLayout(self.sidebar_layout)
        self.sidebar_layout.addWidget(QLabel("Simulation Controls"))
        self.sidebar_layout.addWidget(QLabel("Model"))

        self.model_selector = QComboBox()

        self.model_selector.addItems(["Linear", "Polynomial", "MLP",])
        self.sidebar_layout.addWidget(self.model_selector)
        self.sidebar_layout.addWidget(QLabel("Random Seed"))

        self.seed_input = QSpinBox()

        self.seed_input.setRange(0, 1000000)
        self.seed_input.setValue(42)

        self.sidebar_layout.addWidget(self.seed_input)

        self.start_button = QPushButton("Start")
        self.pause_button = QPushButton("Pause")
        self.resume_button = QPushButton("Resume")
        self.end_button = QPushButton("End Session")

        self.sidebar_layout.addWidget(self.start_button)
        self.sidebar_layout.addWidget(self.pause_button)
        self.sidebar_layout.addWidget(self.resume_button)
        self.sidebar_layout.addWidget(self.end_button)

        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.end_button.setEnabled(False)

        self.status_label = QLabel("Status: Not Started")
        self.step_label = QLabel("Step: 0")
        self.time_label = QLabel("Simulation Time: 0.000 s")

        self.sidebar_layout.addWidget(self.status_label)
        self.sidebar_layout.addWidget(self.step_label)
        self.sidebar_layout.addWidget(self.time_label)

        self.sidebar_layout.addStretch()

        self.tabs = QTabWidget()

        performance_page = QWidget()

        performance_grid = QGridLayout()
        performance_page.setLayout(performance_grid)

        self.tabs.addTab(performance_page, "Model Performance")

        monitoring_page = QWidget()

        monitoring_layout = QGridLayout()
        monitoring_page.setLayout(monitoring_layout)

        self.plot_curves = create_plots(performance_grid, monitoring_layout)

        self.tabs.addTab(monitoring_page, "Motor Monitoring")

        self.root_layout.addWidget(self.sidebar, 1)
        self.root_layout.addWidget(self.tabs, 4)

        self.setCentralWidget(self.central)

        self.session_state = None
        self.loaded_model = None
        self.plot_history = []

        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.advance_simulation)

        self.start_button.clicked.connect(self.start_simulation)
        self.pause_button.clicked.connect(self.pause_simulation)
        self.resume_button.clicked.connect(self.resume_simulation)
        self.end_button.clicked.connect(self.end_simulation)

        self.start_button.setEnabled(True)

    def make_placeholder(self, text):

        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumHeight(150)
        label.setStyleSheet("border: 1px solid #777;" "background: #283342;" "color: white;")

        return label

    def redraw_plots(self):
        update_plots(self.plot_curves, self.plot_history)

    def start_simulation(self):
        current_state = Current_State(0, 0, 21.0)
        model_type = Model[self.model_selector.currentText().upper()]
        seed = self.seed_input.value()

        self.session_state = SessionState(model_type, current_state, seed)
        self.loaded_model = LoadedModel(model_type)

        self.loaded_model.load_model()
        self.plot_history.clear()
        self.redraw_plots()

        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.resume_button.setEnabled(False)
        self.end_button.setEnabled(True)

        self.timer.start()
        self.status_label.setText("Status: Running")

    def advance_simulation(self):
        model_input, model_target, prediction, residual = step(self.session_state, self.loaded_model)

        count = self.session_state.step_count

        self.step_label.setText(f"Step: {count}")
        self.time_label.setText(f"Simulation Time: {count * 0.001:.3f} s")

        self.plot_history.append(np.concatenate((
            [(count - 1) * 0.001],
            model_input.ravel(),
            model_target.ravel(),
            prediction.ravel(),
            residual.ravel()
        )))

        if len(self.plot_history) > 2000:
            self.plot_history.pop(0)

        if count % 10 == 0:
            self.redraw_plots()

    def pause_simulation(self):

        self.timer.stop()

        self.redraw_plots()

        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(True)
        self.end_button.setEnabled(True)

        self.status_label.setText("Status: Paused")

    def resume_simulation(self):

        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.resume_button.setEnabled(False)
        self.end_button.setEnabled(True)

        self.timer.start()
        self.status_label.setText("Status: Running")
        
    def end_simulation(self):

        self.timer.stop()
        self.plot_history.clear()
        self.redraw_plots()

        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.end_button.setEnabled(False)

        self.session_state = None
        self.loaded_model = None

        self.status_label.setText("Status: Not Started")
        self.step_label.setText("Step: 0")
        self.time_label.setText("Simulation Time: 0.000 s")

        self.model_selector.setEnabled(True)
        self.seed_input.setEnabled(True)