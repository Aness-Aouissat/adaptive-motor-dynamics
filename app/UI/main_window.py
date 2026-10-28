from UI.plots import create_plots, update_plots
from Live.session_state import SessionState, SessionPhase
from Live.loaded_model import LoadedModel
from Live.drift import DriftManager
from Live.session_runner import step
from Training.model_type import Model
from cpp_py_interface import Current_State, reset_mutations
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

        self.drift_alarm = QLabel("●  IDLE")
        self.drift_alarm.setAlignment(Qt.AlignCenter)
        self.drift_alarm.setMinimumWidth(190)
        self.drift_alarm.setFixedHeight(30)

        self.tabs.setCornerWidget(self.drift_alarm, Qt.TopRightCorner)

        self.alarm_phase = None
        self.alarm_flash_on = False

        self.alarm_timer = QTimer(self)
        self.alarm_timer.setInterval(450)
        self.alarm_timer.timeout.connect(self.flash_drift_alarm)

        self.set_alarm_style("#263238", "#B0BEC5")

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

    def set_alarm_style(self, background, foreground):
        self.drift_alarm.setStyleSheet(f"""
        QLabel {{
            background-color: {background};
            color: {foreground};
            border-radius: 7px;
            padding: 5px 12px;
            font-size: 12px;
            font-weight: bold;
            }}
        """)

    def flash_drift_alarm(self):
        self.alarm_flash_on = not self.alarm_flash_on

        if self.alarm_flash_on:
            self.set_alarm_style("#DC2626", "#FFFFFF")
        else:
            self.set_alarm_style("#450A0A", "#FCA5A5")

    def update_drift_alarm(self):
        if self.session_state is None:
            self.alarm_phase = None
            self.alarm_timer.stop()
            self.drift_alarm.setText("● IDLE")
            self.set_alarm_style("#263238", "#B0BEC5")
            return

        if self.session_state.drift_status == self.alarm_phase:
            return

        self.alarm_phase = self.session_state.drift_status
        self.alarm_timer.stop()

        if self.alarm_phase == SessionPhase.ADAPTING:
            self.drift_alarm.setText("⚠ DRIFT DETECTED")
            self.alarm_flash_on = False
            self.flash_drift_alarm()
            self.alarm_timer.start()
        elif self.alarm_phase == SessionPhase.STABLE:
            self.drift_alarm.setText("● STABLE")
            self.set_alarm_style("#14532D", "#BBF7D0")
        elif self.alarm_phase == SessionPhase.BASELINE:
            self.drift_alarm.setText("● MONITORING")
            self.set_alarm_style("#1E3A5F", "#BFDBFE")

    def start_simulation(self):
        current_state = Current_State(0, 0, 21.0)
        model_type = Model[self.model_selector.currentText().upper()]
        seed = self.seed_input.value()
        reset_mutations()

        self.session_state = SessionState(model_type, current_state, seed)
        self.loaded_model = LoadedModel(model_type)
        self.loaded_model.load_model()
        
        self.drift_detector = DriftManager(self.session_state, self.loaded_model)
        self.drift_detector.get_initial_CUSUM_reference_values()

        self.update_drift_alarm()

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

        self.drift_detector.sliding_window.append(residual)
        self.drift_detector.maintain_sliding_window()

        self.drift_detector.update_drift_status()
        self.update_drift_alarm()

        #if statement checking if adapting and if adaptation worker is working (in adaptation class)

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
        reset_mutations()

        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.end_button.setEnabled(False)

        self.session_state = None
        self.update_drift_alarm()
        self.loaded_model = None

        self.status_label.setText("Status: Not Started")
        self.step_label.setText("Step: 0")
        self.time_label.setText("Simulation Time: 0.000 s")

        self.model_selector.setEnabled(True)
        self.seed_input.setEnabled(True)