from UI.plots import create_plots, update_plots
from Live.session_state import SessionState, SessionPhase
from Live.loaded_model import LoadedModel
from Live.drift import DriftManager
from Live.online_adapter import OnlineAdapter
from Live.session_runner import step
from Live.config import EVALUATION_BATCH_SIZE, LEARNING_BATCH_SIZE
from Training.model_type import Model
from cpp_py_interface import Current_State, reset_mutations
from PySide6.QtCore import Qt, QTimer, QThreadPool
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
    QPlainTextEdit
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
        self.model_selector.setCurrentText("MLP")  
        self.sidebar_layout.addWidget(self.model_selector)
        self.sidebar_layout.addWidget(QLabel("Random Seed"))

        self.seed_input = QSpinBox()

        self.seed_input.setRange(0, 1000000)
        self.seed_input.setValue(10)

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
        self.stage_label = QLabel("Phase: Idle")
        self.stage_label.setWordWrap(True)
        self.speed_label = QLabel("Playback: —")

        self.sidebar_layout.addWidget(self.status_label)
        self.sidebar_layout.addWidget(self.step_label)
        self.sidebar_layout.addWidget(self.time_label)
        self.sidebar_layout.addWidget(self.stage_label)
        self.sidebar_layout.addWidget(self.speed_label)

        self.sidebar_layout.addWidget(QLabel("System Events"))
        self.event_history = QPlainTextEdit()
        self.event_history.setReadOnly(True)
        self.event_history.setPlaceholderText("Motor changes, drift and recovery will appear here.")
        self.event_history.setMaximumBlockCount(300)
        self.event_history.setMinimumHeight(170)
        self.sidebar_layout.addWidget(self.event_history)

        self.sidebar_layout.addStretch()

        self.tabs = QTabWidget()

        self.drift_alarm = QLabel("●  IDLE")
        self.drift_alarm.setAlignment(Qt.AlignCenter)
        self.drift_alarm.setMinimumWidth(190)
        self.drift_alarm.setFixedHeight(30)

        self.tabs.setCornerWidget(self.drift_alarm, Qt.TopRightCorner)

        self.alarm_phase = None
        self.alarm_flash_on = False
        self.awaiting_post_mutation_detection = False

        self.alarm_timer = QTimer(self)
        self.alarm_timer.setSingleShot(True)
        self.alarm_timer.setInterval(450)
        self.alarm_timer.timeout.connect(self.finish_drift_notification)

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
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.advance_simulation)

        self.start_button.clicked.connect(self.start_simulation)
        self.pause_button.clicked.connect(self.pause_simulation)
        self.resume_button.clicked.connect(self.resume_simulation)
        self.end_button.clicked.connect(self.end_simulation)

        self.start_button.setEnabled(True)

        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(1)

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

    def add_event(self, message, step_number=None):
        if step_number is None:
            step_number = self.session_state.step_count if self.session_state is not None else 0
        self.event_history.appendPlainText(f"Step {step_number:,} | {message}")
        scrollbar = self.event_history.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def finish_drift_notification(self):
        if self.session_state is None:
            return
        if self.session_state.drift_status != SessionPhase.ADAPTING:
            return
        self.drift_alarm.setText("● ADAPTING")
        self.set_alarm_style("#78350F", "#FDE68A")

    def update_adaptation_stage(self):
        if self.session_state is None:
            self.stage_label.setText("Phase: Idle")
            return

        phase = self.session_state.drift_status
        if phase == SessionPhase.BASELINE:
            self.stage_label.setText("Phase: Monitoring baseline")
        elif phase == SessionPhase.STABLE:
            if self.awaiting_post_mutation_detection:
                self.stage_label.setText("Motor changed | Monitoring for drift")
            else:
                remaining = max(0, 700 - self.session_state.stable_step_counter)
                self.stage_label.setText(f"Phase: Stable | Next motor change in ~{remaining} steps")
        elif phase == SessionPhase.ADAPTING:
            adapter = self.online_adapter
            if adapter.currently_learning:
                self.stage_label.setText("Phase: Adapting | Training model")
            elif adapter.currently_evaluating:
                self.stage_label.setText(f"Phase: Adapting | Evaluating recovery " f"({adapter.evaluation_batch_count}/{EVALUATION_BATCH_SIZE})")
            else:
                self.stage_label.setText(f"Phase: Adapting | Collecting samples " f"({adapter.learning_batch_count}/{LEARNING_BATCH_SIZE})")

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
            self.stage_label.setText("Phase: Idle")
            self.speed_label.setText("Playback: —")
            return

        phase = self.session_state.drift_status
        if phase == self.alarm_phase:
            return

        self.alarm_phase = phase
        self.alarm_timer.stop()

        if phase == SessionPhase.ADAPTING:
            self.awaiting_post_mutation_detection = False
            self.last_drift_step = self.session_state.step_count
            self.add_event("CUSUM drift detected; online adaptation started")
            self.drift_alarm.setText("⚠ DRIFT DETECTED")
            self.set_alarm_style("#DC2626", "#FFFFFF")
            self.alarm_timer.start()
            interval = 10
        elif phase == SessionPhase.STABLE:
            self.awaiting_post_mutation_detection = False
            if self.last_drift_step is not None:
                elapsed = self.session_state.step_count - self.last_drift_step
                self.add_event(f"Recovery confirmed ({elapsed:,} adaptation steps)")
            else:
                self.add_event("Stable operation confirmed")
            self.drift_alarm.setText("● STABLE")
            self.set_alarm_style("#14532D", "#BBF7D0")
            interval = 40
        else: 
            self.drift_alarm.setText("● MONITORING")
            self.set_alarm_style("#1E3A5F", "#BFDBFE")
            interval = 30

        self.timer.setInterval(interval)
        self.speed_label.setText(f"Playback: {interval} ms/step")
        self.update_adaptation_stage()

    def start_simulation(self):
        current_state = Current_State(0, 0, 21.0)
        model_type = Model[self.model_selector.currentText().upper()]
        seed = self.seed_input.value()
        reset_mutations()

        self.session_state = SessionState(model_type, current_state, seed)
        self.loaded_model = LoadedModel(model_type)
        self.loaded_model.load_model()
        
        self.drift_manager = DriftManager(self.session_state, self.loaded_model)
        self.drift_manager.get_initial_CUSUM_reference_values()

        self.online_adapter = OnlineAdapter(self.drift_manager)

        self.event_history.clear()
        self.last_drift_step = None
        self.awaiting_post_mutation_detection = False
        self.alarm_phase = None
        self.add_event(f"Session started | {model_type.name} | seed {seed}")
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
        self.update_drift_alarm()
        previous_phase = self.session_state.drift_status
        previous_stable_counter = self.session_state.stable_step_counter

        model_input, model_target, prediction, residual = step(self.session_state, self.loaded_model, schedule_mutations = True)

        if self.session_state.step_count == 500:
            self.add_event("Initial scheduled motor-parameter mutation")
            self.awaiting_post_mutation_detection = True
        elif (previous_phase == SessionPhase.STABLE and previous_stable_counter > 0 and self.session_state.stable_step_counter < previous_stable_counter):
            self.add_event("Scheduled motor-parameter mutation")
            self.awaiting_post_mutation_detection = True

        self.drift_manager.sliding_window.append(residual)
        self.drift_manager.maintain_sliding_window()

        self.drift_manager.update_drift_status()
        self.update_drift_alarm()

        if self.drift_manager.session_state.drift_status == SessionPhase.ADAPTING:
            if self.threadpool.activeThreadCount() == 0:
                if self.online_adapter.currently_learning or (self.online_adapter.currently_evaluating and self.online_adapter.evaluation_batch_count == EVALUATION_BATCH_SIZE):
                    adapter = self.online_adapter
                    self.threadpool.start(lambda: adapter.maintain_adaption(None, None))
                else:
                    self.online_adapter.maintain_adaption(model_input, model_target)

        self.update_adaptation_stage()
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
        self.add_event("Simulation paused")

    def resume_simulation(self):
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.resume_button.setEnabled(False)
        self.end_button.setEnabled(True)

        self.timer.start()
        self.status_label.setText("Status: Running")
        self.add_event("Simulation resumed")
        
    def end_simulation(self):
        final_step = self.session_state.step_count if self.session_state is not None else 0

        self.timer.stop()
        self.threadpool.waitForDone()
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
        self.add_event("Session ended", step_number = final_step)
        self.step_label.setText("Step: 0")
        self.time_label.setText("Simulation Time: 0.000 s")

        self.model_selector.setEnabled(True)
        self.seed_input.setEnabled(True)