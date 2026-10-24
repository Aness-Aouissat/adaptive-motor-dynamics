from PySide6.QtCore import Qt
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

# QMainWindow inherits from QWidget, providing window functionality.
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

        performance_layout = QVBoxLayout()
        performance_page.setLayout(performance_layout)

        performance_layout.addWidget(QLabel("Horizontal Axis"))

        self.x_axis_selector = QComboBox()
        self.x_axis_selector.addItems(["Simulation time", "Voltage", "Load torque", "Motor speed", "Temperature",])
        self.x_axis_selector.setEnabled(False)
        performance_layout.addWidget(self.x_axis_selector)

        performance_grid = QGridLayout()
        performance_layout.addLayout(performance_grid)

        variables = ["Current Δi", "Angular Velocity Δω", "Temperature ΔT",]

        for row, variable in enumerate(variables):

            actual_plot = self.make_placeholder(f"{variable}: Actual vs Predicted")
            residual_plot = self.make_placeholder(f"{variable}: Residual")

            performance_grid.addWidget(actual_plot, row, 0)
            performance_grid.addWidget(residual_plot, row, 1)

        self.tabs.addTab(performance_page, "Model Performance")

        monitoring_page = QWidget()

        monitoring_layout = QGridLayout()
        monitoring_page.setLayout(monitoring_layout)

        monitoring_variables = ["Voltage", "Load Torque", "Motor Current", "Motor Speed", "Temperature",]

        for index, variable in enumerate(monitoring_variables):

            placeholder = self.make_placeholder(variable)
            row = index // 2
            column = index % 2

            monitoring_layout.addWidget(placeholder, row, column)

        self.tabs.addTab(monitoring_page, "Motor Monitoring")

        self.root_layout.addWidget(self.sidebar, 1)
        self.root_layout.addWidget(self.tabs, 4)

        self.setCentralWidget(self.central)

    def make_placeholder(self, text):

        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumHeight(150)
        label.setStyleSheet("border: 1px solid #777;" "background: #283342;" "color: white;")

        return label