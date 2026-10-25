import numpy as np
import pyqtgraph as pg

X_AXIS = {
    "Simulation time": 0,
    "Voltage": 4,
    "Load torque": 5,
    "Motor speed": 2,
    "Temperature": 3,
}

def create_plots(performance_grid, monitoring_grid):
    curves = []

    for i, name in enumerate(("Current", "Speed", "Temperature")):
        comparison = pg.PlotWidget(title=f"{name}: Actual vs Predicted")
        comparison.getPlotItem().addLegend()

        curves.append((comparison, comparison.plot(pen="b", name="Actual"), 6 + i, "b"))
        curves.append((comparison, comparison.plot(pen="y", name="Predicted"), 9 + i, "y"))

        residual = pg.PlotWidget(title=f"{name}: Residual")
        curves.append((residual, residual.plot(pen="r"), 12 + i, "r"))

        performance_grid.addWidget(comparison, i, 0)
        performance_grid.addWidget(residual, i, 1)

    variables = (
        ("Voltage", 4),
        ("Load torque", 5),
        ("Motor current", 1),
        ("Motor speed", 2),
        ("Temperature", 3),
    )

    for i, (name, column) in enumerate(variables):
        plot = pg.PlotWidget(title=name)
        curves.append((plot, plot.plot(pen="g"), column, "g"))
        monitoring_grid.addWidget(plot, i // 2, i % 2)

    return curves


def update_plots(curves, history):
    if not history:
        for plot, curve, column, color in curves:
            curve.setData([], [])
        return

    data = np.asarray(history)

    for plot, curve, column, color in curves:
        curve.setData(
            data[:, 0],
            data[:, column],
            pen=color
        )

        plot.getPlotItem().setLabel(
            "bottom", "Simulation time", units="s"
        )