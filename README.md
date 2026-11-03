# Adaptive Motor Dynamics

I built Adaptive Motor Dynamics to investigate how machine learning models behave when the physical system they're predicting changes over time. The project uses a simulated DC motor and focuses on three questions. Can a change in the motor's behaviour be detected from prediction errors? Can the model adapt using new observations? And can it recover reliably when this happens more than once?

The system combines a C++ simulation engine with a Python training and live inference pipeline. Drift detection, online adaptation, and recovery evaluation are the main focus of the experiment.

## Motor simulation

The C++20 engine models the motor's electrical, mechanical, and thermal dynamics. It tracks armature current, angular velocity, and temperature, with voltage and load torque as inputs. Parameters such as resistance, inductance, inertia, damping, and thermal properties determine how the motor responds. The equations are integrated numerically, and pybind11 exposes the simulator to Python.

The models predict the change in each state variable over the next simulation step. I trained Linear Regression, Polynomial Regression, and a PyTorch multilayer perceptron (MLP) using data generated with the motor's nominal parameters. All three use the same data preparation and offline evaluation pipeline.

## Experimental setup

A live session starts with the trained model predicting motor transitions. Voltage and load torque vary during normal operation. At simulation step 500, the motor's physical parameters are changed while the model continues predicting with what it learned previously.

The system monitors the difference between the predicted and observed state changes. When drift is detected, it updates the model using recent transitions and evaluates whether prediction accuracy has recovered. After 700 stable steps, the motor parameters return to their nominal values before another mutation is introduced.

This allows me to study repeated detection and recovery, not just whether a model can fit a new batch of data. I track prediction errors, detection delays, false alarms, and recovery outcomes throughout each session.

## Drift detection

I implemented two methods for monitoring residuals across current, speed, and temperature.

**Rolling CUSUM** accumulates excess absolute prediction error relative to reference statistics over the most recent 100 observations. Linear Regression and the MLP use it throughout their sessions. Polynomial Regression uses it for the first mutation.

**Page-Hinkley** monitors logarithmically normalized residuals using a running mean, cumulative deviations, and the lowest cumulative sum reached. It keeps these statistics between observations and resets them after recovery. Polynomial Regression switches to Page-Hinkley after its first successful recovery.

The switch came from the experiments. Fixed CUSUM thresholds for Polynomial Regression behaved inconsistently across seeds. A threshold that avoided false alarms in one run could miss real changes in another. Page-Hinkley gave better results under the tested conditions, although it wasn't perfect either.

## Online adaptation

The system keeps a rolling buffer of up to 1,000 recent transitions. Linear and Polynomial Regression are updated by refitting with least squares, with polynomial feature expansion for the latter. The MLP uses Adam and trains on batches of 200 samples drawn from the buffer.

Predictions are made before the corresponding observations are used for learning. An updated model is evaluated on a separate batch of 150 transitions. Recovery requires an R² score of at least 0.90 and acceptable mean absolute error for each output. The regression models also have a small tolerance based on target variability, since their original errors can be close to numerical precision.

After a successful recovery, the adaptation buffer and relevant detector statistics are reset before monitoring resumes. The dashboard runs the simulation and displays its predictions, errors, and detection events, while model training and evaluation work is dispatched outside the main UI thread.

## Results and discussion

I tested the Polynomial CUSUM and Page-Hinkley combination across 20 simulation seeds. Ten were used during detector configuration and ten were held out for validation.

A run only counted as successful if it completed three recoveries, detected four scheduled mutations within 20 simulation steps each, and had no false alarms between mutations.

**The final configuration passed 10 out of 10 calibration seeds and 9 out of 10 held-out seeds, giving 19 out of 20 successful trials overall (95%).** The remaining held-out run failed because of a false alarm.

These numbers describe complete trial success, not model prediction accuracy. They come from the standalone validation experiment. I also checked repeated mutation and recovery cycles in the integrated application, but didn't repeat the full 20-seed test through the GUI.

The main finding was that adaptation and drift detection are separate problems. Polynomial Regression could recover after a parameter change, but the detector still had to distinguish that change from normal variation in prediction errors. Much of the investigation involved testing that distinction across different operating conditions.

## Code and tools

The repository separates preprocessing in `Data/`, model implementations in `Models/`, offline training in `Training/`, live simulation and adaptation in `Live/`, and monitoring in `UI/`. The dashboard is built with PySide6, and the simulator is connected to Python through `cpp_py_interface`.

The main technologies are C++20, CMake, pybind11, Python, NumPy, scikit-learn, PyTorch, and PySide6.

## Limitations

The motor and its parameter changes are simulated rather than measured from real hardware. Results depend on the chosen operating conditions, mutations, and empirically selected detector settings. The held-out false alarm also shows that the final detector isn't completely reliable. The project evaluates this approach under a defined simulation protocol rather than claiming that it will work for every possible motor change.