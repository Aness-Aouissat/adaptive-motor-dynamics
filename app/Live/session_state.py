from Training.model_type import Model
from random import Random
from enum import Enum

import cpp_py_interface as cpi

class SessionPhase(Enum):
    BASELINE = 0
    ADAPTING = 1
    STABLE = 2

class SessionState:
    def __init__(
        self, 
        model_type: Model,
        current_state: cpi.Current_State,
        seed: int,     
    ): 
        self.current_state = current_state
        self.current_input = None
        self.model_type = model_type
        self.seed = seed
        self.rng = Random(self.seed)
        self.step_count = 0
        self.drift_status = SessionPhase.BASELINE