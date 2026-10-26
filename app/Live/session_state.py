from Training.model_type import Model
from random import Random

import cpp_py_interface as cpi

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