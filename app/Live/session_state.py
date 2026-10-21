from Training.model_type import Model
from enum import Enum
from random import Random

class SessionStatus(Enum):
    NOT_STARTED = 0
    RUNNING = 1
    PAUSED = 2
    EXIT = 3

class SessionState:
    def __init__(
        self, 
        model_type: Model,
        current_state,
        seed: int,     
    ): 
        self.current_state = current_state
        self.current_input = None
        self.model_type = model_type
        self.seed = seed
        self.rng = Random(self.seed)
        self.step_count = 0