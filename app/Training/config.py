import Data.preprocessing as pp

#General configs
PYTORCH_SEED = 0
SCALE_TECHNIQUE = pp.Scaler.STANDARDIZE

#MLP Training Hyperparameters
EPOCHS = 1000
LEARNING_RATE = 0.003

#MLP architecture
HIDDEN_LAYERS = [32, 32]