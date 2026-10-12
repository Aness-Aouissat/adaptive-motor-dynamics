import Models.mlp as mlp
import Training.config as config
import torch

def train_mlp(x_train, y_train):

    model = mlp.MultiLayerPerceptron(5, config.HIDDEN_LAYERS, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr = config.LEARNING_RATE)

    tensor_x_data = mlp.from_np_to_tensor(x_train)
    tensor_y_data = mlp.from_np_to_tensor(y_train)

    for epoch in range(config.EPOCHS):

        optimizer.zero_grad()

        curr_prediction = model(tensor_x_data)
        loss = ((curr_prediction - tensor_y_data)**2).mean()

        loss.backward()
        optimizer.step()

    return model