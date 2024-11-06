import torch
import copy
import sys
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
from core.util.early_stop import EarlyStop


def train_one_epoch(
    model: nn.Module,
    optimizer: object,
    loss_function: callable,
    training_loader: DataLoader,
) -> float:
    """Train one epoch."""
    running_loss = 0.0

    for i, (inputs, target) in enumerate(training_loader):
        # Reset the gradients
        optimizer.zero_grad()

        # Make predictions for this batch
        predictions = model(inputs)

        # Compute the loss and its gradient
        target = target.squeeze(-1)
        loss = loss_function(predictions, target)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
    return running_loss / len(training_loader)


def train_model(
    epochs: int,
    model: nn.Module,
    loss_function: callable,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    learning_rate: float,
    early_stopper: EarlyStop,
) -> tuple[list, list, nn.Module]:
    """Train a model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_v_loss = sys.maxsize
    best_model = None

    train_loss = []
    val_loss = []

    for epoch in tqdm(range(epochs), desc="Iterating epochs"):
        # Make sure gradient tracking is on, and do a pass over the data
        model.train()
        avg_loss = train_one_epoch(model, optimizer, loss_function, training_loader)

        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization
        model.eval()

        # Run the validation set, to see how the model performed this epoch
        # Disable gradient computation and reduce memory consumption
        with torch.no_grad():
            running_v_loss = 0.0
            for i, (v_inputs, v_target) in enumerate(validation_loader):
                v_predictions = model(v_inputs)
                v_target = v_target.squeeze(-1)
                running_v_loss += loss_function(v_predictions, v_target).item()

        # Log the running loss averaged per batch
        # for both training and validation
        avg_v_loss = running_v_loss / len(validation_loader)

        # Checks for early stop
        if early_stopper.early_stop(avg_v_loss):
            break

        train_loss.append(avg_loss)
        val_loss.append(avg_v_loss)

        # If this model has the lowest loss, we save its state for later reference
        if avg_v_loss < best_v_loss:
            best_v_loss = avg_v_loss
            best_model = copy.deepcopy(model)

    return (train_loss, val_loss, best_model)


def test_model(
    best_model: nn.Module, loss_function: callable, testing_loader: DataLoader
) -> tuple[float, list]:
    """Test model on test set."""
    best_model.eval()
    predicted = []
    t_loss = 0

    with torch.no_grad():
        for i, t_data in enumerate(testing_loader):
            t_inputs, t_target = t_data
            t_predictions = best_model(t_inputs)
            predicted.append(t_predictions)
            t_loss += loss_function(t_predictions, t_target).item()

    predicted = torch.cat(predicted, dim=0)
    t_loss /= len(testing_loader)
    return (t_loss, predicted)
