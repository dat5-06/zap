import torch
import copy
import sys
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
from core.util.early_stop import EarlyStop

from core.util.get_datasets import cross_validation
from core.util.trefor_dataset import TreforData
from core.util.hyperparameter_configuration import get_hyperparameter_configuration


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


def blocked_training(
    model: nn.Module,
    learning_rate: float,
    device: str,
    batch_size: int,
    early_stopper: EarlyStop,
    features: dict = {},
) -> tuple[list, list, nn.Module]:
    """Train a model with blocked cross validation."""
    _, epochs, horizon, lookback, loss_function, _, folds = (
        get_hyperparameter_configuration()
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_v_loss = float("inf")
    best_model = None

    train_loss = []
    val_loss = []

    for epoch in tqdm(range(epochs), desc="Iterating epochs"):
        # Initialize variables to keep track of block data
        avg_loss = 0
        avg_v_loss = 0
        blocks = 0

        # Iterate over the blocks
        for i, (x_train, y_train, x_val, y_val, _, _, _) in enumerate(
            cross_validation(
                lookback=lookback, horizon=horizon, folds=folds, features=features
            )
        ):
            # convert to dataset that can use dataloaders
            train_dataset = TreforData(x_train, y_train, device)
            val_dataset = TreforData(x_val, y_val, device)

            # initialize the dataloaders, without shuffeling the data between epochs
            training_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=False
            )
            validation_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )

            # Train model on block
            model.train()
            avg_loss += train_one_epoch(
                model, optimizer, loss_function, training_loader
            )

            # Validate model on block
            model.eval()
            with torch.no_grad():
                running_v_loss = 0.0
                for i, (v_inputs, v_target) in enumerate(validation_loader):
                    v_predictions = model(v_inputs)
                    v_target = v_target.squeeze(-1)
                    running_v_loss += loss_function(v_predictions, v_target).item()

            avg_v_loss += running_v_loss / len(validation_loader)
            blocks += 1

        avg_loss = avg_loss / blocks
        avg_v_loss = avg_v_loss / blocks

        # Checks for early stop
        if early_stopper.early_stop(avg_v_loss):
            break

        # Appends average loss for training and validation
        train_loss.append(avg_loss)
        val_loss.append(avg_v_loss)

        # If this model has the lowest loss, we save its state for later reference
        if avg_v_loss < best_v_loss:
            best_v_loss = avg_v_loss
            best_model = copy.deepcopy(model)
    return (train_loss, val_loss, best_model)
