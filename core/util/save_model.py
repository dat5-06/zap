import torch
import torch.nn as nn
import pandas as pd
from core.util.io import get_project_root


def save_model(model: nn.Module, model_name: str, overwrite: bool = False) -> None:
    """Save a model's parameters for later use.

    Arguments:
    ---------
        model (nn.Module): model that should be saved
        model_name (str): name of the file without filetype
        overwrite (bool): if the entry should be overwritten if it already exists

    """
    model_path = get_project_root() / "core/models/saved_models"

    arg_file = pd.read_csv(model_path / "constructor_args.csv", sep=";")
    index = -1

    # Check if the model already exists and get its index if it does
    if model_name in arg_file.get("Model name").to_list():
        if overwrite:
            index = arg_file.index[arg_file["Model name"] == model_name][0]
        else:
            raise Exception("Model name already exists")

    # Convert member values to strings for storage
    constructor_arguments = model.get_members()
    for i in range(len(constructor_arguments)):
        constructor_arguments[i] = str(constructor_arguments[i])

    # Add or change row of the model
    arg_file.loc[index, "Model name"] = model_name
    arg_file.loc[index, "Parameters"] = " ".join(constructor_arguments)
    arg_file.to_csv(model_path / "constructor_args.csv", index=False, sep=";")

    # Save the model as a file
    torch.save(model.state_dict(), model_path / f"{model_name}.pt")


def load_model(model_class: nn.Module, model_name: str, device: str) -> nn.Module:
    """Load in a saved model and returns it.

    Arguments:
    ---------
        model_class (nn.Module): class of the model
        model_name (str): name of model in saved_models folder without filetype
        device (str): device being used (e.g. "cuda:0")

    """
    model_path = get_project_root() / "core/models/saved_models"

    # Check that the model exists
    arg_file = pd.read_csv(model_path / "constructor_args.csv", sep=";")
    if model_name not in arg_file.get("Model name").to_list():
        raise Exception("Model parameters doesn't exist")

    # Get parameter values of the model
    parameters = (
        arg_file.loc[arg_file["Model name"] == model_name].to_numpy()[0].tolist()
    )
    parameters = parameters[1]
    parameters = parameters.split()

    # Convert elements to integers
    for i in range(len(parameters)):
        parameters[i] = int(parameters[i])

    # Initiate and load the model
    model = model_class(*parameters)
    model.load_state_dict(
        torch.load(model_path / f"{model_name}.pt", weights_only=False)
    )
    model.to(device)
    return model


def save_parameters(experiment: str, parameters: dict, overwrite: bool = False) -> None:
    """Save parameters for use in experiments."""
    model_path = get_project_root() / "core/models/saved_models"

    param_file = pd.read_csv(model_path / "experiment_parameters.csv", sep=";")
    index = -1

    # Check if the experiment already exists and get its index if it does
    if experiment in param_file.get("experiment").to_list():
        if overwrite:
            index = param_file.index[param_file["experiment"] == experiment][0]
        else:
            raise Exception("Can't overwrite an entry")

    # Add the parameters and experiment to the file
    param_file.loc[index, "experiment"] = experiment
    for key, val in parameters.items():
        param_file.loc[index, key] = val

    # Write the new parameters to the file
    param_file.to_csv(model_path / "experiment_parameters.csv", index=False, sep=";")


def load_parameters(experiment: str) -> dict:
    """Load in the parameters saved from grid search."""
    model_path = get_project_root() / "core/models/saved_models"

    param_file = pd.read_csv(model_path / "experiment_parameters.csv", sep=";")

    if experiment not in param_file.get("experiment").to_list():
        raise Exception("Experiment doesn't exist")

    # Get the experiment's index and the CSV's headers
    index = param_file.index[param_file["experiment"] == experiment][0]
    columns = param_file.columns.to_list()
    del columns[0]

    # Insert the experiment's parameters in a dictionary
    parameters = {}
    for column in columns:
        parameters[column] = param_file.loc[index, column].item()

    return parameters
