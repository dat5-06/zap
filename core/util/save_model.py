import torch
import torch.nn as nn
import pandas as pd
from core.util.io import get_project_root


def save_model(
    model: nn.Module, model_name: str, overwrite: bool = False, *args: int
) -> None:
    """Save a model's parameters for later use.

    Arguments:
    ---------
        model (nn.Module): model that should be saved
        model_name (str): name of the file without filetype
        overwrite (bool): if the entry should be overwritten if it already exists
        *args (int): constructor arguments used to initialise the model

    """
    model_path = get_project_root() / "core/models/saved_models"

    file = pd.read_csv(model_path / "constructor_args.csv", sep=";")
    index = -1

    # Check if the model already exists and get its index if it does
    if model_name in file.get("Model name").to_list():
        if overwrite:
            index = file.index[file["Model name"] == model_name][0]
        else:
            raise Exception("Model name already exists")

    # Make constructor arguments to a list
    constructor_arguments = []
    for arg in args:
        constructor_arguments.append(str(arg))

    # Convert the list to a single string for storage
    constructor_arguments = " ".join(constructor_arguments)

    # Add or change row of the model
    file.loc[index, "Model name"] = model_name
    file.loc[index, "Parameters"] = constructor_arguments
    file.to_csv(model_path / "constructor_args.csv", index=False, sep=";")

    # Save the model as a file
    torch.save(model.state_dict(), model_path / f"{model_name}.pt")


def load_model(model_class: nn.Module, model_name: str, device: str) -> nn.Module:
    """Load in a saved model and returns it.

    Arguments:
    ---------
        model_class (nn.Module): class of the model
        model_name (str): name of model in saved_models folder without filetype
        device (str): device being used (e.g. cuda:0)

    """
    model_path = get_project_root() / "core/models/saved_models"

    # Check that the model exists
    file = pd.read_csv(model_path / "constructor_args.csv", sep=";")
    if model_name not in file.get("Model name").to_list():
        raise Exception("Model parameters doesn't exist")

    # Get parameter values of the model
    parameters = file.loc[file["Model name"] == model_name].to_numpy()[0].tolist()
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
