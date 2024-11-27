from core.util.loss_functions import zap_loss
from core.util.early_stop import EarlyStop


def get_hyperparameter_configuration() -> dict:
    """Get dictionary of our hyperparameter configuration."""
    hyperparameters = {
        "hidden_size": 16,
        "epochs": 200,
        "horizon": 24,
        "loss_function": zap_loss,
        "dropout_rate": 0.2,
        "train_days": 224,
        "val_days": 28,
        "test_days": 28,
        "early_stopper": EarlyStop(10, 0.005),
    }
    return hyperparameters
