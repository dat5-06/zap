from core.util.loss_functions import zap_loss
from core.util.early_stop import EarlyStop


def get_hyperparameter_configuration() -> dict:
    """Get dictionary of our hyperparameter configuration."""
    days = 20
    hyperparameters = {
        "hidden_size": 16,
        "epochs": 200,
        "horizon": 24,
        "loss_function": zap_loss,
        "dropout_rate": 0.2,
        "train_days": int(days * 0.8),
        "val_days": int(days * 0.1),
        "test_days": int(days * 0.1),
        "early_stopper": EarlyStop(10, 0.005),
    }
    return hyperparameters
