from core.util.loss_functions import zap_loss
from core.util.early_stop import EarlyStop


def get_hyperparameter_configuration() -> (
    tuple[int, int, int, callable, float, int, EarlyStop]
):
    """Get tuple of our hyperparameter configuration."""
    hidden_size = 16
    epochs = 100
    horizon = 24
    loss_function = zap_loss
    dropout_rate = 0.2
    folds = 9
    early_stopper = EarlyStop(5, 0.00)

    return (
        hidden_size,
        epochs,
        horizon,
        loss_function,
        dropout_rate,
        folds,
        early_stopper,
    )
