from core.util.loss_functions import zap_loss


def get_hyperparameter_configuration() -> dict:
    """Get dictionary of our hyperparameter configuration."""
    return {
        "hidden_size": 16,
        "epochs": 10,
        "horizon": 24,
        "lookback": 36,
        "loss_function": zap_loss,
        "dropout_rate": 0.2,
    }
