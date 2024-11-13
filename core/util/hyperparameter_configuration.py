from core.util.loss_functions import zap_loss


def get_hyperparameter_configuration() -> (
    tuple[int, int, int, int, callable, float, int]
):
    """Get tuple of our hyperparameter configuration."""
    hidden_size = 16
    epochs = 10
    horizon = 24
    lookback = 36
    loss_function = zap_loss
    dropout_rate = 0.2
    folds = 9

    return (hidden_size, epochs, horizon, lookback, loss_function, dropout_rate, folds)
