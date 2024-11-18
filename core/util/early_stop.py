class EarlyStop:
    """Class for implementing early stop."""

    def __init__(self, patience: int, min_delta: float = 0) -> None:
        """Initialise early stop object.

        Arguments:
        ---------
            patience (int): how many epochs without improvement before it early stops
            min_delta (float): slack for how much worse the new loss can be

        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss: float) -> bool:
        """Check if the model should early stop."""
        # Check if it has gotten a better validation loss
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        # Check if the validation loss is greater than the allowed slack (min_val+delta)
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            # Stops if patience runs out
            if self.counter >= self.patience:
                return True
        return False

    def reset(self) -> None:
        """Reset the early stop counter."""
        self.counter = 0
        self.min_validation_loss = float("inf")
