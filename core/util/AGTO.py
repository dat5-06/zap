import torch
from torch.optim import Optimizer


class AGTOOptimizer(Optimizer):
    """Artificial Gorilla Optimizer."""

    def __init__(
        self,
        params: any,
        lr: float,
        p1: float,
        p2: float,
        beta: float,
        epoch: int,
        pop_size: int,
    ) -> None:
        """Initialize the optimizer with model parameters and AGTO hyperparameters."""
        defaults = dict(lr=lr, p1=p1, p2=p2, beta=beta, epoch=epoch, pop_size=pop_size)
        super(AGTOOptimizer, self).__init__(params, defaults)

    def step(self, closure: float = None) -> float:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        # Retrieve model parameters
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.data

                # AGTO update equation (customize this part based on AGTO paper)
                a = (torch.cos(2 * torch.rand(1)) + 1) * (
                    1 - group["epoch"] / group["epoch"]
                )
                c = a * (2 * torch.rand(1) - 1)

                # Exploration/Exploitation steps
                if torch.rand(1) < group["p1"]:
                    # Exploration step: random update
                    update = torch.rand_like(param.data) * grad
                else:
                    # Exploitation step: use AGTO equations
                    # (you can customize this further)
                    update = -c * grad * param.data

                # Apply the update to the parameter
                param.data.add_(update, alpha=-group["lr"])

        return loss
