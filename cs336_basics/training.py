import numpy as np
import torch
import os
import typing
import torch.nn as nn

def get_batch(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    # sample random starting positions
    # valid range: 0 to len(x) - context_length - 1
    starts = np.random.randint(0, len(x) - context_length, size=batch_size)

    # build input and target sequences
    inputs  = np.stack([x[s : s + context_length] for s in starts])
    targets = np.stack([x[s + 1 : s + context_length + 1] for s in starts])

    # convert to tensors and move to device
    inputs  = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs, targets

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]