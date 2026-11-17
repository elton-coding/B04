"""Run a small training step on CUDA and report the actual device and build."""

import torch


def cuda_device() -> torch.device:
    if torch.version.cuda != "12.8" or not torch.cuda.is_available():
        raise RuntimeError(f"CUDA 12.8 is required and must be available; torch={torch.__version__}")
    return torch.device("cuda:0")


def check_cuda() -> dict:
    device = cuda_device()
    torch.manual_seed(42)
    model = torch.nn.Conv1d(4, 8, kernel_size=3, padding=1, device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    samples = torch.randn(16, 4, 128, device=device)
    initial_weights = model.weight.detach().clone()
    loss = model(samples).square().mean()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)

    if not torch.isfinite(loss).item() or torch.equal(initial_weights, model.weight):
        raise RuntimeError("The CUDA training step did not produce a finite loss and updated weights")

    return {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "loss": loss.item(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }
