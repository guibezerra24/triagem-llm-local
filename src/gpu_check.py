"""Executa multiplicacao e backward reais na GPU e salva evidencias."""
import platform
import subprocess
import time
from datetime import datetime, timezone

import torch
from .common import ROOT, require_gpu, write_json


def main():
    require_gpu()
    torch.manual_seed(42)
    x = torch.randn(1024, 1024, device="cuda:0", requires_grad=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    loss = (x @ x.T).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert x.grad is not None and torch.isfinite(x.grad).all().item()
    props = torch.cuda.get_device_properties(0)
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "gpu": props.name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "total_vram_mib": props.total_memory / 1024**2,
        "operation_device": str(loss.device), "backward_finite": True,
        "operation_seconds": time.perf_counter() - start,
        "compiled_architectures": torch.cuda.get_arch_list(),
    }
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True).stdout
    write_json(ROOT / "reports" / "gpu.json", result)
    text = "GPU REAL - TESTE CUDA E BACKWARD\n\n" + "\n".join(f"{k}: {v}" for k, v in result.items()) + "\n\n" + smi
    (ROOT / "reports" / "gpu.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
