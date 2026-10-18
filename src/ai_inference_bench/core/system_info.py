"""Hardware and Runtime Environment Profiler."""

import os
import platform
import sys
from typing import Any, Dict, Optional
import psutil
from pydantic import BaseModel


class SystemInfo(BaseModel):
    os_name: str
    os_release: str
    os_architecture: str
    cpu_model: str
    physical_cores: int
    logical_cores: int
    total_ram_gb: float
    available_ram_gb: float
    python_version: str
    pytorch_version: str
    onnx_version: Optional[str] = None
    onnxruntime_version: Optional[str] = None
    cuda_available: bool = False
    cuda_device_name: Optional[str] = None
    cuda_device_count: int = 0
    rocm_available: bool = False
    mps_available: bool = False
    accelerator_type: str = "cpu"


def collect_system_info() -> SystemInfo:
    """Collects genuine system environment, CPU, RAM, and accelerator details."""
    # CPU Model detection
    cpu_model = platform.processor() or "Unknown CPU"
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu_model = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except Exception:
            pass

    # RAM
    mem = psutil.virtual_memory()
    total_ram = round(mem.total / (1024 ** 3), 2)
    available_ram = round(mem.available / (1024 ** 3), 2)

    # PyTorch & Accelerators
    try:
        import torch
        pytorch_version = torch.__version__
        cuda_avail = torch.cuda.is_available()
        cuda_count = torch.cuda.device_count() if cuda_avail else 0
        cuda_name = torch.cuda.get_device_name(0) if cuda_avail and cuda_count > 0 else None
        rocm_avail = hasattr(torch.version, "hip") and torch.version.hip is not None
        mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except Exception:
        pytorch_version = "unavailable"
        cuda_avail = False
        cuda_count = 0
        cuda_name = None
        rocm_avail = False
        mps_avail = False

    # ONNX
    try:
        import onnx
        onnx_ver = onnx.__version__
    except Exception:
        onnx_ver = None

    try:
        import onnxruntime
        ort_ver = onnxruntime.__version__
    except Exception:
        ort_ver = None

    # Accelerator determination
    if cuda_avail:
        acc_type = "cuda"
    elif rocm_avail:
        acc_type = "rocm"
    elif mps_avail:
        acc_type = "mps"
    else:
        acc_type = "cpu"

    return SystemInfo(
        os_name=platform.system(),
        os_release=platform.release(),
        os_architecture=platform.machine(),
        cpu_model=cpu_model,
        physical_cores=psutil.cpu_count(logical=False) or 1,
        logical_cores=psutil.cpu_count(logical=True) or 1,
        total_ram_gb=total_ram,
        available_ram_gb=available_ram,
        python_version=platform.python_version(),
        pytorch_version=pytorch_version,
        onnx_version=onnx_ver,
        onnxruntime_version=ort_ver,
        cuda_available=cuda_avail,
        cuda_device_name=cuda_name,
        cuda_device_count=cuda_count,
        rocm_available=rocm_avail,
        mps_available=mps_avail,
        accelerator_type=acc_type,
    )
