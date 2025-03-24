import logging
import logging.config
import sys
from pathlib import Path
import datetime
import psutil
import torch


def get_system_info() -> str:
    """Get system resource usage information."""
    cpu_usage = psutil.cpu_percent(interval=0.1)
    mem_info = psutil.virtual_memory()
    total_mem = mem_info.total / (1024**3)  # Convert to GB
    used_mem = mem_info.used / (1024**3)  # Convert to GB

    # Format CPU usage: Right-align within 6 characters (e.g., ' 6.60%')
    cpu_str = f"CPU: {cpu_usage:6.2f}%"

    # Format MEM usage: Right-align used and total memory
    mem_str = f"MEM: {used_mem:6.2f}GB/{total_mem:7.2f}GB"

    # GPU information with fixed-width formatting
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        gpu_usage = torch.cuda.memory_allocated(0) / gpu.total_memory * 100
        # Assuming GPU name length varies, but CPU and MEM are fixed
        gpu_str = f" | GPU: {gpu.name}, GPU Usage: {gpu_usage:6.2f}%"
    else:
        gpu_str = ""

    return f"{cpu_str} | {mem_str}{gpu_str}"


class ExtraInfoFilter(logging.Filter):
    """Custom filter to add system info to log records."""

    def filter(self, record):
        record.extra_info = get_system_info()
        return True


def set_up_log(log_level: int = None, suffix: str = None) -> None:
    """Set up logger
    Args:
        log_level (int, optional): log level. Defaults to None.

        log_levels are defined as:
        CRITICAL: 50
        ERROR: 40
        WARNING: 30
        INFO: 20
        DEBUG: 10
        NOTSET: 0"""

    # daettime and calling script name
    if suffix is None:
        calling_script = Path(sys.argv[0]).stem
        suffix = f"{calling_script}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    log_file_path = Path(f"log/{suffix}.log")
    if not log_file_path.parent.exists():
        log_file_path.parent.mkdir()
    log_format = "%(asctime)s | %(levelname)s | %(module)s | %(message)s"

    if log_level is None:
        log_level = 20

    # Advanced
    log = logging.getLogger()

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(log_level)

    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    log.addHandler(file_handler)
    log.addHandler(stream_handler)
    log.setLevel(log_level)

    # Add system info filter
    extra_info_filter = ExtraInfoFilter()
    log.addFilter(extra_info_filter)
