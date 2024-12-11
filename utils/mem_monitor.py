import psutil
import loguru
import os
import pynvml


def monitor_process_memory_once(pid=None, logger= None):
    """
    Monitor the memory usage of a specified process at one time.
    """

    if logger:
        logger.info("CPU_memory##############################")

    if pid is None:
        pid = os.getpid()
    process = psutil.Process(pid)
    mem_info = process.memory_info()

    if logger:
        logger.info(f"Memory usage information of process PID {pid}:")
        logger.info(f"Resident Set Size (RSS): {mem_info.rss / 1024 / 1024:.2f} MB")
        logger.info(f"Virtual Memory Size (VMS): {mem_info.vms / 1024 / 1024:.2f} MB")

    return pid, mem_info.rss / 1024 / 1024


def monitor_process_gpu_memory(pid=None,logger = None):
    """
    Monitor GPU memory usage for specified processes.
    """
    if logger:
        logger.info("GPU_memory##############################")
    if pid is None:
        pid = os.getpid()
    pynvml.nvmlInit()
    gpu_memory_usage = {}
    device_count = pynvml.nvmlDeviceGetCount()
    if device_count == 0:
        if logger:
            logger.info("No NVIDIA GPU device found.")
        return gpu_memory_usage
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        pids = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        for pid_info in pids:
            if pid_info.pid == pid:
                gpu_memory_usage[i] = pid_info.usedGpuMemory / (1024**2)
    if logger:
        for gpu_index, memory_usage in gpu_memory_usage.items():
            logger.info(
                f"Process PID {pid} on GPU {gpu_index} occupies memory: {memory_usage:.2f} MB"
            )
    pynvml.nvmlShutdown()
    return pid, gpu_memory_usage
