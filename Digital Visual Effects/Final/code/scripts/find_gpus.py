import torch
import gpustat


def get_idle_gpus(n):
    gpu_stats = gpustat.GPUStatCollection.new_query()
    gpu_utilizations = [(gpu.entry['index'], float(gpu.entry['utilization.gpu']),
                         float(gpu.entry['memory.total']) - float(gpu.entry['memory.used'])) for gpu in gpu_stats]
    suitable_gpus = [gpu for gpu in gpu_utilizations if gpu[2] >= 6000]  # 6 GB free memory
    suitable_gpus = [gpu for gpu in gpu_utilizations if gpu[0] != 0]  # 6 GB free memory
    suitable_gpus.sort(key=lambda x: x[1])  # Sort by GPU utilization
    most_idle_gpus = [gpu[0] for gpu in suitable_gpus[:n]]  # Select the top 'n' GPUs
    most_idle_gpus = [0, most_idle_gpus[0]]
    return most_idle_gpus


# Check if there are multiple GPUs
if torch.cuda.device_count() > 1:
    gpu_ids = get_idle_gpus(2)
    device = f'cuda:{gpu_ids[0]}'
    print("Using GPU", gpu_ids)
else:
    gpu_ids = [0]
    device = "cuda:0"
