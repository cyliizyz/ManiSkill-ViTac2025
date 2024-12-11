import sys
import re
import matplotlib.pyplot as plt

# Regular expression to match RSS and the value following 'occupies memory'
rss_pattern = re.compile(r'Resident Set Size \(RSS\): (\d+\.\d+) MB')
memory_pattern = re.compile(r'occupies memory: (\d+\.\d+) MB')

def extract_values_from_log(filename):
    rss_values = []
    gpu_memory_values = {f'GPU {i}': [] for i in range(10)}  # Assume there are at most 10 GPUs

    with open(filename, 'r') as file:
        for line in file:
            rss_match = rss_pattern.search(line)
            if rss_match:
                rss_values.append(float(rss_match.group(1)))
            
            memory_match = memory_pattern.search(line)
            if memory_match:
                memory_value = float(memory_match.group(1))
                # Try to match the GPU number
                gpu_match = re.search(r'GPU (\d+)', line)
                if gpu_match:
                    gpu_index = int(gpu_match.group(1))
                    if gpu_index < len(gpu_memory_values):
                        gpu_memory_values[f'GPU {gpu_index}'].append(memory_value)
                else:
                    # Default to GPU 0 if no GPU number is found
                    gpu_memory_values[f'GPU 0'].append(memory_value)
    
    return rss_values, gpu_memory_values

def plot_values(rss_values, gpu_memory_values, output_filename):
    plt.figure(figsize=(12, 6))

    # Plot CPU memory usage
    plt.subplot(1, 2, 1)
    plt.plot(rss_values, label='CPU Memory (MB)')
    plt.title('CPU Memory Usage')
    plt.xlabel('Log Entry Index')
    plt.ylabel('Memory (MB)')
    plt.legend()

    # Plot memory usage for all GPUs
    plt.subplot(1, 2, 2)
    for gpu, values in gpu_memory_values.items():
        if values:  # Only plot GPUs with data
            plt.plot(values, label=gpu)
    plt.title('GPU Memory Usage')
    plt.xlabel('Log Entry Index')
    plt.ylabel('Memory (MB)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Please provide a filename and an output image filename as arguments.")
    else:
        filename = sys.argv[1]
        output_filename = sys.argv[2]
        rss_values, gpu_memory_values = extract_values_from_log(filename)
        plot_values(rss_values, gpu_memory_values, output_filename)