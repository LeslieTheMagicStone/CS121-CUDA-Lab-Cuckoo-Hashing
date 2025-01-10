import matplotlib.pyplot as plt
import os
import numpy as np

def parse_perf_out(filepath):
    data = {
        'experiment1': {'t2': {'exp': [], 'seq': [], 'cuda': []}, 't3': {'exp': [], 'seq': [], 'cuda': []}},
        'experiment2': {'t2': {'i': [], 'seq': [], 'cuda': []}, 't3': {'i': [], 'seq': [], 'cuda': []}},
        'experiment3': {'t2': {'size': [], 'seq': [], 'cuda': []}, 't3': {'size': [], 'seq': [], 'cuda': []}},
        'experiment4': {'t2': {'maxIter': [], 'seq': [], 'cuda': []}, 't3': {'maxIter': [], 'seq': [], 'cuda': []}}
    }
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split()
            if line.startswith('E1'):
                exp = int(parts[2].split('=')[1])
                seq_time = int(parts[4])
                cuda_time = int(parts[10])
                t = parts[1].split('=')[1]
                if t == '2':
                    data['experiment1']['t2']['exp'].append(exp)
                    data['experiment1']['t2']['seq'].append(seq_time)
                    data['experiment1']['t2']['cuda'].append(cuda_time)
                elif t == '3':
                    data['experiment1']['t3']['exp'].append(exp)
                    data['experiment1']['t3']['seq'].append(seq_time)
                    data['experiment1']['t3']['cuda'].append(cuda_time)
            elif line.startswith('E2'):
                i = int(parts[2].split('=')[1])
                seq_time = int(parts[4])
                cuda_time = int(parts[8])
                t = parts[1].split('=')[1]
                if t == '2':
                    data['experiment2']['t2']['i'].append(i)
                    data['experiment2']['t2']['seq'].append(seq_time)
                    data['experiment2']['t2']['cuda'].append(cuda_time)
                elif t == '3':
                    data['experiment2']['t3']['i'].append(i)
                    data['experiment2']['t3']['seq'].append(seq_time)
                    data['experiment2']['t3']['cuda'].append(cuda_time)
            elif line.startswith('E3'):
                size = float(parts[2].split('=')[1].replace('n', ''))
                seq_time = int(parts[4])
                cuda_time = int(parts[10])
                t = parts[1].split('=')[1]
                if t == '2':
                    data['experiment3']['t2']['size'].append(size)
                    data['experiment3']['t2']['seq'].append(seq_time)
                    data['experiment3']['t2']['cuda'].append(cuda_time)
                elif t == '3':
                    data['experiment3']['t3']['size'].append(size)
                    data['experiment3']['t3']['seq'].append(seq_time)
                    data['experiment3']['t3']['cuda'].append(cuda_time)
            elif line.startswith('E4'):
                maxIter = int(parts[2].split('=')[1])
                seq_time = int(parts[5])
                cuda_time = int(parts[11])
                t = parts[1].split('=')[1]
                if t == '2':
                    data['experiment4']['t2']['maxIter'].append(maxIter)
                    data['experiment4']['t2']['seq'].append(seq_time)
                    data['experiment4']['t2']['cuda'].append(cuda_time)
                elif t == '3':
                    data['experiment4']['t3']['maxIter'].append(maxIter)
                    data['experiment4']['t3']['seq'].append(seq_time)
                    data['experiment4']['t3']['cuda'].append(cuda_time)
    return data

def calculate_mop(exp, times):
    mops = []
    for e, t in zip(exp, times):
        mop = (2 ** e) / (t * 1e-6) / 1e6
        mops.append(mop)
    return mops

def plot_experiment1(data, t):
    exp = data[f't{t}']['exp']
    seq_times = data[f't{t}']['seq']
    cuda_times = data[f't{t}']['cuda']

    unique_exp = sorted(set(exp))
    seq_mops = {e: [] for e in unique_exp}
    cuda_mops = {e: [] for e in unique_exp}

    for e, seq_time, cuda_time in zip(exp, seq_times, cuda_times):
        seq_mops[e].append((2 ** e) / (seq_time * 1e-6) / 1e6)
        cuda_mops[e].append((2 ** e) / (cuda_time * 1e-6) / 1e6)

    fig, ax = plt.subplots()
    width = 0.35
    x = np.arange(len(unique_exp))

    seq_means = [np.mean(seq_mops[e]) for e in unique_exp]
    seq_stds = [np.std(seq_mops[e]) for e in unique_exp]
    cuda_means = [np.mean(cuda_mops[e]) for e in unique_exp]
    cuda_stds = [np.std(cuda_mops[e]) for e in unique_exp]

    ax.bar(x - width/2, seq_means, width, yerr=seq_stds, label='Sequential (Bar)', color='blue')
    ax.bar(x + width/2, cuda_means, width, yerr=cuda_stds, label='CUDA (Bar)', color='red')

    ax.plot(x, seq_means, label='Sequential (Line)', marker='o', color='blue')
    ax.plot(x, cuda_means, label='CUDA (Line)', marker='x', color='red')

    ax.set_xlabel('exp (log of total operations)')
    ax.set_ylabel('MOP (Millions of Operations per Second)')
    ax.set_title(f'Experiment 1 (t={t})')
    ax.set_xticks(x)
    ax.set_xticklabels(unique_exp)
    ax.legend()
    ax.grid(True)

    os.makedirs('./figs', exist_ok=True)
    plt.savefig(f'./figs/experiment1_t{t}.png')

def plot_experiment2(data, t):
    i = data[f't{t}']['i']
    seq_times = data[f't{t}']['seq']
    cuda_times = data[f't{t}']['cuda']

    unique_i = sorted(set(i))
    seq_mops = {e: [] for e in unique_i}
    cuda_mops = {e: [] for e in unique_i}

    for e, seq_time, cuda_time in zip(i, seq_times, cuda_times):
        seq_mops[e].append((2 ** 24) / (seq_time * 1e-6) / 1e6)
        cuda_mops[e].append((2 ** 24) / (cuda_time * 1e-6) / 1e6)

    fig, ax = plt.subplots()
    width = 0.35
    x = np.arange(len(unique_i))

    seq_means = [np.mean(seq_mops[e]) for e in unique_i]
    seq_stds = [np.std(seq_mops[e]) for e in unique_i]
    cuda_means = [np.mean(cuda_mops[e]) for e in unique_i]
    cuda_stds = [np.std(cuda_mops[e]) for e in unique_i]

    ax.bar(x - width/2, seq_means, width, yerr=seq_stds, label='Sequential (Bar)', color='blue')
    ax.bar(x + width/2, cuda_means, width, yerr=cuda_stds, label='CUDA (Bar)', color='red')

    ax.plot(x, seq_means, label='Sequential (Line)', marker='o', color='blue')
    ax.plot(x, cuda_means, label='CUDA (Line)', marker='x', color='red')

    ax.set_xlabel('i (percentage of randomly generated keys)')
    ax.set_ylabel('MOP (Millions of Operations per Second)')
    ax.set_title(f'Experiment 2 (t={t})')
    ax.set_xticks(x)
    ax.set_xticklabels(unique_i)
    ax.legend()
    ax.grid(True)

    os.makedirs('./figs', exist_ok=True)
    plt.savefig(f'./figs/experiment2_t{t}.png')

def plot_experiment3(data, t):
    size = data[f't{t}']['size']
    seq_times = data[f't{t}']['seq']
    cuda_times = data[f't{t}']['cuda']

    unique_size = sorted(set(size))
    seq_mops = {s: [] for s in unique_size}
    cuda_mops = {s: [] for s in unique_size}

    for s, seq_time, cuda_time in zip(size, seq_times, cuda_times):
        seq_mops[s].append((2 ** 24) / (seq_time * 1e-6) / 1e6)
        cuda_mops[s].append((2 ** 24) / (cuda_time * 1e-6) / 1e6)

    fig, ax = plt.subplots()
    width = 0.35
    x = np.arange(len(unique_size))

    seq_means = [np.mean(seq_mops[s]) for s in unique_size]
    seq_stds = [np.std(seq_mops[s]) for s in unique_size]
    cuda_means = [np.mean(cuda_mops[s]) for s in unique_size]
    cuda_stds = [np.std(cuda_mops[s]) for s in unique_size]

    ax.bar(x - width/2, seq_means, width, yerr=seq_stds, label='Sequential (Bar)', color='blue')
    ax.bar(x + width/2, cuda_means, width, yerr=cuda_stds, label='CUDA (Bar)', color='red')

    ax.plot(x, seq_means, label='Sequential (Line)', marker='o', color='blue')
    ax.plot(x, cuda_means, label='CUDA (Line)', marker='x', color='red')

    ax.set_xlabel('size (relative to n)')
    ax.set_ylabel('MOP (Millions of Operations per Second)')
    ax.set_title(f'Experiment 3 (t={t})')
    ax.set_xticks(x)
    ax.set_xticklabels(unique_size)
    ax.legend()
    ax.grid(True)

    os.makedirs('./figs', exist_ok=True)
    plt.savefig(f'./figs/experiment3_t{t}.png')

def plot_experiment4(data, t):
    maxIter = data[f't{t}']['maxIter']
    seq_times = data[f't{t}']['seq']
    cuda_times = data[f't{t}']['cuda']

    unique_maxIter = sorted(set(maxIter))
    seq_mops = {m: [] for m in unique_maxIter}
    cuda_mops = {m: [] for m in unique_maxIter}

    for m, seq_time, cuda_time in zip(maxIter, seq_times, cuda_times):
        seq_mops[m].append((2 ** 24) / (seq_time * 1e-6) / 1e6)
        cuda_mops[m].append((2 ** 24) / (cuda_time * 1e-6) / 1e6)

    fig, ax = plt.subplots()
    width = 0.35
    x = np.arange(len(unique_maxIter))

    seq_means = [np.mean(seq_mops[m]) for m in unique_maxIter]
    seq_stds = [np.std(seq_mops[m]) for m in unique_maxIter]
    cuda_means = [np.mean(cuda_mops[m]) for m in unique_maxIter]
    cuda_stds = [np.std(cuda_mops[m]) for m in unique_maxIter]

    ax.bar(x - width/2, seq_means, width, yerr=seq_stds, label='Sequential (Bar)', color='blue')
    ax.bar(x + width/2, cuda_means, width, yerr=cuda_stds, label='CUDA (Bar)', color='red')

    ax.plot(x, seq_means, label='Sequential (Line)', marker='o', color='blue')
    ax.plot(x, cuda_means, label='CUDA (Line)', marker='x', color='red')

    ax.set_xlabel('maxIter (logn)')
    ax.set_ylabel('MOP (Millions of Operations per Second)')
    ax.set_title(f'Experiment 4 (t={t})')
    ax.set_xticks(x)
    ax.set_xticklabels(unique_maxIter)
    ax.legend()
    ax.grid(True)

    os.makedirs('./figs', exist_ok=True)
    plt.savefig(f'./figs/experiment4_t{t}.png')

def main():
    filepath = '/home/magstn/git/CS121-CUDA-Lab-Cuckoo-Hashing/perf.out'
    data = parse_perf_out(filepath)
    plot_experiment1(data['experiment1'], 2)
    plot_experiment1(data['experiment1'], 3)
    plot_experiment2(data['experiment2'], 2)
    plot_experiment2(data['experiment2'], 3)
    plot_experiment3(data['experiment3'], 2)
    plot_experiment3(data['experiment3'], 3)
    plot_experiment4(data['experiment4'], 2)
    plot_experiment4(data['experiment4'], 3)

if __name__ == '__main__':
    main()
