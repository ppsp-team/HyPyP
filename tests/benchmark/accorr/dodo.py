import mne
import time
import numpy as np
import pickle
import json
import pandas as pd
import dfply as df
import seaborn as sns
from pathlib import Path
from collections import OrderedDict
from hypyp import analyses
from tests.hypyp.sync.accorr import accorr_reference
from hypyp.sync.accorr import accorr
import numba

"""
Benchmark script comparing different optimization approaches for the Adjusted Circular Correlation (accorr) metric.
"""

# Define frequency bands as a dictionary
freq_bands = {
    'Alpha-Low': [7.5, 11],
    'Alpha-High': [11.5, 13]
}

# Convert to an OrderedDict to keep the defined order
freq_bands = OrderedDict(freq_bands)
print('Frequency bands:', freq_bands)

preproc_S1 = mne.read_epochs('../../data/preproc_S1.fif')
preproc_S2 = mne.read_epochs('../../data/preproc_S2.fif')

sampling_rate = preproc_S1.info['sfreq']

scaling_results = {method: [] for method in ['original', 'numba']}
scaling_times = {method: [] for method in ['original', 'numba']}

def numba_run(cpu_num):
    def f(complex_signal: np.ndarray, 
          epochs_average: bool = True, 
          show_progress: bool = True):
          numba.set_num_threads(cpu_num)
          return accorr(complex_signal, epochs_average, show_progress, optimization='numba')
    return f


def torch_run(device):
    def f(complex_signal: np.ndarray, 
          epochs_average: bool = True, 
          show_progress: bool = True):
          return accorr(complex_signal, epochs_average, show_progress, 
                        optimization=f'torch_{device}')
    return f


method_dict = {
    'original': accorr_reference,
    'precomputed': accorr,
    'numba4': numba_run(4),
    'numba8': numba_run(8),
    'torch_cpu': torch_run('cpu'),
    'torch_mps': torch_run('mps'),
}

numba_palette = sns.light_palette('C2', 3)
torch_palette = sns.light_palette('C3', 4)
method_palette = OrderedDict({
    'original': 'C0',
    'precomputed': 'C1',
    'numba4': numba_palette[1],
    'numba8': numba_palette[2],
    'torch_cpu': torch_palette[1],
    'torch_mps': torch_palette[2],
})

out_path = Path('results')

def multiply_channels(epochs, i):
    ch_names = [f'{ch}{x}' for ch in epochs.ch_names for x in range(i)]
    n_info = mne.create_info(ch_names, epochs.info['sfreq'])
    return mne.EpochsArray(np.concatenate([epochs.get_data()] * i, axis=1), info=n_info)


def benchmark(method, epoch_multiplier, channel_multiplier):
    # Create scaled dataset by concatenating
    expanded_preproc_S1 = multiply_channels(preproc_S1, channel_multiplier)
    expanded_preproc_S2 = multiply_channels(preproc_S2, channel_multiplier)

    epochs_list_S1 = [expanded_preproc_S1.copy() for _ in range(epoch_multiplier)]
    epochs_list_S2 = [expanded_preproc_S2.copy() for _ in range(epoch_multiplier)]
    
    preproc_S1_scaled = mne.concatenate_epochs(epochs_list_S1)
    preproc_S2_scaled = mne.concatenate_epochs(epochs_list_S2)
    
    # Prepare data for connectivity analysis
    data_inter = np.array([preproc_S1_scaled, preproc_S2_scaled])
    
    complex_signal = analyses.compute_freq_bands(
        data_inter,
        sampling_rate,
        freq_bands,
        filter_length=int(sampling_rate),
        l_trans_bandwidth=5.0,
        h_trans_bandwidth=5.0
    )
    n_epoch, n_ch, n_freq, n_samp = complex_signal.shape[1], complex_signal.shape[2], \
                                    complex_signal.shape[3], complex_signal.shape[4]

    # calculate all epochs at once, the only downside is that the disk may not have enough space
    complex_signal = complex_signal.transpose((1, 3, 0, 2, 4)).reshape(n_epoch, n_freq, 2 * n_ch, n_samp)

    print(complex_signal.shape)
    print(f"\n  Testing {method}...")
    try:
        st = time.perf_counter()
        result = method_dict[method](complex_signal, epochs_average=False, show_progress=True)
        et = time.perf_counter()
        perf_time = et - st
        print(f"    Time: {perf_time:.4f}s")
            
        result_path = out_path / f'original-e{epoch_multiplier}-c{channel_multiplier}_result.pkl'

        is_ok = None
        max_diff = None
        if method != 'original':
            if result_path.is_file():
                with open(result_path, 'rb') as f:
                    orig_result = pickle.load(f)
                    
                # Compare result with orig_result
                max_diff = np.max(np.abs(result - orig_result))
                is_ok = np.allclose(result, orig_result, rtol=1e-9, atol=1e-10)
        
        return {
            'method': method,
            'epoch_multiplier': int(epoch_multiplier),
            'channel_multiplier': int(channel_multiplier),
            'time': float(perf_time),
            'result': result,
            'is_ok': is_ok,
            'max_diff': float(max_diff) if max_diff is not None else None
        }

    except Exception as e:
        raise e
    

benchmark_configs = pd.DataFrame(
    [
        [1, 1],
        [2, 1],
        [3, 1],
        [5, 1],
        [8, 1],
        [10, 1],
        [1, 2],
        [1, 4],
        [1, 8],
        [3, 2],
        [3, 4],
        [3, 8],
    ],
    columns=['epoch_multiplier', 'channel_multiplier']
)

def task_calc_benchmarks():
    "Executes the different optimizations on different problem sizes."
    def benchmark_action(method, epoch_multiplier, channels_multiplier, targets):
        res = benchmark(method, epoch_multiplier, channels_multiplier)

        if len(targets) > 1:
            with open(targets[1], 'wb') as f:
                pickle.dump(res['result'], f)

        del res['result']
        with open(targets[0], 'w') as f:
            print(res)
            json.dump(res, f)

    if not out_path.is_dir():
        out_path.mkdir()

    for _, r in benchmark_configs.iterrows():
        for method in method_dict:
            name = f'{method}-e{r["epoch_multiplier"]}-c{r["channel_multiplier"]}'
            json_out = out_path / (name + '.json')
            result_out = out_path / (name + '_result.pkl')
            yield {
                'name': name,  
                'actions': [(benchmark_action, (method, r['epoch_multiplier'], r['channel_multiplier']))],
                'targets': [json_out] + ([result_out] * (method == 'original')),
                'uptodate': [json_out.is_file()],
                'file_dep': ([str(result_out).replace(method, 'original')] * int(method != 'original'))
            }
            

def get_perfs(): 
    res = []
    for p in out_path.glob('*.json'):
        with open(p) as f:
            j = json.load(f)
            res.append(j)

    base_ch_num = len(preproc_S1.info.ch_names) + len(preproc_S2.info.ch_names)
    base_epoch_num = preproc_S1.get_data().shape[0]
    perfs = (
        pd.DataFrame.from_records(res)
        >> df.mutate(
            channels = df.X.channel_multiplier * base_ch_num,
            epochs = df.X.epoch_multiplier * base_epoch_num,
        )
    )
    return perfs


def task_summary_plots():
    "Generates summary plot of the speed-up of different optimizations"
    def action(targets):
        perfs = get_perfs()
        
        # Calculate speedup relative to original method
        speedup_data = []
        for (ch, ep), group in perfs.groupby(['channels', 'epochs']):
            original_time = group[group['method'] == 'original']['time'].values
            if len(original_time) > 0:
                original_time = original_time[0]
                for _, row in group.iterrows():
                    speedup = original_time / row['time']
                    speedup_data.append({
                        'method': row['method'],
                        'channels': ch,
                        'epochs': ep,
                        'speedup': speedup
                    })
        
        speedup_df = pd.DataFrame(speedup_data)
        
        fg = sns.catplot(
            speedup_df, 
            x='channels', 
            y='speedup', 
            col='epochs', 
            col_wrap=3,
            hue='method',
            hue_order=method_palette.keys(),
            palette=method_palette,
            kind='bar',
            sharey=False
        )
        fg.savefig(targets[0])

    return {
        'actions': [action],
        'targets': [out_path / 'benchmark.pdf'],
        'uptodate': [False]
    }


def task_bad_perfs():
    "Outputs a csv listing optimizations that do not equal the original results within tolerance."
    def action(targets):
        bad_perfs = (
            get_perfs()
            >> df.mask(df.X.is_ok == False)
        )
        bad_perfs.to_csv(targets[0])
    
    return {
        'actions': [action],
        'targets': [out_path / 'bad_perfs.csv'],
        'uptodate': [False]
    }
