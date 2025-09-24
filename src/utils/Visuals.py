import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
from scipy import signal
from matplotlib import cm
import matplotlib.pyplot as plt
from math import comb

def visualize_time_comparison(levels, previous_times, current_times):
    """
    Create a bar chart comparing previous and current time for different question levels.
    
    Parameters:
    levels (dict): A dictionary with keys 'level1', 'level2', 'level3' and values as the number of questions
    previous_times (dict): A dictionary with keys 'level1', 'level2', 'level3' and values as previous times in minutes
    current_times (dict): A dictionary with keys 'level1', 'level2', 'level3' and values as current times in minutes
    """
    
    labels = [f"Level 1\n({levels['level1']} questions)", f"Level 2\n({levels['level2']} questions)", f"Level 3\n({levels['level3']} questions)"]
    
    width = 0.35
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 6))
    
    previous_bars = ax.bar(x - width/2, [previous_times['level1'], previous_times['level2'], previous_times['level3']], width, label='Previous Time')
    current_bars = ax.bar(x + width/2, [current_times['level1'], current_times['level2'], current_times['level3']], width, label='Current Time')
    ax.set_xlabel('Question Levels')
    ax.set_ylabel('Time (minutes)')
    ax.set_title('Time Comparison for Different Question Levels')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    add_labels(previous_bars)
    add_labels(current_bars)
    
    for i in range(len(labels)):
        prev = previous_times[f'level{i+1}']
        curr = current_times[f'level{i+1}']
        diff = prev - curr
        color = 'green' if diff > 0 else 'red'
        ax.plot([x[i] - width/2, x[i] + width/2], [prev, curr], 'o-', color=color, linewidth=2)
        
        ax.annotate(f"{'+' if diff < 0 else ''}{-diff:.1f}", xy=((x[i] - width/2 + x[i] + width/2)/2, (prev + curr)/2),xytext=(0, 5),textcoords="offset points",ha='center', va='bottom',color=color)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def visualize_frequency_comparison(audio_file1, audio_file2, sample_rate=44100):
    """
    Create a plot comparing the Fourier transforms of two voice recordings.
    
    Parameters:
    audio_file1 (str): Path to the first audio file
    audio_file2 (str): Path to the second audio file
    sample_rate (int): Sampling rate of the audio files, defaults to 44100 Hz
    
    Returns:
    matplotlib.figure.Figure: The figure containing the plot
    """
    
    sr1, data1 = wavfile.read(audio_file1)
    sr2, data2 = wavfile.read(audio_file2)
    
    if len(data1.shape) > 1:
        data1 = data1[:, 0]
    if len(data2.shape) > 1:
        data2 = data2[:, 0]
    
    data1 = data1 / np.max(np.abs(data1))
    data2 = data2 / np.max(np.abs(data2))
    
    n1 = len(data1)
    n2 = len(data2)
    
    yf1 = fft(data1)
    yf2 = fft(data2)
    
    xf1 = fftfreq(n1, 1 / sr1)[:n1//2]
    xf2 = fftfreq(n2, 1 / sr2)[:n2//2]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.set_title('Frequency Spectrum - First Recording')
    ax1.plot(xf1, 2.0/n1 * np.abs(yf1[0:n1//2]))
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Amplitude')
    ax1.set_xlim(0, 5000)
    
    ax2.set_title('Frequency Spectrum - Second Recording')
    ax2.plot(xf2, 2.0/n2 * np.abs(yf2[0:n2//2]))
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Amplitude')
    ax2.set_xlim(0, 5000)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def visualize_comparison(data_lists, levels=None):
    """
    Create a bar chart comparing multiple data series for different question levels.
    
    Parameters:
    data_lists (list): A list of lists, where each inner list contains:
                    [label, level1_value, level2_value, level3_value]
    levels (dict): Optional. A dictionary with keys 'level1', 'level2', 'level3' and 
                values as the number of questions
    
    Returns:
    matplotlib.figure.Figure: The figure containing the plot
    """
    if levels:
        labels = [f"Level 1\n({levels['level1']} questions)", 
                f"Level 2\n({levels['level2']} questions)", 
                f"Level 3\n({levels['level3']} questions)"]
    else:
        labels = ["Level 1", "Level 2", "Level 3"]
    
    x = np.arange(len(labels))
    width = 0.8 / len(data_lists)  # Adjust bar width based on number of data series
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars_list = []
    for i, data in enumerate(data_lists):
        label = data[0]
        values = data[1:4]  # Extract the three level values
        position = x + width * (i - (len(data_lists)-1)/2)
        bars = ax.bar(position, values, width, label=label)
        bars_list.append(bars)
    
    ax.set_xlabel('Question Levels')
    ax.set_ylabel('Time (minutes)')
    ax.set_title('Comparison for Different Question Levels')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    for bars in bars_list:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    return fig


def visualize_spectrograms_comparison(audio_file1, audio_file2, title1="First Recording", title2="Second Recording", sample_rate=44100):
    """
    Create spectrograms for two voice recordings one below the other.
    
    Parameters:
    audio_file1 (str): Path to the first audio file
    audio_file2 (str): Path to the second audio file
    title1 (str): Title for the first spectrogram
    title2 (str): Title for the second spectrogram
    sample_rate (int): Sampling rate of the audio files, defaults to 44100 Hz
    
    Returns:
    matplotlib.figure.Figure: The figure containing the spectrograms
    """
    sr1, data1 = wavfile.read(audio_file1)
    sr2, data2 = wavfile.read(audio_file2)
    
    if len(data1.shape) > 1:
        data1 = data1[:, 0]
    if len(data2.shape) > 1:
        data2 = data2[:, 0]
    
    data1 = data1 / np.max(np.abs(data1))
    data2 = data2 / np.max(np.abs(data2))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    nperseg = 1024
    noverlap = 512
    
    f1, t1, Sxx1 = signal.spectrogram(data1, sr1, nperseg=nperseg, noverlap=noverlap)
    f2, t2, Sxx2 = signal.spectrogram(data2, sr2, nperseg=nperseg, noverlap=noverlap)
    
    im1 = ax1.pcolormesh(t1, f1, 10 * np.log10(Sxx1 + 1e-10), shading='gouraud', cmap=cm.viridis)
    im2 = ax2.pcolormesh(t2, f2, 10 * np.log10(Sxx2 + 1e-10), shading='gouraud', cmap=cm.viridis)
    
    fig.colorbar(im1, ax=ax1, label='Power/frequency (dB/Hz)')
    fig.colorbar(im2, ax=ax2, label='Power/frequency (dB/Hz)')
    
    # Set titles and labels
    ax1.set_title(f'Spectrogram - {title1}')
    ax2.set_title(f'Spectrogram - {title2}')
    
    ax1.set_ylabel('Frequency (Hz)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (s)')
    
    ax1.set_ylim(0, 5000)
    ax2.set_ylim(0, 5000)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_spectrogram(audio_file, title="Spectrogram", sample_rate=44100):
    """
    Create a spectrogram for a voice recording.
    
    Parameters:
    audio_file (str): Path to the audio file
    title (str): Title for the spectrogram
    sample_rate (int): Sampling rate of the audio file, defaults to 44100 Hz
    
    Returns:
    matplotlib.figure.Figure: The figure containing the spectrogram
    """
    sr, data = wavfile.read(audio_file)
    
    if len(data.shape) > 1:
        data = data[:, 0]
    
    data = data / np.max(np.abs(data))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    nperseg = 1024
    noverlap = 512
    
    f, t, Sxx = signal.spectrogram(data, sr, nperseg=nperseg, noverlap=noverlap)
    
    im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap=cm.viridis)
    
    fig.colorbar(im, ax=ax, label='Power/frequency (dB/Hz)')
    
    ax.set_title(f'Spectrogram - {title}')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_ylim(0, 5000)
    
    plt.tight_layout()
    plt.show()
    
    return fig
