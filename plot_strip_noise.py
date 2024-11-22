import uproot
import hist as Hist
import numpy as np
import mplhep
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
import sys
import shutil

# invocation: python plot_strip_noise.py module_name [path_to_filename.root] [output_dir]

module_name = sys.argv[1]
filename = Path(sys.argv[2] if len(sys.argv) > 2 else f'~/Hybrid_{module_name}.root')
storage_path = Path(sys.argv[3] if len(sys.argv) > 3 else '/Users/andresf/cernbox/www')
destination_folder = Path(storage_path, module_name)
histname = 'Detector/Board_0/OpticalGroup_0/Hybrid_XX/D_B(0)_O(0)_HybridStripNoiseDistribution_Hybrid(XX)'

# filename = Path('~/Hybrid_PS_26_05_BRN-00102.root')
# destination_folder = Path('/Users/andresf/cernbox/www')
# histname = 'Detector/Board_0/OpticalGroup_0/Hybrid_XX/D_B(0)_O(0)_HybridStripNoiseDistribution_Hybrid(XX)'

def zscore(df, window, thresh=3, return_all=False):
    roll = df.rolling(window=window, min_periods=1, center=True)
    avg = roll.mean()
    std = roll.std(ddof=0)
    z = df.sub(avg).div(std)   
    m = z.between(-thresh, thresh)
    
    if return_all:
        return z, avg, std, m
    return df.where(m, avg)

def make_noise_plot(ax, data, module, hybrid):
    ax.set_xlabel('Sensor number')
    ax.set_ylabel('Noise [ThDAC]')
    max_val = np.max(data.values())
    ax.set_ylim([0, max_val*1.7])
    mplhep.histplot(data, ax=ax, yerr=data.variances(), histtype='errorbar', marker=None, color='g', label='Data')

    window = 20 # moving average
    thresh = 3 # num. sigmas
    df = pd.DataFrame(np.transpose([data.values(), data.variances()]), columns=['noise', 'variance'])
    z, avg, std, m = zscore(df['noise'], window=window, thresh=thresh, return_all=True)
    avg.plot(ax=ax, label=f'Mean ($\\overline{{n}}$ = {window})', color='orange')
    df['noise'].loc[~m].plot(ax=ax, label=f'Outliers ($\\sigma = 3$)', yerr=df['variance'].loc[~m], marker='o', ls='', color='r')

    for chan, noise in zip(df['noise'].loc[~m].index, df['noise'].loc[~m].values):
        ax.text(chan+20, noise, str(chan), color='r')

    ax.text(0.5, 0.93, module, size=12, fontweight='bold', horizontalalignment='center', transform=ax.transAxes)
    ax.text(0.5, 0.85, f"{'Right' if hybrid==0 else 'Left'} hybrid ({hybrid})", fontweight='bold', horizontalalignment='center', size=11, transform=ax.transAxes)
    ax.axhline(y=max_val*1.4, color='black', linestyle='--')
    ax.arrow(0.15, 0.87, -0.05, 0, width=0.01, transform=ax.transAxes)
    ax.text(0.07, 0.92, 'To POH', transform=ax.transAxes)
    ax.arrow(0.85, 0.87, 0.05, 0, width=0.01, transform=ax.transAxes)
    ax.text(0.86, 0.92, 'To ROH', transform=ax.transAxes)
    ax.legend(loc=(0.05, 0.6))


with uproot.open(filename) as f:
    hist_hybrid_0 = f[histname.replace('XX', '0')].to_hist()
    hist_hybrid_1 = f[histname.replace('XX', '1')].to_hist()

# Create artificial broken wirebonds
hist_hybrid_1[450] = [8.0, 1]
hist_hybrid_0[300] = [7.0, 1]

fig0, ax0 = plt.subplots(figsize=(10,5))
fig1, ax1 = plt.subplots(figsize=(10,5))

make_noise_plot(ax0, hist_hybrid_0, filename.stem, 0)
make_noise_plot(ax1, hist_hybrid_1, filename.stem, 1)


Path(destination_folder, filename.stem).mkdir(parents=True, exist_ok=True)
# Copy index.php to destination folders (overwrite ok if already exists)
shutil.copy(Path(storage_path, 'index.php'), Path(destination_folder))
shutil.copy(Path(storage_path, 'index.php'), Path(destination_folder, filename.stem))

print('Outputting plots to path', Path(destination_folder, filename.stem))
fig0.savefig(Path(destination_folder, filename.stem, 'Hybrid0.png'))
fig1.savefig(Path(destination_folder, filename.stem, 'Hybrid1.png'))
fig0.savefig(Path(destination_folder, filename.stem, 'Hybrid0.pdf'))
fig1.savefig(Path(destination_folder, filename.stem, 'Hybrid1.pdf'))

