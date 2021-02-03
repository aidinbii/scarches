import scanpy as sc
import torch
import os
import scarches as sca
from scarches.dataset.trvae.data_handling import remove_sparsity
import matplotlib.pyplot as plt
import numpy as np


def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')


sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)

condition_key = 'study'
cell_type_key = 'cell_type'
target_conditions = []
n_labelled_samples_per_class = 1000


trvae_epochs = 500
surgery_epochs = 500

early_stopping_kwargs = {
    "early_stopping_metric": "val_classifier_loss",
    "threshold": 0,
    "patience": 20,
    "reduce_lr": True,
    "lr_patience": 13,
    "lr_factor": 0.1,
}

adata_all = sc.read(os.path.expanduser(f'~/Documents/benchmarking_datasets/pancreas_normalized.h5ad'))
adata = adata_all.raw.to_adata()
adata = remove_sparsity(adata)
source_adata = adata[~adata.obs[condition_key].isin(target_conditions)]
target_adata = adata[adata.obs[condition_key].isin(target_conditions)]
source_conditions = source_adata.obs[condition_key].unique().tolist()

labeled_ind = []
unlabeled_ind = []
un_labels_r = source_adata.obs[cell_type_key].unique().tolist()
for label in un_labels_r:
    mask = source_adata.obs['cell_type'] == label
    mask = mask.tolist()
    idx = np.where(mask)[0]
    np.random.shuffle(idx)
    labeled_ind += idx[:n_labelled_samples_per_class].tolist()
    unlabeled_ind += idx[n_labelled_samples_per_class:].tolist()

print(len(labeled_ind))
print(len(unlabeled_ind))

tranvae = sca.models.TRANVAE.load(
    dir_path=os.path.expanduser(f'~/Documents/reference_model'),
    adata=source_adata
)

preds, probs = tranvae.classify()
print(np.mean(preds == source_adata.obs.cell_type))
correct_probs = probs[preds == source_adata.obs.cell_type]
incorrect_probs = probs[preds != source_adata.obs.cell_type]
data = [correct_probs,incorrect_probs]

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=True)

ax1.set_title('Default violin plot')
ax1.set_ylabel('Observed values')
ax1.violinplot(data)

ax2.set_title('Customized violin plot')
parts = ax2.violinplot(
        data, showmeans=False, showmedians=False,
        showextrema=False)

for pc in parts['bodies']:
    pc.set_facecolor('#D43F3A')
    pc.set_edgecolor('black')
    pc.set_alpha(1)

# set style for the axes
labels = ['Correct', 'Incorrect']
for ax in [ax1, ax2]:
    set_axis_style(ax, labels)

plt.subplots_adjust(bottom=0.15, wspace=0.05)
plt.show()

preds, probs = tranvae.check_for_unseen()
print(preds)
print(probs)

