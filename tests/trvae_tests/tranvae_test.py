import scanpy as sc
import torch
import os
import scarches as sca
from scarches.dataset.trvae.data_handling import remove_sparsity
import matplotlib.pyplot as plt
import numpy as np

sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)

condition_key = 'study'
cell_type_key = 'cell_type'
target_conditions = []
n_labelled_samples_per_class = 2000


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

tranvae = sca.models.TRANVAE(
    adata=source_adata,
    condition_key=condition_key,
    cell_type_key=cell_type_key,
    labeled_indices=labeled_ind,
    hidden_layer_sizes=[128, 128],
    use_mmd=False,
    n_clusters=20
)
tranvae.model.load_state_dict(torch.load(os.path.expanduser(f'~/Documents/reference_model_state_dict')))
tranvae.train(
    n_epochs=trvae_epochs,
    early_stopping_kwargs=early_stopping_kwargs,
    eta=500,
    tau=0,
)
ref_path = os.path.expanduser(f'~/Documents/reference_model')
tranvae.save(ref_path, overwrite=True)

adata_latent = sc.AnnData(tranvae.get_latent())
adata_latent.obs['cell_type'] = source_adata.obs[cell_type_key].tolist()
adata_latent.obs['batch'] = source_adata.obs[condition_key].tolist()

sc.pp.neighbors(adata_latent, n_neighbors=8)
sc.tl.leiden(adata_latent)
sc.tl.umap(adata_latent)
sc.pl.umap(adata_latent,
           color=['batch', 'cell_type'],
           frameon=False,
           wspace=0.6,
           show=False
           )
plt.savefig(os.path.expanduser(f'~/Documents/umap_ref_tranvae.png'), bbox_inches='tight')