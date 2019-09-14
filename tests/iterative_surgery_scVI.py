import argparse
import os

import scanpy as sc

import surgeon

DATASETS = {
    "pancreas": {"name": "pancreas_hvg", "batch_key": "study", "cell_type_key": "cell_type",
                 "source": ['Pancreas inDrop', 'Pancreas Celseq']},
    "toy": {"name": "toy", "batch_key": "batch", "cell_type_key": "celltype", "source": ["Batch1", "Batch2"]},
    "pbmc": {"name": "pbmc", "batch_key": "study", "cell_type_key": "cell_type", "source": ["inDrops", "Drop-seq"]},
}


def train_and_evaluate(data_dict, freeze=True, count_adata=True):
    data_name = data_dict['name']
    cell_type_key = data_dict['cell_type_key']
    batch_key = data_dict['batch_key']
    source_conditions = data_dict['source']

    path_to_save = f"./results/iterative_surgery/{data_name}/"
    sc.settings.figdir = path_to_save
    os.makedirs(path_to_save, exist_ok=True)

    if count_adata:
        adata = sc.read(f"./data/{data_name}/{data_name}_count.h5ad")
        loss_fn = "nb"
    else:
        adata = sc.read(f"./data/{data_name}/{data_name}_normalized.h5ad")
        loss_fn = "mse"

    adata_for_training = adata[adata.obs[batch_key].isin(source_conditions)]
    other_batches = [batch for batch in adata.obs[batch_key].unique().tolist() if not batch in source_conditions]
    adata_for_training = surgeon.utils.normalize(adata_for_training,
                                                 filter_min_counts=False,
                                                 normalize_input=False,
                                                 size_factors=True,
                                                 logtrans_input=True,
                                                 n_top_genes=-1,
                                                 )
    if count_adata:
        clip_value = 5.0
    else:
        clip_value = 1e6

    train_adata, valid_adata = surgeon.utils.train_test_split(adata_for_training, 0.85)
    n_conditions = len(train_adata.obs[batch_key].unique().tolist())

    network = surgeon.archs.scVI(x_dimension=train_adata.shape[1],
                                 z_dimension=50,
                                 architecture=[512, 256],
                                 n_conditions=n_conditions,
                                 lr=0.001,
                                 alpha=0.000005,
                                 use_batchnorm=True,
                                 scale_factor=1.0,
                                 clip_value=clip_value,
                                 loss_fn=loss_fn,
                                 model_path=f"./models/CVAE/iterative_surgery/before-{data_name}-{loss_fn}/",
                                 dropout_rate=0.1,
                                 output_activation='relu')

    conditions = adata_for_training.obs[batch_key].unique().tolist()
    condition_encoder = surgeon.utils.create_dictionary(conditions, [])

    network.train(train_adata,
                  valid_adata,
                  condition_key=batch_key,
                  cell_type_key=cell_type_key,
                  le=condition_encoder,
                  n_epochs=10000,
                  batch_size=128,
                  early_stop_limit=150,
                  lr_reducer=120,
                  n_per_epoch=0,
                  save=True,
                  retrain=True,
                  verbose=2)

    encoder_labels, _ = surgeon.utils.label_encoder(adata_for_training, label_encoder=network.condition_encoder,
                                                    condition_key=batch_key)

    latent_adata = network.to_latent(adata_for_training)

    sc.pp.neighbors(latent_adata)
    sc.tl.umap(latent_adata)
    sc.pl.umap(latent_adata, color=[batch_key, cell_type_key], wspace=0.7, frameon=False, title="",
               save="_latent_first.pdf")

    new_network = network
    adata_vis = adata_for_training
    for idx, new_batch in enumerate(other_batches):
        print(f"Operating surgery for {new_batch}")
        batch_adata = adata[adata.obs[batch_key] == new_batch]

        batch_adata = surgeon.tl.normalize(batch_adata,
                                           filter_min_counts=False,
                                           normalize_input=False,
                                           size_factors=True,
                                           logtrans_input=True,
                                           n_top_genes=-1)

        new_network = new_network.operate(
            new_conditions=[new_batch],
            init='Xavier',
            freeze=freeze)

        new_network.model_path = f"./models/CVAE/iterative_surgery/after-({idx}:{new_batch})-{data_name}-{loss_fn}-{freeze}/"

        train_adata, valid_adata = surgeon.utils.train_test_split(batch_adata, 0.80)

        new_network.train(train_adata,
                          valid_adata,
                          condition_key=batch_key,
                          cell_type_key=cell_type_key,
                          le=new_network.condition_encoder,
                          n_epochs=10000,
                          batch_size=128,
                          early_stop_limit=150,
                          lr_reducer=130,
                          n_per_epoch=0,
                          save=True,
                          retrain=True,
                          verbose=2)
        adata_vis = adata_vis.concatenate(batch_adata)

        encoder_labels, _ = surgeon.utils.label_encoder(adata_vis, label_encoder=new_network.condition_encoder,
                                                        condition_key=batch_key)

        latent_adata = new_network.to_latent(adata_vis)

        sc.pp.neighbors(latent_adata)
        sc.tl.umap(latent_adata)
        sc.pl.umap(latent_adata, color=[batch_key, cell_type_key], wspace=0.7, frameon=False, title="",
                   save=f"_latent_({idx}:{new_batch}).pdf")

        adata_vis_old = adata_vis[adata_vis.obs[batch_key] != new_batch]
        encoder_labels, _ = surgeon.utils.label_encoder(adata_vis_old, label_encoder=new_network.condition_encoder,
                                                        condition_key=batch_key)

        latent_adata = new_network.to_latent(adata_vis_old)

        sc.pp.neighbors(latent_adata)
        sc.tl.umap(latent_adata)
        sc.pl.umap(latent_adata, color=[batch_key, cell_type_key], wspace=0.7, frameon=False, title="",
                   save=f"_latent_old_{idx}.pdf")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='scVI-like (Keras version)')
    arguments_group = parser.add_argument_group("Parameters")
    arguments_group.add_argument('-d', '--data', type=str, required=True,
                                 help='data name')
    arguments_group.add_argument('-f', '--freeze', type=int, default=1, required=True,
                                 help='freeze')
    arguments_group.add_argument('-c', '--count', type=int, default=0, required=False,
                                 help='latent space dimension')
    args = vars(parser.parse_args())

    freeze = True if args['freeze'] > 0 else False
    count_adata = True if args['count'] > 0 else False
    data_dict = DATASETS[args['data']]

    train_and_evaluate(data_dict=data_dict, freeze=freeze, count_adata=count_adata)