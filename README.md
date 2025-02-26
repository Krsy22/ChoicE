# ChoicE

## Requirements

We used python 3.7 and PyTorch 1.13.1 with cudatoolkit 11.3.

You can install all requirements with:

```shell
pip install -r requirements.txt
```

## Datasets

You can download the datasets from https://drive.google.com/file/d/1u4QthmEboMzRarF_HLYfLDOLcOZeH8Gp/view?usp=drive_link

To use the datasets, place the unzipped data folder in the same directory with the codes. 

## Reproducing the Reported Results

The commands to reproduce the results in our paper:

### VTKG-I

```python
python train.py --data VTKG-I --lr 0.002615994099959632 --dim 52 --num_epoch 150 --valid_epoch 5 --exp best --num_layer_enc_ent 2 --num_layer_enc_rel 2  --num_head 4 --hidden_dim 1246 --dropout 0.083 --emb_dropout 0.63 --vis_dropout 0.47 --txt_dropout 0.37   --smoothing 0.0 --batch_size 128 --decay 0.0 --max_img_num 10 --step_size 50
```

### VTKG-C

```python
python train.py --data VTKG-C --lr 0.0003045227886325736 --dim 212 --num_epoch 150 --valid_epoch 50 --exp best --num_layer_enc_ent 2 --num_layer_enc_rel 1  --num_head 4 --hidden_dim 1433 --dropout 0.08674025072489311 --emb_dropout 0.7432796345246652 --vis_dropout 0.17099615442602828 --txt_dropout 0.04419803944359779   --smoothing 0.0 --batch_size 512 --decay 0.0 --max_img_num 1 --step_size 50
```

### WN18RR++

```python
python train.py --data WN18RR++ --lr 0.0010213298 --dim 256 --num_epoch 750 --valid_epoch 50 --exp best --num_layer_enc_ent 2 --num_layer_enc_rel 1  --num_head 16 --hidden_dim 2048 --dropout 0.1 --emb_dropout 0.9 --vis_dropout 0.3 --txt_dropout 0.1   --smoothing 0.0 --batch_size 1024 --decay 0.0 --max_img_num 1 --step_size 50
```

### FB15K237

```python
python train.py --data FB15K237 --lr 0.000256938979232663 --dim 256 --num_epoch 150 --valid_epoch 50 --exp best --num_layer_enc_ent 1 --num_layer_enc_rel 1  --num_head 64 --hidden_dim 1488 --dropout 0.04120258145778738 --emb_dropout 0.6786708590179708 --vis_dropout 0.029641101227252553 --txt_dropout 0.3208812396423554   --smoothing 0.0 --batch_size 512 --decay 0.0 --max_img_num 1 --step_size 50
```

## Training from Scratch

To train ChoicE from scratch, run `train.py` with arguments. Please refer to `train.py` or `test.py` for the examples of the arguments.

The list of arguments of 'train.py':
- `--data`: name of the dataset
- `--lr`: learning rate
- `--dim`: $d$
- `--num_epoch`: total number of training epochs (only used for `train.py`)
- `--test_epoch`: the epoch to test (only used for `test.py`)
- `--valid_epoch`: the duration of validation
- `--exp`: experiment name
- `--num_layer_enc_ent`: number of the entity encoder layer
- `--num_layer_enc_rel`: number of the relation encoder layer
- `--num_head`: number of attention heads
- `--hidden_dim`: the hidden dimension of the transformers
- `--dropout`: the dropout rate of the transformers
- `--emb_dropout`: the dropout rate of the embedding matrices
- `--vis_dropout`: the dropout rate of the visual representation vectors
- `--txt_dropout`: the dropout rate of the textual representation vectors
- `--smoothing`: label smoothing ratio
- `--batch_size`: the batch size
- `--decay`: the weight decay
- `--max_img_num`: number of visual features for entities and relations
- `--step_size`: the step size of the cosine annealing learning rate scheduler
