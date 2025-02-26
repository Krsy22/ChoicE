# ChoicE

## Requirements

We used python 3.8 and PyTorch 1.12.1 with cudatoolkit 11.3.

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
bash test_VTKG-I.sh
```

### VTKG-C

```python
bash test_VTKG-C.sh
```

### WN18RR++

```python
bash test_WN18RR++.sh
```

### FB15K237

```python
bash test_FB15K237.sh
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
- `--num_layer_enc_ent`: $L$
- `--num_layer_enc_rel`: $\widehat{L}$
- `--num_head`: number of attention heads
- `--hidden_dim`: the hidden dimension of the transformers
- `--dropout`: the dropout rate of the transformers
- `--emb_dropout`: the dropout rate of the embedding matrices
- `--vis_dropout`: the dropout rate of the visual representation vectors
- `--txt_dropout`: the dropout rate of the textual representation vectors
- `--smoothing`: label smoothing ratio
- `--batch_size`: the batch size
- `--decay`: the weight decay
- `--max_img_num`: $k=\hat{k}$
- `--step_size`: the step size of the cosine annealing learning rate scheduler
