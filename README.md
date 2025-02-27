<h1 align="center">
  ChoicE
</h1>
<h4 align="center">Make Your Choice for Multimodal Knowledge Graph Completion</h4>

<h2 align="center">
  Overview of ChoicE
  <img align="center"  src="overview.png" alt="...">
</h2>

This paper has been submitted to the Knowledge-based Systems.

### Dependencies

- python            3.7
- torch             1.21.1
- numpy             1.21.5
- scikit-learn      1.0.2
- scipy             1.7.3


### Dataset:

- We use WN18RR++, FB15K237, VTKG-C and VTKG-I dataset for knowledge graph completion. 
- You should first download the preprocessed data from https://drive.google.com/file/d/1u4QthmEboMzRarF_HLYfLDOLcOZeH8Gp/view?usp=drive_link and put them in the `./data` directory.

### Results:
The results are:

|  Dataset   |   MRR  |   H@1  |   H@3  |  H@10  |
| :--------: | :---:  |  :---: |  :---: |  :---: |
|  WN18RR++  | 0.5856 | 0.5316 | 0.6053 | 0.7014 |
| FB15K237 | 0.4504 | 0.3570 | 0.4903 | 0.6363 |
|  VTKG-I   | 0.4792 | 0.4055 | 0.5133 | 0.6869 |
|  VTKG-C   | 0.4756 | 0.4065 | 0.5021 | 0.6139 |
## How to Run
```
python train.py --data WN18RR++ --lr 0.001 --dim 256 --num_epoch 750 --valid_epoch 50 --exp best --num_layer_enc_ent 2 --num_layer_enc_rel 1  --num_head 16 --hidden_dim 2048 --dropout 0.1 --emb_dropout 0.9 --vis_dropout 0.3 --txt_dropout 0.1   --smoothing 0.0 --batch_size 1024 --decay 0.0 --max_img_num 1 --step_size 50                 ## WN18RR++ dataset
python train.py --data FB15K237 --lr 0.0002 --dim 256 --num_epoch 150 --valid_epoch 50 --exp best --num_layer_enc_ent 1 --num_layer_enc_rel 1  --num_head 64 --hidden_dim 1488 --dropout 0.04 --emb_dropout 0.68 --vis_dropout 0.03 --txt_dropout 0.32   --smoothing 0.0 --batch_size 512 --decay 0.0 --max_img_num 1 --step_size 50            ## FB15K237 dataset
python train.py --data VTKG-I --lr 0.003 --dim 52 --num_epoch 150 --valid_epoch 5 --exp best --num_layer_enc_ent 2 --num_layer_enc_rel 2  --num_head 4 --hidden_dim 1246 --dropout 0.08 --emb_dropout 0.63 --vis_dropout 0.47 --txt_dropout 0.37   --smoothing 0.0 --batch_size 128 --decay 0.0 --max_img_num 10 --step_size 50               ## VTKG-I dataset
python train.py --data VTKG-C --lr 0.0003 --dim 212 --num_epoch 150 --valid_epoch 50 --exp best --num_layer_enc_ent 2 --num_layer_enc_rel 1  --num_head 4 --hidden_dim 1433 --dropout 0.08 --emb_dropout 0.74 --vis_dropout 0.17 --txt_dropout 0.04   --smoothing 0.0 --batch_size 512 --decay 0.0 --max_img_num 1 --step_size 50        ## VTKG-C dataset
```
