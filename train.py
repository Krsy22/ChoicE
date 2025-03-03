from dataset import VTKG
from model import ChoicE
from tqdm import tqdm
from utils import calculate_rank, metrics
import numpy as np
import argparse
import torch
import torch.nn as nn
import datetime
import time
import os
import copy
import math
import random
import distutils
import logging
import optuna
OMP_NUM_THREADS=8
torch.backends.cudnn.benchmark = True
torch.set_num_threads(8)
torch.cuda.empty_cache()

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_format)
logger.addHandler(stream_handler)


parser = argparse.ArgumentParser()
parser.add_argument('--data', default = "FB15K237", type = str)
parser.add_argument('--lr', default=0.000256938979232663, type=float)
parser.add_argument('--dim', default=256, type=int)
parser.add_argument('--num_epoch', default=150, type=int)
parser.add_argument('--valid_epoch', default=50, type=int)
parser.add_argument('--exp', default='best')
parser.add_argument('--no_write', action='store_true')
parser.add_argument('--num_layer_enc_ent', default=1, type=int)
parser.add_argument('--num_layer_enc_rel', default=1, type=int)
parser.add_argument('--num_head', default=64, type=int)
parser.add_argument('--hidden_dim', default = 1488, type = int)
parser.add_argument('--dropout', default = 0.04120258145778738, type = float)
parser.add_argument('--emb_dropout', default = 0.6786708590179708, type = float)
parser.add_argument('--vis_dropout', default = 0.029641101227252553, type = float)
parser.add_argument('--txt_dropout', default = 0.3208812396423554, type = float)
parser.add_argument('--smoothing', default = 0.0, type = float)
parser.add_argument('--batch_size', default = 512, type = int)
parser.add_argument('--decay', default = 0.0, type = float)
parser.add_argument('--max_img_num', default = 1, type = int)
parser.add_argument('--cont', action = 'store_true')
parser.add_argument('--step_size', default = 50, type = int)
args = parser.parse_args()

file_format = "FB15K237"

for arg_name in vars(args).keys():
    if arg_name not in ["data", "exp", "no_write", "num_epoch", "cont", "early_stop"]:
        file_format+=f"_{vars(args)[arg_name]}"



if not args.no_write:
    os.makedirs(f"./result/{args.exp}/{args.data}", exist_ok = True)
    os.makedirs(f"./ckpt/{args.exp}/{args.data}", exist_ok = True)
    os.makedirs(f"./logs/{args.exp}/{args.data}", exist_ok = True)
    if not os.path.isfile(f"ckpt/{args.exp}/args.txt"):
        with open(f"ckpt/{args.exp}/args.txt", "w") as f:
            for arg_name in vars(args).keys():
                if arg_name not in ["data", "exp", "no_write", "num_epoch", "cont", "early_stop"]:
                    f.write(f"{arg_name}\t{type(vars(args)[arg_name])}\n")
else:
    file_format = None
w_rhs = 1.0

file_handler = logging.FileHandler(f"./logs/{args.exp}/{args.data}/{file_format}.log")
file_handler.setFormatter(log_format)
logger.addHandler(file_handler)

logger.info(f"{os.getpid()}")

KG = VTKG(args.data, logger, max_vis_len = args.max_img_num)

KG_Loader = torch.utils.data.DataLoader(KG, batch_size = args.batch_size, shuffle=True)
model = ChoicE(num_ent = KG.num_ent, num_rel = KG.num_rel, ent_vis = KG.ent_vis_matrix, rel_vis = KG.rel_vis_matrix, \
              dim_vis = KG.vis_feat_size, ent_txt = KG.ent_txt_matrix, rel_txt = KG.rel_txt_matrix, dim_txt = KG.txt_feat_size, \
              ent_vis_mask = KG.ent_vis_mask, rel_vis_mask = KG.rel_vis_mask, dim_str = args.dim, num_head = args.num_head, \
              dim_hid = args.hidden_dim, num_layer_enc_ent = args.num_layer_enc_ent, num_layer_enc_rel = args.num_layer_enc_rel, \
              dropout = args.dropout, \
              emb_dropout = args.emb_dropout, vis_dropout = args.vis_dropout, txt_dropout = args.txt_dropout).cuda()

loss_fn = nn.CrossEntropyLoss(label_smoothing = args.smoothing)
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.step_size, T_mult = 2)

last_epoch = 0
if args.cont:
    for ckpt_name in os.listdir(f"./ckpt/{args.exp}/{args.data}/"):
        ckpt_real_name = "_".join(ckpt_name.split("_")[:-1])
        ckpt_epoch = int(ckpt_name.split("_")[-1].split(".")[0])
        if ckpt_real_name == file_format and ckpt_epoch > last_epoch:
            loaded_ckpt = torch.load(f"./ckpt/{args.exp}/{args.data}/{file_format}_{ckpt_epoch}.ckpt")
            model.load_state_dict(loaded_ckpt['model_state_dict'])
            optimizer.load_state_dict(loaded_ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(load_ckpt['scheduler_state_dict'])
            last_epoch = ckpt_epoch

start = time.time()
logger.info("EPOCH\tLOSS\tTOTAL TIME")
all_ents = torch.arange(KG.num_ent).cuda()
all_rels = torch.arange(KG.num_rel).cuda()

best_mrr = 0.0

for epoch in range(last_epoch + 1, args.num_epoch + 1):
    total_loss = 0.0
    #for batch, label in KG_Loader:
    for batch in KG_Loader:


        ent_embs, rel_embs = model()
        rhs_scores, aux_loss = model.score(ent_embs, rel_embs, batch.cuda())
        l_fit = w_rhs * loss_fn(rhs_scores, batch[:, 2].cuda())
        loss = 0.8 * l_fit + 0.2 * aux_loss
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
    scheduler.step()
    logger.info(f"{epoch} \t {total_loss:.6f} \t {time.time() - start:.6f} s")
    if (epoch) % args.valid_epoch == 0:
        model.eval()
        with torch.no_grad():
            
            ent_embs, rel_embs = model()

            lp_list_rank = []
            for triplet in tqdm(KG.valid):
                h,r,t = triplet
                rhs_scores, _ = model.score(ent_embs, rel_embs, torch.tensor([[h, r, t]]).cuda())
                tail_rank = calculate_rank(rhs_scores[0].cpu().numpy(), t, KG.filter_dict[(h, r, -1)])
                lp_list_rank.append(tail_rank)

            lp_list_rank = np.array(lp_list_rank)
            mr, mrr, hit10, hit3, hit1 = metrics(lp_list_rank)
            logger.info("Link Prediction on Validation Set")
            logger.info(f"MR: {mr}")
            logger.info(f"MRR: {mrr}")
            logger.info(f"Hit10: {hit10}")
            logger.info(f"Hit3: {hit3}")
            logger.info(f"Hit1: {hit1}")
            lp_list_rank = []
            for triplet in tqdm(KG.test):
                h,r,t = triplet
                rhs_scores, _ = model.score(ent_embs, rel_embs, torch.tensor([[h, r, t]]).cuda())
                tail_rank = calculate_rank(rhs_scores[0].cpu().numpy(), t, KG.filter_dict[(h, r, -1)])
                lp_list_rank.append(tail_rank)
            lp_list_rank = np.array(lp_list_rank)
            mr, mrr, hit10, hit3, hit1 = metrics(lp_list_rank)
            logger.info("Link Prediction on Test Set")
            logger.info(f"MR: {mr}")
            logger.info(f"MRR: {mrr}")
            logger.info(f"Hit10: {hit10}")
            logger.info(f"Hit3: {hit3}")
            logger.info(f"Hit1: {hit1}")

        if best_mrr < mrr:
            best_mrr = mrr
            patience = 0

        model.train()

        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), \
                    'scheduler_state_dict': scheduler.state_dict()},
                   f"./ckpt/{args.exp}/{args.data}/{file_format}_{epoch}.ckpt")

        model.train()

logger.info("Done!")


