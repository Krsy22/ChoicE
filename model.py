import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from CustomTransformer import *
MIN_NORM = 1e-15
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}



class Artan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        dtype = x.dtype
        x = x.double()
        return x.atan().to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 + input ** 2)
class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        dtype = x.dtype
        x = x.double()
        return (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5).to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)

def artan(x):
    return Artan.apply(x)

def tan(x):
    return x.clamp(-15, 15).tan()

def artanh(x):
    return Artanh.apply(x)


def tanh(x):
    return x.clamp(-15, 15).tanh()


def expmap0(u, c):
    sqrt_c = c ** 0.5
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return project(gamma_1, c)


def logmap0(y, c):
    sqrt_c = c ** 0.5
    y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)


def expmap1(u, c):
    sqrt_c = c ** 0.5
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    gamma_1 = tan(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return project(gamma_1, c)


def logmap1(y, c):
    sqrt_c = c ** 0.5
    y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    return y / y_norm / sqrt_c * artan(sqrt_c * y_norm)

def project(x, c):
    norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    eps = BALL_EPS[x.dtype]
    maxnorm = (1 - eps) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def mobius_add(x, y, c):
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / denom.clamp_min(MIN_NORM)
def givens_rotations(r, x):
    givens = r.view((r.shape[0], -1, 2))
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
    x = x.view((r.shape[0], -1, 2))
    x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_rot.view((r.shape[0], -1))
def hyp_distance_multi_c(x, v, c):
   
    sqrt_c = c ** 0.5
   
    vnorm = torch.norm(v, p=2, dim=-1, keepdim=True).transpose(0, 1)
    xv = x @ v.transpose(0, 1) / vnorm
    
    gamma = tanh(sqrt_c * vnorm) / sqrt_c
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    c1 = 1 - 2 * c * gamma * xv + c * gamma ** 2
    c2 = 1 - c * x2
    num = torch.sqrt((c1 ** 2) * x2 + (c2 ** 2) * (gamma ** 2) - (2 * c1 * c2) * gamma * xv)
    denom = 1 - 2 * c * gamma * xv + (c ** 2) * (gamma ** 2) * x2
    pairwise_norm = num / denom.clamp_min(MIN_NORM)
    dist = artanh(sqrt_c * pairwise_norm)
    
    return 2 * dist / sqrt_c
class Decoder(nn.Module):
    def __init__(self, dim_str, num_rel, num_experts=3):
        super(Decoder, self).__init__()
        self.gate = nn.Linear(dim_str*2, num_experts)
        self.dim_str = dim_str
        self.ent_to_complex = nn.Linear(dim_str, dim_str)
        self.ent_to_real = nn.Linear(dim_str, dim_str)
        self.rel_to_complex = nn.Linear(dim_str, dim_str)
        self.rel_to_real = nn.Linear(dim_str, dim_str)
        self.rank = dim_str // 2
        self.rel_diag = nn.Embedding(num_rel, self.rank).cuda() 
        self.rel_diag.weight.data = 2 * torch.rand((num_rel, self.rank), dtype=torch.float) - 1.0 
        self.c_init = torch.ones((num_rel, 1), dtype=torch.float)
        self.c = nn.Parameter(self.c_init, requires_grad=True)
        self.ent_to_hyper = nn.Linear(dim_str, dim_str // 2)
        self.rel_to_hyper = nn.Linear(dim_str, dim_str)
    def complex(self, ent_embs, rel_embs, triplets):
        
        ent_embs = self.ent_to_complex(ent_embs)
        rel_embs = self.rel_to_complex(rel_embs)
        lhs = ent_embs[triplets[:, 0]]
        rel = rel_embs[triplets[:, 1]]
        rhs = ent_embs[triplets[:, 2]]
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        output_dec_rhs = torch.cat([
                lhs[0] * rel[0] - lhs[1] * rel[1],
                lhs[0] * rel[1] + lhs[1] * rel[0]
            ], 1)
        output_dec_rel = torch.cat([
                lhs[0] * rhs[0] + lhs[1] * rhs[1],
                lhs[0] * rhs[1] - lhs[1] * rhs[0]
            ], 1)

        rhs_scores = torch.inner(output_dec_rhs, ent_embs)
        rel_scores = torch.inner(output_dec_rel, rel_embs)
        return rhs_scores

    def cp(self, ent_embs, rel_embs, triplets):
        ent_embs = self.ent_to_real(ent_embs)
        rel_embs = self.rel_to_real(rel_embs)
        lhs = ent_embs[triplets[:, 0]]
        rel = rel_embs[triplets[:, 1]]
        rhs = ent_embs[triplets[:, 2]]
        rhs_scores = (lhs * rel) @ ent_embs.t()
        rel_scores = (lhs * rhs) @ rel_embs.t()
        return rhs_scores
    def RotH(self, ent_embs, rel_embs, triplets):
        ent_embs = 0.01 * ent_embs
        ent_embs = self.ent_to_hyper(ent_embs)
        rel_embs = 0.01 * rel_embs
        rel_embs = self.rel_to_hyper(rel_embs)
        lhs = ent_embs[triplets[:, 0]]
        rel = rel_embs[triplets[:, 1]]
        rhs = ent_embs[triplets[:, 2]]
        c = F.softplus(self.c[triplets[:, 1]])
        head = expmap0(lhs, c)
        rel1, rel2 = torch.chunk(rel, 2, dim=1)
        rel1 = expmap0(rel1, c)
        rel2 = expmap0(rel2, c)
        lhs = project(mobius_add(head, rel1, c), c)
        res1 = givens_rotations(self.rel_diag(triplets[:, 1]), lhs)
        res2 = mobius_add(res1, rel2, c)
        scores = - hyp_distance_multi_c(res2, ent_embs, c) ** 2
        return scores

    def forward(self, ent_embs, rel_embs, triplets):
        decoder_experts = [
            self.complex,
            self.cp,
            self.RotH
        ]
        lhs = ent_embs[triplets[:, 0]]
        rel = rel_embs[triplets[:, 1]]
        combine = torch.cat([lhs,rel],dim=1)
        gate_output = self.gate(combine)
        gate_probs = F.gumbel_softmax(gate_output, tau=1.0, hard=True, dim=-1) #bsz num_experts
        #加入辅助损失
        expert_importance = gate_probs.sum(dim=0)
        mean_importance = expert_importance.mean()
        std_importance = expert_importance.std()
        cv_squared = (std_importance / mean_importance) ** 2 if mean_importance != 0 else 0

    
        wImportance = 1.0  
        aux_loss = wImportance * cv_squared 

#    # 统计选择次数
#        num_experts = len(decoder_experts)
#        expert_counts = torch.bincount(selected_experts, minlength=num_experts)  # [num_experts]
#    
#        print(f"Decoder selection counts: {expert_counts.tolist()}")  # 输出每个解码器的选择次数
        #print(gate_probs)      
        expert_outputs = torch.stack([expert(ent_embs, rel_embs, triplets) for expert in decoder_experts], dim=1) #bsz num_experts num_entities
        output = torch.bmm(gate_probs.unsqueeze(1), expert_outputs).squeeze(1) #bsz 1 num_experts 
        return output, aux_loss



class VISTA(nn.Module):
    def __init__(self, num_ent, num_rel, ent_vis, rel_vis, dim_vis, ent_txt, rel_txt, dim_txt, ent_vis_mask, rel_vis_mask, \
                 dim_str, num_head, dim_hid, num_layer_enc_ent, num_layer_enc_rel, num_layer_dec, dropout = 0.1, \
                 emb_dropout = 0.6, vis_dropout = 0.1, txt_dropout = 0.1):
        super(VISTA, self).__init__()
        self.dim_str = dim_str
        self.num_head = num_head
        self.dim_hid = dim_hid
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.rank = dim_str // 2
        
        self.rel_diag = nn.Embedding(num_rel, self.rank).cuda() 
        self.rel_diag.weight.data = 2 * torch.rand((num_rel, self.rank), dtype=torch.float) - 1.0 
        self.c_init = torch.ones((num_rel, 1), dtype=torch.float)
        self.c = nn.Parameter(self.c_init, requires_grad=True)
        self.ent_to_hyper = nn.Linear(dim_str, dim_str // 2)
        self.rel_to_hyper = nn.Linear(dim_str, dim_str)
        self.ent_vis = ent_vis
        self.rel_vis = rel_vis
        self.ent_txt = ent_txt.unsqueeze(dim = 1)
        self.rel_txt = rel_txt.unsqueeze(dim = 1)
        self.rel_diag = nn.Embedding(num_rel, self.rank).cuda() 
        self.rel_diag.weight.data = 2 * torch.rand((num_rel, self.rank), dtype=torch.float) - 1.0 
        self.c_init = torch.ones((num_rel, 1), dtype=torch.float)
        self.c = nn.Parameter(self.c_init, requires_grad=True)
        self.ent_to_hyper = nn.Linear(self.dim_str, self.rank)
        self.rel_to_hyper = nn.Linear(self.dim_str, self.dim_str)
        false_ents = torch.full((self.num_ent,1),False).cuda()
        self.ent_mask = torch.cat([false_ents, false_ents, ent_vis_mask, false_ents], dim = 1)
        false_rels = torch.full((self.num_rel,1),False).cuda()
        self.rel_mask = torch.cat([false_rels, false_rels, rel_vis_mask, false_rels], dim = 1)

        self.ent_token = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.rel_token = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.ent_embeddings = nn.Parameter(torch.Tensor(num_ent, 1, dim_str))
        self.rel_embeddings = nn.Parameter(torch.Tensor(num_rel, 1 ,dim_str))
        self.lp_token = nn.Parameter(torch.Tensor(1,dim_str))

        self.str_ent_ln = nn.LayerNorm(dim_str)
        self.str_rel_ln = nn.LayerNorm(dim_str)
        self.vis_ln = nn.LayerNorm(dim_str)
        self.txt_ln = nn.LayerNorm(dim_str)

        self.embdr = nn.Dropout(p = emb_dropout)
        self.visdr = nn.Dropout(p = vis_dropout)
        self.txtdr = nn.Dropout(p = txt_dropout)


        self.pos_str_ent = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_vis_ent = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_txt_ent = nn.Parameter(torch.Tensor(1,1,dim_str))

        self.pos_str_rel = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_vis_rel = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_txt_rel = nn.Parameter(torch.Tensor(1,1,dim_str))

        self.pos_head = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_rel = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_tail = nn.Parameter(torch.Tensor(1,1,dim_str))
        
        self.proj_ent_vis = nn.Linear(dim_vis, dim_str)
        self.proj_txt = nn.Linear(dim_txt, dim_str)

        self.proj_rel_vis = nn.Linear(dim_vis * 3, dim_str)


        
        ent_encoder_layer1 = CustomTransformerEncoderLayer(dim_str, num_head, 3, dim_hid, dropout) 
        
        self.ent_encoder2 = CustomTransformerEncoder(ent_encoder_layer1, num_layer_enc_ent)
       
        rel_encoder_layer1 = CustomTransformerEncoderLayer(dim_str, num_head, 3, dim_hid, dropout)
        
        self.rel_encoder2 = CustomTransformerEncoder(rel_encoder_layer1, num_layer_enc_rel)
        
        self.decoder = Decoder(dim_str, num_rel, 3)
        self.init_weights()
        

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings)
        nn.init.xavier_uniform_(self.rel_embeddings)
        nn.init.xavier_uniform_(self.proj_ent_vis.weight)
        nn.init.xavier_uniform_(self.proj_rel_vis.weight)
        nn.init.xavier_uniform_(self.proj_txt.weight)
        
        
        nn.init.xavier_uniform_(self.ent_token)
        nn.init.xavier_uniform_(self.rel_token)
        nn.init.xavier_uniform_(self.lp_token)
        nn.init.xavier_uniform_(self.pos_str_ent)
        nn.init.xavier_uniform_(self.pos_vis_ent)
        nn.init.xavier_uniform_(self.pos_txt_ent)
        nn.init.xavier_uniform_(self.pos_str_rel)
        nn.init.xavier_uniform_(self.pos_vis_rel)
        nn.init.xavier_uniform_(self.pos_txt_rel)
        nn.init.xavier_uniform_(self.pos_head)
        nn.init.xavier_uniform_(self.pos_rel)
        nn.init.xavier_uniform_(self.pos_tail)

        self.proj_ent_vis.bias.data.zero_()
        self.proj_rel_vis.bias.data.zero_()
        self.proj_txt.bias.data.zero_()
    def visualize_embeddings(self, decoder, ent_embs):
        ent_embs_complex = decoder.ent_to_complex(ent_embs).detach().cpu().numpy()
        ent_embs_real = decoder.ent_to_real(ent_embs).detach().cpu().numpy()
        ent_embs_hyper = decoder.ent_to_hyper(0.01*ent_embs).cuda()
        linear = nn.Linear(106, 212).cuda()
        ent_embs_hyper = linear(ent_embs_hyper).detach().cpu().numpy()
        embeddings = np.vstack([ent_embs_complex, ent_embs_real, ent_embs_hyper])
        labels = np.array([0] * len(ent_embs_complex) + [1] * len(ent_embs_real) + [2] * len(ent_embs_hyper))  # 0=complex, 1=real, 2=hyper
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.8)
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)  
        plt.legend(handles=scatter.legend_elements()[0], labels=["Euclidean", "Complex", "Hyperbolic"])
        plt.title("t-SNE Visualization of Expert Mapped Embeddings")
        plt.xlabel("t-SNE Dim 1")
        plt.ylabel("t-SNE Dim 2")
        plt.savefig("/home/rsy/VISTA-main/figure/embedding.png", dpi=300, bbox_inches='tight') 
    def forward(self):
        ent_tkn = self.ent_token.tile(self.num_ent, 1, 1)
        rep_ent_str = self.embdr(self.str_ent_ln(self.ent_embeddings)) + self.pos_str_ent
        rep_ent_vis = self.visdr(self.vis_ln(self.proj_ent_vis(self.ent_vis))) + self.pos_vis_ent
        rep_ent_txt = self.txtdr(self.txt_ln(self.proj_txt(self.ent_txt))) + self.pos_txt_ent
        ent_seq = torch.cat([ent_tkn, rep_ent_str, rep_ent_vis, rep_ent_txt], dim = 1)
        ent_embs = self.ent_encoder2(ent_seq, src_key_padding_mask = self.ent_mask)[:,0]
        rel_tkn = self.rel_token.tile(self.num_rel, 1, 1)
        rep_rel_str = self.embdr(self.str_rel_ln(self.rel_embeddings)) + self.pos_str_rel
        rep_rel_vis = self.visdr(self.vis_ln(self.proj_rel_vis(self.rel_vis))) + self.pos_vis_rel
        rep_rel_txt = self.txtdr(self.txt_ln(self.proj_txt(self.rel_txt))) + self.pos_txt_rel
        rel_seq = torch.cat([rel_tkn, rep_rel_str, rep_rel_vis, rep_rel_txt], dim = 1)
        rel_embs = self.rel_encoder2(rel_seq, src_key_padding_mask = self.rel_mask)[:,0]
        return torch.cat([ent_embs, self.lp_token], dim = 0), rel_embs
    def cp(self, lhs, rel, rhs, ent_embs, rel_embs):
        rhs_scores = (lhs * rel) @ ent_embs.t()
        rel_scores = (lhs * rhs) @ rel_embs.t()
        return rhs_scores, rel_scores
    def complex(self, lhs, rel, rhs, ent_embs, rel_embs):
        rank = self.dim_str // 2
        lhs = lhs[:, :rank], lhs[:, rank:]
        rel = rel[:, :rank], rel[:, rank:]
        rhs = rhs[:, :rank], rhs[:, rank:]
        output_dec_rhs = torch.cat([
                lhs[0] * rel[0] - lhs[1] * rel[1],
                lhs[0] * rel[1] + lhs[1] * rel[0]
            ], 1)
        # [1000,2000]
        output_dec_rel = torch.cat([
                lhs[0] * rhs[0] + lhs[1] * rhs[1],
                lhs[0] * rhs[1] - lhs[1] * rhs[0]
            ], 1)

        rhs_scores = torch.inner(output_dec_rhs, ent_embs)
        rel_scores = torch.inner(output_dec_rel, rel_embs)
        return rhs_scores, rel_scores
    def RotH(self, ent_embs, rel_embs, triplets):
        
        ent_embs = 0.01 * ent_embs
        ent_embs = self.ent_to_hyper(ent_embs)

        rel_embs = 0.01 * rel_embs
        rel_embs = self.rel_to_hyper(rel_embs)

        lhs = ent_embs[triplets[:, 0]]
        rel = rel_embs[triplets[:, 1]]
        rhs = ent_embs[triplets[:, 2]]
        c = F.softplus(self.c[triplets[:, 1]])
        head = expmap0(lhs, c)
        rel1, rel2 = torch.chunk(rel, 2, dim=1)
        rel1 = expmap0(rel1, c)
        rel2 = expmap0(rel2, c)
        lhs = project(mobius_add(head, rel1, c), c)
        res1 = givens_rotations(self.rel_diag(triplets[:, 1]), lhs)
        res2 = mobius_add(res1, rel2, c)
        scores = - hyp_distance_multi_c(res2, ent_embs, c) ** 2
        return scores
    def score(self, emb_ent, emb_rel, triplets):
        lhs = emb_ent[triplets[:, 0]]
        rel = emb_rel[triplets[:, 1]]
        rhs = emb_ent[triplets[:, 2]]
        scores, aux_loss = self.decoder(emb_ent, emb_rel, triplets)
        return scores, aux_loss