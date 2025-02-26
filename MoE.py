import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, num_experts, in_features, out_features):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, in_features)
        ) for _ in range(num_experts)])
        self.gate = nn.Linear(in_features, num_experts)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x = x.view(-1, d_model)
        gate_score = F.softmax(self.gate(x), dim=1)  #bsz*seq_len num_experts          bsz 1 num_experts
#        score = gate_score.view(batch_size, seq_len, -1)
#        score_str = score[:, 0, :]
#        score_vis = score[:, 1, :]
#        score_txt = score[:, 2, :]
#        str_selected_experts = score_str.argmax(dim=-1)
#        vis_selected_experts = score_vis.argmax(dim=-1)
#        txt_selected_experts = score_txt.argmax(dim=-1)
#        str_results = torch.bincount(str_selected_experts, minlength=3)
#        vis_results = torch.bincount(vis_selected_experts, minlength=3)
#        txt_results = torch.bincount(txt_selected_experts, minlength=3)
        
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)      
        output = torch.bmm(gate_score.unsqueeze(1), expert_outputs).squeeze(1)
        output = output.view(batch_size, seq_len, -1) #bsz*seq_len d_model              bsz d_model
        return output


