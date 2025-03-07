import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveGatedCAHead(nn.Module):
    def __init__(self, input_dim=64, attention_dim=64, num_heads=4, num_classes=2):
        super(AdaptiveGatedCAHead, self).__init__()
        
        # 定义用于EEG和fMRI交叉注意力的多头注意力层
        self.eeg_to_fmri_attention = nn.MultiheadAttention(embed_dim=8, num_heads=num_heads, batch_first=True)
        self.fmri_to_eeg_attention = nn.MultiheadAttention(embed_dim=8, num_heads=num_heads, batch_first=True)
        
        # 将输入特征投影到注意力维度的线性层
        self.eeg_proj = nn.Linear(input_dim, attention_dim)
        self.fmri_proj = nn.Linear(input_dim, attention_dim)
        
        # 门控网络，用于生成自适应融合权重
        self.gate_network = nn.Sequential(
            nn.Linear(attention_dim * 2, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1),
            nn.Sigmoid()  # 输出权重范围在 0 到 1 之间
        )

        # 输出层，将融合后的特征维度调整为所需的输出维度
        self.output_proj = nn.Linear(attention_dim, num_classes)
        self.fusion_feature = 0
        self.eeg_weight = 0

    def forward(self, eeg_features, fmri_features):
        """
        输入:
            eeg_features: Tensor, 形状为 (batch_size, seq_len, input_dim)
            fmri_features: Tensor, 形状为 (batch_size, seq_len, input_dim)
        输出:
            融合后的特征，形状为 (batch_size, seq_len, attention_dim)
        """
        # 将输入特征投影到注意力维度
        eeg_proj = self.eeg_proj(eeg_features)  # (batch_size, seq_len, attention_dim)
        fmri_proj = self.fmri_proj(fmri_features)  # (batch_size, seq_len, attention_dim)
        eeg_proj = eeg_proj.view(eeg_proj.shape[0], 8, 8)
        fmri_proj = fmri_proj.view(fmri_proj.shape[0], 8, 8)
        
        # 交叉注意力：EEG关注fMRI，得到增强EEG特征
        eeg_enhanced, _ = self.eeg_to_fmri_attention(query=eeg_proj, key=fmri_proj, value=fmri_proj)
        
        # 交叉注意力：fMRI关注EEG，得到增强fMRI特征
        fmri_enhanced, _ = self.fmri_to_eeg_attention(query=fmri_proj, key=eeg_proj, value=eeg_proj)
        # 将增强特征拼接，用于计算自适应门控权重
        eeg_enhanced = eeg_enhanced.reshape(eeg_enhanced.shape[0], -1)
        fmri_enhanced = fmri_enhanced.reshape(fmri_enhanced.shape[0], -1)
        combined_features = torch.cat((eeg_enhanced, fmri_enhanced), dim=-1)  # (batch_size, attention_dim * 2)
        # 通过门控网络生成融合权重
        gate_weight = self.gate_network(combined_features)  # (batch_size, seq_len, 1)
        
        # 使用门控权重融合EEG和fMRI的增强特征
        fused_features = gate_weight * eeg_enhanced + (1 - gate_weight) * fmri_enhanced # torch.cat((eeg_enhanced, fmri_enhanced), 1)
        # 消融门控注意力
        # fused_features = eeg_enhanced + fmri_enhanced
        
        # 输出层投影
        output = self.output_proj(fused_features)  # (batch_size, seq_len, attention_dim)

        self.eeg_weight = gate_weight
        self.fusion_feature = fused_features
        
        return output

    def get_weight_feature(self):
        return self.eeg_weight, self.fusion_feature
