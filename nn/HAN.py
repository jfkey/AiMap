# node -> cut(supergate) => gobal delay results.
import torch
import torch.nn as nn


class HierarchicalAttentionNetwork(nn.Module):
    def __init__(
        self,
        node_feature_dim,          #10
        cut_feature_dim,           #61
        cell_feature_dim,          #56
        fusion_dim,               #12
        hidden_dim,                #32
        pdrop,                     # 0.1

    ):
        super(HierarchicalAttentionNetwork, self).__init__()

        # self.node_attention_model = NodeAttention(
        #     node_feature_dim, hidden_dim,
        # )
        #
        # self.cut_attention_model = CutAttention(
        #     4 * (fusion_dim - 3) * (fusion_dim - 3), hidden_dim,
        # )
        self.fusion = FusionModel(cut_feature_dim, cell_feature_dim, fusion_dim, pdrop)
        self.postmlp = nn.Sequential(
            nn.Linear(node_feature_dim + 4 * (fusion_dim - 3) * (fusion_dim - 3), 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear( 2* hidden_dim, 1),
        )

    def forward(self, node_emb, cut_emb, cell_emb):
        node_vecs, node_weights = self.node_attention_model(node_emb)
        fus_emb = self.fusion(cut_emb, cell_emb, fusion_dim)
        fus_vecs, fus_weights = self.cut_attention_model(fus_emb)

        emb = torch.cat((node_vecs, fus_vecs), dim=1)
        output = self.postmlp(emb)
        return output

#  node_feature_dim = 10
#  cut_feature_dim = 61
#  cell_feature_dim = 56  14
class FusionModel(nn.Module):
    def __init__(self, node_feature_dim, cut_feature_dim, cell_feature_dim, fusion_dim, postmlp_dim, pdrop):

        self.fusion_dim = fusion_dim
        super(FusionModel, self).__init__()

        self.attention = AttentionModule(node_feature_dim)

        self.premlp = nn.Sequential(
            nn.Linear(cut_feature_dim + cell_feature_dim, fusion_dim * fusion_dim),
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(pdrop),
            nn.Conv2d(3, 4, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(pdrop),
            nn.Conv2d(4, 2, kernel_size=4, stride=1, padding=1),
            nn.ReLU()
        )
        self.postmlp = nn.Sequential(
            nn.Linear(2 * (fusion_dim - 3) * (fusion_dim - 3) + node_feature_dim, 2 * postmlp_dim),
            nn.ReLU(),
            nn.Linear(2 * postmlp_dim, 1),
        )

    # node_fea: batchx6x28, cut_fea: batchx1x16, cell_fea: batchx1x61
    # batch x 28  batchx1x77 -> batchx12x12  -> 3 CNN batch x k
    # batch x (28+k) MLP
    def forward(self, node_features, cut_features, cell_features):

        if len(node_features) == 2:
            node_features = node_features.unsqueeze(1)
            cut_features = cut_features.unsqueeze(1)
            cell_features = cell_features.unsqueeze(1)
        node_att = self.attention(node_features)

        batch_size = node_features.size(0)

        fused_features = torch.cat((cut_features, cell_features), dim=2)
        fused_features = self.premlp(fused_features)
        fused_features = fused_features.view(batch_size, 1, self.fusion_dim, self.fusion_dim)
        fused_emb = self.cnn(fused_features)
        fused_emb = fused_emb.view(batch_size, -1)
        fused_emb = torch.cat([node_att, fused_emb], dim=-1)
        pre_delay = self.postmlp(fused_emb)
        return pre_delay

    # def forward(self, node_features, cut_features, cell_features):
    #     if len(node_features) == 2:
    #         node_features = node_features.unsqueeze(1)
    #         cut_features = cut_features.unsqueeze(1)
    #         cell_features = cell_features.unsqueeze(1)
    #     batch_size = node_features.size(0)
    #     fused_features = torch.cat((node_features, cut_features, cell_features), dim=2)
    #     fused_features = self.premlp(fused_features)
    #     fused_features = fused_features.view(batch_size, 1, self.fusion_dim, self.fusion_dim)
    #     fused_emb = self.cnn(fused_features)
    #     fused_emb = fused_emb.view(batch_size, -1)
    #     pre_delay = self.postmlp(fused_emb)
    #     return pre_delay




# class Attention(torch.nn.Module):
#     def __init__(self, hidden_size, attention_size):
#         super(Attention, self).__init__()
#         self.hidden_size = hidden_size
#         self.attention_size = attention_size
#         self.weight_matrix = nn.Linear(hidden_size, attention_size)
#         self.init_parameters()
#
#     def init_parameters(self):
#         nn.init.xavier_uniform_(self.weight_matrix.weight)
#
#     def forward(self, embedding):
#         if len(embedding.shape) == 2: embedding = embedding.unsqueeze(1)
#         # batch_size = embedding.size(0)
#         # global_context = torch.mean(
#         #     torch.bmm(embedding, self.weight_matrix.unsqueeze(0).expand(batch_size, -1, -1)), dim=1)
#         # transformed_global = torch.tanh(global_context)
#         # sigmoid_scores = torch.sigmoid(torch.bmm(embedding, transformed_global.view(batch_size, -1, 1)))
#         # representation = torch.bmm(embedding.permute(0, 2, 1), sigmoid_scores)
#
#         batch_size = embedding.size(0)
#         seq_len = embedding.size(1)
#
#         # Compute global context
#         weight_matrix = self.weight_matrix.weight.transpose(0, 1)  # Transpose weight matrix
#         global_context = torch.mean(torch.bmm(embedding, weight_matrix.unsqueeze(0).expand(batch_size, -1, -1)), dim=2)
#
#         transformed_global = torch.tanh(global_context)
#
#         # Compute attention scores
#         sigmoid_scores = torch.sigmoid(torch.bmm(embedding, transformed_global.unsqueeze(2)))
#
#         # Compute representation
#         representation = torch.bmm(embedding.permute(0, 2, 1), sigmoid_scores)
#
#         return representation.squeeze(2)
class AttentionModule(torch.nn.Module): 
    def __init__(self, node_feature_dim):
        super(AttentionModule, self).__init__()
        self.node_feature_dim = node_feature_dim
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.node_feature_dim, self.node_feature_dim))

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):
        if len(embedding) == 2:
            embedding = embedding.unsqueeze(0)
        batch_size = embedding.size(0)
        global_context = torch.mean(torch.matmul(embedding, self.weight_matrix.unsqueeze(0).repeat(batch_size,1,1)), dim=1, keepdim=True)
        transformed_global = torch.tanh(global_context)
        sigmoid_scores = torch.sigmoid(torch.matmul(embedding, torch.transpose(transformed_global,1,2)))
        representation = torch.matmul(torch.transpose(embedding,1,2), sigmoid_scores)
        representation = representation.squeeze(-1)
        return representation




if __name__ == '__main__':
    node_feature_dim = 28
    cut_feature_dim = 16
    cell_feature_dim = 61
    batch_size = 32
    fusion_dim = 12
    postmlp_dim = 32
    pdrop = 0.1
    nodeF =torch.randn(32, 10,  node_feature_dim)
    cutF = torch.randn(32, 1, cut_feature_dim)
    cellF = torch.randn(32, 1, cell_feature_dim)

    fm = FusionModel(node_feature_dim, cut_feature_dim, cell_feature_dim, fusion_dim, postmlp_dim, pdrop)
    a = fm(nodeF, cutF, cellF)
    print(a)

























