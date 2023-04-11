import torch
import torch.nn as nn
from GAT import GraphAttentionLayer

class model(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.GAT_neighbor = GraphAttentionLayer(embed_size, embed_size) # in_features = out_features
        self.GAT_item = GraphAttentionLayer(embed_size, embed_size)
        self.relation_neighbor = nn.Parameter(torch.randn(embed_size))
        self.relation_item = nn.Parameter(torch.randn(embed_size))
        self.relation_self = nn.Parameter(torch.randn(embed_size))
        self.c = nn.Parameter(torch.randn(2 * embed_size))

    def predict(self, user_embedding, item_embedding):
        return torch.matmul(user_embedding, item_embedding.t())


    # model embedding, neibor embedding, item embedding
    def forward(self, feature_self, feature_neighbor, feature_item):
        if type(feature_item) == torch.Tensor:
            f_n = self.GAT_neighbor(feature_self, feature_neighbor)
            f_i = self.GAT_item(feature_self, feature_item)
            # Relational Graph Aggregation (4.2.2)
            e_n = torch.matmul(self.c, torch.cat((f_n, self.relation_neighbor)))
            e_i = torch.matmul(self.c, torch.cat((f_i, self.relation_item)))
            e_s = torch.matmul(self.c, torch.cat((feature_self, self.relation_self)))
            m = nn.Softmax(dim = -1)
            e_tensor = torch.stack([e_n, e_i, e_s])
            e_tensor = m(e_tensor)
            r_n, r_i, r_s = e_tensor
            # Local User Embedding (4.2.2)
            user_embedding = r_s * feature_self + r_n * f_n + r_i * f_i
        else:
            f_n = self.GAT_neighbor(feature_self, feature_neighbor)
            e_n = torch.matmul(self.c, torch.cat((f_n, self.relation_neighbor)))
            e_s = torch.matmul(self.c, torch.cat((feature_self, self.relation_self)))
            m = nn.Softmax(dim = -1)
            e_tensor = torch.stack([e_n, e_s])
            e_tensor = m(e_tensor)
            r_n, r_s = e_tensor
            user_embedding = r_s * feature_self + r_n * f_n

        return user_embedding


"""
该文件是一个PyTorch的模型定义文件，实现了一个基于图卷积神经网络（GCN）的推荐系统模型。该模型包括两个图注意力层（GraphAttentionLayer）和若干个可学习的参数（relation_neighbor、relation_item、relation_self、c）。该模型输入包括三种节点特征（feature_self、feature_neighbor、feature_item），其中feature_self表示目标节点自身的特征，feature_neighbor表示目标节点的邻居节点的特征，feature_item表示与目标节点有交互行为的其他物品的特征。其输出为用户节点的嵌入向量（user_embedding），用于推荐系统中的用户匹配和推荐项排序。该模型中使用了softmax函数进行权重归一化操作。
"""
