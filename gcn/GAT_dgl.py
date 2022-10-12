import dgl.data
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




# 定义GAT神经层
class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        # 数据
        self.g = g
        # 对应公式中1的 W，用于特征的线性变换
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # 对应公式2中的 a, 输入拼接的zi和zj（2 * out_dim），输出eij（一个数值）
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        # 随机初始化需要学习的参数
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # 对应公式2中的拼接操作，即zi || zj
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        # 拼接之后对应公式2中激活函数里的计算操作，即a(zi || zj)
        a = self.attn_fc(z2)
        # 算出来的值经过leakyReLU激活得到eij,保存在e变量中
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # 汇聚信息，传递之前计算好的z（对应节点的特征） 和 e
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # 对应公式3，eij们经过softmax即可得到特征的权重αij
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # 计算出权重之后即可通过 权重αij * 变换后的特征zj 求和计算出节点更新后的特征
        # 不过激活函数并不在这里，代码后面有用到ELU激活函数
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    # 正向传播方式
    def forward(self, h):
        # 对应公式1，先转换特征
        z = self.fc(h)
        # 将转换好的特征保存在z
        self.g.ndata['z'] = z
        # 对应公式2，得出e
        self.g.apply_edges(self.edge_attention)
        # 对应公式3、4计算出注意力权重α并且得出最后的hi
        self.g.update_all(self.message_func, self.reduce_func)
        # 返回并清除hi
        return self.g.ndata.pop('h')

# 定义多头注意力机制的GAT层
class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        # 多头注意力机制的头数（注意力机制的数量）
        self.heads = nn.ModuleList()
        # 添加对应的注意力机制层，即GAT神经层
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge  # 使用拼接的方法，否则取平均

    def forward(self, h):
        # 获取每套注意力机制得到的hi
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # 每套的hi拼接
            return torch.cat(head_outs, dim=1)
        else:
            # 所有的hi对应元素求平均
            return torch.mean(torch.stack(head_outs))

# 定义GAT模型
class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        # 这里需要注意的是，因为第一层多头注意力机制层layer1选择的是拼接
        # 那么传入第二层的参数应该是第一层的 输出维度 * 头数
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)

    def forward(self, h):
        h = self.layer1(h)
        # ELU激活函数
        h = F.leaky_relu(h)
        h = self.layer2(h)
        return h




def load_dataset(): #加载数据
    """
    train_mask：布尔张量，表示节点是否在训练集中。
    val_mask：布尔张量，表示节点是否在验证集中。
    test_mask：布尔张量，表示节点是否在测试集中。
    label：节点类别。
    feat：节点特征。
    """
    dataset=dgl.data.CitationGraphDataset('cora')
    graph=dataset[0].to(device)#DGL数据集对象可以包含一个或多个图。一般情况下，整图分类任务数据集包含多个图，边预测和节点分类数据集只包含一个图，如节点分类任务中的Cora数据集只包含一个图。
    graph=dgl.add_self_loop(graph) #添加自连接
    node_features = graph.ndata['feat']
    node_labels = graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    valid_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    n_features = node_features.shape[1]
    n_labels = int(node_labels.max().item() + 1)
    return node_features, node_labels,train_mask, valid_mask, test_mask, n_features, n_labels,graph

def evaluate(model, graph, features, labels, mask): #模型评估
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

node_features, node_labels,train_mask, valid_mask, test_mask, n_features, n_labels,graph=load_dataset()

def train():
    model = GAT(graph,in_dim=n_features,hidden_dim=100,out_dim=n_labels,num_heads=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    model.train()
    min_epochs = 10
    best_model = None
    min_val_loss = 5
    for epoch in range(100):
        # 使用所有节点(全图)进行前向传播计算
        logits  = model(node_features)
        # 计算损失值
        loss = loss_function(logits [train_mask], node_labels[train_mask])
        # 进行反向传播计算
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # validation
        val_loss = evaluate(model, graph, node_features, node_labels, valid_mask)
        if epoch + 1 >= min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)
        print('Epoch: {:3d} train_Loss: {:.5f} val_loss: {:.5f}'.format(epoch, loss.item(), val_loss))
    return best_model
if __name__ == '__main__':
    model = train()
    print(model)
    # 计算测试集的准确度
    out=evaluate(model, graph, node_features, node_labels, test_mask)
    print('GAT Best Accuracy:{:.4f}'.format(out))