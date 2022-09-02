import dgl.data
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 构建一个2层的GNN模型
class GCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        # 实例化GraphConv，in_feats是输入特征的维度，out_feats是输出特征的维度
        #  .. math::
        #       h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ji}}h_j^{(l)}W^{(l)})
        self.conv1 = dglnn.GraphConv(
            in_feats=in_feats, out_feats=hid_feats, norm='both', weight=True, bias=True)
        self.conv2 = dglnn.GraphConv(
            in_feats=hid_feats, out_feats=out_feats, norm='both', weight=True, bias=True)

    def forward(self, graph, inputs):
        # 输入是节点的特征
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
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
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

node_features, node_labels,train_mask, valid_mask, test_mask, n_features, n_labels,graph=load_dataset()

def train():
    model = GCN(in_feats=n_features, hid_feats=100, out_feats=n_labels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    model.train()
    min_epochs = 10
    best_model = None
    min_val_loss = 5
    for epoch in range(100):
        # 使用所有节点(全图)进行前向传播计算
        logits  = model(graph, node_features)
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
    # 计算测试集的准确度
    out=evaluate(model, graph, node_features, node_labels, test_mask)
    print('GCN Best Accuracy:{:.4f}'.format(out))