import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import time
import argparse
from _thread import start_new_thread
from functools import wraps
import tqdm
import traceback


#### Neighbor sampler

class NeighborSampler(object):
    def __init__(self, g, fanouts):
        self.g = g
        self.fanouts = fanouts

    def sample_blocks(self, seeds):
        seeds = th.LongTensor(np.asarray(seeds))  # 这个seed一开始是dataloader里的batchsampler，按照batch大小依次把graph的id一个个yeild出来
        blocks = []
        for fanout in self.fanouts:  # [10,25]
            # For each seed node, sample ``fanout`` neighbors. 这里的sampler是在v0.4.3版本新加入
            frontier = dgl.sampling.sample_neighbors(graph, seeds, fanout, replace=True)  # 利用1000个seeds得到的10000个边？节点
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier,
                                 seeds)  # to_black操作是把将采样的子图转换为适合计算的二部图,这里特殊的地方在于block.srcdata中的id是包含了dstnodeid的
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]
            # 一个种子的长度是1000，就是一个batch的索引,1000个一个batch，采样10个邻居，得到10000边9640个点，再采样25个点，得到241000个边，105693个点，Blocks里面是两个子图
            blocks.insert(0, block)
        return blocks


class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x  # 第一轮输入的x就是采样两次后的二阶相邻点，维度是10w+*602，其实g.ndata的原始标签
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # We need to first copy the representation of nodes on the RHS from the，两层SAGEConv分别对应两个block
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)，blocks0是左9640右10w+的二部图，block1是左1000右9640的二部图
            h_dst = h[:block.number_of_dst_nodes()]  # 每一阶的节点里面都包含了他的dst节点在序列的最前面，方便计算。但是这个是怎么抽样的？dgl.to_blockd函数定义里有说。。
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h = layer(block.to(th.device('cuda:0')), (h, h_dst))  # block是dglnn.SAGEConv().forward(graph,feat)中的graph,feat=(h,h_dst)，h是10w的起始节点特征，h_dst是目的节点的特征
            if l != len(
                    self.layers) - 1:  # 当汇聚方式是mean的时候，SAGEConv实现了，把h的所有特征发送到dst节点，根据dst节点求平均，加上dst节点的原始特征，接一个fc层输出dst节点的新的特征，如果是gcn的话，其实跟mean基本一样，具体可以看dglnn.SAGEConv里面的4种聚合函数的定义，所谓定义graphsage需要学习的参数权重就是这里SAGEConv内部的权重，比如这里602的维度转换为41的权重。
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        nodes = th.arange(g.number_of_nodes())
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.number_of_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            for start in tqdm.trange(0, len(nodes), batch_size):
                end = start + batch_size
                batch_nodes = nodes[start:end]
                block = dgl.to_block(dgl.in_subgraph(g, batch_nodes), batch_nodes)
                input_nodes = block.srcdata[dgl.NID]

                h = x[input_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                h = layer(block.to(th.device('cuda:0')), (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[start:end] = h.cpu()

            x = y
        return y


def prepare_mp(g):
    """
    Explicitly materialize the CSR, CSC and COO representation of the given graph
    so that they could be shared via copy-on-write to sampler workers and GPU
    trainers.

    This is a workaround before full shared memory support on heterogeneous graphs.
    """
    g.in_degree(0)
    g.out_degree(0)
    g.find_edges([0])


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, inputs, labels, val_mask, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_mask], labels[val_mask])


def load_subtensor(g, labels, seeds, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = g.ndata['feat'][input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels


#### Entry point
def run(args, device, data):
    # Unpack data, in_feats=602 ,nodes=232965 ,edges=114848857,n_classes=41，train_nid 13w训练样本的id
    train_mask, val_mask, in_feats, labels, n_classes, g = data
    train_nid = th.LongTensor(np.nonzero(train_mask)[0])  # np.nonzeros()返回元组(分别描述非0元素的位置二维)
    val_nid = th.LongTensor(np.nonzero(val_mask)[0])
    train_mask = th.BoolTensor(train_mask)
    val_mask = th.BoolTensor(val_mask)

    # Create sampler初始化，默认的fanout是10,25，这个的意思是一阶抽10倍，2阶抽25倍
    sampler = NeighborSampler(g, [int(fanout) for fanout in args.fan_out.split(',')])

    # Create PyTorch DataLoader for constructing blocks,train—id是15w的数据索引，batch=1000，sampler抽样器，
    dataloader = DataLoader(
        dataset=train_nid.numpy(),
        batch_size=args.batch_size,
        collate_fn=sampler.sample_blocks,  # 样本不能被batch整除时，需要的处理函数，这里其实是对1000个种子id做抽样，返回block二部图的方法
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    # Define model and optimizer ，输入维度602，隐层16,n_classes =41
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    avg = 0
    iter_tput = []
    for epoch in range(args.num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        for step, blocks in enumerate(dataloader):
            tic_step = time.time()

            # The nodes for input lies at the LHS side of the first block.
            # The nodes for output lies at the RHS side of the last block.
            input_nodes = blocks[0].srcdata[dgl.NID]  # block0是一个二部图，就是左边9640右边105693个节点(每次采样数量会有变化!!)，边是2410000的二部图，
            seeds = blocks[-1].dstdata[
                dgl.NID]  # seed是种子点1000个，一阶采样是10个边，得到1000-9640的二部图block1,在用9640采样25边，得到9640-105693的二部图block0

            # Load the input features as well as output labels,这里类似把这二hop的105693*603的矩阵作为输出，最后输出的是1000个点。
            batch_inputs, batch_labels = load_subtensor(g, labels, seeds, input_nodes,
                                                        device)  # input_nodes是二阶点的id,batch_inputs是二阶点对应的特征

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print(
                    'Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MiB'.format(
                        epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0 and epoch != 0:
            eval_acc = evaluate(model, g, g.ndata['feat'], labels, val_mask, args.batch_size, device)
            print('Eval Acc {:.4f}'.format(eval_acc))

    print('Avg epoch time: {}'.format(avg / (epoch - 4)))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-epochs', type=int, default=100)
    argparser.add_argument('--num-hidden', type=int, default=100)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=20)
    argparser.add_argument('--lr', type=float, default=0.01)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=0,
                           help="Number of sampling processes. Use 0 for no extra process.")
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    dataset = dgl.data.CitationGraphDataset('cora')
    graph = dataset[0]  # DGL数据集对象可以包含一个或多个图。一般情况下，整图分类任务数据集包含多个图，边预测和节点分类数据集只包含一个图，如节点分类任务中的Cora数据集只包含一个图。
    graph = dgl.add_self_loop(graph)  # 添加自连接
    node_features = graph.ndata['feat']
    node_labels = graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    valid_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    n_features = node_features.shape[1]
    n_labels = int(node_labels.max().item() + 1)
    # prepare_mp(graph)
    # Pack data
    data = train_mask, valid_mask,  n_features , node_labels, n_labels, graph

    run(args, device, data)