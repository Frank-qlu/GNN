import dgl
import torch as th
u,v=th.tensor([0,1,2]),th.tensor([2,3,4])
g=dgl.graph((u,v))
g.ndata['x']=th.randn(5,3)
print(g.device) #cpu
cuda_g=g.to('cuda:0')
print(cuda_g.device) #cuda:0
