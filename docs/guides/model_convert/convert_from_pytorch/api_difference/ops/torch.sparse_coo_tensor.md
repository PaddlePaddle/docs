## [torch 参数更多]torch.sparse_coo_tensor

### [torch.sparse_coo_tensor](https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html?highlight=torch+sparse_coo_tensor#torch.sparse_coo_tensor)

```python
torch.sparse_coo_tensor(indices,values,size=None,*,dtype=None,device=None,requires_grad=False,check_invariants=None)
```

### [paddle.sparse.sparse_coo_tensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/sparse/sparse_coo_tensor_cn.html#sparse-coo-tensor)

```python
paddle.sparse.sparse_coo_tensor(indices, values, shape=None, dtype=None, place=None, stop_gradient=True)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

|     PyTorch      | PaddlePaddle  |                             备注                             |
| :--------------: | :-----------: | :----------------------------------------------------------: |
|     indices      |    indices    |                  表示初始化 tensor 的数据。                  |
|      values      |    values     |                  表示初始化 tensor 的数据。                  |
|      dtype       |     dtype     |                 表示创建 tensor 的数据类型。                 |
|       size       |     shape     |               表示张量的大小，仅参数名不一致。               |
|      device      |     place     |          表示创建tensor的设备位置，仅参数名不一致。          |
|  requires_grad   |       -       |    autograd是否在返回的张量上记录操作，Paddle 无此参数。     |
| check_invariants |       -       |      表示是否检查了稀疏矩阵的不变性，Paddle 无此参数。       |
|        -         | stop_gradient | 表示是否阻断 Autograd 的梯度传导，Pytorch 无此参数，Paddle 保持默认即可。 |

