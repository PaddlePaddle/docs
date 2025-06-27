## [ 输入参数用法不一致 ]torch.sparse.FloatTensor

### [torch.sparse.FloatTensor](https://pytorch.org/docs/stable/generated/torch.cuda.comm.broadcast.html#torch.cuda.comm.broadcast)

```python
torch.sparse.FloatTensor(indices, values, size,  *, device=None)
```

### [paddle.sparse.sparse_coo_tensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/sparse/sparse_coo_tensor_cn.html#sparse-coo-tensor)

```python
paddle.sparse.sparse_coo_tensor(indices, values, shape=None, dtype=None, place=None, stop_gradient=True)
```

其中 PyTorch 与 Paddle 参数不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                         |
| ------- | ------------ | ------------------------------------------------------------ |
| indices  | indices       | 初始化 tensor 的数据。 |
| values | values          | 初始化 tensor 的数据。                                           |
| size     | shape            | 稀疏 Tensor 的形状，仅参数名不一致。          |
| device       | place        | 表示 Tensor 存放设备位置，输入用法不一致，需要转写。    |
| -       | dtype      | 创建 tensor 的数据类型。PyTorch 无此参数，Paddle 保持默认即可。    |
| -       | stop_gradient      | 是否阻断 Autograd 的梯度传导。PyTorch 无此参数，Paddle 保持默认即可。    |
### 转写示例

#### device：输出数据类型

```python
# PyTorch 写法
torch.sparse.FloatTensor(i, v, torch.Size([2, 3]), device='cpu')

# Paddle 写法
paddle.sparse.sparse_coo_tensor(i, v, [2, 3], place="cpu")
