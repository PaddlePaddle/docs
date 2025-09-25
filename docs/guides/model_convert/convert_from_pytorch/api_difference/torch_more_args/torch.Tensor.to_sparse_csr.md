## [ torch 参数更多 ] torch.Tensor.to_sparse_csr

### [torch.Tensor.to_sparse_csr](https://pytorch.org/docs/stable/generated/torch.Tensor.to_sparse_csr.html#torch-tensor-to-sparse-csr)

```python
torch.Tensor.to_sparse_csr(dense_dim=None)
```

### [paddle.Tensor.to_sparse_csr](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#tensor)

```python
paddle.Tensor.to_sparse_csr()
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                                |
| ------------- | ------------ | ----------------------------------------------------------------------------------- |
| dense_dim | -            | 控制稀疏 CSR 张量中稠密部分的维度， Paddle 无此参数，暂无转写方式。    |
