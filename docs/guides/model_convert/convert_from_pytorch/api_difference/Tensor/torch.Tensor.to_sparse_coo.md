## [ paddle 参数更多 ]torch.Tensor.to_sparse_coo

### [torch.Tensor.to_sparse_coo](https://pytorch.org/docs/stable/generated/torch.Tensor.to_sparse_coo.html)

```python
torch.Tensor.to_sparse_coo()
```

### [paddle.Tensor.to_sparse_coo]()

```python
paddle.Tensor.to_sparse_coo(sparse_dim)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                      |
| ------- | ------------ | ----------------------------------------------------------------------------------------- |
| -       | sparse_dim   | 在新的稀疏张量中包含的稀疏维度的数量，pytorch中无此参数，paddle令其为tensor输入维度长度即可。 |
