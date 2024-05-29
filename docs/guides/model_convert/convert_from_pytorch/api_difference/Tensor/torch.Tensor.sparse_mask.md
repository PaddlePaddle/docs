## [ 仅 paddle 参数更多 ] torch.Tensor.sparse_mask

### [torch.Tensor.sparse_mask](https://pytorch.org/docs/stable/generated/torch.Tensor.sparse_mask.html)

```python
torch.Tensor.sparse_mask(mask)
```

### [paddle.sparse.mask_as](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/sparse/mask_as_cn.html)

```python
paddle.sparse.mask_as(x, mask, name=None)
```

PyTorch 作为 Tensor 的方法，Paddle 作为单独的函数调用，两者功能一致，参数用法一致，具体如下：

### 参数映射

| PyTorch    | PaddlePaddle | 备注                                 |
| ---------- | ------------ | ------------------------------------ |
| -          | x            | 输入的 DenseTensor。                  |
| mask       | mask         | 掩码逻辑的 mask，参数完全一致。           |
