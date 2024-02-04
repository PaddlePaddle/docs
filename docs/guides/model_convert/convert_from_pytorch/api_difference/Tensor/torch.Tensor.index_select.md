## [ 仅参数名不一致 ]torch.Tensor.index_select

### [torch.Tensor.index\_select](https://pytorch.org/docs/stable/generated/torch.Tensor.index_select.html)

```python
torch.Tensor.index_select(dim, index)
```

### [paddle.Tensor.index\_select](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#index-select-index-axis-0-name-none)

```python
paddle.Tensor.index_select(index, axis=0, name=None)
```

其中 PyTorch 和 Paddle 功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | -- |
| dim     | axis         | 索引轴，若未指定，则默认选取第 0 维。 |
| index   | index        | 包含索引下标的 1-D Tensor。 |
