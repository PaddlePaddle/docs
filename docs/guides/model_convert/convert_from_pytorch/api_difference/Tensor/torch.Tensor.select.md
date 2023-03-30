## [ torch 参数更多 ]torch.Tensor.select

### [torch.Tensor.select](https://pytorch.org/docs/stable/generated/torch.Tensor.select.html?highlight=select#torch.Tensor.select)

```python
torch.Tensor.select(dim, index)
```

### [paddle.index_select](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#index-select-index-axis-0-name-none)

```python
paddle.Tensor.index_select(index, axis=0, name=None)
```

其中 Pytorch 相⽐ Paddle ⽀持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dim | axis        | 指定进行运算的轴，仅参数名不同。   |
| index | index | 包含索引下标的 1-D Tensor |
