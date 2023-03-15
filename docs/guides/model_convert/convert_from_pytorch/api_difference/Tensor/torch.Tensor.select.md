## [ torch 参数更多 ]torch.Tensor.select

### [torch.Tensor.select](https://pytorch.org/docs/stable/generated/torch.Tensor.select)

```python
torch.select(input, dim, index)
```

### [paddle.index_select](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/index_select_cn.html#index-select)

```python
paddle.index_select(x, index, axis=0, name=None)
```

其中 Pytorch 相⽐ Paddle ⽀持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input    | x           | 表示输入的 Tensor ，仅参数名不同。 |
| dim | axis        | 指定进行运算的轴，仅参数名不同。   |
| index | index | 包含索引下标的 1-D Tensor |
