## [ 仅参数名不一致 ]torch.Tensor.slice_scatter

### [torch.Tensor.slice_scatter](https://pytorch.org/docs/stable/generated/torch.Tensor.slice_scatter.html#torch-tensor-slice-scatter)

```python
Tensor.slice_scatter(src, dim=0, start=None, end=None, step=1)
```

### [paddle.Tensor.slice_scatter](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#slice_scatter-value-axis-0-start-none-stop-none-step-1-name-none)

```python
Tensor.slice_scatter(value, axis=0, start=None, stop=None, step=1, name=None)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| src           | value        | 嵌入的值，仅参数名不一致。 |
| dim           | axis         | 嵌入的维度，仅参数名不一致。 |
| end           | stop         | 嵌入截至索引，仅参数名不一致。 |
