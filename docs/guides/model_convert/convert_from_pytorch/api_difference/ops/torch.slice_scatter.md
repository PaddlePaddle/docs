## [ 仅参数名不一致 ]torch.Tensor.slice_scatter

### [torch.slice_scatter](https://pytorch.org/docs/stable/generated/torch.slice_scatter.html#torch.slice_scatter)

```python
torch.slice_scatter(input, src, dim=0, start=None, end=None, step=1)
```

### [paddle.slice_scatter](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/slice_scatter.html)

```python
paddle.slice_scatter(x, value, axis=0, start=None, stop=None, step=1, name=None)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的目标矩阵, 仅参数名不一致。 |
| src           | value        | 嵌入的值，仅参数名不一致。 |
| dim           | axis         | 嵌入的维度，仅参数名不一致。 |
| end           | stop         | 嵌入截至索引，仅参数名不一致。 |
