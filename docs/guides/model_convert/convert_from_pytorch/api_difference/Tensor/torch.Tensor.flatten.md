## [ 仅参数名不一致 ]torch.Tensor.flatten

### [torch.Tensor.flatten](https://pytorch.org/docs/stable/generated/torch.Tensor.flatten.html?highlight=flatten#torch.Tensor.flatten)

```python
torch.Tensor.flatten(start_dim=0, end_dim=- 1)
```

### [paddle.Tensor.flatten](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#flatten-start-axis-0-stop-axis-1-name-none)

```python
paddle.Tensor.flatten(start_axis=0, stop_axis=-1, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch   | PaddlePaddle | 备注                           |
| --------- | ------------ | ------------------------------ |
| start_dim | start_axis   | 展开的起始维度，仅参数名不一致。 |
| end_dim   | stop_axis    | 展开的结束维度，仅参数名不一致。 |
