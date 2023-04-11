## [ 仅参数名不一致 ]torch.nn.LogSoftmax
### [torch.nn.LogSoftmax](https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html?highlight=nn+logsoftmax#torch.nn.LogSoftmax)

```python
torch.nn.LogSoftmax(dim=None)
```

### [paddle.nn.LogSoftmax](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/LogSoftmax_cn.html#logsoftmax)

```python
paddle.nn.LogSoftmax(axis=- 1, name=None)
```
两者功能一致且参数用法一致，仅参数名不同，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dim           | axis         | 指定对输入 Tensor 进行运算的轴，仅参数名不一致。                          |
