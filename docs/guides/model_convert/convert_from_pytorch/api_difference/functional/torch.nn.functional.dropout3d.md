## [ torch 参数更多 ]torch.nn.functional.dropout3d

### [torch.nn.functional.dropout3d](https://pytorch.org/docs/stable/generated/torch.nn.functional.dropout3d.html#torch.nn.functional.dropout3d)

```python
torch.nn.functional.dropout3d(input, p=0.5, training=True, inplace=False)
```

### [paddle.nn.functional.dropout3d](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/dropout3d_cn.html)

```python
paddle.nn.functional.dropout3d(x, p=0.5, training=True, name=None)
```

PyTorch 对于 dropout1d/dropout2d/dropout3d，是将某个 Channel 以一定概率全部置 0，Paddle 是所有元素以一定概率置 0，但该差异一般不影响网络训练效果。
其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch  | PaddlePaddle | 备注                                                                                                            |
| -------- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| input    | x            | 输入的多维 Tensor，仅参数名不一致。                                                                             |
| p        | p            | 将输入节点置 0 的概率，即丢弃概率。                                                                             |
| training | training     | 标记是否为训练阶段。                                                                                            |
| inplace  | -            | 表示在不更改变量的内存地址的情况下，直接修改变量的值，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
