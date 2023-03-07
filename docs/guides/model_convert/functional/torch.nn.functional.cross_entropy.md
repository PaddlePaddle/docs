## torch.nn.functional.cross_entropy

### [torch.nn.functional.cross_entropy](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html?highlight=cross_#torch.nn.functional.cross_entropy)

```python
torch.nn.functional.cross_entropy(input,
                                 target,
                                 weight=None,
                                 size_average=None,
                                 ignore_index=- 100,
                                 reduce=None,
                                 reduction='mean',
                                 label_smoothing=0.0)
```

### [paddle.nn.functional.cross_entropy](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/cross_entropy_cn.html)

```python
paddle.nn.functional.cross_entropy(input,
                                   label,
                                   weight=None,
                                   ignore_index=- 100,
                                   reduction='mean',
                                   soft_label=False,
                                   axis=- 1,
                                   name=None)
```

两者功能一致，仅参数名不一致，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| x1          | x1         | 表示第一个输入的 Tensor 。                                     |
| x2          | x2         | 表示第二个输入的 Tensor 。                                     |
| dim          | axis         | 表示计算的维度 。                                     |
| eps          | eps         | 表示加到分母上的超参数 。                                     |
