## [ 仅参数名不一致 ]torch.nn.PairwiseDistance
### [torch.nn.PairwiseDistance](https://pytorch.org/docs/stable/generated/torch.nn.PairwiseDistance.html?highlight=nn+pairwisedistance#torch.nn.PairwiseDistance)

```python
torch.nn.PairwiseDistance(p=2.0,
                            eps=1e-06,
                            keepdim=False)
```

### [paddle.nn.PairwiseDistance](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/PairwiseDistance_cn.html#pairwisedistance)

```python
paddle.nn.PairwiseDistance(p=2.,
                            epsilon=1e-6,
                            keepdim=False,
                            name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| p           | p      | 指定 p 阶的范数。                   |
| eps           | epsilon      | 添加到分母的一个很小值，避免发生除零错误，仅参数名不一致。                   |
| keepdim           | keepdim      | 表示是否在输出 Tensor 中保留减小的维度。                   |
