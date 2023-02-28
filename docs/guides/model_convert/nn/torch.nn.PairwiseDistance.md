## torch.nn.PairwiseDistance
### [torch.nn.PairwiseDistance](https://pytorch.org/docs/stable/generated/torch.nn.PairwiseDistance.html?highlight=nn+pairwisedistance#torch.nn.PairwiseDistance)

```python
torch.nn.PairwiseDistance(p=2.0,
                            eps=1e-06,
                            keepdim=False)
```

### [paddle.nn.PairwiseDistance](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/PairwiseDistance_cn.html#pairwisedistance)

```python
paddle.nn.PairwiseDistance(p=2.,
                            epsilon=1e-6,
                            keepdim=False,
                            name=None)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| eps           | epsilon      | 添加到分母的一个很小值，避免发生除零错误。                   |
