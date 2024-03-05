## [ torch 参数更多 ]torch.nn.Mish

### [torch.nn.Mish](https://pytorch.org/docs/stable/generated/torch.nn.Mish.html?highlight=torch+nn+mish)

```python
torch.nn.Mish(inplace=False)
```

### [paddle.nn.Mish](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Mish_cn.html)

```python
paddle.nn.Mish(name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| inplace  | -        | 表示在不更改变量的内存地址的情况下，直接修改变量的值，主要功能为节省显存，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
