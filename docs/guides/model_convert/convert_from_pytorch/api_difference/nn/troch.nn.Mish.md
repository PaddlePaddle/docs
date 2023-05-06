## [ torch 参数更多 ]troch.nn.Mish

### [troch.nn.Mish](https://pytorch.org/docs/1.13/generated/torch.nn.Mish.html?highlight=troch+nn+mish)

```python
troch.nn.Mish(inplace=False)
```

### [paddle.nn.Mish](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Mish_cn.html)

```python
paddle.nn.Mish(name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| inplace  | -        | 表示在不更改变量的内存地址的情况下，直接修改变量的值，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
