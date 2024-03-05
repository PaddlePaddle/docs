## [ torch 参数更多 ]torch.nn.LeakyReLU
### [torch.nn.LeakyReLU](https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html?highlight=leakyrelu#torch.nn.LeakyReLU)

```python
torch.nn.LeakyReLU(negative_slope=0.01,
                   inplace=False)
```

### [paddle.nn.LeakyReLU](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/LeakyReLU_cn.html#leakyrelu)

```python
paddle.nn.LeakyReLU(negative_slope=0.01,
                    name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| negative_slope        | negative_slope            | 表示 x<0 时的斜率。  |
| inplace       | -            | 在不更改变量的内存地址的情况下，直接修改变量的值，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。    |
