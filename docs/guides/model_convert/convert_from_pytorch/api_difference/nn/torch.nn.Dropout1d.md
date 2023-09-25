## [ torch 参数更多 ]torch.nn.Dropout1d

### [torch.nn.Dropout1d](https://pytorch.org/docs/stable/generated/torch.nn.Dropout1d.html#torch.nn.Dropout1d)

```python
torch.nn.Dropout1d(p=0.5,
                 inplace=False)
```

### [paddle.nn.Dropout](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Dropout_cn.html#dropout)

```python
paddle.nn.Dropout(p=0.5,
                  axis=None,
                  mode="upscale_in_train”,
                  name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                                            |
| ------- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| p       | p            | 表示丢弃概率。                                                                                                  |
| inplace | -            | 表示在不更改变量的内存地址的情况下，直接修改变量的值，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| -       | axis         | 指定对输入 Tensor 进行 Dropout 操作的轴，PyTorch 无此参数，Paddle 保持默认即可。                                |
| -       | mode         | 表示丢弃单元的方式，PyTorch 无此参数，Paddle 保持默认即可。                                                     |
