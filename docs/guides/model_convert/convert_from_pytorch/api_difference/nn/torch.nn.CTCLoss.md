## [torch 参数更多]torch.nn.CTCLoss

### [torch.nn.CTCLoss](https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html#torch.nn.CTCLoss)

```python
torch.nn.CTCLoss(blank=0,
                 reduction='mean',
                 zero_infinity=False)
```

### [paddle.nn.CTCLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/CTCLoss_cn.html#ctcloss)

```python
paddle.nn.CTCLoss(blank=0,
                  reduction='mean')
```

其中，PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                         |
| ------------- | ------------ | ------------------------------------------------------------ |
| blank         | blank        | 空格标记的 ID 值。                                           |
| reduction     | reduction    | 表示应用于输出结果的计算方式。                               |
| zero_infinity | -            | 是否将无穷大损失及其梯度置 0，Paddle 无此参数，暂无转写方式。 |
