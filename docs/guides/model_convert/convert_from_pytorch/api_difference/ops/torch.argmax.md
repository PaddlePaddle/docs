## [ 仅参数名不一致 ]torch.argmax
### [torch.argmax](https://pytorch.org/docs/stable/generated/torch.argmax.html?highlight=argmax#torch.argmax)

```python
torch.argmax(input,
             dim=None,
             keepdim=False)
```

### [paddle.argmax](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/argmax_cn.html#argmax)

```python
paddle.argmax(x,
              axis=None,
              keepdim=False,
              dtype='int64',
              name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'>input</font>         | <font color='red'>x</font>            | 输入的多维 Tensor ，仅参数名不一致。                   |
| <font color='red'> dim </font> | <font color='red'> axis </font>    | 指定进行运算的轴，仅参数名不一致。  |
| keepdim |  keepdim | 是否在输出 Tensor 中保留减小的维度。  |
| - | <font color='red'> dtype </font>   | 输出 Tensor 的数据类型， PyTorch 无此参数， Paddle 保持默认即可。  |
