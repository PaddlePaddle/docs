## [ paddle 参数更多 ]torch.fliplr
### [torch.fliplr](https://pytorch.org/docs/stable/generated/torch.fliplr.html?highlight=fliplr#torch.fliplr)

```python
torch.fliplr(input)
```

### [paddle.flip](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/flip_cn.html#flip)

```python
paddle.flip(x,
            axis,
            name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| - | <font color='red'> axis </font> | 表示进行运算的轴，PyTorch 无此参数，Paddle 需设置为 1 。  |
