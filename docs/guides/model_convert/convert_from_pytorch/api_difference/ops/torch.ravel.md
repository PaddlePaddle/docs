## [ 仅参数名不一致 ]torch.ravel
### [torch.ravel](https://pytorch.org/docs/stable/generated/torch.ravel.html?highlight=ravel#torch.ravel)

```python
torch.ravel(input)
```

### [paddle.flatten](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/flatten_cn.html)

```python
paddle.flatten(x, start_axis=0, stop_axis=- 1, name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| -           | start_axis            | 表示 flatten 展开的起始维度， PyTorch 无此参数， Paddle 保持默认即可。               |
| -           | stop_axis            | 表示 flatten 展开的结束维度， PyTorch 无此参数， Paddle 保持默认即可。               |
