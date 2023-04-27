## [ 仅参数名不一致 ]torch.Tensor.ravel
### [torch.Tensor.ravel](https://pytorch.org/docs/stable/generated/torch.Tensor.ravel.html#torch.Tensor.ravel)

```python
torch.Tensor.ravel()
```

### [paddle.Tensor.flatten](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#flatten-start-axis-0-stop-axis-1-name-none)

```python
paddle.Tensor.flatten(start_axis=0, stop_axis=- 1, name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| -           | start_axis            | 表示 flatten 展开的起始维度， PyTorch 无此参数， Paddle 保持默认即可。               |
| -           | stop_axis            | 表示 flatten 展开的结束维度， PyTorch 无此参数， Paddle 保持默认即可。               |
