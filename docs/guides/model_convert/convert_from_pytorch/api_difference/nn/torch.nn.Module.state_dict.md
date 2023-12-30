## [ torch 参数更多 ] torch.nn.Module.state_dict
### [torch.nn.Module.state_dict](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict)

```python
torch.nn.Module.state_dict(*, destination, prefix, keep_vars)
```

### [paddle.nn.Layer.state_dict](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer_cn.html#state-dict-destination-none-include-sublayers-true-use-hook-true)

```python
paddle.nn.Layer.state_dict(destination=None, include_sublayers=True, use_hook=True)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| destination         | destination        | 所有参数和可持久性 buffers 都将存放在 destination 中 。     |
| prefix           | -        | 添加到参数和缓冲区名称的前缀，Paddle 无此参数，暂无转写方式。     |
| keep_vars           | -        |  状态字典中返回的 Tensor 是否与 autograd 分离，Paddle 无此参数，暂无转写方式     |
| -           | include_sublayers        | 包括子层的参数和 buffers, PyTorch 无此参数，Paddle 保持默认即可。     |
| -           | use_hook        | 将_state_dict_hooks 中注册的函数应用于 destination, PyTorch 无此参数，Paddle 保持默认即可。     |
