## [ paddle 参数更多 ] torch.nn.Module.state_dict
### [torch.nn.Module.state_dict](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict)

```python
torch.nn.Module.state_dict(*, destination, prefix='', keep_vars=False)
```

### [paddle.nn.Layer.state_dict](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer_cn.html#state-dict-destination-none-include-sublayers-true-use-hook-true)

```python
paddle.nn.Layer.state_dict(destination=None, include_sublayers=True, structured_name_prefix='', use_hook=True, keep_vars=True)
```

Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| destination         | destination        | 所有参数和可持久性 buffers 都将存放在 destination 中 。     |
| prefix           | structured_name_prefix     | 添加到参数和缓冲区名称的前缀。     |
| keep_vars           | keep_vars        |  状态字典中返回的 Tensor 是否与 autograd 分离，PyTorch 默认值为 False，Paddle 为 True，Paddle 需设置为与 PyTorch 一致。     |
| -           | include_sublayers        | 包括子层的参数和 buffers, PyTorch 无此参数，Paddle 保持默认即可。     |
| -           | use_hook        | 将_state_dict_hooks 中注册的函数应用于 destination, PyTorch 无此参数，Paddle 保持默认即可。     |
