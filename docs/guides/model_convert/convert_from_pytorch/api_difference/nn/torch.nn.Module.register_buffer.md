## [ 仅参数名不一致 ]torch.nn.Module.register_buffer
### [torch.nn.Module.register_buffer](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer)

```python
torch.nn.Module.register_buffer(name, tensor, persistent=True)
```

### [paddle.nn.Layer.register_buffer](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer_cn.html#register-buffer-name-tensor-persistable-true)

```python
paddle.nn.Layer.register_buffer(name, tensor, persistable=True)
```
两者功能一致，仅参数名不一致，具体如下：

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| name       | name    |  注册 buffer 的名字。可以通过此名字来访问已注册的 buffer。                   |
| tensor       | tensor    | 将被注册为 buffer 的变量。                   |
| persistent       | persistable    | 注册的 buffer 是否需要可持久性地保存到 state_dict 中。仅参数名不一致。                   |
