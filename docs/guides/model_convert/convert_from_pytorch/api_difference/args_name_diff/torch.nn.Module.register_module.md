## [ 仅参数名不一致 ]torch.nn.Module.register_module
### [torch.nn.Module.register_module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_module)

```python
torch.nn.Module.register_module(name, module)
```

### [paddle.nn.Layer.add_sublayer](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer_cn.html#add-sublayer-name-sublayer)

```python
paddle.nn.Layer.add_sublayer(name, sublayer)
```
两者功能一致，仅参数名不一致，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| name       | name    |  注册 buffer 的名字。可以通过此名字来访问已注册的 buffer。                   |
| module       | sublayer    | 将被注册为 buffer 的模块。仅参数名不一致。                   |
