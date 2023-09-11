## [仅参数名不一致]torch.nn.Module.register_parameter

### [torch.nn.Module.register_parameter](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_parameter)

```python
torch.nn.Module.register_parameter(name, param)
```

### [paddle.nn.Layer.add_parameter](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer_cn.html#add-parameter-name-parameter)

```python
paddle.nn.Layer.add_parameter(name, parameter)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                             |
| ------- | ------------ | -------------------------------- |
| name    | name         | 参数名。                         |
| param   | parameter    | Parameter 实例，仅参数名不一致。 |
