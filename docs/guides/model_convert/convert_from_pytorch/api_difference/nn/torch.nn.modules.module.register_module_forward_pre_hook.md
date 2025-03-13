## [ 参数完全一致 ]torch.nn.modules.module.register_module_forward_pre_hook

### [torch.nn.modules.module.register_module_forward_pre_hook](https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_pre_hook.html)

```python
torch.nn.modules.module.register_module_forward_pre_hook(hook)
```

### [paddle.nn.Layer.register_forward_pre_hook](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer_cn.html#register-forward-pre-hook-hook)

```python
paddle.nn.Layer.register_forward_pre_hook(hook)
```

其中，PyTorch 为给全局所有 module 注册 hook,而 Paddle 为给单个 Layer 注册 hook。参数完全一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                            |
|---------|--------------|-----------------------------------------------------------------------------------------------|
| hook  | hook       | 被注册为 forward pre-hook 的函数。 |
