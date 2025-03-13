## [ torch 参数更多 ]torch.nn.modules.module.register_module_forward_hook
### [torch.nn.modules.module.register_module_forward_hook](https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html)

```python
torch.nn.modules.module.register_module_forward_hook(hook, *, prepend=False, with_kwargs=False, always_call=False)
```

### [paddle.nn.Layer.register_forward_post_hook](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer_cn.html#register-forward-post-hook-hook)

```python
paddle.nn.Layer.register_forward_post_hook(hook)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| hook  | hook       | 被注册为 forward pre-hook 的函数。 |
| prepend  | -            | 钩子执行顺序控制，Paddle 无此参数，暂无转写方式。 |
| with_kwargs        | -       | 是否传递关键字参数，Paddle 无此参数，暂无转写方式。  |
| always_call        | -       | 是否强制调用钩子，Paddle 无此参数，暂无转写方式。  |
