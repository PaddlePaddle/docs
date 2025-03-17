## [ 组合替代实现 ]torch.nn.modules.module.register_module_forward_hook
### [torch.nn.modules.module.register_module_forward_hook](https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html)

```python
torch.nn.modules.module.register_module_forward_hook(hook, *, always_call=False)
```

### [paddle.nn.Layer.register_forward_post_hook](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer_cn.html#register-forward-post-hook-hook)

```python
paddle.nn.Layer.register_forward_post_hook(hook)
```

其中，PyTorch 为给全局所有 module 注册 hook,而 Paddle 为给单个 Layer 注册 hook。PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| hook  | hook       | 被注册为 forward post-hook 的函数。 |
| always_call        | -       | 是否强制调用钩子，Paddle 无此参数，一般对训练结果影响不大，可直接删除。  |

### 转写示例

```python
# PyTorch 写法
Linear = torch.nn.Linear(2, 4)
Conv2d = torch.nn.Conv2d(3, 16, 3)
Batch2d = torch.nn.BatchNorm2d(10)
torch.nn.modules.module.register_module_forward_hook(hook)

# Paddle 写法
Linear = paddle.nn.Linear(2, 4)
Conv2d = paddle.nn.Conv2d(3, 16, 3)
Batch2d = paddle.nn.BatchNorm2D(10)
Linear.register_forward_post_hook(hook)
Conv2d.register_forward_post_hook(hook)
Batch2d.register_forward_post_hook(hook)
```
