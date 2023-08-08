## [ torch 参数更多 ]torch.nn.Module.register_forward_pre_hook
### [torch.nn.Module.register_forward_pre_hook](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook)

```python
torch.nn.Module.register_forward_pre_hook(hook, *, prepend=False, with_kwargs=False)
```

### [paddle.nn.Layer.register_forward_pre_hook](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer_cn.html#register-forward-pre-hook-hook)

```python
paddle.nn.Layer.register_forward_pre_hook(hook)
```
Pytorch 参数更多，具体如下：

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| hook       | hook    |  被注册的 hook。                   |
| prepend    | -    | 是否在其他 hook 执行前执行，Paddle 无此参数，暂无转写方式。   |
| with_kwargs| -    | 是否传入 forward 函数的参数，Paddle 无此参数，暂无转写方式。  |
