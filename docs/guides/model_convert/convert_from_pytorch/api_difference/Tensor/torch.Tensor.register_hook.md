## [ 参数完全一致 ]torch.Tensor.register_hook
### [torch.Tensor.register_hook](https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html#torch-tensor-register-hook)

```python
torch.Tensor.register_hook(hook)
```

### [paddle.Tensor.register_hook](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#register-hook-hook)

```python
paddle.Tensor.register_hook(hook)
```


两者功能一致，且参数一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| hook         | hook            | 一个需要注册到 Tensor.grad 上的 hook 函数。   |
