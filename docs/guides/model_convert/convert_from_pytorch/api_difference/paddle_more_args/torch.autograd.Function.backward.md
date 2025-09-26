## [ paddle 参数更多 ]torch.autograd.Function.backward

### [torch.autograd.Function.backward](https://pytorch.org/docs/stable/generated/torch.autograd.Function.backward.html#torch.autograd.Function.backward)

```python
torch.autograd.Function.backward(ctx, *grad_outputs)
```

### [paddle.autograd.PyLayer.backward](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/autograd/PyLayer_cn.html#backward-ctx-args-kwargs)

```python
paddle.autograd.PyLayer.backward(ctx, *args, **kwargs)
```

Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注                                                                |
| ------------ | ------------ | ------------------------------------------------------------------- |
| ctx          | ctx          | 上下文对象。                                                        |
| grad_outputs | args         | forward 输出 Tensor 的梯度，仅参数名不一致。                        |
| -            | kwargs       | forward 输出 Tensor 的梯度，PyTorch 无此参数，Paddle 保持默认即可。 |
