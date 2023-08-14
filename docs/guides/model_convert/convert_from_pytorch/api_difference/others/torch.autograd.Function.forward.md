## [参数完全一致]torch.autograd.Function.forward

### [torch.autograd.Function.forward](https://pytorch.org/docs/stable/generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward)

```python
torch.autograd.Function.forward(ctx, *args, **kwargs)
```

### [paddle.autograd.PyLayer.forward](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/autograd/PyLayer_cn.html#forward-ctx-args-kwargs)

```python
paddle.autograd.PyLayer.forward(ctx, *args, **kwargs)
```

两者功能一致，参数完全一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注               |
| ------- | ------------ | ------------------ |
| ctx     | ctx          | 上下文对象。       |
| args    | args         | 自定义算子的输入。 |
| kwargs  | kwargs       | 自定义算子的输入。 |
