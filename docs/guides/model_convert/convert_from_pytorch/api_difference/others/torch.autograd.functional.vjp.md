## [torch 参数更多]torch.autograd.functional.vjp

### [torch.autograd.functional.vjp](https://pytorch.org/docs/stable/generated/torch.autograd.functional.vjp.html#torch.autograd.functional.vjp)

```python
torch.autograd.functional.vjp(func, inputs, v=None, create_graph=False, strict=False)
```

### [paddle.incubate.autograd.vjp](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/incubate/autograd/vjp_cn.html)

```python
paddle.incubate.autograd.vjp(func, xs, v=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注                                                                |
| ------------ | ------------ | ------------------------------------------------------------------- |
| func         | func         | Python 函数。                                                       |
| inputs       | xs           | 函数 func 的输入参数。                                              |
| v            | v            | 用于计算 vjp 的输入向量。                                           |
| create_graph | -            | 是否创建图，Paddle 无此参数，暂无转写方式。                                   |
| strict       | -            | 是否在存在一个与所有输出无关的输入时抛出错误，Paddle 无此参数，暂无转写方式。 |
