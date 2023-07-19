## [torch 参数更多]torch.autograd.functional.hessian

### [torch.autograd.functional.hessian](https://pytorch.org/docs/stable/generated/torch.autograd.functional.hessian.html#torch.autograd.functional.hessian)

```python
torch.autograd.functional.hessian(func, inputs, create_graph=False, strict=False, vectorize=False, outer_jacobian_strategy='reverse-mode')
```

### [paddle.incubate.autograd.Hessian](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/incubate/autograd/Hessian_cn.html)

```python
paddle.incubate.autograd.Hessian(func, xs, is_batched=False)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch                 | PaddlePaddle | 备注                                                                |
| ----------------------- | ------------ | ------------------------------------------------------------------- |
| func                    | func         | Python 函数。                                                       |
| inputs                  | xs           | 函数 func 的输入参数。                                              |
| create_graph            | -            | 是否创建图，Paddle 无此参数，暂无转写方式。                                   |
| strict                  | -            | 是否在存在一个与所有输出无关的输入时抛出错误，Paddle 无此参数，暂无转写方式。 |
| vectorize               | -            | 体验中功能，Paddle 无此参数，暂无转写方式。                                   |
| outer_jacobian_strategy | -            | AD 计算模式，Paddle 无此参数，暂无转写方式。                                  |
| -                       | is_batched   | 表示包含 batch 维，PyTorch 无此参数，Paddle 保持默认即可。          |
