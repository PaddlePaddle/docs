## [torch 参数更多]torch.autograd.backward

### [torch.autograd.backward](https://pytorch.org/docs/stable/generated/torch.autograd.backward.html#torch.autograd.backward)

```python
torch.autograd.backward(tensors, grad_tensors=None, retain_graph=None, create_graph=False, grad_variables=None, inputs=None)
```

### [paddle.autograd.backward](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/autograd/backward_cn.html)

```python
paddle.autograd.backward(tensors, grad_tensors=None, retain_graph=False)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch        | PaddlePaddle | 备注                                  |
| -------------- | ------------ | ------------------------------------- |
| tensors        | tensors      | 将要计算梯度的 Tensors 列表。         |
| grad_tensors   | grad_tensors | tensors 的初始梯度值。                |
| retain_graph   | retain_graph | 如果为 False，反向计算图将被释放。    |
| create_graph   | -            | 是否创建图，Paddle 无此参数，暂无转写方式。     |
| grad_variables | -            | 创建图关联变量，Paddle 无此参数，暂无转写方式。 |
| inputs         | -            | 将累积的梯度，Paddle 无此参数，暂无转写方式。   |
