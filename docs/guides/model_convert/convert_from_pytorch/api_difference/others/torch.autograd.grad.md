## [torch 参数更多]torch.autograd.grad

### [torch.autograd.grad](https://pytorch.org/docs/stable/generated/torch.autograd.grad.html#torch.autograd.grad)

```python
torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False, is_grads_batched=False)
```

### [paddle.grad](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/grad_cn.html)

```python
paddle.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False, no_grad_vars=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch          | PaddlePaddle | 备注                                                         |
| ---------------- | ------------ | ------------------------------------------------------------ |
| outputs          | outputs      | 用于计算梯度的图的输出变量。                                 |
| inputs           | inputs       | 用于计算梯度的图的输入变量。                                 |
| grad_outputs     | grad_outputs | outputs 变量梯度的初始值。                                   |
| retain_graph     | retain_graph | 是否保留计算梯度的前向图。                                   |
| create_graph     | create_graph | 是否创建计算过程中的反向图。                                 |
| only_inputs      | only_inputs  | 是否只计算 inputs 的梯度。                                   |
| allow_unused     | allow_unused | 决定当某些 inputs 变量不在计算图中时抛出错误还是返回 None。  |
| is_grads_batched | -            | 是否反向使用批量，Paddle 无此参数，暂无转写方式。            |
| -                | no_grad_vars | 指明不需要计算梯度的变量，PyTorch 无此参数，Paddle 保持默认即可。 |
