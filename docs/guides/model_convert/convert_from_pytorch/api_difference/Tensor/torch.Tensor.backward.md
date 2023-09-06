## [ torch 参数更多 ] torch.Tensor.backward

### [torch.Tensor.backward](https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html#torch.Tensor.backward)

```python
torch.Tensor.backward(gradient=None, retain_graph=None, create_graph=False, inputs=None)
```

### [paddle.Tensor.backward](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#backward-grad-tensor-none-retain-graph-false)

```python
paddle.Tensor.backward(grad_tensor=None, retain_graph=False)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                                |
| ------------- | ------------ | ----------------------------------------------------------------------------------- |
| gradient | grad_tensor            | 当前 Tensor 的初始梯度值。仅参数名不一致。    |
| retain_graph | retain_graph            | 是否保留计算图。    |
| create_graph | -            | 是否创建梯度图，Paddle 无此参数，暂无转写方式。    |
| inputs | -            | 计算的起始输入 tensor，Paddle 无此参数，暂无转写方式。     |
