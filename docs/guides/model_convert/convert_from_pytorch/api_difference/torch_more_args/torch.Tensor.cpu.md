## [ torch 参数更多 ]torch.Tensor.cpu

### [torch.Tensor.cpu](https://pytorch.org/docs/stable/generated/torch.Tensor.cpu.html?highlight=torch+tensor+cpu#torch.Tensor.cpu)

```python
torch.Tensor.cpu(memory_format=torch.preserve_format)
```

### [paddle.Tensor.cpu](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#cpu)

```python
paddle.Tensor.cpu()
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch                             | PaddlePaddle | 备注                                                                    |
| ----------------------------------- | ------------ | ----------------------------------------------------------------------- |
| memory_format | -            | 表示内存格式，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
