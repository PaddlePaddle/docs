## [ torch 参数更多 ]torch.Tensor.cpu

### [torch.Tensor.cpu](https://pytorch.org/docs/stable/generated/torch.Tensor.cpu.html?highlight=torch+tensor+cpu#torch.Tensor.cpu)

```python
Tensor.cpu(memory_format=torch.preserve_format)
```

### [paddle.Tensor.cpu](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#cpu)

```python
Tensor.cpu()
```

两者功能一致且参数用法一致，torch 参数更多，具体如下：

### 参数映射

| PyTorch                                                | PaddlePaddle         | 备注                             |
| ------------------------------------------------------ | -------------------- | -------------------------------- |
| <center> memory_format=torch.preserve_format </center> | <center> - </center> | pytorch:返回张量所需的内存格式。 |
