## [ 仅参数名不一致 ]torch.Tensor.

### [torch.Tensor.element_size](https://pytorch.org/docs/stable/generated/torch.Tensor.element_size.html?highlight=element_size#torch.Tensor.element_size)

```python
Tensor.element_size()
```

### [paddle.Tensor.element_size](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#element-size)

```python
Tensor.element_size()
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

### 参数映射

| PyTorch              | PaddlePaddle         | 备注                                                |
| -------------------- | -------------------- | --------------------------------------------------- |
| <center> - </center> | <center> - </center> | 返回 Tensor 单个元素在计算机中所分配的 bytes 数量。 |
