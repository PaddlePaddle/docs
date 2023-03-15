## [ paddle 参数更多 ]torch.Tensor.fill\_

### [torch.Tensor.fill\_](https://pytorch.org/docs/stable/generated/torch.Tensor.fill_.html?highlight=fill_#torch.Tensor.fill_)

```python
Tensor.fill_(value)
```

### [paddle.Tensor.fill\_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#fill-x-value-name-none)

```python
Tensor.fill_(x, value, name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

### 参数映射

| PyTorch                  | PaddlePaddle             | 备注                                |
| ------------------------ | ------------------------ | ----------------------------------- |
| <center> - </center>     | <center> x </center>     | paddle:修改的原始 Tensor。          |
| <center> value </center> | <center> value </center> | 输入 value 值修改原始 Tensor 元素。 |
