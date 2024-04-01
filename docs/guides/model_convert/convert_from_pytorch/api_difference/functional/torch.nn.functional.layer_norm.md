## [ 仅参数名不一致 ]torch.nn.functional.layer_norm

### [torch.nn.functional.layer_norm](https://pytorch.org/docs/stable/generated/torch.nn.functional.layer_norm.html#torch.nn.functional.layer_norm)

```python
torch.nn.functional.layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05)
```

### [paddle.nn.functional.layer_norm](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/layer_norm_cn.html#layer-norm)
```python
paddle.nn.functional.layer_norm(x, normalized_shape, weight=None, bias=None, epsilon=1e-05, name=None)
```

两者功能相同，仅参数名不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> normalized_shape </font>             | <font color='red'> normalized_shape </font>  | 需规范化的 shape               |
| <font color='red'> weight </font>   | <font color='red'> weight </font>   | 权重的 Tensor               |
| <font color='red'> bias </font>   | <font color='red'> bias </font>   | 偏置的 Tensor               |
| <font color='red'> eps  </font>         |    <font color='red'> epsilon  </font>         | 为了数值稳定加在分母上的值             |
