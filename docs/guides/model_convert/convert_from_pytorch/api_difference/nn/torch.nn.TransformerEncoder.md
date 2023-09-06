## [torch 参数更多]torch.nn.TransformerEncoder

### [torch.nn.TransformerEncoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html#torch.nn.TransformerEncoder)

```python
torch.nn.TransformerEncoder(encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True)
```

### [paddle.nn.TransformerEncoder](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/TransformerEncoder_cn.html)

```python
paddle.nn.TransformerEncoder(encoder_layer, num_layers, norm=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch              | PaddlePaddle  | 备注                                       |
| -------------------- | ------------- | ------------------------------------------ |
| encoder_layer        | encoder_layer | TransformerEncoderLayer 的一个实例。       |
| num_layers           | num_layers    | TransformerEncoderLayer 层的叠加数量。     |
| norm                 | norm          | 层标准化（Layer Normalization）。          |
| enable_nested_tensor | -             | 是否转为嵌套 Tensor，Paddle 无此参数，暂无转写方式。 |
| mask_check           | -             | mask_check 参数，Paddle 无此参数，暂无转写方式。     |
