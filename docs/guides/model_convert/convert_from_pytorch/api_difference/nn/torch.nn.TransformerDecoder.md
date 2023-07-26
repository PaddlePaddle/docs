## [参数完全一致]torch.nn.TransformerDecoder

### [torch.nn.TransformerDecoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html#transformerdecoder)

```python
torch.nn.TransformerDecoder(decoder_layer, num_layers, norm=None)
```

### [paddle.nn.TransformerDecoder](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/TransformerDecoder_cn.html)

```python
paddle.nn.TransformerDecoder(decoder_layer, num_layers, norm=None)
```

其中功能一致, 参数完全一致，具体如下：

### 参数映射

| PyTorch              | PaddlePaddle  | 备注                                       |
| -------------------- | ------------- | ------------------------------------------ |
| decoder_layer        | decoder_layer | TransformerDecoderLayer 的一个实例。       |
| num_layers           | num_layers    | TransformerDecoderLayer 层的叠加数量。     |
| norm                 | norm          | 层标准化（Layer Normalization）。          |
