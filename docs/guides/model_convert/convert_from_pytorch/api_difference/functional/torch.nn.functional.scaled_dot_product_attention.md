## [ 仅参数名称不一致 ]torch.nn.functional.scaled_dot_product_attention

### [torch.nn.functional.scaled_dot_product_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch-nn-functional-scaled-dot-product-attention)

```python
torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None)
```

### [paddle.nn.functional.scaled_dot_product_attention](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/scaled_dot_product_attention_cn.html#scaled-dot-product-attention)

```python
paddle.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, training=True, name=None)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch   | PaddlePaddle | 备注                                                                                                            |
| --------- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| input     | x            | 输入的 Tensor，仅参数名不一致。                                                                                 |
| threshold | threshold    | thresholded_relu 激活计算公式中的 threshold 值。                                                                |
| value     | value        | 不在指定 threshold 范围时的值。                                                  |
