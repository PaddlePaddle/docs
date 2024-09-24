## [ torch 参数更多 ]torch.nn.functional.scaled_dot_product_attention

### [torch.nn.functional.scaled_dot_product_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch-nn-functional-scaled-dot-product-attention)

```python
torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None)
```

### [paddle.nn.functional.scaled_dot_product_attention](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/scaled_dot_product_attention_cn.html#scaled-dot-product-attention)

```python
paddle.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, training=True, name=None)
```

两者功能基本一致，参数不一致，具体如下：

### 参数映射

| PyTorch   | PaddlePaddle | 备注                                                                                                            |
| --------- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| query     | query        | 注意力模块中的查询张量。                                                                                        |
| key       | key          | 注意力模块中的关键张量。                                                                                        |
| value     | value        | 注意力模块中的值张量。                                                                                          |
| attn_mask | attn_mask    | 与添加到注意力分数的 `query`、 `key`、 `value` 类型相同的浮点掩码, 默认值为 `None`。                            |
| dropout_p | dropout_p    | `dropout` 的比例, 默认值为 0.00 即不进行正则化。                                                                |
| is_causal | is_causal    | 是否启用因果关系, 默认值为 `False` 即不启用。                                                                   |
| scale     | -            | 在 softmax 之前应用的缩放因子。默认与 Paddle 行为一致。Paddle 无此参数，暂无转写方式。                            |
| -         | training     | 是否处于训练阶段, 默认值为 `True` 即处于训练阶段。Pytorch 无此参数，默认行为等同与 `training=True`，Paddle 保持默认即可。|
