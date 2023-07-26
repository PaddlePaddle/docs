# [torch 参数更多 ]torch.nn.LayerNorm
### [torch.nn.LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html?highlight=layernorm#torch.nn.LayerNorm)

```python
torch.nn.LayerNorm(normalized_shape,
                   eps=1e-05,
                   elementwise_affine=True,
                   device=None,
                   dtype=None)
```

### [paddle.nn.LayerNorm](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/LayerNorm_cn.html#layernorm)

```python
paddle.nn.LayerNorm(normalized_shape,
                    epsilon=1e-05,
                    weight_attr=None,
                    bias_attr=None,
                    name=None)
```

两者功能一致但参数不一致，torch 参数更多，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| elementwise_affine        | -            | 是否进行仿射变换，Paddle 无此参数，需要进行转写。         |
| device        | -            | 设备类型，Paddle 无此参数，可直接删除。 |
| dtype         | -            | 参数类型，PaddlePaddle 无此参数，可直接删除。         |
| eps           | epsilon      | 为了数值稳定加在分母上的值。                                     |
| -             | weight_attr  | 指定权重参数属性的对象。如果为 False, 则表示每个通道的伸缩固定为 1，不可改变。默认值为 None，表示使用默认的权重参数属性。 |
| -             | bias_attr    | 指定偏置参数属性的对象。如果为 False, 则表示每一个通道的偏移固定为 0，不可改变。默认值为 None，表示使用默认的偏置参数属性。 |


### 转写示例
#### elementwise_affine：是否进行仿射变换
```python
# 当 PyTorch 的 elementwise_affine 为`False`，表示 weight 和 bias 不进行更新，torch 写法
torch.nn.LayerNorm(normalized_shape=(256, 256), eps=1e-05, elementwise_affine=False)

# paddle 写法
paddle.nn.GroupNorm(normalized_shape=(256, 256), epsilon=1e-05, weight_attr=False, bias_attr=False)

# 当 PyTorch 的 elementwise_affine 为`True`，torch 写法
torch.nn.LayerNorm(normalized_shape=(256, 256), eps=1e-05, elementwise_affine=True)

# paddle 写法
paddle.nn.LayerNorm(normalized_shape=(256, 256), epsilon=1e-05)
```
