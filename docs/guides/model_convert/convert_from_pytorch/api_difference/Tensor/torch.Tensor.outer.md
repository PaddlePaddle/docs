## [ torch参数更多 ] torch.Tensor.outer

### [torch.Tensor.outer](https://pytorch.org/docs/stable/generated/torch.Tensor.outer.html?highlight=outer#torch.Tensor.outer)

```python
torch.Tensor.outer(input, vec2, *, out=None)
```

### [paddle.Tensor.outer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/outer_cn.html=)

```python
paddle.outer(x, y, name=None)
```

两者功能一致，仅参数名不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input      | x         | 一个 多维 `Tensor`，数据类型为 `float16` 、 `float32` 、 `float64` 、 `int32` 或 `int64` ，仅参数名不一致。 |
| exponent | y         | 如果类型是多维 `Tensor`，其数据类型应该和 `x` 相同，仅参数名不一致。 |
| -      | name      | 一般无需设置，默认值为 None。 |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数，需要进行转写。                              |

### 转写示例

#### out：指定输出
```python
# Pytorch 写法
torch.Tensor.outer(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]), out = y) # 同 y = torch.Tensor.outer(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
# Paddle 写法
y = paddle.outer(paddle.to_tensor([[1, 2], [3, 4]]), paddle.to_tensor([[1, 1], [4, 4]]))
```
