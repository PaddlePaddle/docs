## [ torch 参数更多 ] torch.sparse.softmax

### [torch.sparse.softmax](https://pytorch.org/docs/1.13/generated/torch.sparse.softmax.html#torch.sparse.softmax)

```python
torch.sparse.softmax(input, dim, dtype=None)
```

### [paddle.sparse.nn.functional.softmax](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/sparse/nn/functional/softmax_cn.html)

```python
paddle.sparse.nn.functional.softmax(x, axis=-1, name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

 PyTorch  | PaddlePaddle |  备注
 --------|  -------------| --------------------------------------------------------------------------------------
 input  |x       |  输入的 Tensor，仅参数名不一致。
 dim   |      axis|   输入的第二个 Tensor，仅参数名不一致。
 dtype | -  | 指定数据类型，PaddlePaddle 无此功能。
