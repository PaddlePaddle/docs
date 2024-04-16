## [ torch 参数更多 ] torch.sparse.softmax

### [torch.sparse.softmax](https://pytorch.org/docs/stable/generated/torch.sparse.softmax.html#torch.sparse.softmax)

```python
torch.sparse.softmax(input, dim, dtype=None)
```

### [paddle.sparse.nn.functional.softmax](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/sparse/nn/functional/softmax_cn.html)

```python
paddle.sparse.nn.functional.softmax(x, axis=-1, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

 |PyTorch  | PaddlePaddle |  备注       |
 |--------|  -------------| --------------------------------------------------------------------------------------|
 |input  |x       |  输入的稀疏 Tensor，仅参数名不一致。|
 |dim   |      axis|   指定对输入 SparseTensor 计算 softmax 的轴，Paddle 的默认值：-1。仅参数名不一致。|
 |dtype | -  | 指定数据类型，可选项，PyTorch 默认值为 None，Paddle 无此参数，需要转写。|
### 转写示例
#### dytpe：指定数据类型
```Python
# PyTorch 写法
y = torch.sparse.softmax(x, dim=-1, dtype=torch.float32)

# Paddle 写法
y = paddle.sparse.cast(x, value_dtype='float32')
y = paddle.sparse.nn.functional.softmax(y)
```
