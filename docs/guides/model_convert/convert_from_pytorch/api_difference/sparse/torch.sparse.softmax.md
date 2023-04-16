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

 |PyTorch  | PaddlePaddle |  备注       |
 |--------|  -------------| --------------------------------------------------------------------------------------|
 |input  |x       |  输入的稀疏 Tensor，仅参数名不一致。|
 |dim   |      axis|   指定对输入 SparseTensor 计算 softmax 的轴，Paddle 的默认值：-1。仅参数名不一致。|
 |dtype | -  | 指定数据类型，可选项，Pytorch 默认值为 None，PaddlePaddle 无此功能。|
### 转写示例
#### dytpe：指定数据类型
```Python
# torch
torch.sparse.softmax(x,-1,dtype=torch.float32)
# paddle 转化 values 的 dtype 到 float32 数据类型
x = paddle.sparse.cast(x, index_dtype=None, value_dtype='float32')
paddle.sparse.nn.functional.softmax(x,-1)
```
