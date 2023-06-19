## [ 参数不一致 ] torch.sparse.sum

### [torch.sparse.sum](https://pytorch.org/docs/stable/generated/torch.sparse.sum.html?highlight=sparse+sum#torch.sparse.sum)

```python
torch.sparse.sum(input, dim=None, dtype=None)
```

### [paddle.sparse.sum](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/sparse/sum_cn.html#sum)

```python
paddle.sparse.sum(x, axis=None, dtype=None, keepdim=False, name=None)
```

输入参数不一致和返回类型不一致，具体如下：

### 参数映射

 |PyTorch |  PaddlePaddle |  备注|
 |--------|  ------------- | ------|
 |input | x|         输入的 Tensor，仅参数名不一致。|
 |dim   |      axis   |输入的第二个 Tensor，仅参数名不一致。|
 |dtype   |      dtype   |输出数据的类型，参数类型不一致。 Pytorch 为 torch.dtype 类型, Paddle 为 str 类型。 需要转写。|
 |-  |      keepdim   |是否留减少的维度， Pytorch 无此参数， Paddle 保持默认即可。|

### 转写示例
#### dytpe：指定数据类型
```Python
# Pytorch 写法
y = torch.sparse.sum(x, dim=-1, dtype=torch.float32)

# Paddle 写法
y = paddle.sparse.sum(x, axis=-1, dtype='float32')
```

#### 返回类型：当不指定 dim 时，Pytorch 返回 0D Tensor， Paddle 返回 Sparse Tensor。
```Python
# Pytorch 写法
y = torch.sparse.sum(x)

# Paddle 写法
y = paddle.sparse.sum(x).values()
```
