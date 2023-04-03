## [ 组合替代实现 ] torch.Tensor.index_copy_

### [torch.Tensor.index_copy_](https://pytorch.org/docs/stable/generated/torch.Tensor.index_copy_.html?highlight=index_copy_#torch.Tensor.index_copy_)

```python
torch.Tensor.index_copy_(dim, index, tensor)
```

### [paddle.scatter_](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/scatter__cn.html)

```python
paddle.scatter_(x, index, updates, overwrite=True, name=None)
```

两者功能类似，参数不一致，但 torch 是类成员方式，paddle 是 funtion 调用方式，具体如下：
### 参数映射
| PyTorch | PaddlePaddle | 备注                        |
|---------|--------------|---------------------------|
| dim     | -            | 索引的维度值。 |
| -     | x            | 表示输入的 Tensor。 |
| index   | index          | 选择的需要更新的 Tensor 索引。 |
| tensor  | updates          | 根据 index 来更新 Tensor。 |
| -       | overwrite          | 更新输出的方式，True 为覆盖模式，False 为累加模式。 |


### 转写示例
#### 示例 1: 索引的维度为 0
```python
# torch 写法
x.index_copy_(0, index, t)

# paddle 写法
paddle.scatter_(x, index, t)
```
#### 示例 2: 索引的维度不为 0
```python
# torch 写法
y = x.index_copy_(2, index, t)

# paddle 写法
dim = list(x.shape)
for i0 in range(dim[0]):
    for i1 in range(dim[1]):
        x[i0, i1, :] = paddle.scatter_(x[i0][i1], index, t[i0][i1])
y = x.clone()
```
