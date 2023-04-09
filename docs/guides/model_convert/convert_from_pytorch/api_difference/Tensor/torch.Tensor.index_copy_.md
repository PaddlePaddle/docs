## [torch 参数更多 ] torch.Tensor.index_copy_

### [torch.Tensor.index_copy_](https://pytorch.org/docs/stable/generated/torch.Tensor.index_copy_.html?highlight=index_copy_#torch.Tensor.index_copy_)

```python
torch.Tensor.index_copy_(dim, index, tensor)
```

### [paddle.Tensor.scatter_](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#id17)

```python
paddle.Tensor.scatter_(index, updates, overwrite=True, name=None)
```

两者功能类似，torch 参数更多，具体如下：
### 参数映射
| PyTorch | PaddlePaddle | 备注                        |
|---------|--------------|---------------------------|
| dim     | -            | 表示需要更新 Tensor 索引的维度值， Paddle 无此参数，需要进行转写。 |
| index   | index          | 表示需要更新的 Tensor 索引。 |
| tensor  | updates          | 表示根据 index 用来更新的 Tensor, 仅参数名一致 |
| -       | overwrite          | 更新输出的方式，True 为覆盖模式，False 为累加模式。Pytorch 无此参数，Paddle 保持默认即可。 |


### 转写示例
#### dim: 索引的维度为 0
```python
# torch 写法
x.index_copy_(0, index, t)

# paddle 写法
x.scatter_(index, t)
```
#### dim: 索引的维度不为 0
```python
# torch 写法
x = torch.zeros(2, 1, 3, 3)
t = torch.tensor([
    [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
    [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=torch.float)
index = torch.tensor([0, 1, 2])
x.index_copy_(2, index, t)

# paddle 写法
x = paddle.zeros(shape=[2, 1, 3, 3])
t = paddle.to_tensor(data=[[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], [[[1, 2, 3],
    [4, 5, 6], [7, 8, 9]]]], dtype='float32')
index = paddle.to_tensor(data=[0, 1, 2])
times, temp_shape, temp_index = paddle.prod(paddle.to_tensor(x.shape[:2])), x.shape, index
x, new_t = x.reshape([-1] + temp_shape[2 + 1:]), t.reshape([-1] + temp_shape[2 + 1:])
for i in range(1, times):
    temp_index = paddle.concat([temp_index, index + len(index) * i])
x.scatter_(temp_index, new_t).reshape(temp_shape)
```
