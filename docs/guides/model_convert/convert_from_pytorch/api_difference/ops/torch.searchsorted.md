## [ torch 参数更多]torch.searchsorted

### [torch.searchsorted](https://pytorch.org/docs/1.13/generated/torch.searchsorted.html#torch-searchsorted)

```python
torch.searchsorted(sorted_sequence,
                   values,
                   *,
                   out_int32=False,
                   right=False,
                   side='left',
                   out=None,
                   sorter=None)
```

### [paddle.searchsorted](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/searchsorted_cn.html#searchsorted)

```python
paddle.searchsorted(sorted_sequence,
                    values,
                    out_int32=False,
                    right=False,
                    name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch | PaddlePaddle | 备注                                                |
| ------- | ------------ | --------------------------------------------------- |
| sorted_sequence   | sorted_sequence            | 表示待查找的 Tensor 。          |
| values   | values            | 表示用于查找的 Tensor。           |
| out_int32     |    out_int32     | 表示输出的数据类型。 |
| right     |    right     | 表示查找对应的上边界或下边界。 |
| side     | -       | 表示查找对应的上边界或下边界，Paddle 无此参数，需要进行转写。 |
| out     | -       | 表示输出的 Tensor，Paddle 无此参数，需要进行转写。 |
| sorter     | -       | 表示 sorted_sequence 元素无序时对应的升序索引，Paddle 无此参数，一般对网络训练结果影响不大。 |

### 转写示例
#### side：指定查找对应的上边界或下边界

```python
# Pytorch 写法
torch.searchsorted(x,y, side='right')

# Paddle 写法
paddle.searchsorted(x,y,right=True)
```

#### out：指定输出

```python
# Pytorch 写法
torch.searchsorted(x,y, out=output)

# Paddle 写法
paddle.assign(paddle.searchsorted(x,y), output)
```

#### sorter: 提供 sorted_sequence 为无序 Tensor 时，相对应的升序索引

```python
# Pytorch 写法
torch.searchsorted(x,y, sorter=sorter)

# Paddle 写法
paddle.searchsorted(x.take_along_axis(axis = -1, indices = sorter), y)
```
