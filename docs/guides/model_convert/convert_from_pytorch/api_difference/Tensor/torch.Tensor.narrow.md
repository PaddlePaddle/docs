## [ 参数不一致 ]torch.Tensor.narrow
### [torch.Tensor.narrow](https://pytorch.org/docs/stable/generated/torch.Tensor.narrow.html#torch.Tensor.narrow)

```python
torch.Tensor.narrow(dim, start, length)
```

### [paddle.slice](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/slice_cn.html#slice)
```python
paddle.slice(input,
             axes,
             starts,
             ends)
```

其中 Pytorch 的 length 与 Paddle 的 ends 用法不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| -           | input         | 表示输入的 Tensor 。                                           |
| dim           | axes         | 表示切片的轴。                                           |
| start         | starts       | 表示起始位置。                                           |
| length        | -            | 到结束位置的长度，Paddle 无此参数，需要转写。Paddle 应改写 ends。                                       |
| -             | ends         | 表示结束位置，Pytorch 无此参数，应设为 start + length 。                                         |

### 转写示例

```python
# Pytorch 写法
y = a.narrow(1, 1, 4)

# Paddle 写法
# Paddle 可通过设置 ends-starts=length 来实现 Pytorch 的 length 功能
y = paddle.slice(a, [1], [1], [5])
```
