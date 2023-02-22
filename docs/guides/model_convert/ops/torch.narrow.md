## torch.narrow
### [torch.narrow](https://pytorch.org/docs/stable/generated/torch.narrow.html?highlight=narrow#torch.narrow)
```python
torch.narrow(input,
             dim,
             start,
             length)
```


### [paddle.slice](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/slice_cn.html#slice)
```python
paddle.slice(input,
             axes,
             starts,
             ends)
```

其中 Pytorch 的 length 与 Paddle 的 ends 用法不一致，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dim           | axes         | 表示切片的轴。                                           |
| start         | starts       | 表示起始位置。                                           |
| length        | -            | 到结束位置的长度。                                       |
| -             | ends         | 表示结束位置。                                           |


### 转写示例
``` python
# PyTorch 写法：
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
torch.narrow(x, 0, 1, 2)

# Paddle 写法：
# Paddle 可通过设置 ends-starts=length 来实现 Pytorch 的 length 功能
x = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
paddle.slice(x, [0], [1], [3])
```
