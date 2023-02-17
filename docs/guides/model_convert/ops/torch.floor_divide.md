## torch.floor_divide
### [torch.floor_divide](https://pytorch.org/docs/stable/generated/torch.floor_divide.html?highlight=floor_divide#torch.floor_divide)

```python
torch.floor_divide(input,
                   other,
                   *,
                   out=None)
```

### [paddle.floor_divide](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/floor_divide_cn.html#floor-divide)

```python
paddle.floor_divide(x,
                    y,
                    name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 被除数。                                      |
| other         | y            | 除数。                                       |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.floor_divide([2, 3, 8, 7], [1, 5, 3, 3], out=y)

# Paddle 写法
y = paddle.floor_divide([2, 3, 8, 7], [1, 5, 3, 3])
```
