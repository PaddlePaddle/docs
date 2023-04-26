## [torch 参数更多]torch.floor_divide
### [torch.floor_divide](https://pytorch.org/docs/1.13/generated/torch.floor_divide.html?highlight=torch+floor_divide#torch.floor_divide)

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
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|  input  |  x  | 表示输入的被除数 Tensor ，仅参数名不一致。  |
|  other  |  y  | 表示输入的除数 Tensor ，仅参数名不一致。  |
|  out  | -  | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。    |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.floor_divide(input, other, out=y)

# Paddle 写法
paddle.assign(paddle.floor_divide(input, other), y)
```
