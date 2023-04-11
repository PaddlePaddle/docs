## [torch 参数更多 ]torch.floor
### [torch.floor](https://pytorch.org/docs/stable/generated/torch.floor.html?highlight=floor#torch.floor)

```python
torch.floor(input,
            *,
            out=None)
```

### [paddle.floor](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/floor_cn.html#floor)

```python
paddle.floor(x,
             name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> out </font> | - |  表示输出的 Tensor ， Paddle 无此参数，需要进行转写。    |

### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.floor([-0.4, -0.2, 0.1, 0.3], out=y)

# Paddle 写法
paddle.assign(paddle.floor([-0.4, -0.2, 0.1, 0.3]), y)
```
