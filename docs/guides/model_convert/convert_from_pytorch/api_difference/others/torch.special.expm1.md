## [ torch 参数更多 ]torch.special.expm1
### [torch.special.expm1](https://pytorch.org/docs/stable/special.html#torch.special.expm1)

```python
torch.special.expm1(input,
                *,
                out=None)
```

### [paddle.expm1](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/expm1_cn.html)

```python
paddle.expm1(x,
        name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          |  x           | 该 OP 的输入为多维 Tensor。数据类型为：float16、float32、float64，仅参数名不一致。  |
| out         | -         | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。 |

### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.special.expm1([2, 3, 8, 7], [1, 5, 3, 3], out=y)

# Paddle 写法
paddle.assign(paddle.expm1([2, 3, 8, 7], [1, 5, 3, 3]), y)
```
