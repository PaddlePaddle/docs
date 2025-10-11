## [torch 参数更多]torch.vdot

### [torch.vdot](https://pytorch.org/docs/stable/generated/torch.vdot.html#torch.vdot)

```python
torch.vdot(input, other, *, out=None)
```

### [paddle.dot](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/dot_cn.html#dot)

```python
paddle.dot(x, y, name=None)
```

torch 参数更多，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|  input |  x  | 输入的向量。   |
|  other |  y  | 被乘的向量。   |
|  out |  -  | 指定输出。Paddle 无此参数，需要转写   |

### 转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.vdot(x, y, out=out)

# Paddle 写法
paddle.assign(paddle.dot(x, y), out)
```
