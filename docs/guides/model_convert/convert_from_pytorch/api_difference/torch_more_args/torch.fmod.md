## [ 参数一致 ]torch.fmod
### [torch.fmod](https://docs.pytorch.org/docs/stable/generated/torch.fmod.html#torch.fmod)
```python
torch.fmod(input,
           other,
           *,
           out=None)
```

### [paddle.fmod](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/fmod_cn.html#paddle.fmod)
```python
paddle.fmod(x,
           y,
           name=None,
           *,
           out=None)
```

PyTorch 与 Paddle 参数完全一致，均支持 Tensor 和 Python Number 类型的输入。

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|  input  |  x  | 表示输入的被除数，仅参数名不一致。  |
|  other  |  y  | 表示输入的除数，两者均支持 Tensor 和 scalar 类型。  |
|  out  | out  | 表示输出的 Tensor，两者均支持。    |

### 转写示例
#### 参数名差异
```python
# PyTorch 写法
result = torch.fmod(input=x, other=y)

# Paddle 写法
result = paddle.fmod(x=x, y=y)
```

#### out 参数
```python
# PyTorch 写法
torch.fmod(x, y, out=output)

# Paddle 写法
paddle.fmod(x, y, out=output)
```
