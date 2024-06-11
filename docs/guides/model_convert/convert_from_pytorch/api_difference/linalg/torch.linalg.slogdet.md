## [ 返回参数类型不一致 ]torch.linalg.slogdet
### [torch.linalg.slogdet](https://pytorch.org/docs/stable/generated/torch.linalg.slogdet.html#torch.linalg.slogdet)

```python
torch.linalg.slogdet(A, *, out=None)
```

### [paddle.linalg.slogdet](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/slogdet_cn.html#slogdet)

```python
paddle.linalg.slogdet(x)
```

两者功能一致，返回参数个数不同，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> A </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> out </font> | - | 表示输出 Tensor，Paddle 无此参数，需要转写。  |
| 返回值 | 返回值 | PyTorch 返回两个 Tensor，Paddle 返回一个 Tensor，需要转写。  |


### 转写示例

#### 返回值
```python
# PyTorch 写法
torch.linalg.slogdet(x)

# Paddle 写法
y = paddle.linalg.slogdet(x)
(y[0], y[1])
```

#### out：输出的 Tensor
```python
# PyTorch 写法
torch.linalg.slogdet(x, out=(y1, y2))

# Paddle 写法
z = paddle.linalg.slogdet(a)
paddle.assign(z[0], output=y1)
paddle.assign(z[1], output=y2)
```
