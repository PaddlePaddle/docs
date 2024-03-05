## [ 返回参数类型不一致 ]torch.Tensor.slogdet

### [torch.Tensor.slogdet](https://pytorch.org/docs/stable/generated/torch.Tensor.slogdet.html)

```python
torch.Tensor.slogdet()
```

### [paddle.linalg.slogdet](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/slogdet_cn.html#slogdet)

```python
paddle.linalg.slogdet(x)
```

两者功能一致，返回参数的个数不同，PyTorch 返回两个 Tesnor，Paddle 返回一个 Tensor，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> self </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| 返回值 | 返回值 | PyTorch 返回两个 Tesnor，Paddle 返回一个 Tensor，需要转写。 |



### 转写示例

#### 返回值
```python
# PyTorch 写法
x.slogdet()

# Paddle 写法
y = paddle.linalg.slogdet(x)
(y[0], y[1])
```
