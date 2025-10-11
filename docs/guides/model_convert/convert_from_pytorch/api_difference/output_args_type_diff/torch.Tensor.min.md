## [ 返回参数类型不一致 ]torch.Tensor.min

该 api 有两组参数列表重载，因此有两组差异分析。

-----------------------------------------------

### [torch.Tensor.min](https://pytorch.org/docs/stable/generated/torch.Tensor.min.html)

```python
torch.Tensor.min(dim=None, keepdim=False)
```

### [paddle.Tensor.min](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#min-axis-none-keepdim-false-name-none)

```python
paddle.Tensor.min(axis=None, keepdim=False, name=None)
```

其中 PyTorch 与 Paddle 指定 `dim` 后返回值不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dim           | axis         | 求最小值运算的维度， 仅参数名不一致。                                      |
| keepdim       | keepdim      | 是否在输出 Tensor 中保留减小的维度。  |
| 返回值           | 返回值            | 表示返回结果，当指定 dim 后，PyTorch 会返回比较结果和元素索引， Paddle 不会返回元素索引，需要转写。               |

### 转写示例

#### 指定 dim 后的返回值
```python
# PyTorch 写法
result = x.min(dim=1)

# Paddle 写法
result = x.min(dim=1), x.argmin(dim=1)
```

--------------------------------------------------------------

### [torch.Tensor.min](https://pytorch.org/docs/stable/generated/torch.Tensor.min.html)

```python
torch.Tensor.min(other)
```

### [paddle.Tensor.minimum](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#minimum-y-axis-1-name-none)

```python
paddle.Tensor.minimum(y)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch           | PaddlePaddle           | 备注                                 |
| ----------------- | ---------------------- | ------------------------------------ |
|  other            |             y          | 输⼊ Tensor ，仅参数名不一致。         |
