## torch.arcsin
### [torch.arcsin](https://pytorch.org/docs/stable/generated/torch.arcsin.html?highlight=arcsin#torch.arcsin)

```python
torch.arcsin(input,
            *,
            out=None)
```

### [paddle.asin](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/asin_cn.html#asin)

```python
paddle.asin(x,
            name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                                      |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |


### 代码示例
``` python
# PyTorch 示例：
a = torch.randn(4)
a
# 输出
# tensor([-0.5962,  1.4985, -0.4396,  1.4525])
torch.arcsin(a)
# 输出
# tensor([-0.6387,     nan, -0.4552,     nan])
```

``` python
# PaddlePaddle 示例：
x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.asin(x)
print(out)
# 输出
# [-0.41151685 -0.20135792  0.10016742  0.30469265]
```
