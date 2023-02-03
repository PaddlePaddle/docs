## torch.sinh
### [torch.sinh](https://pytorch.org/docs/stable/generated/torch.sinh.html?highlight=sinh#torch.sinh)

```python
torch.sinh(input,
            *,
            out=None)
```

### [paddle.sinh](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/sinh_cn.html#sinh)

```python
paddle.sinh(x,
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
# tensor([ 0.5380, -0.8632, -0.1265,  0.9399])
torch.sinh(a)
# 输出
# tensor([ 0.5644, -0.9744, -0.1268,  1.0845])
```

``` python
# PaddlePaddle 示例：
x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.sinh(x)
print(out)
# 输出
# [-0.41075233 -0.201336    0.10016675  0.30452029]
```
