## torch.arctan
### [torch.arctan](https://pytorch.org/docs/stable/generated/torch.arctan.html?highlight=arctan#torch.arctan)

```python
torch.arctan(input,
            *,
            out=None)
```

### [paddle.atan](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/atan_cn.html#atan)

```python
paddle.atan(x,
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
# tensor([ 0.2341,  0.2539, -0.6256, -0.6448])
torch.arctan(a)
# 输出
# tensor([ 0.2299,  0.2487, -0.5591, -0.5727])
```

``` python
# PaddlePaddle 示例：
x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.atan(x)
print(out)
# 输出
# [-0.38050638 -0.19739556  0.09966865  0.29145679]
```
