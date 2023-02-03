## torch.abs
### [torch.abs](https://pytorch.org/docs/stable/generated/torch.abs.html?highlight=abs#torch.abs)

```python
torch.abs(input,
            *,
            out=None)
```

### [paddle.abs](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/abs_cn.html#abs)

```python
paddle.abs(x,
            name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                                     |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |


### 代码示例
``` python
# PyTorch 示例：
torch.abs(torch.tensor([-1, -2, 3]))
# 输出
# tensor([ 1,  2,  3])
```

``` python
# PaddlePaddle 示例：
x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.abs(x)
print(out)
# 输出
# [0.4 0.2 0.1 0.3]
```
