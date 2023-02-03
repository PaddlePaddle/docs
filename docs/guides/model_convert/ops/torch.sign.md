## torch.sign
### [torch.sign](https://pytorch.org/docs/stable/generated/torch.sign.html?highlight=sign#torch.sign)

```python
torch.sign(input,
            *,
            out=None)
```

### [paddle.sign](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/sign_cn.html#sign)

```python
paddle.sign(x,
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
a = torch.tensor([0.7, -1.2, 0., 2.3])
a
# 输出
# tensor([ 0.7000, -1.2000,  0.0000,  2.3000])
torch.sign(a)
# 输出
# tensor([ 1., -1.,  0.,  1.])
```

``` python
# PaddlePaddle 示例：
x = paddle.to_tensor([3.0, 0.0, -2.0, 1.7], dtype='float32')
out = paddle.sign(x=x)
print(out)
# 输出
# [1.0, 0.0, -1.0, 1.0]
```
