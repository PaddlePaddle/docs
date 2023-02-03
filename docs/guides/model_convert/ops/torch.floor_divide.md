## torch.floor_divide
### [torch.floor_divide](https://pytorch.org/docs/stable/generated/torch.floor_divide.html?highlight=floor_divide#torch.floor_divide)

```python
torch.floor_divide(input, 
                    other, 
                    *, 
                    out=None)
```

### [paddle.floor_divide](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/floor_divide_cn.html#floor-divide)

```python
paddle.floor_divide(x, 
                    y, 
                    name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 被除数。                                      |
| other         | y            | 除数。                                       |
| out           | -            | 表示输出的Tensor，PaddlePaddle无此参数。               |


### 代码示例
``` python
# PyTorch示例：
a = torch.tensor([4.0, 3.0])
b = torch.tensor([2.0, 2.0])
torch.floor_divide(a, b)
# 输出
# tensor([2.0, 1.0])
torch.floor_divide(a, 1.4)
# 输出
# tensor([2.0, 2.0])
```

``` python
# PaddlePaddle示例：
x = paddle.to_tensor([2, 3, 8, 7])
y = paddle.to_tensor([1, 5, 3, 3])
z = paddle.floor_divide(x, y)
print(z)  
# 输出
# [2, 0, 2, 2]
```
