## torch.equal
### [torch.equal](https://pytorch.org/docs/stable/generated/torch.equal.html?highlight=equal#torch.equal)

```python
torch.equal(input,
            other)
```

### [paddle.equal](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/equal_cn.html#equal)

```python
paddle.equal(x,
            y,
            name=None)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input        | x            | 输入的 Tensor。                   |
| other        | y            | 输入的 Tensor。                   |


### 功能差异

#### 使用方式
***PyTorch***：返回 bool 类型。
***PaddlePaddle***：返回 0D bool Tensor。


### 代码示例
``` python
# PyTorch 示例：
torch.equal(torch.tensor([1, 2]), torch.tensor([1, 2]))
# 输出
# True
```

``` python
# PaddlePaddle 示例：
x = paddle.to_tensor([1, 2, 3])
y = paddle.to_tensor([1, 2, 3])
z = paddle.to_tensor([1, 4, 3])
result1 = paddle.equal_all(x, y)
print(result1)
# 输出
# result1 = [True ]
result2 = paddle.equal_all(x, z)
print(result2)
# 输出
# result2 = [False ]
```
