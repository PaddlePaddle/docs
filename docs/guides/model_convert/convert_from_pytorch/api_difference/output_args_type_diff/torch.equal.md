## [ 返回参数类型不一致 ]torch.equal
### [torch.equal](https://pytorch.org/docs/stable/generated/torch.equal.html?highlight=equal#torch.equal)

```python
torch.equal(input,
            other)
```

### [paddle.equal_all](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/equal_all_cn.html#equal-all)

```python
paddle.equal_all(x,
                 y,
                 name=None)
```

两者功能一致但返回参数类型不同，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                          |
| ------- | ------------ | ----------------------------- |
| input   | x            | 输入 Tensor，仅参数名不一致。 |
| other   | y            | 输入 Tensor，仅参数名不一致。 |
| 返回值   | 返回值        | PyTorch 返回 bool 类型，Paddle 返回 0-D bool Tensor，需要转写。|

### 转写示例
#### 返回值
```Python
# torch 中的写法
torch.equal(x, y)

# paddle 中的写法
paddle.equal_all(x, y).item()
```
