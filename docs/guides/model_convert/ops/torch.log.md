## torch.log
### [torch.log](https://pytorch.org/docs/stable/generated/torch.log.html?highlight=log#torch.log)

```python
torch.log(input, 
            *, 
            out=None)
```

### [paddle.log](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/log_cn.html#log)

```python
paddle.log(x, 
            name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                                      |
| out           | -            | 表示输出的Tensor，PaddlePaddle无此参数。               |


### 代码示例
``` python
# PyTorch示例：
a = torch.rand(5) * 5
a
# 输出
# tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739])
torch.log(a)
# 输出
# tensor([ 1.5637,  1.4640,  0.1952, -1.4226,  1.5204])
```

``` python
# PaddlePaddle示例：
x = [[2,3,4], [7,8,9]]
x = paddle.to_tensor(x, dtype='float32')
res = paddle.log(x)
# 输出
# [[0.693147, 1.09861, 1.38629], [1.94591, 2.07944, 2.19722]]
```
