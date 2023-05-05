## [仅 torch 参数更多]torch.bmm

### [torch.bmm](https://pytorch.org/docs/stable/generated/torch.bmm.html?highlight=bmm#torch.bmm)

```python
torch.bmm(input,mat2,*,out=None)
```

### [paddle.bmm](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/bmm_cn.html)

```python
paddle.bmm(x,y,name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射
|PyTorch|PaddlePaddle|备注|
| ------- | ------- | ------- |
|input|x|矩阵x|
|mat2|y|矩阵y|
|out||输出矩阵，Paddle无此字段，需要进行转写|

### 转写示例

```python
# Pytorch 写法
>>> input = torch.tensor([[[1.0, 1.0, 1.0],
                    	[2.0, 2.0, 2.0]],
                    	[[3.0, 3.0, 3.0],
                    	[4.0, 4.0, 4.0]]])
>>> mat2 = torch.tensor([[[1.0, 1.0],[2.0, 2.0],[3.0, 3.0]],
                    [[4.0, 4.0],[5.0, 5.0],[6.0, 6.0]]]) 
>>> torch.bmm(input, mat2, out=y)

# Paddle 写法
 
import paddle

# In imperative mode:
# size x: (2, 2, 3) and y: (2, 3, 2)
x = paddle.to_tensor([[[1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0]],
                    [[3.0, 3.0, 3.0],
                    [4.0, 4.0, 4.0]]])
y = paddle.to_tensor([[[1.0, 1.0],[2.0, 2.0],[3.0, 3.0]],
                    [[4.0, 4.0],[5.0, 5.0],[6.0, 6.0]]])
out = paddle.bmm(x, y)
print(out)
# Tensor(shape=[2, 2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [[[6. , 6. ],
#          [12., 12.]],

#         [[45., 45.],
#          [60., 60.]]])

```
