## [torch 参数更多]torch.set_printoptions

### [torch.set_printoptions](https://pytorch.org/docs/stable/generated/torch.set_printoptions.html?highlight=torch+set_printoptions#torch.set_printoptions)

```python
torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=None)
```

### [paddle.set_printoptions][https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/set_printoptions_cn.html]

```python
paddle.set_printoptions(precision=None, threshold=None, edgeitems=None, sci_mode=None, linewidth=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

|  PyTorch  | PaddlePaddle |                             备注                             |
| :-------: | :----------: | :----------------------------------------------------------: |
| precision |  precision   |                    表示浮点数的小数位数。                    |
| threshold |  threshold   |                   表示打印的元素个数上限。                   |
| edgeitem  |   edgeitem   |           表示以缩略形式打印时左右两边的元素个数。           |
| linewidth |  linewidth   |              仅参数名不一致，表示每行的字符数。              |
|  profile  |      -       | Paddle 无此参数，需要转写，表示启用(True)或禁用(False)科学计数法。 |
| sci_mode  |   sci_mode   |                  表示是否以科学计数法打印。                  |
