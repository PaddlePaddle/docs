## [ 仅 API 调用方式不一致 ]transformers.AddedToken
### [transformers.AddedToken](https://github.com/huggingface/transformers/blob/d625294d79341662784495551abdf45e6cb9372f/src/transformers/tokenization_utils_base.py#L84)
```python
transformers.AddedToken(content: str, single_word=False, lstrip=False, rstrip=False, special=False, normalized=None)
```

### [paddleformers.transformers.AddedToken](https://github.com/PaddlePaddle/PaddleFormers/blob/8a544d93fab8984c223218d2e31328ffefb01f67/paddleformers/transformers/legacy/tokenizer_utils_base.py#L84)
```python
paddleformers.transformers.AddedToken(content: str, single_word=False, lstrip=False, rstrip=False, special=False, normalized=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
transformers.AddedToken("newToken", single_word=True)

# Paddle 写法
paddleformers.transformers.AddedToken("newToken", single_word=True)
```
