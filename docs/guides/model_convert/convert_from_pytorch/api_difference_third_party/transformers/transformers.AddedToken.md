## [参数完全一致]transformers.AddedToken

### [transformers.AddedToken](https://github.com/huggingface/transformers/blob/d625294d79341662784495551abdf45e6cb9372f/src/transformers/tokenization_utils_base.py#L84)

```python
transformers.AddedToken(content: str, single_word=False, lstrip=False, rstrip=False, special=False, normalized=None)
```

### [paddlenlp.transformers.AddedToken](https://github.com/PaddlePaddle/PaddleNLP/blob/e336e78c338d2514ee6c937982ce5d8c960b85ff/paddlenlp/transformers/tokenizer_utils_base.py#L48)

```python
paddlenlp.transformers.AddedToken(content: str = field(default_factory=str), single_word: bool = False， lstrip: bool = False, rstrip: bool = False, normalized: bool = True, special: bool = True)
```

功能一致，参数完全一致，具体如下：

### 参数映射

| transformers | PaddlePaddle | 备注                   |
| ------------ | ------------ | ---------------------- |
| content      | content      | 待添加的 Token 内容。  |
| single_word  | single_word  | token 是否视为独立的词。 |
| lstrip       | lstrip       | 是否移除左侧空白符。    |
| rstrip       | rstrip       | 是否移除左侧空白符。    |
| special      | special      | 是否有特殊的处理方式。  |
| normalized   | normalized  | 是否进行规范化处理。  |
