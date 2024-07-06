# Paddle 中的类型提示与 Q&A

Python 在 3.5 版本通过 [PEP 484 – Type Hints](https://peps.python.org/pep-0484) 正式规范了 `类型提示` 功能，以帮助开发者提高代码质量。Paddle 推荐开发者使用此特性，并且将静态类型检查工具 (如 `mypy` ) 集成在 CI 流水线中，以保证基础的类型标注的准确性。但是，由于 Paddle 中存在较多非公开 API 与 c++ 接口，目前版本 (2.6.0) 不声明 Paddle 具有类型标注的完备性

## Paddle 中的类型提示

Paddle 中的 `类型提示` 主要关注以下几个部分：

- 函数输入参数的类型
- 函数输出参数的类型

以如下函数为例：

``` python
def greeting(
    name: str                                   # (1)
) -> str:                                       # (2)
    """
    Say hello to your friend!

    Args:
        name (str): The name of your friend.    # (3)

    Returns:
        str, The greeting.                      # (4)
    """
    return 'Hello ' + name
```

其中：

- (1) 函数输入参数的类型
- (2) 函数输出参数的类型
- (3) 函数文档中输入参数的类型
- (4) 函数文档中输出参数的类型

其中 (1) 和 (3) ，以及 (2) 和 (4) 需要一一对应。

## Paddle 中类型提示的实现方案

Python 的类型标注可以通过不同的方式实现，参考 [PEP 561 – Distributing and Packaging Type Information](https://peps.python.org/pep-0561/#implementation) ：

> - The package maintainer would like to add type information inline.
> - The package maintainer would like to add type information via stubs.
> - A third party or package maintainer would like to share stub files for a package, but the maintainer does not want to include them in the source of the package.

Paddle 采用 `Inline type annotation + Stub files in package` 的方案，即：

- Python 接口，使用 `inline` 方式标注，如：

    ``` python
    def greeting(name):
        return 'Hello ' + name
    ```

    需要标注为：

    ``` python
    def greeting(name: str) -> str:
        return 'Hello ' + name
    ```

- 非 Python 接口，提供 `stub` 标注文件，如：

    存在一个 c++ 实现的模块

    ``` shell
    foo
    └── bar.py
    ```

    则应在同一个文件夹下添加 `stub` 文件 `bar.pyi`

    ``` shell
    foo
    ├── bar.py
    └── bar.pyi
    ```

    `stub` 文件不需要实现具体的代码逻辑，只需要保留函数定义，具体可以参考 [PEP 561 – Distributing and Packaging Type Information](https://peps.python.org/pep-0561/#implementation)

Python 目前 (`3.12` 版本) 已经完成的相关 `PEP` 有 `21` 个，具体的实现方案可参考 [Typing PEPs](https://peps.python.org/topic/typing/) 。

## Q&A

### **问:** 我该如何下手？

答：Python 的类型标注特性一直在完善，目前已经是个相对庞大的体系了。

可以先学习一下 Python 官方的文档：[Static Typing with Python](https://typing.readthedocs.io/en/latest/)，熟悉一下相关的 PEP 。

以 `通过 CI 检查` 作为最基础的实现目标。

另外，目前 Paddle 添加了 `_typing` 模块，对于一些常用的公用类型做了统一整理，如：

``` pyton
# python/paddle/_typing/layout.py
DataLayout2D: TypeAlias = Literal["NCHW", "NHCW"]
DataLayout3D: TypeAlias = Literal["NCDHW", "NDHWC"]
```

标注时应尽量使用 `_typing` 模块中的类型，以方便后续维护。

### **问:** docstring 中的 Args 与 type annotation 有什么区别？

答：Paddle 之前的版本 (2.6.0 及以前) 未统一进行类型标注，而在 docstring 中描述了参数类型。
docstring 中 Args 的参数类型以方便用户理解为目的，在与 type annotation 不冲突的前提下，可以保持简洁。如：

``` python
def test(a: int | list[int] | tuple[int, ...]) -> None:
    """
    ...

    Args:
        a (int|list|tuple): xxx

    Returns:
        None, xxx

    ...
    """
```

### **问:** docstring 中的 Args 与 type annotation 不一致怎么办？

答：首先需要保证 type annotation 的正确性，如果 docstring 原有 Args 中的类型不正确，需要进行修改，并且，同时检查此接口的 `中文文档` (即 `docs`)是否正确，如发现错误，需要对 `docs` 单独提 PR 进行修改。

### **问:** 该使用 `Union` 还是 `|` 以及 `from __future__ import annotations` ？

答：尽可能的使用 `|` ，通常需要 `from __future__ import annotations` 。

由于目前 Paddle (2.6.0) 支持的 Python 最低版本为 `3.8` ，因此，`|` 只能在类型标注的情况下使用，而不能在表达式中使用，并且，同时需要  `from __future__ import annotations`，如：

``` python
from __future__ import annotations
def test(a: int | str): ...
```

而在表达式中仍使用 `Union` ：

``` python
from typing import Union
t = Union[int, str]
```

可参考 [PEP 563 – Postponed Evaluation of Annotations](https://peps.python.org/pep-0563/) 。

### **问:** 如果测试无法通过怎么办？

答：可以使用 `# type: ignore` 进行规避。

Paddle 通过工具 (如 `mypy`) 对接口的示例代码进行检查，进而保证类型标注的正确性。

类型标注的过程中，难免产生接口依赖问题，如果依赖的是 `私有接口` 或 `外部接口` ，则可以使用 `# type: ignore` 规避相应的类型检查，如：

``` python
>>> import abcde # type: ignore
>>> print('ok')
```

或者规避整个代码检查：

``` python
>>> # type: ignore
>>> import abcde
>>> print('ok')
```

### **问:** 能否使用 `Any` 类型？

答：可以，但应尽量避免。

### **问:** 如果出现 `circular import` 错误怎么办？

答：出现此情况可以参考以下处理方法：

- 添加 `from __future__ import annotations`
- 将类型单独通过 `typing.TYPE_CHECKING` 引入，如：

  ``` python
  from typing import TYPE_CHECKING
  if TYPE_CHECKING:
      import paddle.xxx as xxx

  def tmp() -> xxx: ...
  ```

  另外，如果标注的类型仅用作 type hints，也尽可能的使用 `TYPE_CHECKING` ，以减少不必要的模块导入。

### **问:** 使用 `Tensor` 还是 `Variable`？

答：尽量使用 `Tensor` ，不将静态图的 `Variable/Value` 概念暴露给用户。

更详细的讨论可以参考 https://github.com/PaddlePaddle/community/pull/858#discussion_r1564552690

### **问:** 如果遇到需要根据不同输入类型有不同输出类型的函数怎么办？

答：出现此情况可以参考以下处理方法：

- 添加 `from typing import overload`
- 标注多个同名函数，并用装饰器装饰，如：

  ``` python
  from typing import overload

  @overload
  def array_length(array: list[Any]) -> int:...

  @overload
  def array_length(array: paddle.Tensor) -> paddle.Tensor:...

  def array_length(array): ... # 具体实现的代码，不再进行标注
  ```

### **问:** 什么时候用 `Sequence` ，什么时候用 `list` 和 `tuple`？

答：Python 的 PEP 中有提示：

> Note: Dict, DefaultDict, List, Set and FrozenSet are mainly useful for annotating return values. For arguments, prefer the abstract collection types defined below, e.g. Mapping, Sequence or AbstractSet.

也就是说，输入中用 `Sequence` ，返回值用 `list` 。

但是，如果代码中使用到了 `list` 的方法，如 `append` ，或者明确表示此输入只能是 `list` ，则不应再使用 `Sequence` 。

### **问:** 标注的时候用 `Tensor` 还是 `paddle.Tensor`？

答：两者皆可。

若文件中出现较多 `paddle.Tensor` ，出于简洁的考虑，可以使用 `Tensor` 代替，但是需要在导入包时注意：

``` python
if TYPE_CHECKING:
    from paddle import Tensor
```

可参考讨论：https://github.com/PaddlePaddle/Paddle/pull/65073#discussion_r1636116450

### **问:** 该用 `paddle.framework.Block`  还是 `paddle.pir.Block`？

答：统一使用 `paddle.pir.Block`。

可参考讨论：https://github.com/PaddlePaddle/Paddle/pull/65095#discussion_r1637570850

## 参考资料

- [PEP 484 – Type Hints](https://peps.python.org/pep-0484/)
- [PEP 563 – Postponed Evaluation of Annotations](https://peps.python.org/pep-0563/)
- [PEP 561 – Distributing and Packaging Type Information](https://peps.python.org/pep-0561/#implementation)
- [Typing PEPs](https://peps.python.org/topic/typing/)
- [Static Typing with Python](https://typing.readthedocs.io/en/latest/index.html#)
