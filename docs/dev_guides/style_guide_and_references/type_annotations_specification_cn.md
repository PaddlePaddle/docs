# Python 类型提示标注规范

Python 是一门动态类型语言，变量的类型在运行时确定，因此不需要在函数参数和返回值声明类型。这虽然给开发带来了很大的灵活性，让开发者不再需要苦恼于解决类型错误，但同时也给大型项目的维护和代码的可读性带来了挑战。为了避免这一问题，Python 从 3.0 开始引入类型注解功能，并不断迭代和规范，如今已经形成了一套静态类型检查体系。这不仅为开发者提供了标注类型的能力，增加代码可读性，也催生了 [Mypy](https://github.com/python/mypy)、[Pyright](https://github.com/microsoft/pyright) 等优秀的静态类型检查工具，使开发者能够在静态检查阶段就发现浅显的类型错误，提升代码的鲁棒性。

虽然 Python 为开发者提供了标注类型注解的功能，但并不强制用户使用。相比于静态类型的语言（诸如 Rust、C++ 等），Python 没有强制根据类型注解来强制检查的编译时，相比于其它动态类型语言（诸如 JavaScript/TypeScript 等），Python 现有的类型提示的规范尚在迭代完善中，现有的静态类型表达能力相对较弱，这导致了相关的资料并不多。因此大多数项目在实际开发中，类型提示的覆盖率还是比较低的，即便有些项目在代码中添加了类型提示，也可能存在类型提示不准确、不完整的问题。

为了确保 Paddle 内所标注的类型提示有着良好的一致性和准确性，我们根据过往项目的开发经验，总结出了一系列的类型提示集成的最佳实践，并将其应用于 Paddle 项目中。在此基础上，我们制定了以下的类型提示规范，确保所有开发者能够最快写出符合规范的类型提示。

## 类型提示标注范围

Paddle 是一个大型开源项目，包含了数十万行的 Python 代码，不乏有历史悠久的代码，为全部函数都标注类型提示是不可取的，也是不现实的。因此，我们需要明确界定哪些代码是需要标注类型提示的，哪些代码是不需要标注类型提示的。

### 公开 API 必须标注类型提示

公开 API 是指会被用户直接使用的函数、类等，这些 API 是用户与 Paddle 交互的接口，因此必须标注完善且准确的类型提示，以确保用户拥有最佳的开发体验。

这里以一个函数为例：

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

这里 `(1)`、`(2)` 为函数签名中的类型提示，`(3)`、`(4)` 为函数文档中的类型提示，这两者需要一一对应。

再以一个类为例：

``` python
from typing import ClassVar

class User:
    count: ClassVar[int] = 0

    name: str

    def __init__(self, name: str):
        self.name = name
        self._age = 18
```

这里 `count: ClassVar[int]` 为类属性的类型提示，`name: str` 为实例属性的类型提示。这里对于实例属性 `name: str`，虽然 `__init__` 方法中已经为参数 `name` 标注了 `str` 类型，对于大多数静态类型检查工具来说，也能够正确推断出 `name` 的类型，但为了确保用户类型提示的准确性，我们建议尽可能为这样的公开属性标注类型提示，私有属性则可以不标注。

### 非公开 API 可选标注类型提示

对于非公开 API，我们不强制要求标注类型提示，是否标注按照团队和开发者的习惯来决定。但是，我们建议开发者在编写新的代码时，尽量标注类型提示，以提高代码的可读性和可维护性。

## 类型提示正确性保证

类型提示的正确性只能由静态类型检查工具来约束，因此我们在 CI 中引入了静态类型检查工具 `Mypy`。鉴于示例代码是用户学习和使用 Paddle 的重要参考，我们要求示例代码必须通过静态类型检查，以同时保障示例代码和公开 API 的类型提示正确性。

当开发者对公开 API 进行修改时，CI 会利用 `Mypy` 对该 API 的示例代码进行检查，确保修改后类型提示仍然是正确的。

## 最佳实践

我们根据过往项目的开发经验以及在 Paddle 项目中类型提示的集成经验，总结出了一系列的最佳实践，以供开发者参考。

### 使用 PEP 563，延迟类型注解计算

[PEP 563](https://peps.python.org/pep-0563/) 提出了一种延迟类型注解计算的方式，可以通过在代码首行添加 `from __future__ import annotations` 来引入该功能（后续简称 PEP 563）。该功能有如下优点：

- 前向引用：可以在类型注解中提前使用后续定义的类型
- 消除导入模块时类型提示的计算开销
- 在类型注解上下文中，在低版本 Python 使用部分高版本才能使用的特性（如 [PEP 585](https://peps.python.org/pep-0585/) 标准集合类型、[PEP 604](https://peps.python.org/pep-0604/) 简化的 Union type 写法 `X | Y` 等），使代码更加现代化，并降低跨版本兼容成本。

比如在考虑 Python 3.8 兼容性的考虑下，如果不使用 PEP 563，则需要编写如下代码：

```python
from typing import List, Union, Sequence

def search(user: "User", keywords: Sequence[str]) -> Union[List[str], None]:
    ...

class User: ...
```

而使用 PEP 563 后则可以简化为：

```python
from __future__ import annotations

from collections.abc import Sequence

def search(user: User, keywords: Sequence[str]) -> list[str] | None:
    ...

class User: ...
```

因此我们总是建议使用 PEP 563，以确保 Paddle 类型提示的简洁性和现代化。

> 注意 PEP 563 会在 Python 3.14 被 [PEP 649](https://peps.python.org/pep-0649/) 取代，但在 Python 3.13 及以前，PEP 563 仍然有着重要的作用。因此我们会继续使用 PEP 563 直到 Python 3.13 退场。

值得注意的是 PEP 563 的生效范围仅为类型注解上下文，对于其它语法上下文是无效的，一种典型的 case 就是 `TypeAlias`，比如：

```python
from __future__ import annotations

from collections.abc import Sequence

from typing import Literal

from typing_extensions import TypeAlias

IntOrStr: TypeAlias = int | str  # 不生效，3.8 仍然会报错

class SequenceInt(Sequence[int]): ...  # 不生效，3.8 仍然会报错
```

对于此类情况，我们仍然需要使用 Python 3.8 兼容的形式，如：

```python
from __future__ import annotations

from typing import Literal, Union, Sequence

from typing_extensions import TypeAlias

IntOrStr: TypeAlias = Union[int, str]

class SequenceInt(Sequence[int]): ...
```

> 后续示例代码默认使用 PEP 563，不再重复说明。

### 仅类型提示相关的导入放在 `if TYPE_CHECKING` 下

我们在标注类型时，经常会需要额外 import 一些其他模块定义的类型，这些类型通常只在类型提示时使用，而在运行时并不需要。为了避免不必要的模块导入，我们建议将这些类型提示相关的导入放在 `if TYPE_CHECKING` 下，如：

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paddle import Tensor
    from another_module import AnotherType
```

这一方面避免了运行时的额外模块导入带来的开销，也能够避免循环导入问题，同时也能够更好的区分类型提示相关的导入和运行时相关的导入，使导入部分代码更加清晰。

### 尽可能使用通用类型

为了确保框架内类型提示标注的一致性和可重用性，我们整理了一系列通用类型，放在 `_typing` 模块中，涵盖了 shape、dtype、device、data layout 等常用类型。开发者在标注类型时，应尽可能使用这些类型，同时，也应该以类型提示为指导，在 API 设计上尽可能涵盖所有可能的输入类型。

比如对于如下函数：

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paddle._typing import TensorLike
    from paddle import Tensor

# TensorLike 定义如下
# TensorLike: TypeAlias = Union[npt.NDArray[Any], "Tensor", Numberic]

def add(a: Tensor, b: TensorLike) -> Tensor:
    if isinstance(b, np.ndarray):
        return dispatch_np_add(a, b)
    elif isinstance(b, paddle.Tensor):
        return dispatch_paddle_add(a, b)
    else:
        return dispatch_numberic_add(a, b)
```

这里 `TensorLike` 是一个通用类型，包含了 `np.ndarray`、`paddle.Tensor`、`Numberic` 三种类型，因此在实现中也应该考虑到这三种类型的输入。

### 使用更加明确的类型以提供更好的提示效果

在类型提示的标注过程中，我们应该尽可能使用更加明确的类型，以提供最佳的提示效果。比如对于如下函数：

```python
from typing import Any

def save(options: dict[str, Any]) -> None: ...
```

这里 `options` 的类型是 `dict[str, Any]`，这样的类型提示虽然能够正确的提示 `options` 的类型是一个字典，但是对于字典的键值对的类型并没有进行明确的标注，这样的类型提示对于开发者来说并不友好。因此我们建议如果可能，使用 [`TypedDict`](https://docs.python.org/3/library/typing.html#typing.TypedDict) 为 `dict` 标注更加明确的类型，如：

```python
from typing import TypedDict

class SaveOptions(TypedDict):
    path: str
    id: int

def save(options: SaveOptions) -> None: ...
```

这样的类型提示不仅能够提示 `options` 是一个字典，还能够提示 `options` 的键值对的类型，这会带来如下几点优点：

- 对于输入参数 key，IDE 可以提供下拉菜单直接选择，提高开发效率
- 对于输入参数 value，IDE 可以提供类型检查，减少低级错误
- 对于输出参数，可以为下游使用提供更好的提示

除去 `TypedDict`，我们还建议使用 [`NamedTuple`](https://docs.python.org/3/library/typing.html#typing.NamedTuple) 代替 [`collections.namedtuple`](https://docs.python.org/3/library/collections.html#collections.namedtuple)，以及使用 `Literal` 代替 `str` 等更宽泛的类型。

比如：

```python
from collections import namedtuple

User = namedtuple("User", ["name", "age", "type"])

def get_user() -> User: ...

def filter_user(user: list[User], type: str) -> list[User]: ...
```

可以改写为：

```python
from typing import NamedTuple, Literal

from typing_extensions import TypeAlias

UserType: TypeAlias = Literal["admin", "user"]

class User(NamedTuple):
    name: str
    age: int
    type: Literal["admin", "user"]

def get_user() -> User: ...

def filter_user(user: list[User], type: UserType) -> list[User]: ...
```

> 值得注意的是，使用 `Literal` 来替代 `str` 等类型并不是一种绝对的最佳实践，而是根据具体情况而定。因为 `Literal` 表示的是在静态检查阶段就确定了的值，事实上，一些静态检查阶段无法确定的值也有可能是合法的，比如用户输入的字符串 `"user"`（比如通过 `input` 函数获取的）。如果想要使用有限集合，使用 `Enum` 永远是更好的选择。
>
> 对 Paddle 来说，有大量存量 API 使用字符串类型来直接表示某些特定的值，这些值是有限的，且大多数情况都是直接作为字面量直接传参，因此使用 `Literal` 可以提供更好的提示效果，虽然从类型检查角度来说可能导致传入 `str` 类型的值无法通过检查，但这种写法并不常见。

### 参数应尽可能使用抽象类型，返回值应尽可能使用具体类型

对于函数输入参数，如果允许，我们应该尽可能使用 [Protocal](https://docs.python.org/3/library/typing.html#typing.Protocol)，如 [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)、[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping) 、[Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable) 等抽象类型，以提高函数的通用性。而对于函数返回值，我们应该尽可能使用具体类型，以确保下游使用时能得到更好的提示效果。

比如相比于如下写法：

```python
from collections.abc import Sequence, Iterable, Mapping

def map_it(input: list[str]) -> list[int]: ...
def create_map() -> Mapping[str, int]: ...
```

我们更推荐如下写法：

```python
from collections.abc import Sequence, Iterable, Mapping

def map_it(input: Sequence[str]) -> list[int]: ...
def create_map() -> dict[str, int]: ...
```

类似地，我们应该尽可能避免在返回值出现 `Union` 类型，因为这会导致下游需要 `isinstance` 判断返回值的类型，比如：

```python
from collections.abc import Iterable, Callable

def process_data(
    data: Iterable[str],
    processor: Callable[[str], str],
    with_index: bool = True,
) -> list[str] | list[tuple[int, str]]:
    if with_index:
        return list(enumerate(map(processor, data)))
    else:
        return list(map(processor, data))

data = ["a", "b", "c"]
processor = lambda x: x.upper()

processed_data = process_data(data, processor, with_index=True)
reveal_type(processed_data)  # -> Revealed type is 'list[str] | list[tuple[int, str]]'
```

对于这个 case，返回值是一个 `Union`，下游如果想要确定类型就需要额外判断，而对于这个 case，我们完全可以根据 `with_index` 的字面量来确定返回值的类型，这可以通过 [`overload`](https://docs.python.org/3/library/typing.html#typing.overload) 来实现，因此我们更推荐如下写法：

```python
from collections.abc import Callable, Iterable
from typing import Literal, overload


@overload
def process_data(
    data: Iterable[str],
    processor: Callable[[str], str],
    with_index: Literal[False] = ...,
) -> list[str]: ...


@overload
def process_data(
    data: Iterable[str],
    processor: Callable[[str], str],
    with_index: Literal[True] = ...,
) -> list[tuple[int, str]]: ...


@overload
def process_data(
    data: Iterable[str],
    processor: Callable[[str], str],
    with_index: bool = ...,
) -> list[str] | list[tuple[int, str]]: ...


def process_data(
    data,
    processor,
    with_index=True,
):
    if with_index:
        return list(enumerate(data))
    else:
        return list(data)

data = ["a", "b", "c"]
processor = lambda x: x.upper()

processed_data_with_index = process_data(data, processor, with_index=True)
reveal_type(processed_data_with_index)  # -> Revealed type is 'list[tuple[int, str]]'

processed_data_without_index = process_data(data, processor, with_index=False)
reveal_type(processed_data_without_index)  # -> Revealed type is 'list[str]'

def get_bool() -> bool: ...
with_index = get_bool()
processed_data_with_index_or_not = process_data(data, processor, with_index=get_bool())
reveal_type(
    processed_data_with_index_or_not
)  # -> Revealed type is 'list[tuple[int, str]]' | 'list[str]'
```

这里通过添加两个 `Literal` 的 `overload` 来明确返回值的类型，这样可以尽可能避免下游对返回值类型的判断。当然这个技巧也不是所有情况都适用的，对于一些场景返回值就是 `Union` 类型的情况，可以考虑拆分函数、使用泛型参数等方式来解决。

### 区分异构数组和同构数组类型

对于数组类型，我们可以将其分为异构数组和同构数组两种，异构数组表示数组内的元素类型不一致，往往拥有多个泛型参数，同构数组表示数组内的元素类型一致，往往只有一个泛型参数。

Python 的 `list` 类型是同构数组类型，因此它只能拥有一个泛型参数，比如 `list[int]`，表示该 list 只包含 `int` 类型的元素。但不限定元素数量。

Python 的 `tuple` 类型是异构数组类型，因此它可以拥有多种不同类型的元素，比如 `tuple[int, str, bool]`，表示该 tuple 只包含三个元素，依次分别为 `int`、`str`、`bool` 类型。如果想要表示一个同构的 `tuple` 类型，则需要使用 `tuple[int, ...]`，表示该 tuple 只包含 `int` 类型的元素，但不限定元素数量。这是初学者常见的误区，经常会误以为 `tuple` 也是同构数组类型，因此使用 `tuple[int]` 来表示不定长的 `int` 类型 tuple，但是实际上这只表示仅包含一个 `int` 类型元素的 tuple。

### `*args` 和 `**kwargs` 的类型标注

对于 `*args` 和 `**kwargs`，类型注解所提供的类型是每个元素的类型，比如：

```python
def fn(*args: int, **kwargs: bool) -> None: ...
```

表示的是 `*args` 里所有元素都是 `int` 类型，也即 `args` 类型为 `tuple[int, ...]`，`**kwargs` 里所有元素都是 `bool` 类型，也即 `kwargs` 类型为 `dict[str, bool]`。

对于大多数情况来说，`*args` 和 `**kwargs` 内的元素并不是相同的，在这种情况下，我们可以利用 `Unpack` 配合 `TypedDict` 等类型来提供更好的约束，比如：

```python
from typing import TYPE_CHECKING, TypedDict

from typing_extensions import Unpack

if TYPE_CHECKING:
    class _Options(TypedDict, total=False):
        x: int
        y: str

def fn(*args: Unpack[tuple[int, str, bool]], **kwargs: Unpack[_Options]) -> None: ...
```

这里将 `*args` 的类型标注为 `Unpack[tuple[int, str, bool]]`，表示 `args` 必须传递三个参数，分别是 `int`、`str`、`bool` 类型。这种用法并不常见，因为 `*args` 很少会限制参数的个数，只在极少数场景下有用。

而 `**kwargs` 的类型标注为 `Unpack[_Options]`，表示 `kwargs` 只能传递 `x` 和 `y` 两个参数，其中 `x` 的类型是 `int`，`y` 的类型是 `str`，而 `total=False` 表示所有参数都是可选的。这种用法非常常见，建议所有 `**kwargs` 都使用此种类型进行标注。

### 对于重复出现的类型，使用类型别名减少冗余代码

如果一个复杂的类型在代码中需要用到多次，我们可以使用类型别名来减少冗余代码，提高代码的可读性。比如：

```python
from typing_extensions import TypeAlias

UserType: TypeAlias = Literal["admin", "user"]

def get_user_type() -> UserType: ...
def set_user_type(user_type: UserType) -> None: ...
```

### 泛型参数命名规范

泛型参数应统一以大写字母结尾（如 `T`），以与类型别名区分。常见泛型参数后缀如下：

- `T`：表示任意类型
- `K`：表示键类型
- `V`：表示值类型
- `P`：表示参数类型（`ParamSpec`）

特别地，对于序列类型（`TypeVarTuple`），我们建议在泛型参数后加上 `s`，如 `Ts`，以与单个元素的类型区分。

此外还有一些后缀用于表示该泛型参数的特性：

- `_co`：表示协变类型
- `_contra`：表示逆变类型

非暴露的泛型参数应以 `_` 开头，如 `_T`。

比如：

```python
from typing import TypeVar

from typing_extensions import ParamSpec, TypeVarTuple

_T = TypeVar('_T')
_T_co = TypeVar('_T_co', covariant=True)
_T_contra = TypeVar('_T_contra', contravariant=True)

_Ts = TypeVarTuple('_Ts')

_K = TypeVar('_K')
_V = TypeVar('_V')

_InputT = ParamSpec('_InputT')
_RetT = TypeVar('_RetT')
```

### 使用泛型类时必须写明参数类型

Python 允许泛型类省略掉泛型参数，会隐式地将泛型参数视为 `Any` 类型，这可能会导致类型检查不够精准，且部分静态类型检查工具在严格模式下无法进行正确的类型推导，因此我们建议在使用泛型类时必顽写明泛型参数，即便使用 `Any`，如：

因此相比于如下写法：

```python
from collections.abc import Callable, Sequence

def process_data(data: Sequence, processor: Callable) -> None: ...
```

我们更推荐如下写法：

```python
from collections.abc import Callable, Sequence
from typing import Any

def process_data(data: Sequence[int], processor: Callable[[int], int]) -> None: ...
# 如果泛型参数真的可以是任意类型，可以使用 Any，但不要省略
def process_data(data: Sequence[Any], processor: Callable[..., Any]) -> None: ...
```

### 当输出类型与输入类型一同变化时应考虑使用泛型

如果一个函数的输出类型与输入类型有关联，我们可以考虑使用泛型参数来表示这种关联。比如：

```python
from typing import TypeVar

_T = TypeVar('_T')

def process_data(data: list[_T]) -> list[_T]: ...
```

这里 `process_data` 函数的输入类型与输出类型是一致的，我们可以使用泛型参数 `_T` 来表示这种关联。

如果需要限定泛型参数的类型，可以传入 `constraints`，如：

```python
from typing import TypeVar

_T = TypeVar('_T', int, float)  # _T 只能是 int 或 float

def process_data(data: list[_T]) -> list[_T]: ...
```

### 如果函数使用了装饰器，装饰器同样应该标注类型

如果函数使用了装饰器，装饰器同样应该标注类型，以确保类型提示能够正确推导和传递。比如：

```python
from collections.abc import Callable
from typing import TypeVar

from typing_extensions import ParamSpec

_InputT = ParamSpec('_InputT')
_RetT = TypeVar('_RetT')


def decorator(fn: Callable[_InputT, _RetT]) -> Callable[_InputT, _RetT]:
    def wrapper(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
        ...
    return wrapper

@decorator
def fn(x: int) -> str:
    ...
```

### 使用 `Protocol` 表示复杂的函数类型

当我们将函数作为参数时，可以直接使用 [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) 来进行标注，比如：

```python
from collections.abc import Callable

def process_data(data: list[int], processor: Callable[[int], int]) -> list[int]: ...
```

但对于某些特殊函数，`Callable` 的表达能力可能不够，比如包含 `overload` 的场景：

```python
from typing import Protocol, overload

class Processor(Protocol):
    @overload
    def __call__(self, value: int) -> int: ...
    @overload
    def __call__(self, value: tuple[int]) -> int: ...

def process_data(data: list[int], processor: Processor) -> list[int]: ...
```

### 如果函数用于确定输入类型，应当使用 `TypeGuard` 或 `TypeIs`

如果函数用于确定输入类型，应当使用 `TypeGuard` 或 `TypeIs` 来标注返回值，以确保使用该函数结果为条件的控制流能够正确 Narrow 类型。比如：

```python
from typing_extensions import TypeIs


def is_str(x: int | str) -> TypeIs[str]:
    return isinstance(x, str)

x: int | str = ...

if is_str(x):
    reveal_type(x)  # -> Revealed type is 'str'
else:
    reveal_type(x)  # -> Revealed type is 'int'
```

### Docstring 中的类型信息

Docstring 中 Args 的参数类型以方便用户理解为目的，在与类型提示不冲突的前提下，可以保持简洁。如：

``` python
def fn(a: int | list[int] | tuple[int, ...]) -> None:
    """
    ...

    Args:
        a (int|list|tuple): xxx

    Returns:
        None, xxx
    """
```

如果类型提示与 Docstring 发生了不一致，首先需要保证类型提示的正确性，如果 Docstring 原有 Args 中的类型不正确，需要进行修改，并且，同时检查此接口的中文文档是否正确，如发现错误，需要向 `docs` repo 单独提 PR 进行修改。

### 动静统一 API 应统一使用 Tensor 标注

Paddle 大多数组网 API 是动静统一的，对于这些 API 是同时支持输入动态图 `Tensor` 和静态图 `Value` 的，但是我们不希望将静态图的 `Value` 暴露给用户，因此我们建议这些 API 统一使用 `Tensor` 标注，如：

``` python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paddle import Tensor

def add(a: Tensor, b: Tensor) -> Tensor: ...
```

### 如有必要，可以使用 `# type: ignore` 规避类型检查

对于部分示例代码，可能会因为检查过于严格或者类型检查工具自身的局限性而无法通过，但修改示例代码又会导致示例代码的可读性下降，这时可以使用 `# type: ignore` 规避类型检查，但应该注意 ignore 时应该尽可能精确到具体的类型检查错误，以避免忽略掉潜在的其它类型错误，如：

``` python
>>> x = paddle.rand([10, 2, 2], dtype=dtype)  # type: ignore[arg-type]
```

如果需要 ignore 整个代码块，可以使用整行形式的 `# type: ignore` 注释：

```python
>>> # type: ignore
>>> x = paddle.rand([10, 2, 2], dtype=dtype)
```

## 参考资料

- [Static Typing with Python](https://typing.readthedocs.io/en/latest/index.html#)
- [PEP 484 – Type Hints](https://peps.python.org/pep-0484/)
- [PEP 563 – Postponed Evaluation of Annotations](https://peps.python.org/pep-0563/)
- [Typing PEPs](https://peps.python.org/topic/typing/)
- [typing library documentation](https://docs.python.org/3/library/typing.html)
