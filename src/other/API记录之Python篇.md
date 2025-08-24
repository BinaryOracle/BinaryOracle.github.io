---
title: API记录之Python篇
icon: file
category:
  - tools
tag:
  - 已发布
footer: 技术共建，知识共享
date: 2025-06-11
author:
  - BinaryOracle
---

`API记录之Python篇` 

<!-- more -->

## Python

### 作用域

Python 的作用域遵循 **LEGB（Local → Enclosing → Global → Built-in）** 原则。

* **Local**：函数或代码块内部定义的名字。

* **Enclosing**：外层函数的作用域。

* **Global**：模块文件的顶层作用域。

* **Built-in**：Python 内置命名空间。

关键点是：

➡️ **Python 没有像 C/C++ 那样的“块级作用域”**。

也就是说，在 `if`、`for`、`while` 这些代码块里定义的变量，**并不会限制在这个代码块内部**，而是直接存在于函数作用域里。


**Python 代码块与作用域对照表**:

| 代码块类型                      | 示例                       | 是否产生新作用域？ | 说明                                                         |
| -------------------------- | ------------------------ | --------- | ---------------------------------------------------------- |
| **模块（module）**             | 一个 `.py` 文件              | ✅         | 文件顶层的名字都在模块作用域内（全局作用域）。                                    |
| **函数定义（def / lambda）**     | `def f(): ...`           | ✅         | 每次调用函数都会创建一个新的局部作用域（Local）。                                |
| **类定义（class）**             | `class A: ...`           | ✅         | 类体代码在独立的命名空间里执行，成员存入类的属性字典。                                |
| **if / else / elif**       | `if cond: x = 1`         | ❌         | 不产生新作用域，变量提升到所在函数/模块作用域。                                   |
| **for / while**            | `for i in range(3): ...` | ❌         | 循环变量在循环体外仍然可见。                                             |
| **try / except / finally** | `try: ... except: ...`   | ❌         | 不产生新作用域，里面的变量外面也能用。                                        |
| **with**                   | `with open(...) as f:`   | ❌         | 不产生新作用域，`f` 在外面仍然可见。                                       |
| **推导式 (Python 3.x)**       | `[x for x in range(5)]`  | ✅（局部作用域）  | 列表/字典/集合/生成器推导式里的循环变量 **只在推导式内部有效**，不会泄漏到外部（Python 2 会泄漏）。 |

### 位置参数与关键字参数

- 位置参数（Positional Argument）：按 位置顺序 传入函数的参数。

- 关键字参数（Keyword Argument）：用 key=value 的形式明确指定的参数。

| 场景      | 位置参数 `*`      | 关键字参数 `**` |
| ------- | ------------- | ---------- |
| **调用时** | 解包 tuple/list | 解包 dict    |
| **定义时** | 收集成 tuple     | 收集成 dict   |


### 闭包与高阶导数

#### 什么是高阶函数？

**高阶函数**（Higher-Order Function）满足以下两个条件之一即可：

1. 函数接收另一个函数作为参数；

2. 函数返回一个函数。

Python 中的 `map`、`sorted`、`functools.partial` 都是高阶函数。

比如这个函数就是高阶函数：

```python
def outer(func):  # 接收函数作为参数
    def inner():
        print("调用前")
        func()
        print("调用后")
    return inner  # 返回一个函数
```

#### 什么是闭包？

**闭包**是一个函数，它“记住”了它定义时的 **外部作用域变量**，即使外部函数已经执行完毕，这些变量依然存在。

例如：

```python
def outer():
    x = 10
    def inner():
        print(x)  # inner 记住了 x
    return inner

f = outer()
f()  # 输出 10
```

这里 `inner` 是一个闭包，因为它引用了 `outer` 中的变量 `x`，而 `outer` 已经返回了。

> 正常情况下：局部变量会在函数执行完后被释放; 但如果我们在内部函数中引用了外部函数的变量，Python 会自动把这些变量“绑定”到这个内部函数上，也就是形成闭包, 变量“被引用”而不会释放。



#### 装饰器的实现用到了什么？

现在看一个典型的装饰器例子：

```python
def my_decorator(func):                 # ✅ 高阶函数（接收函数并返回函数）
    def wrapper(*args, **kwargs):       # ✅ wrapper 是闭包（记住了 func）
        print("Before call")
        result = func(*args, **kwargs)
        print("After call")
        return result
    return wrapper

@my_decorator
def greet(name):
    print(f"Hello, {name}")
```

* `my_decorator` 是 **高阶函数**，因为它接收 `func` 并返回 `wrapper`。

* `wrapper` 是 **闭包**，因为它访问了其外部作用域的变量 `func`，并在被调用时依然保留这个引用。

> **“装饰器 = 高阶函数 + 闭包”** 的意思是：
>
> 一个装饰器的实现，**必须用高阶函数**（来接收和返回函数），而在返回的内部函数中，**依赖闭包机制**来记住原函数的引用，从而实现对原函数行为的增强或修改。



### 装饰器

**装饰器**是 Python 中的一种语法结构，本质是一个 **函数（或类）**，它接收一个函数或类作为参数，对其进行加工，并返回一个新的函数或类对象。

简而言之：

> 装饰器 = 高阶函数 + 闭包

装饰器主要用于在 **不修改原始函数代码的前提下，动态增加其功能**，这在日志记录、性能测试、权限校验等场景中非常常见。

#### 最基本的函数装饰器

```python
def my_decorator(func):
    def wrapper():
        print("调用前")
        func()
        print("调用后")
    return wrapper

@my_decorator
def say_hello():
    print("Hello")

say_hello()
```

输出：

```
调用前
Hello
调用后
```
说明：

* `@my_decorator` 相当于：`say_hello = my_decorator(say_hello)`
* `wrapper()` 是闭包，持有对 `func` 的引用。
* 返回的 `wrapper` 函数替代了原来的 `say_hello` 函数。


#### 带参数的函数装饰器

装饰器支持原函数有参数的情况：

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("开始")
        result = func(*args, **kwargs)
        print("结束")
        return result
    return wrapper

@my_decorator
def add(a, b):
    return a + b

print(add(3, 5))
```

> 使用 `*args` 和 `**kwargs` 是为了支持任意参数签名。


#### 带参数的装饰器（装饰器工厂）

如果你希望装饰器 **本身接受参数**，则需要再多一层函数嵌套：

```python
def log(prefix):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"{prefix} 开始调用 {func.__name__}")
            result = func(*args, **kwargs)
            print(f"{prefix} 结束调用 {func.__name__}")
            return result
        return wrapper
    return decorator

@log("DEBUG")
def multiply(a, b):
    return a * b
```

执行顺序：

1. `@log("DEBUG")` 先返回 `decorator`
2. 然后 `decorator(multiply)` 返回 `wrapper`

#### 使用 `functools.wraps` 保留原函数元信息

装饰器会改变函数的元信息:

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before call")
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name):
    """Say hello to someone"""
    print(f"Hello, {name}")

print(greet.__name__)   # ⚠️ 输出 wrapper，不是 greet
print(greet.__doc__)    # ⚠️ 输出 None，不是函数原文档
```

@my_decorator 返回的是 wrapper 函数，所以 greet 实际上变成了 wrapper，它的名字和文档字符串也被覆盖了，所以使用装饰器会导致原函数的 `__name__`、`__doc__` 等属性丢失。

Python 提供了 functools.wraps(func) 装饰器，作用是：

- 把原函数的 `__name__、__doc__、__module__` 等元信息“复制”到 wrapper 函数上，让被装饰函数看起来仍然像原来的函数。

```python
import functools

def my_decorator(func):
    @functools.wraps(func)  # ✅ 这一步很关键
    def wrapper(*args, **kwargs):
        print("Before call")
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name):
    """Say hello to someone"""
    print(f"Hello, {name}")

print(greet.__name__)   # ✅ greet
print(greet.__doc__)    # ✅ Say hello to someone
```

这在调试、文档生成、类型检查、元编程、反射中都非常重要。例如：

- help(greet)：没有 wraps 就看不到真实文档了

- 使用 inspect 模块查看参数、注解、类型签名会失效

- 多个装饰器嵌套时更容易出错

#### 装饰类方法（普通方法 / 类方法 / 静态方法）

```python
def log_method(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"调用方法 {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

class MyClass:
    @log_method
    def hello(self):
        print("Hello from method")
```

#### 装饰整个类

```python
def decorate_class(cls):
    cls.version = "1.0"
    return cls

@decorate_class
class MyService:
    pass

print(MyService.version)  # 1.0
```

#### 装饰器的底层原理与执行过程

本质：装饰器 = 函数替换器

一个装饰器：

```python
@decorator
def func():
    pass
```

等价于：

```python
func = decorator(func)
```

即：**把 `func` 传给 `decorator` 函数，并用它的返回值替换 `func` 本身。**

#### 多个装饰器叠加时的执行顺序（从内到外）

```python
@d1
@d2
def func():
    pass
```

等价于：

```python
func = d1(d2(func))
```

即，**先应用最内层的 `d2`，再由外层 `d1` 包裹起来。**

#### 类装饰器

类装饰器通常通过实现 `__call__` 方法来模拟函数行为：

```python
class MyDecorator:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        print("调用前")
        result = self.func(*args, **kwargs)
        print("调用后")
        return result

@MyDecorator
def greet(name):
    print(f"Hi, {name}")

greet("Alice")
```
#### 总结

| 类型      | 例子                  | 含义                        |
| ------- | ------------------- | ------------------------- |
| 最基本装饰器  | `@func`             | `f = func(f)`             |
| 装饰器工厂   | `@decorator(x)`     | `f = decorator(x)(f)`     |
| 对象方法装饰器 | `@obj.method`       | `f = obj.method(f)`       |
| 对象方法工厂  | `@obj.method(args)` | `f = obj.method(args)(f)` |


#### 典型应用场景举例

1. **日志记录**：

   ```python
   def log(func):
       @wraps(func)
       def wrapper(*args, **kwargs):
           print(f"调用 {func.__name__} 参数: {args}, {kwargs}")
           return func(*args, **kwargs)
       return wrapper
   ```

2. **权限控制**：

   ```python
   def require_admin(func):
       @wraps(func)
       def wrapper(*args, **kwargs):
           if not user_is_admin():
               raise PermissionError("需要管理员权限")
           return func(*args, **kwargs)
       return wrapper
   ```

3. **性能测试（统计函数运行时间）**：

   ```python
   import time

   def timing(func):
       @wraps(func)
       def wrapper(*args, **kwargs):
           start = time.time()
           result = func(*args, **kwargs)
           print(f"{func.__name__} 耗时: {time.time() - start:.4f}s")
           return result
       return wrapper
   ```

### 地板除 “//” 

`//` 是 **地板除（floor division）** 运算符，表示**向下取整的除法**。

| 表达式      | 结果         | 类型    |
| -------- | ---------- | ----- |
| `7 / 3`  | `2.333...` | float |
| `7 // 3` | `2`        | int   |

### Ellipsis (...)

`Ellipsis`（用 `...` 表示）是 Python 中的语法，用于表示 **多维索引中的省略维度**。在多维数组或张量索引时，`...` 可以代替多个冒号 `:`，表示选择剩余所有维度。

示例：

```python
import torch
x = torch.randn(2, 3, 4, 5)

# 取第 0 维第 1 个元素，后面所有维度都选中
y = x[1, ...]       # 等价于 x[1, :, :, :]
print(y.shape)      # torch.Size([3, 4, 5])

# 在最后加维度
z = x[..., None]    # shape: [2, 3, 4, 5, 1]
```

总结：`...` 用于 **简化多维索引** 或 **保留剩余维度**。
