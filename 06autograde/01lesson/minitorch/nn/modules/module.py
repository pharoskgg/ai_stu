"""网络模块的调用入口、参数注册和子模块管理。"""

from __future__ import annotations

from ..parameter import Parameter


class Module:
    """继承后先调用 super().__init__()，再定义参数和子模块。"""

    def __init__(self):
        self._parameters: dict[str, Parameter | None] = {}
        self._modules: dict[str, Module | None] = {}
        self.training = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} 必须实现 forward()")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __getattr__(self, name):
        # 注册对象仅存放在对应字典中，避免属性与注册信息不同步。
        for registry_name in ("_parameters", "_modules"):
            registry = self.__dict__.get(registry_name, {})
            if name in registry:
                return registry[name]
        raise AttributeError(f"{type(self).__name__} 没有属性 {name!r}")

    def __setattr__(self, name, value):
        parameters = self.__dict__.get("_parameters")
        modules = self.__dict__.get("_modules")
        if isinstance(value, Parameter):
            self._register_parameter(name, value, replace_attribute=True)
        elif parameters is not None and name in parameters:
            if value is not None:
                raise TypeError(f"参数 {name!r} 只能赋值为 Parameter 或 None")
            self.register_parameter(name, value)
        elif isinstance(value, Module):
            self._register_module(name, value, replace_attribute=True)
        elif modules is not None and name in modules:
            if value is not None:
                raise TypeError(f"子模块 {name!r} 只能赋值为 Module 或 None")
            self.add_module(name, value)
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        for registry_name in ("_parameters", "_modules"):
            registry = self.__dict__.get(registry_name, {})
            if name in registry:
                del registry[name]
                return
        object.__delattr__(self, name)

    def _check_registration_name(self, name, registry, replace_attribute):
        if not isinstance(name, str):
            raise TypeError("注册名称必须是字符串")
        if not name or "." in name:
            raise KeyError("注册名称不能为空或包含 '.'")
        if name in ("_parameters", "_modules", "training") or hasattr(type(self), name):
            raise KeyError(f"注册名称 {name!r} 与模块已有接口冲突")
        if not replace_attribute and hasattr(self, name) and name not in registry:
            raise KeyError(f"属性 {name!r} 已存在")

    def _register_parameter(self, name, param, replace_attribute=False):
        parameters = self.__dict__.get("_parameters")
        if parameters is None:
            raise AttributeError("注册参数前必须调用 Module.__init__()")
        self._check_registration_name(name, parameters, replace_attribute)
        if param is not None:
            if not isinstance(param, Parameter):
                raise TypeError("参数必须是 Parameter 或 None")
            if param.grad_fn is not None:
                raise ValueError("Parameter 必须是叶子 Tensor")
        # 先完成检查，再替换属性，失败时保留原注册信息。
        if replace_attribute:
            self.__dict__.pop(name, None)
            self._modules.pop(name, None)
        parameters[name] = param

    def register_parameter(self, name: str, param: Parameter | None):
        """显式注册参数；None 可用于关闭可选参数，例如 bias。"""
        self._register_parameter(name, param)

    def _register_module(self, name, module, replace_attribute=False):
        modules = self.__dict__.get("_modules")
        if modules is None:
            raise AttributeError("注册子模块前必须调用 Module.__init__()")
        self._check_registration_name(name, modules, replace_attribute)
        if module is not None:
            if not isinstance(module, Module):
                raise TypeError("子模块必须是 Module 或 None")
            if any(child is self for child in module.modules()):
                raise ValueError("子模块不能包含自身或祖先模块")
        if replace_attribute:
            self.__dict__.pop(name, None)
        modules[name] = module

    def add_module(self, name: str, module: Module | None):
        """显式注册子模块，参数遍历和模式切换会递归进入它。"""
        self._register_module(name, module)

    def named_parameters(self, prefix: str = "", recurse: bool = True,
                         remove_duplicate: bool = True):
        """按注册顺序返回名称和参数，默认对共享参数去重。"""
        modules = (self.named_modules(prefix=prefix, remove_duplicate=remove_duplicate)
                   if recurse else ((prefix, self),))
        visited = set()
        for module_prefix, module in modules:
            for name, param in module._parameters.items():
                if param is None or (remove_duplicate and id(param) in visited):
                    continue
                visited.add(id(param))
                full_name = f"{module_prefix}.{name}" if module_prefix else name
                yield full_name, param

    def parameters(self, recurse: bool = True):
        """返回参数迭代器，包括 requires_grad=False 的注册参数。"""
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def named_children(self):
        """返回直接子模块，共享子模块仅出现一次。"""
        visited = set()
        for name, module in self._modules.items():
            if module is not None and id(module) not in visited:
                visited.add(id(module))
                yield name, module

    def children(self):
        for _, module in self.named_children():
            yield module

    def named_modules(self, memo=None, prefix: str = "", remove_duplicate: bool = True):
        """递归返回模块，包含名称为空的当前模块。"""
        if memo is None:
            memo = set()
        if self in memo:
            return
        if remove_duplicate:
            memo.add(self)
        yield prefix, self
        for name, module in self._modules.items():
            if module is not None:
                full_name = f"{prefix}.{name}" if prefix else name
                yield from module.named_modules(memo, full_name, remove_duplicate)

    def modules(self):
        for _, module in self.named_modules():
            yield module

    def train(self, mode: bool = True):
        """递归设置训练模式；模式切换不会改变是否记录计算图。"""
        if not isinstance(mode, bool):
            raise ValueError("训练模式必须是 bool")
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def zero_grad(self, set_to_none: bool = True):
        """递归清空参数梯度；False 时将已有梯度数组原地置零。"""
        for param in self.parameters():
            if param.grad is not None:
                if set_to_none:
                    param.zero_grad()
                else:
                    param.grad.fill(0)

    def __repr__(self):
        lines = []
        for name, module in self._modules.items():
            if module is not None:
                description = repr(module).replace("\n", "\n  ")
                lines.append(f"  ({name}): {description}")
        if not lines:
            return f"{type(self).__name__}()"
        return f"{type(self).__name__}(\n" + "\n".join(lines) + "\n)"
