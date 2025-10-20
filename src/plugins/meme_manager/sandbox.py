from nonebot import logger
from pebble import asynchronous
from concurrent.futures import TimeoutError as PebbleTimeoutError
from concurrent.futures import TimeoutError
import os
# —— 这些符号就按你本地源码精确导入 ——
from RestrictedPython import safe_builtins, compile_restricted
from RestrictedPython.Eval import (
    default_guarded_getiter,    # 你源码里提供
    default_guarded_getitem,    # 你源码里提供
)
from RestrictedPython.Guards import (
    full_write_guard,           # 由 _full_write_guard() 生成
    guarded_unpack_sequence,    # 你源码里提供
    guarded_iter_unpack_sequence,  # 你源码里提供
    # guarded_setattr, guarded_delattr 也在 Guards 里，
    # 但源码已把它们注册到 safe_builtins['setattr'/'delattr']，无需再放进 globals
)

# safer_getattr 已在你的 Guards 源码里注册为：
# safe_builtins['_getattr_'] = safer_getattr
# 所以直接从 safe_builtins 里取用即可，不需要再 import 一个 guarded_getattr


# ---- 可信环境（沙盒外）放置 ----
import importlib
import re

# 只这一个白名单：允许这些“包或模块前缀”及其所有子模块（无限深度）
ALLOWED_PREFIXES = {
    "math",
    "io",
    "base64",
    "PIL",   # 允许 PIL 及其所有子模块（PIL.Image、PIL.ImageDraw、…）
}

# 名称校验，避免奇怪标识符
_MODULE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# 简单缓存，减少重复 import
_import_cache: dict[str, object] = {}

def _is_allowed(name: str) -> bool:
    return any(name == p or name.startswith(p + ".") for p in ALLOWED_PREFIXES)

def custom_safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    # 禁止相对导入
    if level and level != 0:
        raise ImportError("SECURITY ERROR: Relative import is not allowed.")

    # 基本校验
    if not isinstance(name, str) or not _MODULE_NAME_RE.match(name):
        raise ImportError(f"SECURITY ERROR: Invalid module name '{name}'.")

    # 检查白名单
    if not _is_allowed(name):
        raise ImportError(f"SECURITY ERROR: Importing '{name}' is not allowed.")

    # 导入模块（带缓存）
    if name in _import_cache:
        mod = _import_cache[name]
    else:
        mod = importlib.import_module(name)
        _import_cache[name] = mod

    # 处理 fromlist：from X import Y, Z
    if fromlist:
        for item in fromlist:
            if not isinstance(item, str) or not _IDENT_RE.match(item):
                raise ImportError(f"SECURITY ERROR: Invalid fromlist item '{item}'.")
            # 先尝试顶层属性（某些包把对象直接挂在 __init__ 里）
            try:
                setattr(mod, item, getattr(mod, item))
                continue
            except Exception:
                pass
            # 当作子模块导入
            subname = f"{name}.{item}"
            if not _is_allowed(subname):
                raise ImportError(f"SECURITY ERROR: Importing '{subname}' is not allowed.")
            if subname in _import_cache:
                submod = _import_cache[subname]
            else:
                submod = importlib.import_module(subname)
                _import_cache[subname] = submod
            setattr(mod, item, submod)

    return mod

import pickle

# ===== 1) 放在文件顶部（custom_safe_import 附近）=====

EXTRA_SAFE_BUILTINS = {
    # 标准数值/序列操作
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "len": len,
    "divmod": divmod,
    "pow": pow,          # 允许 pow(a, b[, mod])
    
    # 排序与遍历
    "sorted": sorted,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "reversed": reversed,
    "slice": slice,
    "map": map,
    "filter": filter,
    "any": any,
    "all": all,

    # 基础类型与构造器
    "int": int,
    "float": float,
    "bool": bool,
    "str": str,
    "bytes": bytes,
    "bytearray": bytearray,
    "memoryview": memoryview,

    "list": list,
    "tuple": tuple,
    "set": set,
    "frozenset": frozenset,
    "dict": dict,

    # 类型/字符工具
    "isinstance": isinstance,
    "issubclass": issubclass,
    "chr": chr,
    "ord": ord,

    # 其他常用
    "hash": hash,        # 注：对不可哈希对象会抛错，仍是安全的
    "format": format,
    "print": print,      # 若不想暴露 print，可去掉；你已有 _print_ → logger.info
}



@asynchronous.process(timeout=5)
def _execute_in_process(code: str, global_vars: dict, local_vars: dict):
    """
    这个函数将会在子进程中被独立执行。
    它必须是模块的顶层函数，才能被 pickle。
    """
    try:
        byte_code = compile_restricted(code, '<string>', 'exec')
        # __builtins__ 必须在子进程中重新构建，而不是通过参数传递
        safe_globals = {
                "__builtins__": {**safe_builtins, "__import__": custom_safe_import, **EXTRA_SAFE_BUILTINS},
                # —— RestrictedPython 运行时钩子（按你源码的名字来）——
                "_getattr_": safe_builtins["_getattr_"],      # = safer_getattr
                "_getitem_": default_guarded_getitem,
                "_getiter_": default_guarded_getiter,
                "_unpack_sequence_": guarded_unpack_sequence,
                "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
                "_write_": full_write_guard,
                "_print_": logger.info,
                **global_vars
            }
        exec(byte_code, safe_globals, local_vars)
        def picklable(x):
            try:
                pickle.dumps(x)
                return True
            except Exception:
                return False
        # 对 globals 和 locals 进行过滤，只保留可被 pickle 的对象
        safe_globals = {k: v for k, v in safe_globals.items() if picklable(v)}
        local_vars = {k: v for k, v in local_vars.items() if picklable(v)}
        return safe_globals, local_vars
    except MemoryError:
        raise MemoryError("Code execution exceeded memory limit.")
    except Exception as e:
        # 将子进程中的其他异常也抛出，以便主进程捕获
        raise e

async def worker_with_limits(python_code: str, globals: dict, locals: dict, timeout: int = 5):
    """
    使用 pebble 内建的 timeout 功能来管理子进程的生命周期，代码更简洁。
    """
    try:
        result = await _execute_in_process(code=python_code, global_vars=globals, local_vars=locals) # type: ignore
        return result
    except PebbleTimeoutError:
        # 3. 捕获 pebble 的超时异常
        logger.error(f"Code execution was terminated by pebble after exceeding {timeout} seconds.")
        # pebble 已经自动处理了子进程，我们只需向上抛出异常即可
        raise TimeoutError(f"Code execution exceeded time limit of {timeout}s.")
    except Exception as e:
        logger.error(f"Worker process or future failed: {type(e).__name__}: {e}")
        raise e