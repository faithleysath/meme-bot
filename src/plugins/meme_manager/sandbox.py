# === 标准库导入 ===
import pickle
import re
import importlib
from concurrent.futures import TimeoutError as PebbleTimeoutError, TimeoutError

# === 第三方库导入 ===
from nonebot import logger
from pebble import asynchronous
from RestrictedPython import safe_builtins, compile_restricted
from RestrictedPython.Eval import default_guarded_getiter, default_guarded_getitem
from RestrictedPython.PrintCollector import PrintCollector
from RestrictedPython.Guards import (
    full_write_guard,
    guarded_unpack_sequence,
    guarded_iter_unpack_sequence,
)

# ============================================================================
# 配置常量
# ============================================================================

# === 沙盒安全配置 ===
# 允许导入的模块前缀白名单（包括所有子模块）
ALLOWED_PREFIXES = {
    "math",      # 数学函数
    "io",        # 输入输出操作
    "base64",    # Base64 编解码
    "PIL",       # Python Imaging Library 及其子模块
}

# === 正则表达式模式 ===
# 模块名称验证模式
_MODULE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
# 标识符验证模式
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# === 性能优化配置 ===
# 导入模块缓存，减少重复导入开销
_import_cache: dict[str, object] = {}

# === 执行配置 ===
# 默认代码执行超时时间（秒）
DEFAULT_TIMEOUT = 5

# ============================================================================
# 工具函数
# ============================================================================

def _is_allowed(name: str) -> bool:
    """检查模块名称是否在允许的前缀白名单中"""
    return any(name == p or name.startswith(p + ".") for p in ALLOWED_PREFIXES)

def custom_safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """
    安全的模块导入函数，仅允许白名单中的模块

    Args:
        name: 要导入的模块名称
        globals: 全局命名空间（未使用，保持兼容性）
        locals: 局部命名空间（未使用，保持兼容性）
        fromlist: from 语句中要导入的对象列表
        level: 导入级别（0 表示绝对导入）

    Returns:
        导入的模块对象

    Raises:
        ImportError: 当导入不被允许时抛出
    """
    # 禁止相对导入
    if level and level != 0:
        raise ImportError("SECURITY ERROR: Relative import is not allowed.")

    # 基本校验
    if not isinstance(name, str) or not _MODULE_NAME_RE.match(name):
        raise ImportError(f"SECURITY ERROR: Invalid module name '{name}'.")

    # 检查白名单
    if not _is_allowed(name):
        raise ImportError(f"SECURITY ERROR: Importing '{name}' is not allowed.")

    # 导入模块（带缓存优化）
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

            # 先尝试获取模块的顶层属性
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

# ============================================================================
# 安全内置函数配置
# ============================================================================

EXTRA_SAFE_BUILTINS = {
    # === 数值与数学操作 ===
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "len": len,
    "divmod": divmod,
    "pow": pow,          # 支持 pow(a, b[, mod]) 形式

    # === 序列与迭代操作 ===
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

    # === 基础类型构造器 ===
    "int": int,
    "float": float,
    "bool": bool,
    "str": str,
    "bytes": bytes,
    "bytearray": bytearray,
    "memoryview": memoryview,

    # === 容器类型 ===
    "list": list,
    "tuple": tuple,
    "set": set,
    "frozenset": frozenset,
    "dict": dict,

    # === 类型检查与转换 ===
    "isinstance": isinstance,
    "issubclass": issubclass,
    "chr": chr,
    "ord": ord,

    # === 其他工具函数 ===
    "hash": hash,        # 对不可哈希对象会抛出异常，仍是安全的
    "format": format,
}



# ============================================================================
# 沙盒执行核心函数
# ============================================================================

@asynchronous.process(timeout=DEFAULT_TIMEOUT)
def _execute_in_process(code: str, global_vars: dict, local_vars: dict):
    """
    在独立子进程中安全执行 Python 代码的核心函数

    此函数将在子进程中被独立执行，必须是模块的顶层函数才能被 pickle 序列化。
    使用 RestrictedPython 编译代码并在受限环境中执行。

    Args:
        code: 要执行的 Python 代码字符串
        global_vars: 传入的全局变量字典
        local_vars: 传入的局部变量字典

    Returns:
        tuple: (filtered_globals, filtered_locals) - 经过序列化过滤的变量字典

    Raises:
        MemoryError: 当代码执行超过内存限制时
        Exception: 其他执行异常
    """
    try:
        # 编译受限制的代码
        byte_code = compile_restricted(code, '<string>', 'exec')

        # 构建安全的全局变量环境
        safe_globals = _build_safe_globals(global_vars)

        # 执行代码
        exec(byte_code, safe_globals, local_vars)

        # 过滤并返回可序列化的结果
        return _filter_serializable_objects(safe_globals, local_vars)

    except MemoryError:
        raise MemoryError("Code execution exceeded memory limit.")
    except Exception as e:
        # 将子进程中的异常传播到主进程
        raise e


def _build_safe_globals(global_vars: dict) -> dict:
    """构建安全的全局变量环境"""
    return {
        "__builtins__": {
            **safe_builtins,
            "__import__": custom_safe_import,
            **EXTRA_SAFE_BUILTINS
        },
        # RestrictedPython 运行时安全钩子
        "_getattr_": safe_builtins["_getattr_"],      # 更安全的属性访问
        "_getitem_": default_guarded_getitem,        # 安全的索引访问
        "_getiter_": default_guarded_getiter,        # 安全的迭代器
        "_unpack_sequence_": guarded_unpack_sequence,      # 安全的序列解包
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,  # 安全的迭代解包
        "_write_": full_write_guard,                 # 写入保护
        "_print_": PrintCollector,                   # 安全的打印输出
        **global_vars
    }


def _filter_serializable_objects(safe_globals: dict, local_vars: dict) -> tuple:
    """过滤出可序列化的对象，确保能在进程间传递"""
    def is_picklable(obj):
        try:
            pickle.dumps(obj)
            return True
        except Exception:
            return False

    filtered_globals = {k: v for k, v in safe_globals.items() if is_picklable(v)}
    filtered_locals = {k: v for k, v in local_vars.items() if is_picklable(v)}

    return filtered_globals, filtered_locals

# ============================================================================
# 主要接口函数
# ============================================================================

async def worker_with_limits(python_code: str, globals: dict, locals: dict, timeout: int = DEFAULT_TIMEOUT):
    """
    带限制的异步代码执行工作器

    使用 pebble 库的内置超时功能管理子进程生命周期，提供安全可靠的代码执行环境。

    Args:
        python_code: 要执行的 Python 代码字符串
        globals: 传入代码的全局变量字典
        locals: 传入代码的局部变量字典
        timeout: 执行超时时间（秒），默认使用 DEFAULT_TIMEOUT

    Returns:
        tuple: (execution_globals, execution_locals) - 代码执行后的变量状态

    Raises:
        TimeoutError: 当代码执行超过指定时间限制时
        Exception: 其他执行相关异常
    """
    try:
        result = await _execute_in_process(  # type: ignore
            code=python_code,
            global_vars=globals,
            local_vars=locals
        )
        return result

    except PebbleTimeoutError:
        # 捕获 pebble 的超时异常并转换为标准 TimeoutError
        logger.error(f"Code execution terminated after exceeding {timeout} seconds")
        raise TimeoutError(f"Code execution exceeded time limit of {timeout}s")

    except Exception as e:
        logger.error(f"Worker process execution failed: {type(e).__name__}: {e}")
        raise e