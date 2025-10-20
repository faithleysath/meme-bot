# sandbox.py

# ============================================================================
# 标准库导入
# ============================================================================

import importlib
import pickle
import re
import sys
from collections.abc import Callable

# ============================================================================
# 第三方库导入
# ============================================================================

from nonebot import logger, require
from pebble import asynchronous
# 恢复您原来的 TimeoutError 导入
from concurrent.futures import TimeoutError as PebbleTimeoutError, TimeoutError
from RestrictedPython import safe_builtins, compile_restricted
from RestrictedPython.Eval import default_guarded_getiter, default_guarded_getitem
from RestrictedPython.Guards import (
    full_write_guard,
    guarded_iter_unpack_sequence,
    guarded_unpack_sequence,
)
from RestrictedPython.PrintCollector import PrintCollector

# ============================================================================
# 插件依赖导入
# ============================================================================

require("nonebot_plugin_localstore")

# ============================================================================
# 本地模块导入
# ============================================================================

from .dependency_manager import DependencyManager

# ============================================================================
# 配置常量
# ============================================================================

BUILTIN_MODULE_PREFIXES = {
    "math", "io", "base64", "PIL",
}
MODULE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_import_cache: dict[str, object] = {}
DEFAULT_TIMEOUT = 5

# ============================================================================
# 安全模块导入 (在子进程中运行)
# ============================================================================

def _is_module_allowed(module_name: str, allowed_prefixes: list[str]) -> bool:
    return any(
        module_name == prefix or module_name.startswith(f"{prefix}.")
        for prefix in allowed_prefixes
    )

def _validate_module_name(name: str) -> bool:
    return isinstance(name, str) and bool(MODULE_NAME_PATTERN.match(name))

def _validate_identifier(identifier: str) -> bool:
    return isinstance(identifier, str) and bool(IDENTIFIER_PATTERN.match(identifier))

def _create_safe_import(allowed_prefixes: list[str]) -> Callable:
    def custom_safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level != 0:
            raise ImportError("SECURITY ERROR: 相对导入被禁止")
        if not _validate_module_name(name):
            raise ImportError(f"SECURITY ERROR: 无效的模块名称 '{name}'")
        if not _is_module_allowed(name, allowed_prefixes):
            raise ImportError(f"SECURITY ERROR: 不允许导入模块 '{name}'")

        if name in _import_cache:
            module = _import_cache[name]
        else:
            module = importlib.import_module(name)
            _import_cache[name] = module

        if fromlist:
            for item in fromlist:
                if not _validate_identifier(item):
                    raise ImportError(f"SECURITY ERROR: 无效的导入项 '{item}'")
                try:
                    setattr(module, item, getattr(module, item))
                    continue
                except AttributeError:
                    pass

                submodule_name = f"{name}.{item}"
                if not _is_module_allowed(submodule_name, allowed_prefixes):
                    raise ImportError(f"SECURITY ERROR: 不允许导入子模块 '{submodule_name}'")
                
                if submodule_name in _import_cache:
                    submodule = _import_cache[submodule_name]
                else:
                    submodule = importlib.import_module(submodule_name)
                    _import_cache[submodule_name] = submodule
                setattr(module, item, submodule)
        
        return module
    
    return custom_safe_import

# ============================================================================
# 安全内置函数配置
# ============================================================================

_NUMERIC_BUILTINS = {
    "min": min, "max": max, "sum": sum, "abs": abs, "round": round, "len": len, "divmod": divmod, "pow": pow,
}
_SEQUENCE_BUILTINS = {
    "sorted": sorted, "range": range, "enumerate": enumerate, "zip": zip, "reversed": reversed, "slice": slice, "map": map, "filter": filter, "any": any, "all": all,
}
_TYPE_BUILTINS = {
    "int": int, "float": float, "bool": bool, "str": str, "bytes": bytes, "bytearray": bytearray, "memoryview": memoryview,
}
_CONTAINER_BUILTINS = {
    "list": list, "tuple": tuple, "set": set, "frozenset": frozenset, "dict": dict,
}
_TYPE_CHECK_BUILTINS = {
    "isinstance": isinstance, "issubclass": issubclass, "chr": chr, "ord": ord,
}
_UTILITY_BUILTINS = {
    "hash": hash, "format": format,
}
EXTRA_SAFE_BUILTINS = {
    **_NUMERIC_BUILTINS, **_SEQUENCE_BUILTINS, **_TYPE_BUILTINS, **_CONTAINER_BUILTINS, **_TYPE_CHECK_BUILTINS, **_UTILITY_BUILTINS,
}

# ============================================================================
# 沙盒执行核心函数 (在子进程中运行)
# ============================================================================

# 恢复您原来的装饰器，由 pebble 控制固定的超时
@asynchronous.process(timeout=DEFAULT_TIMEOUT)
def _execute_in_process(
    code: str, 
    global_vars: dict, 
    local_vars: dict,
    allowed_prefixes: list[str] # 这是唯一需要的修改，传入依赖列表
):
    """在独立子进程中安全执行 Python 代码的核心函数"""
    try:
        logger.debug("开始编译受限制代码")
        byte_code = compile_restricted(code, '<string>', 'exec')

        logger.debug("构建安全的全局变量环境")
        importer = _create_safe_import(allowed_prefixes)
        safe_globals = _build_safe_globals(global_vars, importer)

        logger.debug("执行代码")
        exec(byte_code, safe_globals, local_vars)

        logger.debug("过滤可序列化对象")
        return _filter_serializable_objects(safe_globals, local_vars)

    except MemoryError:
        logger.error("代码执行超出内存限制")
        raise MemoryError("代码执行超出内存限制")
    except Exception as e:
        logger.error(f"子进程执行失败: {type(e).__name__}: {e}")
        raise e

def _build_safe_globals(global_vars: dict, importer: Callable) -> dict:
    return {
        "__builtins__": {
            **safe_builtins,
            "__import__": importer,
            **EXTRA_SAFE_BUILTINS
        },
        "_getattr_": safe_builtins["_getattr_"], "_getitem_": default_guarded_getitem,
        "_getiter_": default_guarded_getiter, "_unpack_sequence_": guarded_unpack_sequence,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence, "_write_": full_write_guard,
        "_print_": PrintCollector, **global_vars
    }

def _is_serializable(obj) -> bool:
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False

def _filter_serializable_objects(safe_globals: dict, local_vars: dict) -> tuple:
    filtered_globals = {k: v for k, v in safe_globals.items() if _is_serializable(v)}
    filtered_locals = {k: v for k, v in local_vars.items() if _is_serializable(v)}
    logger.debug(f"过滤后的全局变量: {len(filtered_globals)} 个")
    logger.debug(f"过滤后的局部变量: {len(filtered_locals)} 个")
    return filtered_globals, filtered_locals

# ============================================================================
# 主要接口函数 (在主进程中运行)
# ============================================================================

async def worker_with_limits(
    python_code: str,
    globals: dict,
    locals: dict,
    # 注意：此处的 timeout 参数未生效，实际超时由 @asynchronous.process 装饰器中的 DEFAULT_TIMEOUT 控制。
    # 这也是您原来的实现方式。
    timeout: int = DEFAULT_TIMEOUT
) -> tuple:
    """带限制的异步代码执行工作器"""
    if not isinstance(python_code, str):
        raise TypeError("python_code 必须是字符串")
    if not isinstance(globals, dict):
        raise TypeError("globals 必须是字典")
    if not isinstance(locals, dict):
        raise TypeError("locals 必须是字典")
    
    # 这里我们不再检查 timeout 参数，因为它是固定的
    
    logger.info(f"开始执行沙盒代码，超时限制: {DEFAULT_TIMEOUT} 秒")
    logger.debug(f"代码长度: {len(python_code)} 字符")

    try:
        # 1. (异步问题修复) 确保依赖存在并获取依赖列表
        await DependencyManager.ensure_dependencies()
        allowed_deps = await DependencyManager.get_dependency_names()
        allowed_prefixes = list(BUILTIN_MODULE_PREFIXES) + allowed_deps
        logger.debug(f"沙盒允许的模块前缀: {allowed_prefixes}")

        # 2. (恢复原状) 直接 await pebble 返回的 future
        result = await _execute_in_process( # type: ignore
            code=python_code,
            global_vars=globals,
            local_vars=locals,
            allowed_prefixes=allowed_prefixes
        )

        logger.info("沙盒代码执行成功")
        return result

    # 3. (恢复原状) 捕获 pebble 的超时错误
    except PebbleTimeoutError:
        error_msg = f"代码执行超时，超过 {DEFAULT_TIMEOUT} 秒限制"
        logger.error(error_msg)
        raise TimeoutError(error_msg)

    except Exception as e:
        error_msg = f"工作进程执行失败: {type(e).__name__}: {e}"
        logger.error(error_msg)
        raise e

# ============================================================================
# 公共 API 导出
# =================================e===========================================

__all__ = [
    "worker_with_limits",
    "DEFAULT_TIMEOUT",
    "BUILTIN_MODULE_PREFIXES",
]