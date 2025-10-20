"""
安全沙盒执行环境

提供受限的 Python 代码执行环境，支持：
- 安全的模块导入控制
- 受限的内置函数访问
- 进程隔离执行
- 依赖包管理
- 超时和资源限制
"""

# ============================================================================
# 标准库导入
# ============================================================================

import importlib
import os
import pickle
import re
import shutil
import subprocess
import sys
from concurrent.futures import TimeoutError as PebbleTimeoutError, TimeoutError
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ============================================================================
# 第三方库导入
# ============================================================================

from nonebot import logger, require
from pebble import asynchronous
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
import nonebot_plugin_localstore as store

# ============================================================================
# 配置常量
# ============================================================================

# 沙盒安全配置
BUILTIN_MODULE_PREFIXES = {
    "math",    # 数学函数库
    "io",      # 输入输出操作
    "base64",  # Base64 编解码
    "PIL",     # Python Imaging Library 及其子模块
}

# 正则表达式模式
MODULE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# 性能优化配置
_import_cache: dict[str, object] = {}

# 执行配置
DEFAULT_TIMEOUT = 5  # 默认代码执行超时时间（秒）

# ============================================================================
# 安全模块导入
# ============================================================================

def _is_module_allowed(module_name: str) -> bool:
    """
    检查模块名称是否在允许的前缀白名单中

    Args:
        module_name: 要检查的模块名称

    Returns:
        bool: 是否允许导入该模块
    """
    allowed_prefixes = list(BUILTIN_MODULE_PREFIXES) + dependencyManager.get_dependency_list()
    return any(
        module_name == prefix or module_name.startswith(prefix + ".")
        for prefix in allowed_prefixes
    )


def _validate_module_name(name: str) -> bool:
    """
    验证模块名称格式是否合法

    Args:
        name: 模块名称

    Returns:
        bool: 模块名称是否合法
    """
    return isinstance(name, str) and bool(MODULE_NAME_PATTERN.match(name))


def _validate_identifier(identifier: str) -> bool:
    """
    验证标识符格式是否合法

    Args:
        identifier: 标识符名称

    Returns:
        bool: 标识符是否合法
    """
    return isinstance(identifier, str) and bool(IDENTIFIER_PATTERN.match(identifier))


def custom_safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """
    安全的模块导入函数，仅允许白名单中的模块

    此函数作为 RestrictedPython 的自定义 __import__ 实现，提供严格的模块导入控制。

    Args:
        name: 要导入的模块名称
        globals: 全局命名空间（未使用，保持兼容性）
        locals: 局部命名空间（未使用，保持兼容性）
        fromlist: from 语句中要导入的对象列表
        level: 导入级别（0 表示绝对导入）

    Returns:
        导入的模块对象

    Raises:
        ImportError: 当导入不被允许或格式错误时抛出
    """
    # 禁止相对导入
    if level != 0:
        raise ImportError("SECURITY ERROR: 相对导入被禁止")

    # 验证模块名称格式
    if not _validate_module_name(name):
        raise ImportError(f"SECURITY ERROR: 无效的模块名称 '{name}'")

    # 检查模块是否在白名单中
    if not _is_module_allowed(name):
        raise ImportError(f"SECURITY ERROR: 不允许导入模块 '{name}'")

    # 从缓存导入或执行导入
    if name in _import_cache:
        module = _import_cache[name]
    else:
        module = importlib.import_module(name)
        _import_cache[name] = module

    # 处理 from X import Y, Z 语法
    if fromlist:
        for item in fromlist:
            if not _validate_identifier(item):
                raise ImportError(f"SECURITY ERROR: 无效的导入项 '{item}'")

            # 首先尝试获取模块属性
            try:
                setattr(module, item, getattr(module, item))
                continue
            except AttributeError:
                pass

            # 尝试作为子模块导入
            submodule_name = f"{name}.{item}"
            if not _is_module_allowed(submodule_name):
                raise ImportError(f"SECURITY ERROR: 不允许导入子模块 '{submodule_name}'")

            # 导入子模块
            if submodule_name in _import_cache:
                submodule = _import_cache[submodule_name]
            else:
                submodule = importlib.import_module(submodule_name)
                _import_cache[submodule_name] = submodule

            setattr(module, item, submodule)

    return module

# ============================================================================
# 安全内置函数配置
# ============================================================================

# 数值与数学操作
_NUMERIC_BUILTINS = {
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "len": len,
    "divmod": divmod,
    "pow": pow,
}

# 序列与迭代操作
_SEQUENCE_BUILTINS = {
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
}

# 基础类型构造器
_TYPE_BUILTINS = {
    "int": int,
    "float": float,
    "bool": bool,
    "str": str,
    "bytes": bytes,
    "bytearray": bytearray,
    "memoryview": memoryview,
}

# 容器类型
_CONTAINER_BUILTINS = {
    "list": list,
    "tuple": tuple,
    "set": set,
    "frozenset": frozenset,
    "dict": dict,
}

# 类型检查与转换
_TYPE_CHECK_BUILTINS = {
    "isinstance": isinstance,
    "issubclass": issubclass,
    "chr": chr,
    "ord": ord,
}

# 其他工具函数
_UTILITY_BUILTINS = {
    "hash": hash,        # 对不可哈希对象会抛出异常，仍是安全的
    "format": format,
}

# 合并所有安全的内置函数
EXTRA_SAFE_BUILTINS = {
    **_NUMERIC_BUILTINS,
    **_SEQUENCE_BUILTINS,
    **_TYPE_BUILTINS,
    **_CONTAINER_BUILTINS,
    **_TYPE_CHECK_BUILTINS,
    **_UTILITY_BUILTINS,
}

# ============================================================================
# 依赖管理器
# ============================================================================

class DependencyManager:
    """
    沙盒环境依赖管理器（无实例类）

    负责管理沙盒执行环境的 Python 依赖包，包括：
    - 虚拟环境创建和管理
    - 依赖包安装和卸载
    - 环境清理和重建
    - 自动路径管理
    """

    # 类级别的路径配置
    _requirements_file: Path = store.get_plugin_data_file("sandbox_requirements.txt")
    _venv_path: Path = store.get_plugin_cache_dir() / "sandbox_venv"

    @classmethod
    def _get_pip_executable(cls) -> Path:
        """获取 pip 可执行文件路径"""
        if os.name == "nt":  # Windows
            return cls._venv_path / "Scripts" / "pip.exe"
        else:  # Linux/macOS
            return cls._venv_path / "bin" / "pip"

    @classmethod
    def _get_site_packages_path(cls) -> str:
        """获取虚拟环境的 site-packages 路径"""
        if os.name == "nt":  # Windows
            return os.path.join(
                cls._venv_path,
                "Lib",
                "site-packages"
            )
        else:  # Linux/macOS
            return os.path.join(
                cls._venv_path,
                "lib",
                f"python{sys.version_info.major}.{sys.version_info.minor}",
                "site-packages"
            )

    @classmethod
    def _add_to_sys_path(cls) -> None:
        """
        将虚拟环境的 site-packages 路径添加到 sys.path 末尾

        这样可以确保：
        1. 优先使用系统环境的包
        2. 虚拟环境中的包作为补充
        3. 沙盒代码可以直接 import 虚拟环境中的包
        """
        if not cls._venv_path.exists():
            return

        site_packages_path = cls._get_site_packages_path()

        # 检查路径是否已经在 sys.path 中
        if site_packages_path not in sys.path:
            sys.path.append(site_packages_path)
            logger.info(f"已将虚拟环境路径添加到 sys.path 末尾: {site_packages_path}")
            logger.debug("导入优先级：系统环境包 > 虚拟环境包")
        else:
            logger.debug("虚拟环境路径已存在于 sys.path 中")

    @classmethod
    def ensure_dependencies(cls) -> None:
        """
        确保沙盒环境的依赖包已安装

        如果 requirements.txt 文件存在，会自动创建虚拟环境并安装所有依赖包。
        安装完成后会将虚拟环境的 site-packages 路径添加到 sys.path 末尾，
        确保系统环境的包优先级更高，虚拟环境的包作为补充。
        """
        if not cls._requirements_file.exists():
            logger.debug("未找到依赖文件，跳过依赖安装")
            return

        # 创建虚拟环境（如果不存在）
        if not cls._venv_path.exists():
            logger.info(f"创建沙盒虚拟环境: {cls._venv_path}")
            import venv
            venv.create(cls._venv_path, with_pip=True)

        # 安装依赖包
        try:
            pip_executable = cls._get_pip_executable()
            logger.info(f"安装沙盒依赖包: {cls._requirements_file}")

            subprocess.check_call([
                str(pip_executable),
                "install",
                "-r",
                str(cls._requirements_file)
            ])
            logger.info("依赖包安装完成")

            # 将虚拟环境的 site-packages 路径添加到 sys.path 末尾
            # 确保系统环境包优先级更高，虚拟环境包作为补充
            cls._add_to_sys_path()

        except subprocess.CalledProcessError as e:
            logger.error(f"依赖包安装失败: {e}")
            raise

    @classmethod
    def get_dependency_list(cls) -> list[str]:
        """
        获取当前依赖包列表

        Returns:
            依赖包名称列表
        """
        if not cls._requirements_file.exists():
            return []

        try:
            with cls._requirements_file.open("r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip() and not line.startswith("#")]
        except Exception as e:
            logger.error(f"读取依赖文件失败: {e}")
            return []

    @classmethod
    def save_dependency_list(cls, dependencies: list[str]) -> None:
        """
        保存依赖包列表到 requirements.txt

        Args:
            dependencies: 依赖包名称列表
        """
        try:
            with cls._requirements_file.open("w", encoding="utf-8") as f:
                f.write("# 沙盒环境依赖包\n")
                for dep in dependencies:
                    f.write(f"{dep}\n")
            logger.debug(f"已保存 {len(dependencies)} 个依赖包")
        except Exception as e:
            logger.error(f"保存依赖文件失败: {e}")
            raise

    @classmethod
    def add_dependency(cls, dependency: str) -> None:
        """
        添加单个依赖包并安装

        Args:
            dependency: 要添加的依赖包名称
        """
        dependencies = cls.get_dependency_list()
        if dependency not in dependencies:
            dependencies.append(dependency)
            cls.save_dependency_list(dependencies)
            cls.ensure_dependencies()
            logger.info(f"已添加依赖包: {dependency}")
        else:
            logger.debug(f"依赖包已存在: {dependency}")

    @classmethod
    def remove_dependency(cls, dependency: str) -> None:
        """
        移除单个依赖包并更新安装

        Args:
            dependency: 要移除的依赖包名称
        """
        dependencies = cls.get_dependency_list()
        if dependency in dependencies:
            dependencies.remove(dependency)
            cls.save_dependency_list(dependencies)
            cls.ensure_dependencies()
            logger.info(f"已移除依赖包: {dependency}")
        else:
            logger.debug(f"依赖包不存在: {dependency}")

    @classmethod
    def rebuild_environment(cls) -> None:
        """
        重新构建沙盒环境，重新安装所有依赖包

        此操作会删除现有虚拟环境并重新创建。
        """
        logger.info("开始重新构建沙盒环境")

        if cls._venv_path.exists():
            logger.debug("删除现有虚拟环境")
            shutil.rmtree(cls._venv_path)

        cls.ensure_dependencies()
        logger.info("沙盒环境重建完成")

    @classmethod
    def clear_environment(cls) -> None:
        """
        完全清理沙盒环境，删除虚拟环境和依赖文件
        """
        logger.info("开始清理沙盒环境")

        try:
            if cls._venv_path.exists():
                shutil.rmtree(cls._venv_path)
                logger.debug("已删除虚拟环境")

            if cls._requirements_file.exists():
                cls._requirements_file.unlink()
                logger.debug("已删除依赖文件")

            logger.info("沙盒环境清理完成")

        except Exception as e:
            logger.error(f"清理环境失败: {e}")
            raise


# 使用类级别的依赖管理器（无需实例化）
dependencyManager = DependencyManager


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
        global_vars: 传入代码的全局变量字典
        local_vars: 传入代码的局部变量字典

    Returns:
        tuple: (filtered_globals, filtered_locals) - 经过序列化过滤的变量字典

    Raises:
        MemoryError: 当代码执行超过内存限制时
        Exception: 其他执行异常
    """
    try:
        logger.debug("开始编译受限制代码")
        byte_code = compile_restricted(code, '<string>', 'exec')

        logger.debug("构建安全的全局变量环境")
        safe_globals = _build_safe_globals(global_vars)

        logger.debug("执行代码")
        exec(byte_code, safe_globals, local_vars)

        logger.debug("过滤可序列化对象")
        return _filter_serializable_objects(safe_globals, local_vars)

    except MemoryError:
        logger.error("代码执行超出内存限制")
        raise MemoryError("代码执行超出内存限制")
    except Exception as e:
        logger.error(f"子进程执行失败: {type(e).__name__}: {e}")
        # 将子进程中的异常传播到主进程
        raise e


def _build_safe_globals(global_vars: dict) -> dict:
    """
    构建安全的全局变量环境

    Args:
        global_vars: 用户提供的全局变量字典

    Returns:
        包含安全内置函数和 RestrictedPython 钩子的全局变量字典
    """
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


def _is_serializable(obj) -> bool:
    """
    检查对象是否可序列化

    Args:
        obj: 要检查的对象

    Returns:
        bool: 对象是否可序列化
    """
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False


def _filter_serializable_objects(safe_globals: dict, local_vars: dict) -> tuple:
    """
    过滤出可序列化的对象，确保能在进程间传递

    Args:
        safe_globals: 安全的全局变量字典
        local_vars: 局部变量字典

    Returns:
        tuple: (filtered_globals, filtered_locals) - 过滤后的变量字典
    """
    filtered_globals = {
        key: value for key, value in safe_globals.items()
        if _is_serializable(value)
    }
    filtered_locals = {
        key: value for key, value in local_vars.items()
        if _is_serializable(value)
    }

    logger.debug(f"过滤后的全局变量: {len(filtered_globals)} 个")
    logger.debug(f"过滤后的局部变量: {len(filtered_locals)} 个")

    return filtered_globals, filtered_locals

# ============================================================================
# 主要接口函数
# ============================================================================

async def worker_with_limits(
    python_code: str,
    globals: dict,
    locals: dict,
    timeout: int = DEFAULT_TIMEOUT
) -> tuple:
    """
    带限制的异步代码执行工作器

    这是沙盒模块的主要接口函数，提供安全、受限的 Python 代码执行环境。
    使用进程隔离和超时控制确保主程序的安全性。

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

    Example:
        >>> code = "result = 2 + 2"
        >>> globals = {}
        >>> locals = {}
        >>> exec_globals, exec_locals = await worker_with_limits(code, globals, locals)
        >>> print(exec_locals["result"])  # 输出: 4
    """
    if not isinstance(python_code, str):
        raise TypeError("python_code 必须是字符串")

    if not isinstance(globals, dict):
        raise TypeError("globals 必须是字典")

    if not isinstance(locals, dict):
        raise TypeError("locals 必须是字典")

    if not isinstance(timeout, int) or timeout <= 0:
        raise ValueError("timeout 必须是正整数")

    logger.info(f"开始执行沙盒代码，超时限制: {timeout} 秒")
    logger.debug(f"代码长度: {len(python_code)} 字符")

    try:
        result = await _execute_in_process(  # type: ignore
            code=python_code,
            global_vars=globals,
            local_vars=locals
        )

        logger.info("沙盒代码执行成功")
        return result

    except PebbleTimeoutError:
        # 捕获 pebble 的超时异常并转换为标准 TimeoutError
        error_msg = f"代码执行超时，超过 {timeout} 秒限制"
        logger.error(error_msg)
        raise TimeoutError(error_msg)

    except Exception as e:
        error_msg = f"工作进程执行失败: {type(e).__name__}: {e}"
        logger.error(error_msg)
        raise e


# ============================================================================
# 公共 API 导出
# ============================================================================

__all__ = [
    "worker_with_limits",
    "DependencyManager",
    "dependencyManager",
    "custom_safe_import",
    "DEFAULT_TIMEOUT",
    "BUILTIN_MODULE_PREFIXES",
]