"""
沙盒依赖管理器

负责管理沙盒执行环境的 Python 依赖包，包括：
- 虚拟环境创建和管理
- 依赖包安装和卸载
- 环境清理和重建
- 自动路径管理
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
import venv

import nonebot_plugin_localstore as store
from nonebot import logger


class DependencyManager:
    """
    沙盒环境依赖管理器（无实例类）

    通过管理一个独立的虚拟环境，为沙盒代码提供 Python 依赖支持。
    主要功能包括：
    - 自动创建和管理虚拟环境。
    - 通过 requirements.txt 文件安装、更新和移除依赖包。
    - 将虚拟环境路径添加到 sys.path，使沙盒代码可以导入其中的包。
    """

    _requirements_file: Path = store.get_plugin_data_file("sandbox_requirements.txt")
    _venv_path: Path = store.get_plugin_cache_dir() / "sandbox_venv"

    @classmethod
    def _get_pip_executable(cls) -> Path:
        if os.name == "nt":
            return cls._venv_path / "Scripts" / "pip.exe"
        else:
            return cls._venv_path / "bin" / "pip"

    @classmethod
    def _get_site_packages_path(cls) -> str:
        if os.name == "nt":
            return str(cls._venv_path / "Lib" / "site-packages")
        else:
            py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
            return str(cls._venv_path / "lib" / py_version / "site-packages")

    @classmethod
    def _get_package_name(cls, dependency: str) -> str | None:
        match = re.match(r"^[a-zA-Z0-9][a-zA-Z0-9\-_.]*", dependency)
        return match.group(0) if match else None

    @classmethod
    def _add_to_sys_path(cls) -> None:
        if not cls._venv_path.exists():
            return
        site_packages_path = cls._get_site_packages_path()
        if site_packages_path not in sys.path:
            sys.path.append(site_packages_path)
            logger.info(f"已将沙盒环境路径添加到 sys.path: {site_packages_path}")
        else:
            logger.debug("沙盒环境路径已存在于 sys.path 中")

    @classmethod
    def ensure_dependencies(cls) -> None:
        if not cls._requirements_file.exists():
            cls._requirements_file.touch()
            logger.debug("依赖文件不存在，已创建空文件。")

        if not cls._venv_path.exists() or not cls._get_pip_executable().exists():
            logger.info(f"虚拟环境不存在或不完整，正在创建: {cls._venv_path}")
            venv.create(cls._venv_path, with_pip=True)

        if cls._requirements_file.read_text(encoding="utf-8").strip():
            try:
                pip_executable = cls._get_pip_executable()
                logger.info(f"正在检查并同步沙盒依赖于: {cls._requirements_file}")
                subprocess.check_call([
                    str(pip_executable), "install", "-r", str(cls._requirements_file)
                ])
                logger.debug("依赖同步完成。")
            except subprocess.CalledProcessError as e:
                logger.error(f"依赖同步失败，建议尝试重建环境: {e}")
                raise
        else:
            logger.debug("依赖文件为空，跳过同步。")

        cls._add_to_sys_path()

    @classmethod
    def rebuild_environment(cls) -> None:
        """
        [精简版] 强制重新构建沙盒环境。

        此操作通过删除现有虚拟环境，然后调用 `ensure_dependencies` 来完成
        一个完整的、纯净的重建过程。
        """
        logger.info("开始重建沙盒环境...")

        if cls._venv_path.exists():
            logger.debug(f"删除现有虚拟环境: {cls._venv_path}")
            shutil.rmtree(cls._venv_path)

        # 删除后，ensure_dependencies 会自动处理创建和安装的全部流程
        cls.ensure_dependencies()

        logger.info("沙盒环境重建完成。")

    @classmethod
    def get_dependency_list(cls) -> list[str]:
        if not cls._requirements_file.exists():
            return []
        try:
            with cls._requirements_file.open("r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip() and not line.startswith("#")]
        except Exception as e:
            logger.error(f"读取依赖文件失败: {e}")
            return []

    @classmethod
    def get_dependency_names(cls) -> list[str]:
        return [name for dep in cls.get_dependency_list() if (name := cls._get_package_name(dep))]

    @classmethod
    def save_dependency_list(cls, dependencies: list[str]) -> None:
        try:
            with cls._requirements_file.open("w", encoding="utf-8") as f:
                if dependencies:
                    f.write("# 沙盒环境依赖包\n")
                    f.write("\n".join(dependencies) + "\n")
            logger.debug(f"已将 {len(dependencies)} 个依赖包保存到 requirements.txt")
        except Exception as e:
            logger.error(f"保存依赖文件失败: {e}")
            raise

    @classmethod
    def add_dependency(cls, dependency: str) -> None:
        package_name = cls._get_package_name(dependency)
        if not package_name:
            logger.warning(f"无效的依赖包格式: {dependency}")
            return
        dependencies = cls.get_dependency_list()
        updated = False
        for i, existing_dep in enumerate(dependencies):
            if cls._get_package_name(existing_dep) == package_name:
                logger.info(f"更新依赖包: {existing_dep} -> {dependency}")
                dependencies[i] = dependency
                updated = True
                break
        if not updated:
            logger.info(f"添加新依赖包: {dependency}")
            dependencies.append(dependency)
        cls.save_dependency_list(dependencies)
        cls.rebuild_environment()

    @classmethod
    def remove_dependency(cls, package_name: str) -> None:
        package_name_to_remove = cls._get_package_name(package_name)
        if not package_name_to_remove:
            logger.warning(f"无效的包名格式: {package_name}")
            return
        dependencies = cls.get_dependency_list()
        original_count = len(dependencies)
        remaining_deps = [
            dep for dep in dependencies
            if cls._get_package_name(dep) != package_name_to_remove
        ]
        if len(remaining_deps) < original_count:
            logger.info(f"已从依赖列表中移除包: {package_name_to_remove}")
            cls.save_dependency_list(remaining_deps)
            cls.rebuild_environment()
        else:
            logger.debug(f"依赖包不存在，无需移除: {package_name_to_remove}")

    @classmethod
    def clear_environment(cls) -> None:
        logger.info("开始清理沙盒环境...")
        try:
            if cls._venv_path.exists():
                shutil.rmtree(cls._venv_path)
                logger.debug("已删除虚拟环境。")
            if cls._requirements_file.exists():
                cls._requirements_file.unlink()
                logger.debug("已删除依赖文件。")
            logger.info("沙盒环境清理完成。")
        except Exception as e:
            logger.error(f"清理环境失败: {e}")
            raise


# 公共 API 导出
__all__ = [
    "DependencyManager",
]