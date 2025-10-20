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

    # 类级别的路径配置
    _requirements_file: Path = store.get_plugin_data_file("sandbox_requirements.txt")
    _venv_path: Path = store.get_plugin_cache_dir() / "sandbox_venv"

    @classmethod
    def _get_pip_executable(cls) -> Path:
        """获取虚拟环境中 pip 的可执行文件路径"""
        if os.name == "nt":  # Windows
            return cls._venv_path / "Scripts" / "pip.exe"
        else:  # Linux/macOS
            return cls._venv_path / "bin" / "pip"

    @classmethod
    def _get_site_packages_path(cls) -> str:
        """获取虚拟环境的 site-packages 路径"""
        if os.name == "nt":  # Windows
            return str(cls._venv_path / "Lib" / "site-packages")
        else:  # Linux/macOS
            py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
            return str(cls._venv_path / "lib" / py_version / "site-packages")

    @classmethod
    def _get_package_name(cls, dependency: str) -> str | None:
        """从依赖字符串中提取包名 (e.g., 'requests==2.28.1' -> 'requests')"""
        match = re.match(r"^[a-zA-Z0-9][a-zA-Z0-9\-_.]*", dependency)
        return match.group(0) if match else None

    @classmethod
    def _add_to_sys_path(cls) -> None:
        """
        将虚拟环境的 site-packages 路径添加到 sys.path 末尾。

        设计说明:
        这种方式将沙盒环境作为主环境的“扩展”，而不是完全“隔离”。
        - 优点: 沙盒代码可以无缝访问主环境已安装的包，避免重复安装。
        - 注意: 如果主环境和沙盒环境存在同名但版本不同的包，
          会优先使用主环境的版本 (因为其路径在 sys.path 中靠前)，
          这可能导致非预期的行为。
        """
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
        """
        确保沙盒环境和依赖项已正确安装。

        如果 requirements.txt 存在，则创建虚拟环境并安装所有依赖。
        此方法是初始化和检查环境的主要入口点。
        """
        if not cls._requirements_file.exists():
            # 确保依赖文件至少是存在的，即使是空的
            cls._requirements_file.touch()
            logger.debug("依赖文件不存在，已创建空文件。")

        # 确保虚拟环境存在
        if not cls._venv_path.exists() or not cls._get_pip_executable().exists():
            logger.info(f"沙盒虚拟环境不存在或不完整，将重新构建。")
            cls.rebuild_environment()
        else:
            # 如果环境已存在，只需确保路径已加载
            cls._add_to_sys_path()
            logger.debug("沙盒环境已存在，跳过安装。")

    @classmethod
    def rebuild_environment(cls) -> None:
        """
        强制重新构建沙盒环境。

        此操作会删除现有虚拟环境（如果存在），然后根据 requirements.txt 重新创建并安装所有依赖。
        这是确保环境纯净和一致的最可靠方法。
        """
        logger.info("开始重建沙盒环境...")

        if cls._venv_path.exists():
            logger.debug(f"删除现有虚拟环境: {cls._venv_path}")
            shutil.rmtree(cls._venv_path)

        logger.info(f"创建沙盒虚拟环境: {cls._venv_path}")
        import venv
        venv.create(cls._venv_path, with_pip=True)

        if cls._requirements_file.exists() and cls._requirements_file.read_text().strip():
            try:
                pip_executable = cls._get_pip_executable()
                logger.info(f"从 {cls._requirements_file} 安装依赖包...")

                subprocess.check_call([
                    str(pip_executable), "install", "-r", str(cls._requirements_file)
                ])
                logger.info("依赖包安装完成。")

            except subprocess.CalledProcessError as e:
                logger.error(f"依赖包安装失败: {e}")
                raise
        else:
            logger.info("依赖文件为空，无需安装。")

        cls._add_to_sys_path()
        logger.info("沙盒环境重建完成。")

    @classmethod
    def get_dependency_list(cls) -> list[str]:
        """获取当前 `requirements.txt` 中的依赖列表（包含版本号）。"""
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
        """获取当前依赖包的名称列表（不含版本号）。"""
        return [name for dep in cls.get_dependency_list() if (name := cls._get_package_name(dep))]

    @classmethod
    def save_dependency_list(cls, dependencies: list[str]) -> None:
        """将依赖列表保存回 `requirements.txt`。"""
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
        """
        添加或更新单个依赖包，并重建环境以应用更改。

        如果包已存在，此方法会更新其版本。

        Args:
            dependency: 依赖包，例如 "requests" 或 "requests==2.28.1"
        """
        package_name = cls._get_package_name(dependency)
        if not package_name:
            logger.warning(f"无效的依赖包格式: {dependency}")
            return

        dependencies = cls.get_dependency_list()
        # 查找是否已存在同名包
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
        """
        移除单个依赖包，并重建环境以应用更改。

        Args:
            package_name: 要移除的包名（无需版本号，例如 "requests"）
        """
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
        """
        完全清理沙盒环境，删除虚拟环境和依赖文件。
        """
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