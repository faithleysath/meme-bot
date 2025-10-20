"""
沙盒依赖管理器

负责管理沙盒执行环境的 Python 依赖包，包括：
- 虚拟环境创建和管理
- 依赖包安装和卸载
- 环境清理和重建
- 自动路径管理
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import nonebot_plugin_localstore as store
from nonebot import logger


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


# 公共 API 导出
__all__ = [
    "DependencyManager",
    "dependencyManager",
]