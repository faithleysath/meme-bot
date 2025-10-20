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
    def _get_python_executable(cls) -> Path:
        if os.name == "nt":
            return cls._venv_path / "Scripts" / "python.exe"
        else:
            return cls._venv_path / "bin" / "python"

    @classmethod
    def _get_site_packages_path(cls) -> str:
        # 使用子进程从虚拟环境中动态获取 site-packages 路径，更可靠
        try:
            python_executable = cls._get_python_executable()
            command = [
                str(python_executable),
                "-c",
                "import site; print(site.getsitepackages()[0])",
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"无法获取 site-packages 路径: {e}")
            # Fallback to the old method if the new one fails
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
    def _run_pip_command(cls, *args: str) -> None:
        """通用 pip 命令执行器，增加日志和错误捕获"""
        if not cls._venv_path.exists() or not cls._get_pip_executable().exists():
            logger.error("虚拟环境或 pip 不存在，无法执行命令。")
            raise FileNotFoundError("vnev or pip not found")
        
        command = [str(cls._get_pip_executable()), *args]
        logger.info(f"正在执行命令: {' '.join(command)}")
        try:
            # 使用 subprocess.run 捕获输出，便于调试
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                check=True,
                encoding="utf-8"
            )
            if result.stdout:
                logger.debug(f"Pip stdout:\n{result.stdout}")
            if result.stderr:
                logger.debug(f"Pip stderr:\n{result.stderr}")
            logger.info(f"命令 {' '.join(args)} 执行成功。")
        except subprocess.CalledProcessError as e:
            logger.error(f"Pip 命令执行失败: {' '.join(command)}")
            logger.error(f"返回码: {e.returncode}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            raise

    @classmethod
    def _add_to_sys_path(cls) -> None:
        if not cls._venv_path.exists():
            return
        site_packages_path = cls._get_site_packages_path()
        if site_packages_path and site_packages_path not in sys.path:
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

        # 这里的判断非常高效，是正确的选择
        if cls._requirements_file.read_text(encoding="utf-8").strip():
            logger.info(f"正在检查并同步沙盒依赖于: {cls._requirements_file}")
            cls._run_pip_command("install", "-r", str(cls._requirements_file))
        else:
            logger.debug("依赖文件为空，跳过同步。")

        cls._add_to_sys_path()

    @classmethod
    def rebuild_environment(cls) -> None:
        """
        强制彻底重新构建沙盒环境。

        此操作会删除现有的虚拟环境，然后根据 requirements.txt 文件
        重新安装所有依赖。这是一个耗时操作，用于修复环境问题。
        """
        logger.info("开始重建沙盒环境...")

        if cls._venv_path.exists():
            logger.debug(f"删除现有虚拟环境: {cls._venv_path}")
            shutil.rmtree(cls._venv_path)

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
        """
        [高效版] 添加或更新一个依赖包，并只安装这一个包。
        """
        package_name = cls._get_package_name(dependency)
        if not package_name:
            logger.warning(f"无效的依赖包格式: {dependency}")
            return
        
        # 确保环境存在
        cls.ensure_dependencies()
        
        logger.info(f"正在安装/更新依赖包: {dependency}")
        cls._run_pip_command("install", "--upgrade", dependency)

        # 安装成功后，再更新 requirements.txt 文件
        dependencies = cls.get_dependency_list()
        updated = False
        for i, existing_dep in enumerate(dependencies):
            if cls._get_package_name(existing_dep) == package_name:
                logger.debug(f"更新依赖文件中的包: {existing_dep} -> {dependency}")
                dependencies[i] = dependency
                updated = True
                break
        if not updated:
            logger.debug(f"向依赖文件添加新包: {dependency}")
            dependencies.append(dependency)
        cls.save_dependency_list(dependencies)

    @classmethod
    def remove_dependency(cls, package_name: str) -> None:
        """
        [高效版] 移除一个依赖包，并只卸载这一个包。
        """
        package_name_to_remove = cls._get_package_name(package_name)
        if not package_name_to_remove:
            logger.warning(f"无效的包名格式: {package_name}")
            return

        # 确保环境存在
        cls.ensure_dependencies()

        dependencies = cls.get_dependency_list()
        original_count = len(dependencies)
        remaining_deps = [
            dep for dep in dependencies
            if cls._get_package_name(dep) != package_name_to_remove
        ]

        if len(remaining_deps) < original_count:
            logger.info(f"正在卸载包: {package_name_to_remove}")
            cls._run_pip_command("uninstall", "-y", package_name_to_remove)

            logger.info(f"已从依赖列表中移除包: {package_name_to_remove}")
            cls.save_dependency_list(remaining_deps)
        else:
            logger.warning(f"依赖包不存在，无需移除: {package_name_to_remove}")

    @classmethod
    def clear_environment(cls) -> None:
        logger.info("开始清理沙盒环境...")
        try:
            site_packages_path = cls._get_site_packages_path()
            if site_packages_path in sys.path:
                sys.path.remove(site_packages_path)

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