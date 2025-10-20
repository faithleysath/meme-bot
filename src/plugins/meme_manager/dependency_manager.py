import asyncio
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
import venv

import aiofiles
import nonebot_plugin_localstore as store
from nonebot import logger

class DependencyManager:
    """
    沙盒环境依赖管理器（异步版）

    通过管理一个独立的虚拟环境，为沙盒代码提供 Python 依赖支持。
    所有 I/O 操作均为异步，以避免阻塞事件循环。
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
    def _get_package_name(cls, dependency: str) -> str | None:
        match = re.match(r"^[a-zA-Z0-9][a-zA-Z0-9\-_.]*", dependency)
        return match.group(0) if match else None

    
    @classmethod
    async def _get_site_packages_path(cls) -> str:
        """异步获取 site-packages 路径，更可靠"""
        try:
            python_executable = cls._get_python_executable()
            command = [
                str(python_executable),
                "-c",
                "import site; print(site.getsitepackages()[0])",
            ]
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"获取 site-packages 路径失败: {stderr.decode(errors='ignore')}")
                raise RuntimeError("Failed to get site-packages path")

            return stdout.decode().strip()
        except FileNotFoundError as e:
            logger.error(f"无法获取 site-packages 路径，Python 可执行文件未找到: {e}")
            raise

    @classmethod
    async def _run_pip_command(cls, *args: str) -> None:
        """异步通用 pip 命令执行器"""
        pip_executable = cls._get_pip_executable()
        if not await asyncio.to_thread(pip_executable.exists):
            logger.error("虚拟环境 pip 不存在，无法执行命令。")
            raise FileNotFoundError("vnev pip not found")

        command = [str(pip_executable), *args]
        logger.info(f"正在执行命令: {' '.join(command)}")

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            assert process.returncode is not None
            error_message = (
                f"Pip 命令执行失败: {' '.join(command)}\n"
                f"返回码: {process.returncode}\n"
                f"Stdout: {stdout.decode(errors='ignore')}\n"
                f"Stderr: {stderr.decode(errors='ignore')}"
            )
            logger.error(error_message)
            raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)
        else:
            logger.debug(f"Pip stdout:\n{stdout.decode(errors='ignore')}")
            logger.info(f"命令 {' '.join(args)} 执行成功。")


    @classmethod
    async def _add_to_sys_path(cls) -> None:
        if not await asyncio.to_thread(cls._venv_path.exists):
            return
        try:
            site_packages_path = await cls._get_site_packages_path()
            if site_packages_path and site_packages_path not in sys.path:
                sys.path.append(site_packages_path)
                logger.info(f"已将沙盒环境路径添加到 sys.path: {site_packages_path}")
            else:
                logger.debug("沙盒环境路径已存在于 sys.path 中")
        except (FileNotFoundError, RuntimeError):
            logger.warning("无法添加沙盒环境路径到 sys.path，环境可能不完整。")

    
    @classmethod
    async def ensure_dependencies(cls) -> None:
        """确保虚拟环境和依赖存在并同步"""
        if not await asyncio.to_thread(cls._requirements_file.exists):
            async with aiofiles.open(cls._requirements_file, "w") as f:
                await f.write("") # 异步创建空文件
            logger.debug("依赖文件不存在，已创建空文件。")

        pip_executable = cls._get_pip_executable()
        if not await asyncio.to_thread(pip_executable.exists):
            logger.info(f"虚拟环境不存在或不完整，正在创建: {cls._venv_path}")
            # venv.create 是阻塞的，所以放到线程中执行
            await asyncio.to_thread(venv.create, cls._venv_path, with_pip=True)

        async with aiofiles.open(cls._requirements_file, "r", encoding="utf-8") as f:
            content = await f.read()
        
        if content.strip():
            logger.info(f"正在检查并同步沙盒依赖于: {cls._requirements_file}")
            await cls._run_pip_command("install", "-r", str(cls._requirements_file))
        else:
            logger.debug("依赖文件为空，跳过同步。")

        await cls._add_to_sys_path()

    @classmethod
    async def rebuild_environment(cls) -> None:
        """强制彻底地重新构建沙盒环境（异步）"""
        logger.info("开始重建沙盒环境...")

        if await asyncio.to_thread(cls._venv_path.exists):
            logger.debug(f"删除现有虚拟环境: {cls._venv_path}")
            # shutil.rmtree 是阻塞的
            await asyncio.to_thread(shutil.rmtree, cls._venv_path)

        await cls.ensure_dependencies()
        logger.info("沙盒环境重建完成。")

    @classmethod
    async def get_dependency_list(cls) -> list[str]:
        if not await asyncio.to_thread(cls._requirements_file.exists):
            return []
        try:
            async with aiofiles.open(cls._requirements_file, "r", encoding="utf-8") as f:
                lines = await f.readlines()
                return [line.strip() for line in lines if line.strip() and not line.startswith("#")]
        except Exception as e:
            logger.error(f"读取依赖文件失败: {e}")
            return []

    @classmethod
    async def get_dependency_names(cls) -> list[str]:
        deps = await cls.get_dependency_list()
        return [name for dep in deps if (name := cls._get_package_name(dep))]

    @classmethod
    async def save_dependency_list(cls, dependencies: list[str]) -> None:
        try:
            content = ""
            if dependencies:
                content = "# 沙盒环境依赖包\n" + "\n".join(dependencies) + "\n"
            
            async with aiofiles.open(cls._requirements_file, "w", encoding="utf-8") as f:
                await f.write(content)
            logger.debug(f"已将 {len(dependencies)} 个依赖包保存到 requirements.txt")
        except Exception as e:
            logger.error(f"保存依赖文件失败: {e}")
            raise

    @classmethod
    async def add_dependency(cls, dependency: str) -> None:
        """[高效异步版] 添加或更新一个依赖包"""
        package_name = cls._get_package_name(dependency)
        if not package_name:
            logger.warning(f"无效的依赖包格式: {dependency}")
            return
        
        await cls.ensure_dependencies()
        
        logger.info(f"正在安装/更新依赖包: {dependency}")
        await cls._run_pip_command("install", "--upgrade", dependency)

        dependencies = await cls.get_dependency_list()
        updated = False
        for i, existing_dep in enumerate(dependencies):
            if cls._get_package_name(existing_dep) == package_name:
                dependencies[i] = dependency
                updated = True
                break
        if not updated:
            dependencies.append(dependency)
        await cls.save_dependency_list(dependencies)

    @classmethod
    async def remove_dependency(cls, package_name: str) -> None:
        """[高效异步版] 移除一个依赖包"""
        package_name_to_remove = cls._get_package_name(package_name)
        if not package_name_to_remove:
            logger.warning(f"无效的包名格式: {package_name}")
            return

        await cls.ensure_dependencies()

        dependencies = await cls.get_dependency_list()
        remaining_deps = [
            dep for dep in dependencies
            if cls._get_package_name(dep) != package_name_to_remove
        ]

        if len(remaining_deps) < len(dependencies):
            logger.info(f"正在卸载包: {package_name_to_remove}")
            await cls._run_pip_command("uninstall", "-y", package_name_to_remove)
            await cls.save_dependency_list(remaining_deps)
        else:
            logger.warning(f"依赖包不存在，无需移除: {package_name_to_remove}")

    @classmethod
    async def clear_environment(cls) -> None:
        """完全清理沙盒环境（异步）"""
        logger.info("开始清理沙盒环境...")
        try:
            # 移除 sys.path 是同步操作，因为它直接操作内存中的列表
            try:
                site_packages_path = await cls._get_site_packages_path()
                if site_packages_path in sys.path:
                    sys.path.remove(site_packages_path)
            except (FileNotFoundError, RuntimeError):
                pass # 环境可能已经不完整，忽略错误

            if await asyncio.to_thread(cls._venv_path.exists):
                await asyncio.to_thread(shutil.rmtree, cls._venv_path)
                logger.debug("已删除虚拟环境。")
            
            if await asyncio.to_thread(cls._requirements_file.exists):
                # os.remove 是阻塞的
                await asyncio.to_thread(os.remove, cls._requirements_file)
                logger.debug("已删除依赖文件。")
            
            logger.info("沙盒环境清理完成。")
        except Exception as e:
            logger.error(f"清理环境失败: {e}")
            raise


# 公共 API 导出
__all__ = [
    "DependencyManager",
]