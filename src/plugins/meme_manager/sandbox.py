import os
from RestrictedPython import compile_restricted, safe_builtins

def _execute_in_process(code: str, global_vars: dict, local_vars: dict, memory_mb: int):
    """
    这个函数将会在子进程中被独立执行。
    它不依赖任何 NoneBot 的运行时环境。
    """
    if os.name == 'posix':
        import resource
        memory_limit_bytes = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
    
    try:
        byte_code = compile_restricted(code, '<string>', 'exec')
        safe_globals = {"__builtins__": safe_builtins, **global_vars}
        exec(byte_code, safe_globals, local_vars)
        return safe_globals, local_vars
    except MemoryError:
        raise MemoryError("Code execution exceeded memory limit.")
    except Exception as e:
        raise e