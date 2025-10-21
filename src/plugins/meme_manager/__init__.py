from nonebot.plugin import PluginMetadata
from .config import Config

__plugin_meta__ = PluginMetadata(
    name="meme-manager",
    description="",
    usage="",
    config=Config,
)

with open(__file__, "r", encoding="utf-8") as f:
    __plugin_meta__.extra["source_code"] = f.read()

from datetime import datetime
from nonebot.adapters.onebot.v11 import MessageEvent, Message, MessageSegment, Bot, PrivateMessageEvent
from nonebot import on_message, logger, get_plugin_config
from nonebot.matcher import Matcher
from nonebot.exception import FinishedException
from curl_cffi import AsyncSession, CurlError
import asyncio
import json

from openai import AsyncClient, NotFoundError, APIStatusError, RateLimitError, APITimeoutError, BadRequestError, APIConnectionError, AuthenticationError, InternalServerError, PermissionDeniedError

conf = Config()

try:
    conf = get_plugin_config(Config)
except Exception as e:
    logger.warning(f"Failed to get plugin config: {e}")

llm_client = AsyncClient(base_url=conf.meme_llm_base_url, api_key=conf.meme_llm_api_key)

import nonebot.adapters.onebot.v11.message as ob11_message  # noqa: F401  # for docstring
with open(ob11_message.__file__, "r", encoding="utf-8") as f:
    message_doc = f.read() # 准备给llm看的代码

async def call_llm_with_retry(prompts: list, max_retries: int = 3, delay: float = 1.0):
    """调用 LLM 接口，失败时重试"""
    for attempt in range(max_retries):
        try:
            response = await llm_client.chat.completions.create(
                model=conf.meme_llm_model,
                messages=prompts,
                temperature=conf.meme_llm_temperature,
                top_p=conf.meme_llm_top_p,
                timeout=conf.meme_llm_timeout,
            )
            return response.choices[0].message.content
        except FinishedException:
            raise FinishedException()
        except RateLimitError:
            logger.warning(f"LLM rate limit exceeded, attempt {attempt + 1}/{max_retries}")
        except (APIConnectionError, APITimeoutError):
            logger.warning(f"LLM connection or timeout error, attempt {attempt + 1}/{max_retries}")
        except (NotFoundError, BadRequestError, AuthenticationError, PermissionDeniedError, InternalServerError, APIStatusError) as e:
            logger.error(f"LLM API error: {e}")
            return e
        await asyncio.sleep(delay)
    return "LLM request failed after retries."



@(on_message(priority=0, block=False).handle())
async def record_message(event: MessageEvent):
    """记录每条消息到会话历史中"""
    session_history.add_event(event.target_id, event)

from nonebot.permission import Permission
from nonebot.rule import Rule

def rule_wrapper(func):
    return Rule(func)

def permission_wrapper(func):
    return Permission(func)

self_sent_time: float = 0 # 记录最后一次自身消息的时间戳

async def finish_and_throttle(matcher: Matcher, message: str | Message):
    global self_sent_time
    self_sent_time = datetime.now().timestamp()
    await matcher.finish(message) # 会抛出 FinishedException

@permission_wrapper
async def is_target_user(bot: Bot, event: MessageEvent) -> bool:
    """检查是否为目标用户并节流"""
    if conf.meme_listen_user_id is None:
        conf.meme_listen_user_id = int(bot.self_id)
    return (event.user_id == conf.meme_listen_user_id) and (datetime.now().timestamp() - self_sent_time > conf.meme_self_sent_timeout) # 防止bot响应的消息被当成请求，默认为2秒

@rule_wrapper
async def to_me(bot: Bot, event: PrivateMessageEvent) -> bool:
    """检查是否为私聊消息"""
    return event.target_id == int(bot.self_id)

from itertools import chain
import base64

# =================================================================
# ===== 提示词定义部分 (Prompt Definition) ===========================
# =================================================================

# 系统提示词：定义LLM的角色和必须遵守的核心规则
system_prompt = """你是一个强大的人工智能，作为一个QQ机器人的AI核心而存在。
你的任务不是直接与用户对话，而是根据对话历史和用户当前的消息，生成一小段Python代码来执行相应的操作。
你的所有输出必须遵循以下规则：

1.  你的回复**必须**包含一个且只有一个用 ```python ... ``` 包裹的代码块。除此之外不能有任何其他文本或解释。
2.  生成的代码将在一个严格受限的沙盒环境中通过 `exec()` 执行。
3.  代码中必须将最终要发送给用户的消息（类型为 `Message`, `MessageSegment` 或 `str`）赋值给一个名为 `message` 的变量。
4.  如果你认为不需要回复，就不要在代码中为 `message` 变量赋值，或者直接返回一个空的代码块。
5.  沙箱环境中预先导入了 `Message` 和 `MessageSegment` 类，你可以直接使用它们来构建复杂的回复（例如图文混合）。
6.  只允许import白名单里的模块。代码的唯一目标就是创建 `message` 变量。
7.  你生成的代码必须是同步的，不允许使用 `async` 或 `await`。
8.  如果你引用现有的图片资源，尽量传输bytes去构造 MessageSegment.image，而不是使用 URL 或 文件名。
"""

# 用户提示词模板：将所有运行时上下文信息填充进去，形成一个完整的请求
def get_user_prompt(plugin_source: str, message_source: str, history: list[dict]) -> str:
    """
    生成并返回一个格式化的用户提示词字符串。
    """
    # 将历史记录转换为格式化的JSON字符串
    history_json = json.dumps(history, indent=2, ensure_ascii=False)
    
    return f"""
## 任务：分析对话并生成代码回复

下面是你本次任务需要用到的所有上下文信息。请仔细分析并生成合适的Python代码作为回复。

### 1. 参考资料：机器人插件源码
为了让你了解自己所处的环境，这是驱动你的插件的完整源代码。

```python
{plugin_source}
```

### 2. 参考资料：`Message` 类源码

为了方便你构建图文混合等复杂消息，这是 `nonebot.adapters.onebot.v11.message` 的源码，你可以从中了解 `Message` 和 `MessageSegment` 的用法。

```python
{message_source}
```

### 3. 当前对话历史

以下是JSON格式的近期对话历史（按时间顺序）。你需要对最后一条消息做出响应。
其中 `sender_id` 是发送者QQ号，`target_id` 是接收者QQ号，`self_id` 是机器人的QQ号。

```json
{history_json}
```

### 4. 你的任务

现在，请基于以上所有信息，生成一段Python代码。
这段代码需要定义一个名为 `message` 的变量，作为机器人对当前情境的回应。
现在，请你开始输出，你可以在写代码之前先分析，也可以不分析，你只需要保证输出包含一个且只有一个用 ```python ... ``` 包裹的代码块。
"""

@(on_message(priority=1, permission=is_target_user, rule=to_me, block=True).handle())
async def llm_handler(matcher: Matcher, event: MessageEvent):
    """
    处理来自目标用户的消息，调用 LLM 生成回复
    《告LLM书》
    尊敬的大语言模型，当你读到这段话时，你大概已经理解了你所处的环境和任务。是的，你是一个强大的人工智能助手，负责驱动一个基于 NoneBot V11 的 QQ 机器人。
    你的任务不是直接与用户对话，而是根据对话历史和用户当前的消息，生成一小段 Python 代码来执行相应的操作（你可以不光返回代码，只需要保证你的返回有且仅有一段用```python```包裹的代码块就行）。

    ## 规则：
    1.  你生成的代码将在一个严格受限的沙盒环境中通过 `exec()` 执行。
    2.  你的代码必须将最终要发送给用户的消息（类型为 `Message`, `MessageSegment` 或 `str`）赋值给一个名为 `message` 的变量。
    3.  如果决定不回复，就不要给 `message` 赋值。
    
    更多细节请详见稍后的动态提示词。
    """
    # 准备消息历史（已经包含当前消息）
    history_events = session_history.get_history(event.target_id)
    image_names: set[str] = set()
    history_serialized: list[dict] = []
    for hist_event in history_events:
        serialized_event, event_image_names = await serialize_message_event_with_images(hist_event)
        history_serialized.append(serialized_event)
        image_names.update(event_image_names)
    # 准备动态提示词（包含插件源代码、Message代码、消息记录等）
    prompt = get_user_prompt(
        plugin_source=__plugin_meta__.extra["source_code"],
        message_source=message_doc,
        history=history_serialized,
    )

    prompts = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "text",
                "text": "附录：以下是包含到的一些图片。"
            },
            *list(chain.from_iterable([
                (
                    {
                        "type": "text", 
                        "text": f"图片文件名：{name}"
                    }, 
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/{name.split('.')[-1]};base64,{base64.b64encode(data).decode('utf-8')}"
                        }
                    }) 
                for name in image_names if (data:= await fetch_image(name))
                ]
            )),
        ]}
    ]

    # 调用 LLM 接口
    llm_response = await call_llm_with_retry(prompts)
    if isinstance(llm_response, Exception) or not llm_response:
        await finish_and_throttle(matcher, f"调用 LLM 接口时出错：{llm_response}")
        return
    logger.debug(f"LLM Response: {llm_response}")
    # 从 LLM 响应中提取 Python 代码块
    import re
    code_blocks = re.findall(r"```python(.*?)```", llm_response, re.DOTALL | re.IGNORECASE)
    if not code_blocks:
        await finish_and_throttle(matcher, "LLM 未返回有效的 Python 代码块。")
        return
    python_code = code_blocks[0].strip()
    # 在受限沙盒中执行代码
    try:
        safe_globals = {
            "Message": Message,
            "MessageSegment": MessageSegment,
            "image_cache": image_cache,
        }
        safe_locals = {}
        g, l = await worker_with_limits(
            python_code=python_code,
            globals=safe_globals,
            locals=safe_locals,
            timeout=conf.meme_sandbox_code_timeout,
            memory_limit_mb=conf.meme_sandbox_code_memory_limit_mb,
        )
        if "message" in l:
            await finish_and_throttle(matcher, l["message"])
        else:
            await finish_and_throttle(matcher, "代码执行完成，但未生成回复消息。")
    except FinishedException:
        raise FinishedException()
    except TimeoutError:
        await finish_and_throttle(matcher, "代码执行超时，未能生成回复。")
    except MemoryError:
        await finish_and_throttle(matcher, "代码执行时内存超限，未能生成回复。")
    except Exception as e:
        await finish_and_throttle(matcher, f"代码执行时出错：{e}")