from collections import deque

class SessionHistory:
    """存储每个会话的消息历史记录"""
    def __init__(self, max_history: int = 20):
        super().__init__()
        self.max_history = max_history
        self._history: dict[int, deque[MessageEvent]] = {}

    def add_event(self, session_id: int, event: MessageEvent):
        """将事件添加到指定会话的历史记录中"""
        if session_id not in self._history:
            self._history[session_id] = deque(maxlen=self.max_history)
        self._history[session_id].append(event)

    def get_history(self, session_id: int) -> list[MessageEvent]:
        """获取指定会话的历史记录"""
        return list(self._history.get(session_id, deque()))
    
    def clear_history(self, session_id: int):
        """清除指定会话的历史记录"""
        if session_id in self._history:
            del self._history[session_id]

    # 重载一些字典方法
    def __contains__(self, key: int) -> bool:
        return key in self._history and len(self._history[key]) > 0
    def __getitem__(self, key: int) -> list[MessageEvent]:
        return self.get_history(key)
    def __setitem__(self, key: int, value: list[MessageEvent]):
        self._history[key] = deque(value, maxlen=self.max_history)
    # 重载一些运算符
    def __len__(self) -> int:
        return len(self._history)

session_history = SessionHistory(max_history=conf.meme_max_history_messages)  # 会话历史记录，最多存储20条消息