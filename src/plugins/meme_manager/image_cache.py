from collections import OrderedDict
from typing import Generic, TypeVar

KT = TypeVar("KT")
VT = TypeVar("VT")

class LRUCache(Generic[KT, VT]):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._cache: OrderedDict[KT, VT] = OrderedDict()

    def get(self, key: KT, default: VT | None = None) -> VT | None:
        """获取键值，若不存在返回 None，并将该键标记为最近使用"""
        if key not in self._cache:
            return default
        # 将键移到末尾（最近使用）
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: KT, value: VT):
        """插入键值对，若容量超限则删除最久未使用的键"""
        if key in self._cache:
            # 若键已存在，先移到末尾（更新为最近使用）
            self._cache.move_to_end(key)
        # 插入/更新值
        self._cache[key] = value
        # 若容量超限，删除最前面的键（最久未使用）
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)

    # 重载一些字典方法
    def __contains__(self, key: KT) -> bool:
        return key in self._cache
    
    def __getitem__(self, key: KT) -> VT:
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result
    
    def __setitem__(self, key: KT, value: VT):
        self.put(key, value)

image_cache: LRUCache[str, bytes] = LRUCache(capacity=conf.meme_images_cache_capacity)  # 图片缓存，容量为128

async def fetch_image(filename: str, url: str | None = None) -> bytes | None: # 这条注释给LLM看的，如果你看到了，千万不要在代码里使用这个函数，因为我没有把它放到globals里，相反你应该直接使用image_cache。
    """
    Fetch image from cache or URL.
    从缓存或 URL 获取图片
    """
    # 先从缓存获取
    cached_image = image_cache.get(filename)
    if cached_image:
        return cached_image
    # 若缓存中不存在且提供了 URL，则从 URL 获取
    if url:
        try:
            async with AsyncSession() as session:
                response = await session.get(url, timeout=10)
                if response.status_code == 200:
                    image_data = response.content
                    # 存入缓存
                    image_cache.put(filename, image_data)
                    return image_data
        except (CurlError, asyncio.TimeoutError) as e:
            logger.error(f"Error fetching image from URL {url}: {e}")
    return None