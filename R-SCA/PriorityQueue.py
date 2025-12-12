import heapq

# 定义优先队列控制，节点生长优先级  吸引子影响节点生长优先级以及枝干粗细程度
class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0
        self._removed = []

    def push(self, item, priority):
        heapq.heappush(self._queue, (-priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]

    def peek(self):
        return self._queue[0][-1] if self._queue else None

    def is_empty(self):
        return len(self._queue) == 0

    def contains(self, item):
        return any(item.position == element[2].position for element in self._queue)

    def remove(self, item):
        for i, (_, _, element) in enumerate(self._queue):
            if element == item:
                self._removed.append(i)

        new_queuq = [element for i, element in enumerate(self._queue) if i not in self._removed]
        heapq.heapify(new_queuq)
        self._queue = new_queuq
        self._removed.clear()
    def length(self):
        return len(self._queue)

    def print(self):
        for node in self._queue:
            print(node[-1].position)

