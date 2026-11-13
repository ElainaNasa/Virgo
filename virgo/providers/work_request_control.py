"""进程内工作Agent请求起始限速，路由与上下文构建不使用此入口。"""
import threading
import time
from contextvars import ContextVar
from contextlib import contextmanager

_WORK_REQUEST=ContextVar("virgo_work_request",default=False)
_WORK_GUARD=ContextVar("virgo_work_guard",default=None)

@contextmanager
def work_request_scope(guard=None):
    token=_WORK_REQUEST.set(True)
    guard_token=_WORK_GUARD.set(guard)
    try:
        yield
    finally:
        _WORK_REQUEST.reset(token)
        _WORK_GUARD.reset(guard_token)

def throttle_work_transport():
    if _WORK_REQUEST.get():
        wait_for_work_request()
        guard=_WORK_GUARD.get()
        if guard is not None and not guard():
            raise RuntimeError("等待期间该模型渠道已失败，不再发送请求")

_LOCK=threading.Lock()
_NEXT_START=0.0


def wait_for_work_request():
    global _NEXT_START
    with _LOCK:
        delay=max(0.0,_NEXT_START-time.monotonic())
        if delay:
            time.sleep(delay)
        _NEXT_START=time.monotonic()+8.0
