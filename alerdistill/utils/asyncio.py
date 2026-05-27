from __future__ import annotations
import asyncio
from typing import Any, Coroutine

def run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """
    Run an async coroutine from sync code, compatible with notebooks and scripts.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        # If already running (e.g., notebook), create a new loop in a thread
        import threading
        result_container = {"res": None, "err": None}
        def _runner():
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                result_container["res"] = new_loop.run_until_complete(coro)
            except BaseException as e:
                result_container["err"] = e
            finally:
                try:
                    new_loop.close()
                except Exception:
                    pass
        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        t.join()
        if result_container["err"] is not None:
            raise result_container["err"]
        return result_container["res"]
    else:
        return loop.run_until_complete(coro)
