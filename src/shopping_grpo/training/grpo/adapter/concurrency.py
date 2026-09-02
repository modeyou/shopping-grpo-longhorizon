'''Process-local concurrency control for ShopSimulator leases.'''

from __future__ import annotations

import asyncio
import weakref


_ENVIRONMENT_SEMAPHORES = weakref.WeakKeyDictionary()


def environment_semaphore(limit: int) -> asyncio.Semaphore:
    '''Return one lease limiter shared by every trajectory in this worker loop.'''
    limit = int(limit)
    if limit < 1:
        raise ValueError('max_concurrent_environments_per_worker must be positive')
    loop = asyncio.get_running_loop()
    existing = _ENVIRONMENT_SEMAPHORES.get(loop)
    if existing is None:
        semaphore = asyncio.BoundedSemaphore(limit)
        _ENVIRONMENT_SEMAPHORES[loop] = (limit, semaphore)
        return semaphore
    configured_limit, semaphore = existing
    if configured_limit != limit:
        raise RuntimeError(
            'one AgentLoopWorker event loop cannot use multiple environment '
            f'concurrency limits: {configured_limit} and {limit}'
        )
    return semaphore
