import asyncio

import pytest

from shopping_grpo.training.grpo.adapter.concurrency import environment_semaphore


def test_environment_semaphore_bounds_one_worker_event_loop():
    async def exercise():
        active = 0
        peak = 0
        limiter = environment_semaphore(2)

        async def trajectory():
            nonlocal active, peak
            async with limiter:
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.01)
                active -= 1

        await asyncio.gather(*(trajectory() for _ in range(7)))
        return peak

    assert asyncio.run(exercise()) == 2


def test_environment_semaphore_rejects_conflicting_limits_in_one_worker():
    async def exercise():
        environment_semaphore(2)
        with pytest.raises(RuntimeError, match='multiple environment concurrency limits'):
            environment_semaphore(3)

    asyncio.run(exercise())
