"""Parallel execution utilities for DeepSearch reasoning."""
import asyncio
from typing import AsyncIterator, Iterable, Tuple


class ParallelExecutionManager:
    """Schedule plan sub-agents with a configurable concurrency limit."""

    def __init__(self, *, max_concurrency: int = 1):
        self._max_concurrency = max(1, int(max_concurrency))

    async def run(self, agents: Iterable["PlanSubAgent"]) -> AsyncIterator[Tuple[int, "SubAgentOutcome"]]:
        """Yield (index, outcome) pairs as soon as sub-agents finish."""

        from .subagent import PlanSubAgent, SubAgentOutcome  # Local import to avoid cycles

        agent_list = list(agents)
        if not agent_list:
            return

        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def _runner(agent: PlanSubAgent) -> Tuple[int, SubAgentOutcome]:
            async with semaphore:
                outcome = await agent.run()
            return agent.step_index, outcome

        tasks = [asyncio.create_task(_runner(agent)) for agent in agent_list]
        try:
            for task in asyncio.as_completed(tasks):
                yield await task
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
