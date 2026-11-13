"""Multi-agent debate (Du et al., 2023), Game of 24 only.

`agents` copies of the model answer the task independently (round 1, the baseline prompt, each
with its own seed). In every later round each agent keeps its own conversation: it sees its
earlier answer and the latest answers of all other agents, and gives an updated answer. After
`rounds` rounds the final answer is the majority vote over the agents' last answers (the same
vote as self-consistency; ties go to the first agent). No verifier is used. The defaults, 3
agents and 2 rounds, are the main setting of the paper.

Part of the feedback study (separate from the main results): a paper-like multi-agent setup to
set next to our planner / solver / critic Multi-agent approach.
"""

from collections.abc import Iterator
from typing import Any

from agentic.approaches.base import Caller, load_prompt
from agentic.approaches.self_consistency import vote
from agentic.events import AgentMessage, Body, FinalAnswer
from agentic.llm.base import LLMClient
from agentic.tasks.base import Item, Task

DEFAULTS = {"agents": 3, "rounds": 2}


class Debate:
    name = "debate"

    def run(self, item: Item, task: Task, llm: LLMClient, *, seed: int,
            config: dict[str, Any]) -> Iterator[Body]:
        if task.name != "game24":
            raise NotImplementedError("Debate is built for Game of 24 only")
        settings = {**DEFAULTS, **config}
        numbers = " ".join(str(n) for n in item.payload["numbers"])
        caller = Caller(llm, seed)
        chats = [[{"role": "user", "content": task.build_prompt(item)}]
                 for _ in range(settings["agents"])]
        latest: list[str] = []

        for round_no in range(1, settings["rounds"] + 1):
            if round_no > 1:
                for i, chat in enumerate(chats):
                    others = "\n\n".join(f"Agent {j + 1}: {text}"
                                         for j, text in enumerate(latest) if j != i)
                    chat.append({"role": "user", "content": load_prompt(
                        "debate_update_game24").format(others=others, numbers=numbers)})
            replies = []
            for i, chat in enumerate(chats):
                call = caller.chat(f"agent {i + 1}", list(chat))
                yield call
                yield AgentMessage(round=round_no, role=f"agent {i + 1}", text=call.response,
                                   call_id=call.call_id)
                replies.append(call.response)
            for chat, reply in zip(chats, replies, strict=True):
                chat.append({"role": "assistant", "content": reply})
            latest = replies

        answers = [task.extract_answer(item, text) for text in latest]
        winner = vote(answers)
        yield FinalAnswer(answer="" if winner is None else answers[winner])
