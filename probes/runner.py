# -*- coding: utf-8 -*-
"""阶段 3：双臂运行器（基线 vs 1~2 轮澄清）。

用法：
  python probes/runner.py --mode mock            # 离线冒烟（不依赖环境/API）
  python probes/runner.py --mode real --limit 5  # 真实运行（需 ShopSimulator 服务 + LLM API）

设计：
  - EnvAdapter 抽象 reset/step，真实模式包装参考项目的 ShopAgentEnv；
  - LLM 抽象 complete(messages, tools)；澄清臂多一个 ask_user 工具，由 harness
    拦截后调用 user_simulator 回答（不进入环境）。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from user_simulator import answer_question  # noqa: E402

MAX_STEPS = 35
MAX_CLARIFY = 2

ASK_USER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "ask_user",
        "description": "向用户提出一个澄清问题以获取需求中缺失的信息（如预算、品牌、颜色、规格）。最多使用 2 次。",
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string", "description": "要问用户的中文问题"}},
            "required": ["question"],
            "additionalProperties": False,
        },
    },
}


class EnvAdapter:
    """reset(task_id)->observation_text ; step(action)->(observation_text, done)。"""

    def reset(self, task_id):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def release(self):
        pass

    def last_reward(self):
        """终端 reward（真实环境提供；mock 返回 None）。"""
        return None


class FakeEnv(EnvAdapter):
    """离线冒烟用：返回最小合法 observation，让 MockLLM 能走完整循环。"""

    def __init__(self):
        self.page = "search_home"
        self.done = False

    def _obs(self):
        footer = "搜索功能是否可用: True\n可点击的按钮: [\"search\"]"
        if self.page == "search_home":
            return f"[SHOPPING_OBSERVATION_V2]\npage_type: search_home\n{footer}"
        return f"[SHOPPING_OBSERVATION_V2]\npage_type: search_results\n{footer}"

    def reset(self, task_id):
        self.page = "search_home"
        self.done = False
        return self._obs()

    def step(self, action):
        if action.startswith("search["):
            self.page = "search_results"
        elif "Buy Now" in action or action.startswith("click[Buy Now]"):
            self.done = True
        elif action.startswith("click["):
            self.page = "product_detail"
        elif action.startswith("finish"):
            self.done = True
        return self._obs(), self.done


class RealEnv(EnvAdapter):
    """包装参考项目 ShopAgentEnv；渲染 observation 供模型读取。"""

    def __init__(self, base_url="http://127.0.0.1:5700", timeout=60):
        from shopping_grpo.environment.client import ShopAgentEnv
        from shopping_grpo.environment.observation import render_structured_observation

        self._client = ShopAgentEnv(base_url=base_url, timeout=timeout)
        self._render = render_structured_observation
        self._last = None

    def reset(self, task_id):
        result = self._client.reset(task_id)
        state = result.get("observation_state")
        if state is not None:
            return self._render(state)
        # 兜底：直接返回原始 result 的文本表示
        return json.dumps(result, ensure_ascii=False)

    def step(self, action):
        result = self._client.step(action)
        self._last = result
        state = result.get("observation_state")
        obs = self._render(state) if state is not None else json.dumps(result, ensure_ascii=False)
        return obs, bool(result.get("done", False))

    def release(self):
        self._client.release()

    def last_reward(self):
        if not self._last:
            return None
        return self._last.get("reward")


class BaseLLM:
    """LLM 抽象：complete(messages, tools) -> list[tool_call_dict] 或 None。"""

    def complete(self, messages, tools):
        raise NotImplementedError


class MockLLM(BaseLLM):
    """离线冒烟策略：ask(澄清臂) -> search -> buy，带状态计数推进。"""

    def __init__(self, arm):
        self.arm = arm
        self._n = 0

    def complete(self, messages, tools):
        self._n += 1
        tool_names = [t["function"]["name"] for t in tools]
        if self.arm == "clarify" and self._n == 1 and "ask_user" in tool_names:
            return [{"id": "ask_1", "type": "function",
                     "function": {"name": "ask_user", "arguments": json.dumps({"question": "请问您的预算大概多少？"}, ensure_ascii=False)}}]
        if "search_products" in tool_names and self._n <= 2:
            return [{"id": "s1", "type": "function",
                     "function": {"name": "search_products", "arguments": json.dumps({"query": "默认查询"}, ensure_ascii=False)}}]
        if "buy_now" in tool_names:
            return [{"id": "b1", "type": "function",
                     "function": {"name": "buy_now", "arguments": "{}"}}]
        return None


class RealLLM(BaseLLM):
    """OpenAI 兼容 chat completions 客户端。"""

    def __init__(self, model=None, base_url=None, api_key=None, temperature=0.7):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key or None, base_url=base_url or None)
        self.model = model or "deepseek-chat"
        self.temperature = temperature

    def complete(self, messages, tools):
        kwargs = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            kwargs["tools"] = tools
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        if msg.tool_calls:
            return [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        return None


def tool_schemas_for(arm):
    """基线臂只有购物工具；澄清臂额外加 ask_user。"""
    from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS

    schemas = [json.loads(json.dumps(s)) for s in SHOP_TOOL_SCHEMAS]
    if arm == "clarify":
        schemas.append(json.loads(json.dumps(ASK_USER_SCHEMA)))
    return schemas


def convert_tool_call(tc):
    from shopping_grpo.environment.tools import tool_call_to_action

    fn = tc["function"]
    try:
        args = json.loads(fn.get("arguments", "{}") or "{}")
    except json.JSONDecodeError:
        args = {}
    return fn["name"], args, tool_call_to_action(fn["name"], args)


def run_arm(task, arm, llm, env, out_path):
    """跑一条轨迹并落盘。返回 record dict。"""
    record = {
        "task_id": task["task_id"],
        "arm": arm,
        "query": task["under_query"],
        "clarify_turns": 0,
        "steps": [],
        "terminal": {"done": False, "reward": None},
    }
    tools = tool_schemas_for(arm)
    obs = env.reset(task["task_id"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT[arm]},
        {"role": "user", "content": "Instruction: " + task["under_query"]},
        {"role": "assistant", "content": "（当前页面）\n" + obs},
    ]

    for step_i in range(MAX_STEPS):
        try:
            tool_calls = llm.complete(messages, tools)
        except Exception as exc:  # noqa: BLE001
            record["terminal"]["error"] = str(exc)
            break
        if not tool_calls:
            record["terminal"]["no_tool_call"] = True
            break

        tc = tool_calls[0]
        name = tc["function"]["name"]
        if name == "ask_user":
            if record["clarify_turns"] >= MAX_CLARIFY:
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": "不能再追问了，请基于现有信息继续购物。"})
                continue
            record["clarify_turns"] += 1
            args = json.loads(tc["function"].get("arguments", "{}") or "{}")
            question = args.get("question", "")
            answer, _info = answer_question(question, task.get("fake_profile", {}))
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": answer})
            record["steps"].append({"step": step_i, "kind": "ask", "question": question, "answer": answer})
            continue

        try:
            action = convert_tool_call(tc)[2]
        except Exception as exc:  # noqa: BLE001
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": f"工具调用无法转换（{exc}），请换一个合法动作。"})
            record["steps"].append({"step": step_i, "kind": "invalid", "detail": str(exc)})
            continue

        obs, done = env.step(action)
        messages.append({"role": "assistant", "content": f"{name}({tc['function'].get('arguments')})"})
        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": obs})
        record["steps"].append({"step": step_i, "kind": "action", "action": action})
        if done:
            record["terminal"]["done"] = True
            record["terminal"]["reward"] = env.last_reward()
            break

    env.release()
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


SYSTEM_PROMPT = {
    "baseline": (
        "你是购物 Agent，负责在 ShopSimulator 中替用户完成一次购物任务。"
        "用户需求只会在开始时给出一句。不得向用户追问、确认；只调用提供的工具与商店交互。"
        "目标是找到整体最符合需求的商品并购买；经过有效探索仍无法完成时合理结束。"
    ),
    "clarify": (
        "你是购物 Agent，负责在 ShopSimulator 中替用户完成一次购物任务。"
        "用户需求可能不完整。如果需求存在明显缺失（如预算、品牌、颜色、规格不明），"
        "你可以先调用 ask_user 向用户提 1~2 个澄清问题，再根据回答继续搜索与购买。"
        "澄清后应基于回答完成购物；不要过度追问。"
    ),
}

def main():
    import argparse

    parser = argparse.ArgumentParser(description="B 探针双臂运行器")
    parser.add_argument("--mode", choices=["mock", "real"], default="mock")
    parser.add_argument("--tasks", default=str(Path(__file__).resolve().parent / "data" / "tasks.json"))
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 个任务（冒烟用）")
    parser.add_argument("--arm", choices=["both", "baseline", "clarify"], default="both")
    parser.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "outputs"))
    args = parser.parse_args()

    tasks = json.loads(open(args.tasks, "r", encoding="utf-8").read())
    if args.limit:
        tasks = tasks[: args.limit]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    arms = ["baseline", "clarify"] if args.arm == "both" else [args.arm]

    if args.mode == "mock":
        env = FakeEnv()
        llm_maker = lambda arm: MockLLM(arm)  # noqa: E731
    else:
        env = RealEnv()
        llm_maker = lambda arm: RealLLM()  # noqa: E731

    started = time.time()
    for task in tasks:
        for arm in arms:
            out_path = outdir / f"trajectories_{arm}.jsonl"
            rec = run_arm(task, arm, llm_maker(arm), env, out_path)
            print(f"[{arm}] task={task['task_id']} steps={len(rec['steps'])} "
                  f"clarify={rec['clarify_turns']} done={rec['terminal']['done']}")
    print(f"done in {time.time() - started:.1f}s -> {outdir}")


if __name__ == "__main__":
    main()
