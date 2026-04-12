import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from Modified.environment import AnomalyAction  # <-- your env

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

TASK_NAME = "anomaly-detection"
BENCHMARK = "anomaly_env"

MAX_STEPS = 5
TEMPERATURE = 0.3
MAX_TOKENS = 120
SUCCESS_SCORE_THRESHOLD = 0.6


SYSTEM_PROMPT = """
You are analyzing images to detect if something is unusual or concerning.

Given image metadata, decide:
- label: normal or anomaly
- confidence: 0.0 to 1.0
- reason: short explanation

Respond STRICTLY in this format:
label=<normal|anomaly>; confidence=<float>; reason=<text>
"""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, observation, last_reward: float) -> str:
    return textwrap.dedent(f"""
    Step: {step}
    Image path: {observation.image_path}
    Context: {observation.context}
    Previous reward: {last_reward:.2f}

    Decide if this image is normal or anomalous.
    """).strip()


def parse_response(text: str):
    try:
        parts = dict(item.split("=") for item in text.split(";"))
        return {
            "label": parts.get("label", "normal").strip(),
            "confidence": float(parts.get("confidence", 0.5)),
            "reason": parts.get("reason", "").strip()
        }
    except Exception:
        return {"label": "normal", "confidence": 0.5, "reason": "fallback"}


def get_model_action(client: OpenAI, step: int, observation, last_reward: float):
    user_prompt = build_user_prompt(step, observation, last_reward)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        return parse_response(text), text
    except Exception:
        return {"label": "normal", "confidence": 0.5, "reason": "error"}, "error"


async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = await AnomalyEnv1.from_docker_image(IMAGE_NAME)

    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    try:
        result = await env.reset()
        observation = result.observation
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_dict, raw_text = get_model_action(client, step, observation, last_reward)

            action = AnomalyAction(**action_dict)

            result = await env.step(action)

            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps_taken = step
            observation = result.observation
            last_reward = reward

            log_step(step, raw_text, reward, done, None)

            if done:
                break

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        await env.close()
        log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    asyncio.run(main())