# Anomaly Detection Environment (OpenEnv Compatible)

from uuid import uuid4
from typing import List, Dict, Tuple

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import AnomalyAction, AnomalyObservation
except ImportError:
    from models import AnomalyAction, AnomalyObservation


class AnomalyDetectionEnvironment(Environment):
    """
    Environment for detecting anomalies in images.

    Agent must:
    - classify image as normal/anomaly
    - assign severity (for harder tasks)
    - explain reasoning

    Reward is based on correctness + explanation quality.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

        # 🔥 Inline tasks (you can move to JSON later)
        self._tasks: List[Tuple[str, Dict]] = [
            # EASY
            ("easy", {
                "image_path": "clean_room.jpg",
                "description": "A clean and organized room",
                "label": "normal"
            }),
            ("easy", {
                "image_path": "broken_window.jpg",
                "description": "A shattered window with glass pieces",
                "label": "anomaly"
            }),

            # MEDIUM
            ("medium", {
                "image_path": "person_fallen.jpg",
                "description": "A person lying on the ground",
                "label": "anomaly",
                "keywords": ["person", "fallen", "ground"]
            }),

            # HARD
            ("hard", {
                "image_path": "fire.jpg",
                "description": "A building with visible fire and smoke",
                "label": "anomaly",
                "severity": "high",
                "keywords": ["fire", "smoke", "danger"]
            }),
        ]

        self._current_idx = 0

    def reset(self) -> AnomalyObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        self._current_idx = 0

        return self._build_observation(
            message="Environment ready. Analyze the image.",
            reward=0.0,
            done=False
        )

    def step(self, action: AnomalyAction) -> AnomalyObservation:  # type: ignore[override]
        self._state.step_count += 1

        difficulty, task = self._tasks[self._current_idx]

        # 🔥 Compute reward
        reward = self._grade(action, task, difficulty)

        # Move to next task
        self._current_idx += 1
        done = self._current_idx >= len(self._tasks)

        return self._build_observation(
            message="Processed",
            reward=reward,
            done=done,
            metadata={
                "difficulty": difficulty,
                "step": self._state.step_count,
                "ground_truth": task
            }
        )

    def _build_observation(self, message, reward, done, metadata=None):
        if self._current_idx < len(self._tasks):
            _, task = self._tasks[self._current_idx]
            image_path = task["image_path"]
            description = task.get("description")
        else:
            image_path = None
            description = None

        return AnomalyObservation(
            message=message,
            image_path=image_path,
            description=description,
            reward=reward,
            done=done,
            metadata=metadata or {}
        )

    # 🔥 Grading logic (core of your assignment)
    def _grade(self, action: AnomalyAction, task: Dict, difficulty: str) -> float:
        label_correct = action.label == task.get("label")
        severity_correct = action.severity == task.get("severity")

        explanation = (action.explanation or "").lower()
        keywords = task.get("keywords", [])

        keyword_hits = sum(1 for kw in keywords if kw in explanation)
        explanation_score = (keyword_hits / len(keywords)) if keywords else 0.0

        if difficulty == "easy":
            return 1.0 if label_correct else 0.0

        elif difficulty == "medium":
            return 0.6 * float(label_correct) + 0.4 * explanation_score

        elif difficulty == "hard":
            return (
                0.4 * float(label_correct) +
                0.3 * float(severity_correct) +
                0.3 * explanation_score
            )

        return 0.0

    @property
    def state(self) -> State:
        return self._state