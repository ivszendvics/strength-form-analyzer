"""Exercise registry: maps a CLI/config exercise name to its implementation.

Adding a new exercise means adding a class here and registering it below,
plus a matching ``configs/<name>.yaml``.
"""

from __future__ import annotations

from src.exercises.base import Exercise
from src.exercises.bench_press import BenchPressExercise
from src.exercises.deadlift import DeadliftExercise
from src.exercises.lunge import LungeExercise
from src.exercises.squat import SquatExercise

EXERCISE_REGISTRY: dict[str, type[Exercise]] = {
    "squat": SquatExercise,
    "deadlift": DeadliftExercise,
    "lunge": LungeExercise,
    "bench_press": BenchPressExercise,
}


def get_exercise(name: str) -> Exercise:
    try:
        exercise_cls = EXERCISE_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(EXERCISE_REGISTRY))
        raise ValueError(f"Unsupported exercise '{name}'. Available exercises: {available}") from exc
    return exercise_cls()
