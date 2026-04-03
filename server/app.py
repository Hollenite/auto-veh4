"""FastAPI application entrypoint for the traffic-control environment."""

import os
from typing import Any, Dict, List

try:
    from openenv.core.env_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv-core is required to run the traffic-control environment server."
    ) from exc

try:
    from ..models import TrafficControlAction, TrafficControlObservation
    from ..task_bank import list_tasks
    from .traffic_control_environment import TrafficControlEnvironment
except ImportError:
    from models import TrafficControlAction, TrafficControlObservation
    from task_bank import list_tasks
    from server.traffic_control_environment import TrafficControlEnvironment

# Make the local debugging UI available by default during development.
os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")


app = create_app(
    TrafficControlEnvironment,
    TrafficControlAction,
    TrafficControlObservation,
    env_name="traffic_control_env",
    max_concurrent_envs=1,
)


@app.get("/tasks")
def get_tasks() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "tasks": [
            {
                "task_id": task.task_id.value,
                "name": task.name,
                "description": task.description,
                "horizon_steps": task.horizon_steps,
            }
            for task in list_tasks()
        ]
    }


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


def main() -> None:
    run_server()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
