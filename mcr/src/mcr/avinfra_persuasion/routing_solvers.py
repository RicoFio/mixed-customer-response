from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from .datastructures import Node, Scenario, World
from .mosp import DEFAULT_SCALE, solve_mosp_routes
from .opt import (
    RoutingSolution,
    build_benpy_model_sample_average,
    build_turn_state_benpy_model_sample_average,
)


RoutingBackend: TypeAlias = Literal["benpy", "mosp"]


@dataclass(frozen=True)
class RoutingSolverConfig:
    backend: RoutingBackend = "benpy"
    use_turn_state: bool = True
    mosp_scale: float = DEFAULT_SCALE
    benpy_options: Mapping[str, Any] | None = None


RoutingSolverConfigLike: TypeAlias = RoutingSolverConfig | RoutingBackend | None


def solve_routes(
    *,
    world: World,
    source: Node,
    target: Node,
    scenarios: Sequence[Scenario],
    config: RoutingSolverConfigLike = None,
    use_average: bool = True,
) -> RoutingSolution:
    """Solve one routing query using the selected backend."""
    solver_config = coerce_routing_solver_config(config)
    realized_scenarios = tuple(scenarios)

    if solver_config.backend == "benpy":
        return _solve_routes_benpy(
            world=world,
            source=source,
            target=target,
            scenarios=realized_scenarios,
            config=solver_config,
            use_average=use_average,
        )

    if solver_config.backend == "mosp":
        if not solver_config.use_turn_state:
            raise ValueError("MOSP routing currently requires use_turn_state=True.")
        return solve_mosp_routes(
            world=world,
            source=source,
            target=target,
            scenarios=realized_scenarios,
            use_average=use_average,
            scale=solver_config.mosp_scale,
        )

    raise ValueError(f"Unknown routing backend: {solver_config.backend!r}.")


def coerce_routing_solver_config(
    config: RoutingSolverConfigLike,
) -> RoutingSolverConfig:
    if config is None:
        return RoutingSolverConfig()
    if isinstance(config, RoutingSolverConfig):
        return config
    if isinstance(config, str) and config in {"benpy", "mosp"}:
        return RoutingSolverConfig(backend=config)
    raise ValueError(f"Unknown routing solver config: {config!r}.")


def _solve_routes_benpy(
    *,
    world: World,
    source: Node,
    target: Node,
    scenarios: Sequence[Scenario],
    config: RoutingSolverConfig,
    use_average: bool,
) -> RoutingSolution:
    builder = (
        build_turn_state_benpy_model_sample_average
        if config.use_turn_state
        else build_benpy_model_sample_average
    )
    model = builder(
        V=world.ordered_V,
        A=world.ordered_A,
        L=world.ordered_L,
        s=source,
        t=target,
        scenarios=scenarios,
        use_average=use_average,
    )
    solver_options: dict[str, Any] = {}
    if config.benpy_options is not None:
        solver_options.update(config.benpy_options)
    solver_options["solution"] = True

    raw_solution = model.solve(options=solver_options)
    return RoutingSolution.from_benpy_solution(
        raw_solution=raw_solution,
        model=model,
    )
