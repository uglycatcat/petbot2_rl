"""Future backend interfaces. This file deliberately provides no fake simulator."""

from dataclasses import dataclass
from typing import Protocol

from .control import ActuatorTargets
from .evaluation import EvaluationSample
from .observations import MotionEstimate, Proprioception, TerrainPacket


@dataclass(frozen=True)
class SensorSample:
    dt_s: float
    specific_force_body_m_s2: tuple[float, float, float]
    proprioception: Proprioception


class MotionEstimator(Protocol):
    def reset(self) -> None:
        """Initial stationary pose; calibrate origin and sensor offsets."""
        ...

    def update(self, sample: SensorSample) -> MotionEstimate:
        """Integrate own sensor data; no ground-truth pose or terrain refresh."""
        ...


class SimulationBackend(Protocol):
    def reset(self) -> tuple[TerrainPacket, SensorSample]:
        """Create source-default pose; place terrain relative to actual initial whole-body COM."""
        ...

    def step(self, targets: ActuatorTargets) -> tuple[SensorSample, EvaluationSample]:
        """Separate sensors from privileged reward/evaluation evidence.

        Accumulate collision events at physical substeps, not just policy ticks.
        Target adapters honor source effort limits and hold head/ears at home.
        """
        ...
