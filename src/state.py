"""Trade state machine for the options agent."""

import logging
from enum import Enum, auto

logger = logging.getLogger(__name__)


class AgentState(Enum):
    IDLE = auto()             # Waiting for a signal
    SIGNAL_DETECTED = auto()  # Cross signal seen, evaluating entry
    ENTERING = auto()         # Order submitted, awaiting fill
    POSITION_OPEN = auto()    # Holding an open option position
    CLOSING = auto()          # Exit order submitted, awaiting fill
    HALTED = auto()           # Daily loss limit hit or other halt condition


_VALID_TRANSITIONS = {
    AgentState.IDLE: {AgentState.SIGNAL_DETECTED, AgentState.HALTED},
    AgentState.SIGNAL_DETECTED: {AgentState.ENTERING, AgentState.IDLE, AgentState.HALTED},
    AgentState.ENTERING: {AgentState.POSITION_OPEN, AgentState.IDLE, AgentState.HALTED},
    AgentState.POSITION_OPEN: {AgentState.CLOSING, AgentState.HALTED},
    AgentState.CLOSING: {AgentState.IDLE, AgentState.HALTED},
    AgentState.HALTED: {AgentState.IDLE},  # Manual reset only
}


class StateMachine:
    """Tracks the lifecycle of a single trade opportunity."""

    def __init__(self):
        self._state = AgentState.IDLE

    @property
    def state(self) -> AgentState:
        return self._state

    def transition(self, new_state: AgentState) -> None:
        allowed = _VALID_TRANSITIONS.get(self._state, set())
        if new_state not in allowed:
            raise ValueError(
                f"Invalid transition: {self._state.name} → {new_state.name}. "
                f"Allowed: {[s.name for s in allowed]}"
            )
        logger.info("State transition: %s → %s", self._state.name, new_state.name)
        self._state = new_state

    def reset(self) -> None:
        """Force-reset to IDLE (use after order fills or errors)."""
        logger.info("State machine reset to IDLE from %s", self._state.name)
        self._state = AgentState.IDLE

    def is_idle(self) -> bool:
        return self._state == AgentState.IDLE

    def is_halted(self) -> bool:
        return self._state == AgentState.HALTED
