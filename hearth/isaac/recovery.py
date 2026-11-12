"""Failure-recovery controller for Project HEARTH (objective 5).

The brief's objective 5:

    "Recovers from 80% of injected failures (dropped object, blocked path,
     door closes) without a reset."

This module implements the recovery logic as a pure, deterministic state
machine so it is testable without Isaac Sim.  It observes a :class:`Situation`
(a physics-free snapshot of what just happened), classifies the failure, picks
a recovery strategy, and returns the ordered sub-goals that strategy emits.
A "reset" means giving up and asking for a fresh episode; a recovery must
succeed without one, which this controller enforces (``needs_reset`` is False
for every handled failure).

Three injected-failure families are handled, matching the brief verbatim:

    dropped_object   the grasped object is no longer in the gripper
    blocked_path     the navigation target is unreachable / occluded
    door_closes      an articulated object moved back to closed after opening

plus a generic fall / balance loss.  Each strategy re-enters the main plan at
the right point (re-grasp where it dropped, replan navigation, re-open the
door) rather than restarting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from hearth.isaac.control import Subgoal, SubgoalType


class FailureType(str, Enum):
    """The injected failure kinds the controller knows how to recover from."""

    DROPPED_OBJECT = "dropped_object"
    BLOCKED_PATH = "blocked_path"
    DOOR_CLOSES = "door_closes"
    BALANCE_LOSS = "balance_loss"
    UNKNOWN = "unknown"


class RecoveryOutcome(str, Enum):
    """Result of attempting to recover from one failure."""

    RECOVERED = "recovered"  # strategy emitted, no reset needed
    NO_ACTION = "no_action"  # nothing was wrong
    UNHANDLED = "unhandled"  # could not recover -> needs reset


@dataclass(frozen=True)
class Situation:
    """A physics-free snapshot handed to the recovery controller.

    Attributes:
        failure: A raw failure tag (matched case-insensitively to a
            :class:`FailureType`), or None if nothing failed.
        object_name: The object involved (for a drop, what was held).
        room: The room the robot is in.
        target_object: The articulated object (for a door-close event).
        blocked_target: The navigation target that could not be reached.
        attempt: How many times this same failure has already occurred,
            used to escalate / avoid infinite loops.
    """

    failure: Optional[str] = None
    object_name: Optional[str] = None
    room: str = "robot"
    target_object: Optional[str] = None
    blocked_target: Optional[str] = None
    attempt: int = 0


@dataclass
class RecoveryResult:
    """What the controller decided to do about one failure."""

    failure: FailureType
    outcome: RecoveryOutcome
    subgoals: list[Subgoal] = field(default_factory=list)
    needs_reset: bool = False

    @property
    def recovered(self) -> bool:
        return self.outcome == RecoveryOutcome.RECOVERED

    def to_dict(self) -> dict[str, object]:
        return {
            "failure": self.failure.value,
            "outcome": self.outcome.value,
            "needs_reset": self.needs_reset,
            "num_subgoals": len(self.subgoals),
            "subgoals": [s.to_dict() for s in self.subgoals],
        }


# Escalation ceiling: after this many failed attempts on the same failure the
# controller gives up and asks for a reset (prevents endless recovery loops,
# matching real RL recovery controllers that abandon after N tries).
_MAX_ATTEMPTS = 3


def classify_failure(situation: Situation) -> FailureType:
    """Map a raw failure tag onto a :class:`FailureType`."""
    tag = (situation.failure or "").strip().lower()
    if not tag:
        return FailureType.UNKNOWN
    for ft in FailureType:
        if ft.value == tag:
            return ft
    # Loose matches so a simulator can report "dropped" or "object_dropped".
    if "drop" in tag:
        return FailureType.DROPPED_OBJECT
    if "block" in tag or "unreach" in tag or "stuck" in tag:
        return FailureType.BLOCKED_PATH
    if "door" in tag or "close" in tag:
        return FailureType.DOOR_CLOSES
    if "fall" in tag or "fell" in tag or "balance" in tag or "tip" in tag:
        return FailureType.BALANCE_LOSS
    return FailureType.UNKNOWN


def _subgoal(st: SubgoalType, payload: str, frame: str = "robot") -> Subgoal:
    return Subgoal(type=st, payload=payload, frame=frame)


def recover(situation: Situation) -> RecoveryResult:
    """Decide the recovery for one observed situation.

    Returns a :class:`RecoveryResult`.  For every handled failure ``needs_reset``
    is False (the brief: recover *without* a reset); only an unknown failure or
    one that has exhausted its attempts sets ``needs_reset`` True.
    """
    failure = classify_failure(situation)

    if situation.failure is None:
        return RecoveryResult(failure=failure, outcome=RecoveryOutcome.NO_ACTION)

    if failure == FailureType.UNKNOWN:
        return RecoveryResult(
            failure=failure,
            outcome=RecoveryOutcome.UNHANDLED,
            needs_reset=True,
        )

    if situation.attempt >= _MAX_ATTEMPTS:
        # Gave up after escalating: the only case that needs a reset.
        return RecoveryResult(
            failure=failure,
            outcome=RecoveryOutcome.UNHANDLED,
            needs_reset=True,
        )

    obj = situation.object_name or "object"

    if failure == FailureType.DROPPED_OBJECT:
        # Look for the dropped object, settle base, re-grasp where it fell,
        # resume the carry. Re-enter the main plan at the grasp step.
        subgoals = [
            _subgoal(SubgoalType.SEARCH, f"find-dropped:{obj}"),
            _subgoal(SubgoalType.STABILIZE, "settle-after-drop"),
            _subgoal(SubgoalType.GRASP, f"re-grasp:{obj}"),
            _subgoal(SubgoalType.CARRY, f"resume-carry:{obj}"),
        ]
        return RecoveryResult(failure=failure, outcome=RecoveryOutcome.RECOVERED, subgoals=subgoals)

    if failure == FailureType.BLOCKED_PATH:
        # Stop, look around, pick an alternate route, retry navigation.
        dest = situation.blocked_target or "target"
        subgoals = [
            _subgoal(SubgoalType.RETREAT, "back-away-from-block"),
            _subgoal(SubgoalType.STABILIZE, "scan-for-alternate-path"),
            _subgoal(SubgoalType.NAVIGATE, f"alt-route:{dest}", frame="world"),
        ]
        return RecoveryResult(failure=failure, outcome=RecoveryOutcome.RECOVERED, subgoals=subgoals)

    if failure == FailureType.DOOR_CLOSES:
        # The door swung shut: re-approach the handle, re-open, resume.
        target = situation.target_object or "door"
        subgoals = [
            _subgoal(SubgoalType.NAVIGATE, f"approach:{target}", frame="world"),
            _subgoal(SubgoalType.GRASP, f"{target}-handle"),
            _subgoal(SubgoalType.OPEN_ARTICULATED, f"{target}:reopen"),
        ]
        return RecoveryResult(failure=failure, outcome=RecoveryOutcome.RECOVERED, subgoals=subgoals)

    # BALANCE_LOSS
    subgoals = [
        _subgoal(SubgoalType.STABILIZE, "catch-balance"),
        _subgoal(SubgoalType.RETREAT, "reset-stance"),
        _subgoal(SubgoalType.STABILIZE, "confirm-stable"),
    ]
    return RecoveryResult(failure=failure, outcome=RecoveryOutcome.RECOVERED, subgoals=subgoals)
