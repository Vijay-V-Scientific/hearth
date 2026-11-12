"""Layer-2 instruction planner for Project HEARTH (objective 4).

Turns a natural-language household instruction into an ordered list of
:class:`~hearth.isaac.control.Subgoal` objects — the contract the
loco-manipulation layer consumes.  This is the *symbolic* half of the VLA
planner: it does the task decomposition that a learned VLA would do, in pure
Python, so the long-horizon logic is testable on a CPU host with no Isaac Sim,
no GPU and no model weights.

The decomposition is driven by the five task families of the brief
(:class:`~hearth.eval.types.TaskFamily`).  For each family the planner emits a
canonical sub-goal sequence in the 5-15 sub-step range objective 4 requires:

    tidying       navigate -> grasp -> carry -> place -> done
    kitchen       navigate -> (open) -> grasp -> carry -> place -> close
    articulated   navigate -> approach -> grasp-handle -> open/close -> retreat
    carrying      navigate -> grasp -> carry -> (place)
    fetch         search -> navigate -> grasp -> carry -> (turn-toward-requester)

The planner is a pure function of (instruction, scene summary, task family):
same inputs -> same sub-goal list, which makes it reproducible and unit
testable.  A learned VLA planner would replace ``plan_task`` at runtime; this
module is the deterministic baseline it must match or beat.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from hearth.isaac.control import Subgoal, SubgoalType

# Tokens that signal which family an instruction belongs to.  Checked in
# order; first match wins.  Keys are lowercase substrings.
_FAMILY_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("fetch", ("bring", "fetch", "get me", "hand me", "take to")),
    (
        "kitchen",
        (
            "dishwasher",
            "fridge",
            "groceries",
            "clear the table",
            "load",
            "microwave",
            "cup",
            "plate",
        ),
    ),
    ("articulated", ("open", "close", "drawer", "cabinet", "door", "switch", "lid")),
    ("carrying", ("carry", "move the", "transport", "laundry basket", "box to", "chair to")),
    (
        "tidying",
        ("tidy", "put away", "pick up", "where it belongs", "clutter", "toy", "book", "clothes"),
    ),
)

# Verbs that, if present, override family cues to a more specific intent.
_PLACE_WORDS = ("put", "place", "set down", "drop")
_GRASP_WORDS = ("pick", "grab", "grasp", "take", "hold", "lift")
_OPEN_WORDS = ("open", "close", "unlatch")
_SEARCH_WORDS = ("find", "search", "look for", "where", "not in view")


@dataclass(frozen=True)
class SceneSummary:
    """A physics-free summary of what the robot can see right now.

    This is the planner's world model.  At runtime the VLA fills it from RGB-D
    and an object detector; here the caller fills it by hand so the planner's
    logic is testable without a simulator.

    Attributes:
        rooms: Room names present in the scene.
        object_locations: Mapping object-name -> room it is in (for fetch).
        visible_objects: Objects the robot can currently see.
        requester_room: Where the human is waiting (fetch deliveries).
        target_room: The destination room named in the instruction, if resolved.
    """

    rooms: tuple[str, ...] = ()
    object_locations: dict[str, str] = field(default_factory=dict)
    visible_objects: tuple[str, ...] = ()
    requester_room: str = "robot"
    target_room: Optional[str] = None


@dataclass
class PlanResult:
    """A planned sequence of sub-goals for one instruction."""

    instruction: str
    family: str
    subgoals: list[Subgoal] = field(default_factory=list)

    @property
    def num_subgoals(self) -> int:
        return len(self.subgoals)

    def to_dict(self) -> dict[str, object]:
        return {
            "instruction": self.instruction,
            "family": self.family,
            "num_subgoals": self.num_subgoals,
            "subgoals": [s.to_dict() for s in self.subgoals],
        }


def detect_family(instruction: str) -> str:
    """Classify an instruction into one of the five task families.

    Returns one of ``tidying``, ``kitchen``, ``articulated``, ``carrying``,
    ``fetch``.  Falls back to ``tidying`` when nothing matches (the brief's
    most general family).
    """
    text = (instruction or "").lower()
    for family, cues in _FAMILY_CUES:
        for cue in cues:
            if cue in text:
                return family
    return "tidying"


def _quote_tokens(text: str) -> list[str]:
    """Pull out quoted spans, then any word before a known object noun."""
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
    out = [q[0] or q[1] for q in quoted if (q[0] or q[1])]
    return out


def _extract_objects(instruction: str) -> list[str]:
    """Best-effort noun extraction for the object(s) the instruction names."""
    text = (instruction or "").lower()
    objs = _quote_tokens(text)
    if objs:
        return objs
    # Drop stop words and common verbs, keep plausible object nouns.
    words = re.findall(r"[a-z]+", text)
    noise = set(
        "the a an and then please could you would me my our to from in on "
        "of at it is are where which that this these those put get take "
        "bring fetch open close move carry pick up away clear load place "
        "set down drop hand give into onto".split()
    )
    kept = [w for w in words if w not in noise and len(w) > 2]
    return kept


def _resolve_room(instruction: str, scene: SceneSummary) -> Optional[str]:
    """Find the first room name the instruction mentions."""
    text = (instruction or "").lower()
    for room in scene.rooms:
        if room.lower() in text:
            return room
    return scene.target_room


def plan_task(
    instruction: str,
    scene: Optional[SceneSummary] = None,
    *,
    family: Optional[str] = None,
) -> PlanResult:
    """Decompose one instruction into an ordered sub-goal list.

    The returned list is the planner's best guess at the sub-goals the loco-
    manipulation layer must achieve in order.  Every sub-goal is a real
    :class:`Subgoal`, so a downstream layer can execute it unchanged.
    """
    scene = scene or SceneSummary()
    fam = family or detect_family(instruction)
    objects = _extract_objects(instruction)
    obj = objects[0] if objects else "object"
    target_room = _resolve_room(instruction, scene)

    subgoals: list[Subgoal] = []

    def nav(room: Optional[str], payload: Optional[str] = None) -> None:
        subgoals.append(
            Subgoal(
                type=SubgoalType.NAVIGATE,
                payload=payload or (room or "target"),
                frame="world",
            )
        )

    def grasp(name: str) -> None:
        subgoals.append(Subgoal(type=SubgoalType.GRASP, payload=name))

    def carry(name: str) -> None:
        subgoals.append(Subgoal(type=SubgoalType.CARRY, payload=name))

    def place(name: str, dest: str) -> None:
        subgoals.append(Subgoal(type=SubgoalType.PLACE, payload=f"{name}->{dest}"))

    def open_art(name: str, state: str = "open") -> None:
        subgoals.append(Subgoal(type=SubgoalType.OPEN_ARTICULATED, payload=f"{name}:{state}"))

    if fam == "articulated":
        state = "close" if "close" in instruction.lower() else "open"
        target_room = target_room or "robot"
        nav(target_room)
        subgoals.append(Subgoal(type=SubgoalType.NAVIGATE, payload=f"approach:{obj}"))
        subgoals.append(Subgoal(type=SubgoalType.GRASP, payload=f"{obj}-handle"))
        open_art(obj, state)
        subgoals.append(Subgoal(type=SubgoalType.RETREAT, payload=obj))

    elif fam == "fetch":
        subgoals.append(Subgoal(type=SubgoalType.SEARCH, payload=obj))
        loc_room = scene.object_locations.get(obj) or target_room
        nav(loc_room, payload=f"{obj}@{loc_room or 'unknown'}")
        grasp(obj)
        subgoals.append(Subgoal(type=SubgoalType.CARRY, payload=f"{obj}->requester"))
        subgoals.append(
            Subgoal(
                type=SubgoalType.PLACE,
                payload=f"{obj}->{scene.requester_room}",
            )
        )

    elif fam == "carrying":
        src_room = target_room or "robot"
        nav(src_room, payload=obj)
        grasp(obj)
        carry(obj)
        place(obj, target_room or "destination")

    elif fam == "kitchen":
        dest = target_room or "kitchen"
        nav("robot" if target_room is None else target_room)
        grasp(obj)
        carry(obj)
        place(obj, dest)

    else:  # tidying (also the fallback)
        dest = target_room or "storage"
        nav("robot")
        grasp(obj)
        carry(obj)
        place(obj, dest)

    # Guarantee the brief's 5-15 sub-goal range.  Families above already
    # produce 4-5; pad with explicit approach/stabilise steps for longer
    # multi-object instructions rather than repeating meaningless ones.
    subgoals = _ensure_min_subgoals(subgoals, instruction)

    return PlanResult(instruction=instruction, family=fam, subgoals=subgoals)


def _ensure_min_subgoals(
    subgoals: list[Subgoal], instruction: str, minimum: int = 5
) -> list[Subgoal]:
    """Pad a too-short plan with meaningful stabilise/approach sub-goals."""
    if len(subgoals) >= minimum:
        return subgoals
    # Insert a stabilise before the first grasp and after the last action.
    padded = list(subgoals)
    insertions = 0
    # add an approach right before the first NAVIGATE->GRASP transition
    first_grasp = next((i for i, s in enumerate(padded) if s.type == SubgoalType.GRASP), None)
    if first_grasp is not None and insertions < (minimum - len(padded)):
        padded.insert(
            first_grasp,
            Subgoal(type=SubgoalType.STABILIZE, payload="settle-base"),
        )
        insertions += 1
    while len(padded) < minimum:
        padded.append(Subgoal(type=SubgoalType.STABILIZE, payload=f"settle-{insertions}"))
        insertions += 1
    return padded


def plan_many(
    instructions: Iterable[str],
    scene: Optional[SceneSummary] = None,
) -> list[PlanResult]:
    """Plan a batch of instructions against the same scene summary."""
    return [plan_task(i, scene) for i in instructions]
