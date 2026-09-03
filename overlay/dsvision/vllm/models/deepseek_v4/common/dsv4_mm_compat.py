"""Prompt-update planner backported from upstream vLLM (the PR #54566 tree) for
the older vendored vllm inside the sparkinfer image, which has no
`_plan_prompt_updates`. Self-contained: it imports only types that exist in both
generations of vllm.multimodal.
"""

from collections import deque
from collections.abc import Callable, Generator, Sequence
from typing import NamedTuple, TypeAlias

from vllm.multimodal.processing.processor import (
    MultiModalPromptUpdates,
    MultiModalPromptUpdatesApplyResult,
    PromptIndex,
    PromptTargetMatch,
    ResolvedPromptUpdate,
    UpdateMode,
    UpdateTarget,
)

class _MatchedUpdate(NamedTuple):
    """A resolved update selected for a match in the original prompt."""

    priority: int
    """The original item order used to preserve match tie-breaking."""

    update: ResolvedPromptUpdate
    """The selected update for the multimodal item."""

    update_idx: int
    """The selected update's index within the item's alternatives."""

    match: PromptTargetMatch
    """The target range in the original prompt."""


_UpdateQueue: TypeAlias = deque[tuple[int, Sequence[ResolvedPromptUpdate]]]


_QueueMatch: TypeAlias = tuple[_UpdateQueue, PromptTargetMatch, int]


def _target_key(target: UpdateTarget) -> tuple[str, object]:
    """Return a hashable key that preserves target matching semantics."""
    if isinstance(target, PromptIndex):
        return ("index", id(target))

    return ("tokens", tuple(target))


def _compile_prompt_update_queues(
    mm_prompt_updates: "MultiModalPromptUpdates",
) -> dict[
    tuple[tuple[UpdateMode, tuple[str, object]], ...],
    _UpdateQueue,
]:
    """Group items with identical match rules into ordered queues."""
    queues_by_signature = dict[
        tuple[tuple[UpdateMode, tuple[str, object]], ...],
        _UpdateQueue,
    ]()
    priority = 0

    for modality_updates in mm_prompt_updates.values():
        for updates in modality_updates:
            signature = tuple(
                (update.mode, _target_key(update.target)) for update in updates
            )
            queues_by_signature.setdefault(signature, deque()).append(
                (priority, updates)
            )
            priority += 1

    return queues_by_signature


_IterMatches: TypeAlias = Callable[
    [ResolvedPromptUpdate, int],
    Generator[PromptTargetMatch, None, None],
]


def _find_queue_match(
    queue: _UpdateQueue,
    iter_matches: _IterMatches,
    *,
    start_idx: int,
    mode: UpdateMode | None = None,
) -> tuple[PromptTargetMatch, int] | None:
    """Find the first matching alternative for the next queued item."""
    _, updates = queue[0]
    for update_idx, update in enumerate(updates):
        if mode is not None and update.mode != mode:
            continue

        match = next(iter_matches(update, start_idx), None)
        if match is not None:
            return match, update_idx

    return None


def _next_priority(queue: _UpdateQueue) -> int:
    """Return the original priority of the next queued item."""
    priority, _ = queue[0]
    return priority


def _plan_prompt_updates_with(
    mm_prompt_updates: "MultiModalPromptUpdates",
    iter_matches: _IterMatches,
) -> tuple[list[_MatchedUpdate], "MultiModalPromptUpdatesApplyResult"]:
    """Plan non-overlapping prompt updates before rendering the output."""
    queues = list(_compile_prompt_update_queues(mm_prompt_updates).values())
    result: MultiModalPromptUpdatesApplyResult = {
        modality: [None] * len(items) for modality, items in mm_prompt_updates.items()
    }
    planned_updates = list[_MatchedUpdate]()
    prev_end_idx = 0

    while queues:
        first_matches = list[_QueueMatch]()
        for queue in queues:
            queue_match = _find_queue_match(
                queue,
                iter_matches,
                start_idx=prev_end_idx,
            )
            if queue_match is not None:
                prompt_match, update_idx = queue_match
                first_matches.append((queue, prompt_match, update_idx))

        if not first_matches:
            break

        mode_queue, _, mode_update_idx = min(
            first_matches,
            key=lambda item: _next_priority(item[0]),
        )
        _, mode_updates = mode_queue[0]
        mode = mode_updates[mode_update_idx].mode

        mode_matches = list[_QueueMatch]()
        for queue, prompt_match, first_update_idx in first_matches:
            if queue[0][1][first_update_idx].mode == mode:
                mode_matches.append((queue, prompt_match, first_update_idx))
                continue

            queue_match = _find_queue_match(
                queue,
                iter_matches,
                start_idx=prev_end_idx,
                mode=mode,
            )
            if queue_match is not None:
                prompt_match, update_idx = queue_match
                mode_matches.append((queue, prompt_match, update_idx))

        updates_to_apply = list[_MatchedUpdate]()
        non_empty_replacements = list[_QueueMatch]()
        for queue, match, update_idx in mode_matches:
            if mode == UpdateMode.REPLACE and match.start_idx != match.end_idx:
                non_empty_replacements.append((queue, match, update_idx))
            else:
                while queue:
                    priority, updates = queue.popleft()
                    updates_to_apply.append(
                        _MatchedUpdate(
                            priority=priority,
                            update=updates[update_idx],
                            update_idx=update_idx,
                            match=match,
                        )
                    )

        if non_empty_replacements:
            queue, match, update_idx = min(
                non_empty_replacements,
                key=lambda item: (item[1], _next_priority(item[0])),
            )
            priority, updates = queue.popleft()
            updates_to_apply.append(
                _MatchedUpdate(
                    priority=priority,
                    update=updates[update_idx],
                    update_idx=update_idx,
                    match=match,
                )
            )

        updates_to_apply.sort(key=lambda item: (item.match, item.priority))
        for matched_update in updates_to_apply:
            update = matched_update.update
            result[update.modality][update.item_idx] = matched_update.update_idx
            prev_end_idx = matched_update.match.end_idx
        planned_updates.extend(updates_to_apply)
        queues = [queue for queue in queues if queue]

    return planned_updates, result


def _plan_prompt_updates(
    prompt: list[int],
    mm_prompt_updates: "MultiModalPromptUpdates",
) -> tuple[list[_MatchedUpdate], "MultiModalPromptUpdatesApplyResult"]:
    """Plan non-overlapping prompt updates before rendering the output."""
    return _plan_prompt_updates_with(
        mm_prompt_updates,
        lambda update, start_idx: update.iter_token_matches(
            prompt, start_idx=start_idx
        ),
    )

