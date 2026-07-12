"""Prompt merging derived from Deathspike/ComfyUI-NegPipPromptMerge (GPL-3.0)."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator


class _Group:
    def __init__(self, default_weight: float, parent: _Group | None):
        self.default_weight = default_weight
        self.parent = parent
        self.weight = default_weight

    def calculated_weight(self) -> float:
        parent_weight = self.parent.calculated_weight() if self.parent else 1.0
        return parent_weight * self.weight


class _TagBuilder:
    def __init__(self, group: _Group):
        self.groups = [group]
        self.pieces = [""]

    def append(self, character: str) -> None:
        self.pieces[-1] += character

    def enter(self, group: _Group) -> None:
        self.groups.append(group)
        self.pieces.append("")

    def build(self) -> _Tag | None:
        while self.pieces and not self.pieces[0].strip():
            self.groups.pop(0)
            self.pieces.pop(0)
        if not self.pieces:
            return None

        index = 1
        while index < len(self.pieces):
            if not self.pieces[index]:
                self.groups.pop(index)
                self.pieces.pop(index)
            elif not self.pieces[index].strip():
                self.pieces[index - 1] = self.pieces[index - 1].rstrip() + " "
                self.groups.pop(index)
                self.pieces.pop(index)
            elif self.pieces[index][0].isspace():
                self.pieces[index - 1] = self.pieces[index - 1].rstrip() + " "
                self.pieces[index] = self.pieces[index].lstrip()
                index += 1
            else:
                index += 1

        self.pieces[0] = self.pieces[0].lstrip()
        self.pieces[-1] = self.pieces[-1].rstrip()
        pieces = list(zip(self.pieces, (group.calculated_weight() for group in self.groups)))
        return _Tag("".join(self.pieces), pieces)


class _TagParser:
    def __init__(self, text: str):
        self.text = text

    def __iter__(self) -> Iterator[_Tag]:
        group = _Group(1.0, None)
        builders = [_TagBuilder(group)]
        index = 0

        while index < len(self.text):
            current = self.text[index]
            previous = self.text[index - 1] if index else None

            if current == "(" and previous != "\\":
                group = _Group(1.1, group)
                builders[-1].enter(group)
            elif current == ")" and previous != "\\":
                group = group.parent or _Group(1.0, None)
                builders[-1].enter(group)
            elif current == ":":
                weight, consumed = self._parse_weight(index + 1)
                if weight is None:
                    builders[-1].append(current)
                else:
                    group.weight = weight
                    group = _Group(group.default_weight, group.parent)
                    builders[-1].enter(group)
                    index += consumed
            elif current in ",\n":
                group = group if group.parent else _Group(1.0, None)
                builders.append(_TagBuilder(group))
            else:
                builders[-1].append(current)
            index += 1

        for builder in builders:
            tag = builder.build()
            if tag:
                yield tag

    def _parse_weight(self, start: int) -> tuple[float | None, int]:
        end = start
        while end < len(self.text):
            character = self.text[end]
            if character in ".-+" or character.isdigit() or character.isspace():
                end += 1
            else:
                break

        candidate = self.text[start:end]
        match = re.search(r"-?\s*(?:\d*\s*\.\s*\d+|\d+\s*\.\s*\d*|\d+)", candidate)
        if not match:
            return None, 0
        value = float(re.sub(r"\s", "", match.group(0)))
        return value, len(candidate.rstrip())


class _Tag:
    def __init__(self, name: str, pieces: list[tuple[str, float]]):
        self.name = name
        self.pieces = pieces


def _group_end(weight: float) -> str:
    return ")" if weight == 1.1 else f":{weight:.2g})"


def _render(tags: Iterable[_Tag]) -> str:
    output: list[str] = []
    separator: str | None = None
    active_weight = 1.0

    for tag in tags:
        for text, weight in tag.pieces:
            if weight != active_weight:
                if active_weight != 1.0:
                    output.append(_group_end(active_weight))
                if separator:
                    output.append(separator)
                    separator = None
                if weight != 1.0:
                    output.append("(")
                active_weight = weight
            elif separator:
                output.append(separator)
                separator = None
            output.append(text)
        separator = " " if tag.pieces[-1][0].endswith(".") else ", "

    if active_weight != 1.0:
        output.append(_group_end(active_weight))
    return "".join(output)


def merge_prompts(positive: str, negative: str) -> str:
    """Fold negative tags into the positive prompt with negative weights."""

    tags = list(_TagParser(positive))
    for tag in _TagParser(negative):
        tag.pieces = [(text, -abs(weight)) for text, weight in tag.pieces]
        tags.append(tag)
    return _render(tags)
