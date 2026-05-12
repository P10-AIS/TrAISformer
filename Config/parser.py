import dataclasses

import yaml
from dataclasses import is_dataclass, fields
from typing import Any, Callable, Dict, Type, get_type_hints
from typing import Type, TypeVar

T = TypeVar("T")

CUSTOM_PARSERS: Dict[Type, Callable[[Any], Any]] = {}


def register_parser(type_: Type, fn: Callable[[Any], Any]):
    CUSTOM_PARSERS[type_] = fn


def parse_dataclass(cls: Type[T], data: Any) -> T:
    if cls in CUSTOM_PARSERS:
        return CUSTOM_PARSERS[cls](data)

    if not is_dataclass(cls):
        return data

    kwargs = {}
    hints = get_type_hints(cls)

    for f in fields(cls):
        field_type = hints[f.name]
        value = data[f.name]

        kwargs[f.name] = parse_dataclass(field_type, value)

    return cls(**kwargs)


def parse_config(path: str, root_cls: Type[T]) -> T:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return parse_dataclass(root_cls, raw)


def init_parsers():
    import Config.adapters


def print_config(cfg, indent=0):
    prefix = "  " * indent
    for field in dataclasses.fields(cfg):
        value = getattr(cfg, field.name)
        if dataclasses.is_dataclass(value):
            print(f"{prefix}{field.name}:")
            print_config(value, indent + 1)
        else:
            print(f"{prefix}{field.name}: {value}")
