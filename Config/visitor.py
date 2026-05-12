from typing import Type, Callable, Dict, Any


class ConfigVisitorRegistry:
    _visitors: Dict[Type, Callable[[dict], Any]] = {}

    @classmethod
    def register(cls, config_type: Type, visitor: Callable[[dict], Any]):
        cls._visitors[config_type] = visitor

    @classmethod
    def visit(cls, config_type: Type, data: dict):
        visitor = cls._visitors.get(config_type)
        if not visitor:
            raise ValueError(
                f"No visitor registered for {config_type.__name__}")
        return visitor(data)
