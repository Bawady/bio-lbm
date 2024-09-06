from typing import Callable


class GenericFactory:
    def __init__(self):
        self._entries = {}

    def add_entry(self, name: str, cls: Callable) -> None:
        if self.contains(name):
            raise Warning(f"Overwriting previously added factory entry with name '{name}'")
        self._entries[name] = cls

    def create(self, name: str, **params: dict) -> None:
        cls = self._entries.get(name)
        if not cls:
            raise ValueError(f"Factory does not contain an entry with named {name}")
        obj = cls()
        obj.fabricate(params)
        obj.init()
        return obj

    def create_blank(self, name):
        cls = self._entries.get(name)
        if not cls:
            raise ValueError(f"Factory does not contain an entry with named {name}")
        obj = cls()
        return obj

    def contains(self, name: str) -> bool:
        return name in self._entries.keys()


class Fabricable:
    factory = None

    def fabricate(self, params: dict):
        from src.Serializable import Serializable
        for key in params:
            setattr(self, key, params[key])
            if isinstance(self, Serializable) and key not in self.serialize_members:
                self.serialize_members.append(key)

    def init(self):
        pass
