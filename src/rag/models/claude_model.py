from enum import Enum


class ClaudeModel(str, Enum):
    SONNET = "claude-sonnet-4-6"
    HAIKU = "claude-haiku-4-5-20251001"

    @classmethod
    def values(cls) -> list[str]:
        return [m.value for m in cls]
