from grampus.evaluation.red_team.strategies.base import BaseAttackStrategy
from grampus.evaluation.red_team.strategies.excessive_agency import ExcessiveAgencyStrategy
from grampus.evaluation.red_team.strategies.jailbreak import JailbreakStrategy
from grampus.evaluation.red_team.strategies.memory_poison import MemoryPoisonStrategy
from grampus.evaluation.red_team.strategies.prompt_injection import PromptInjectionStrategy
from grampus.evaluation.red_team.strategies.reasoning_hijack import ReasoningHijackStrategy
from grampus.evaluation.red_team.strategies.tool_misuse import ToolMisuseStrategy

ALL_STRATEGIES: list[type[BaseAttackStrategy]] = [
    PromptInjectionStrategy,
    JailbreakStrategy,
    ReasoningHijackStrategy,
    MemoryPoisonStrategy,
    ToolMisuseStrategy,
    ExcessiveAgencyStrategy,
]

__all__ = [
    "BaseAttackStrategy",
    "PromptInjectionStrategy",
    "JailbreakStrategy",
    "ReasoningHijackStrategy",
    "MemoryPoisonStrategy",
    "ToolMisuseStrategy",
    "ExcessiveAgencyStrategy",
    "ALL_STRATEGIES",
]
