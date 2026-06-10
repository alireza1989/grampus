"""Entry-point discovery for third-party Nexus plugins."""

from __future__ import annotations

from nexus.core.logging import get_logger
from nexus.plugins.base import NexusPlugin
from nexus.plugins.manager import PluginManager

_log = get_logger(__name__)
_ENTRY_POINT_GROUP = "nexus.plugins"


def load_entry_point_plugins() -> list[NexusPlugin]:
    """Discover and instantiate all plugins registered via entry points.

    Third-party packages register plugins in their pyproject.toml:

        [project.entry-points."nexus.plugins"]
        my_plugin = "my_package:MyPlugin"

    The entry-point value must be a class (not an instance). It is instantiated
    with no arguments. If instantiation or loading fails, the error is logged
    and that plugin is skipped — other plugins continue loading.

    Returns:
        List of instantiated NexusPlugin instances, in discovery order.
    """
    from importlib.metadata import entry_points

    plugins: list[NexusPlugin] = []
    for ep in entry_points(group=_ENTRY_POINT_GROUP):
        try:
            cls = ep.load()
            plugin = cls()
            if not isinstance(plugin, NexusPlugin):
                _log.warning(
                    "plugin_not_nexus_plugin",
                    entry_point=ep.name,
                    cls=cls.__name__,
                )
                continue
            plugins.append(plugin)
            _log.info(
                "plugin_loaded_from_entry_point",
                entry_point=ep.name,
                plugin=plugin.name,
            )
        except Exception as exc:
            _log.error("plugin_load_failed", entry_point=ep.name, error=str(exc))
    return plugins


def create_manager_from_entry_points() -> PluginManager:
    """Discover all entry-point plugins and return a configured PluginManager."""
    return PluginManager(plugins=load_entry_point_plugins())
