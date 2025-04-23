# SPDX-License-Identifier: Apache-2.0

from .base import MultiModalPlaceholderMap, MultiModalPlugin
from .hasher import MultiModalHashDict, MultiModalHasher
from .inputs import (BatchedTensorInputs, ModalityData, MultiModalDataBuiltins,
                     MultiModalDataDict, MultiModalKwargs,
                     MultiModalPlaceholderDict, NestedTensors)
from .registry import MultiModalRegistry
from .internvl_audio_plugin import AudioPlugin
MULTIMODAL_REGISTRY = MultiModalRegistry()
"""
The global :class:`~MultiModalRegistry` is used by model runners to
dispatch data processing according to the target model.

See also:
    :ref:`mm-processing`
"""

try:
    MULTIMODAL_REGISTRY.register_plugin(AudioPlugin())
    print("INFO: Successfully registered AudioPlugin.") # Add a print for confirmation
except AttributeError:
    print("ERROR: Could not find 'register_plugin' method on MultiModalRegistry. Check registry.py for the correct method name.")
except Exception as e:
    print(f"ERROR: Failed to register AudioPlugin: {e}")

__all__ = [
    "BatchedTensorInputs",
    "ModalityData",
    "MultiModalDataBuiltins",
    "MultiModalDataDict",
    "MultiModalHashDict",
    "MultiModalHasher",
    "MultiModalKwargs",
    "MultiModalPlaceholderDict",
    "MultiModalPlaceholderMap",
    "MultiModalPlugin",
    "NestedTensors",
    "MULTIMODAL_REGISTRY",
    "MultiModalRegistry",
]