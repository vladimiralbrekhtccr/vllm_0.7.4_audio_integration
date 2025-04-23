# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections import UserDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import (TYPE_CHECKING, Any, Generic, NamedTuple, Optional, TypeVar,
                    Union)

import numpy as np
import torch
from PIL.Image import Image
from transformers import BatchFeature
from typing_extensions import TypeAlias, TypeGuard, assert_never

from vllm.utils import is_list_of

from .audio import resample_audio
from .inputs import (AudioItem, HfAudioItem, HfImageItem, HfVideoItem,
                     ImageItem, ModalityData, MultiModalDataDict,
                     MultiModalFieldConfig, MultiModalKwargs, VideoItem)

_T = TypeVar("_T")
_I = TypeVar("_I")


class ModalityDataItems(ABC, Generic[_T, _I]):
    """
    Represents data items for a modality in :class:`MultiModalDataItems`.
    """

    def __init__(self, data: _T, modality: str) -> None:
        super().__init__()

        self.data = data
        self.modality = modality

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(modality={self.modality!r}, "
                f"len={len(self)})")

    def __len__(self) -> int:
        return self.get_count()

    def __getitem__(self, index: int) -> _I:
        return self.get(index)

    if TYPE_CHECKING:
        # Auto-generated
        def __iter__(self) -> Iterator[_I]:
            ...

    @abstractmethod
    def get_count(self) -> int:
        """Get the number of data items."""
        raise NotImplementedError

    @abstractmethod
    def get(self, index: int) -> _I:
        """Get a data item by its index."""
        raise NotImplementedError

    def get_all(self) -> list[_I]:
        """Get all data items."""
        return [self.get(idx) for idx in range(self.get_count())]

    @abstractmethod
    def get_processor_data(self) -> Mapping[str, object]:
        """Get the data to pass to the HF processor."""
        raise NotImplementedError

    @abstractmethod
    def get_passthrough_data(self) -> Mapping[str, object]:
        """Get the data to pass directly to the model."""
        raise NotImplementedError


class ProcessorBatchItems(ModalityDataItems[Sequence[_T], _T]):
    """Base class for data items that are arranged in a list."""

    def get_count(self) -> int:
        return len(self.data)

    def get(self, index: int) -> _T:
        return self.data[index]

    def get_processor_data(self) -> Mapping[str, object]:
        return {f"{self.modality}s": self.data}

    def get_passthrough_data(self) -> Mapping[str, object]:
        return {}


class EmbeddingItems(ModalityDataItems[Union[torch.Tensor, list[torch.Tensor]],
                                       torch.Tensor]):
    """
    Base class for data items that are expressed as a batched embedding tensor,
    or a list of embedding tensors (one per item).
    """

    def get_count(self) -> int:
        return len(self.data)

    def get(self, index: int) -> torch.Tensor:
        return self.data[index]

    def get_processor_data(self) -> Mapping[str, object]:
        return {}

    def get_passthrough_data(self) -> Mapping[str, object]:
        return {f"{self.modality}_embeds": self.data}

    def get_feature_size(self, item_idx: int) -> int:
        return len(self.get(item_idx))


class DictEmbeddingItems(ModalityDataItems[Mapping[str, torch.Tensor],
                                           Mapping[str, torch.Tensor]]):
    """
    Base class for data items that are expressed as a dictionary of tensors.

    Usually, the dictionary keys correspond to the outputs of HF processor.
    """

    def __init__(
        self,
        data: Mapping[str, torch.Tensor],
        modality: str,
        required_fields: set[str],
        fields_factory: Callable[
            [Mapping[str, torch.Tensor]],
            Mapping[str, MultiModalFieldConfig],
        ],
    ) -> None:
        super().__init__(data, modality)

        missing_required_data_keys = required_fields - data.keys()
        if missing_required_data_keys:
            data_keys = set(data.keys())
            msg = (f"The data should contain the fields: {required_fields}, "
                   f"but only found the following keys: {data_keys}")
            raise ValueError(msg)

        fields_config = fields_factory(data)
        missing_required_fields = required_fields - fields_config.keys()
        if missing_required_fields:
            fields = set(fields_config.keys())
            msg = f"{required_fields=} should be a subset of {fields=}"
            raise ValueError(msg)

        self.fields_config = fields_config
        self.required_fields = required_fields

        self._kwargs = MultiModalKwargs.from_hf_inputs(
            BatchFeature(dict(data)),
            fields_config,
        )

    def get_count(self) -> int:
        return self._kwargs.get_item_count(self.modality)

    def get(self, index: int) -> Mapping[str, torch.Tensor]:
        return {
            k: v.data
            for k, v in self._kwargs.get_item(self.modality, index).items()
        }

    def get_processor_data(self) -> Mapping[str, object]:
        return {}

    def get_passthrough_data(self) -> Mapping[str, object]:
        return self.data


class AudioProcessorItems(ProcessorBatchItems[HfAudioItem]):

    def __init__(self, data: Sequence[HfAudioItem]) -> None:
        super().__init__(data, "audio")

class InternVLAudioFeatureItems(ModalityDataItems[Mapping[str, list], Mapping[str, torch.Tensor]]):
    """
    Handles pre-processed audio data passed as a dictionary containing features and metadata.
    The input 'data' is expected to be like:
    {
        "audio_features": [Tensor[C, T], ...],
        "audio_len_after_cnn": [Tensor[], ...],
        "audio_token_num": [Tensor[], ...]
    }
    """
    def __init__(self, data: Mapping[str, list], modality: str) -> None:
        # Basic validation
        required_keys = {"audio_features", "audio_len_after_cnn", "audio_token_num"}
        if not required_keys.issubset(data.keys()):
            raise ValueError(f"InternVLAudioFeatureItems requires keys {required_keys}")

        # Ensure values are lists of tensors
        if not (is_list_of(data.get("audio_features"), torch.Tensor) and
                is_list_of(data.get("audio_len_after_cnn"), torch.Tensor) and
                is_list_of(data.get("audio_token_num"), torch.Tensor)):
            raise TypeError("InternVLAudioFeatureItems expects a dictionary with lists of tensors.")

        num_items = len(data["audio_features"])
        if not (len(data.get("audio_len_after_cnn", [])) == num_items and
                len(data.get("audio_token_num", [])) == num_items):
            raise ValueError("All lists in InternVLAudioFeatureItems must have the same length.")

        super().__init__(data, modality)
        self._num_items = num_items
        print(f"DEBUG: Initialized InternVLAudioFeatureItems with {self._num_items} items.") # Debug print

    def get_count(self) -> int:
        return self._num_items

    def get(self, index: int) -> Mapping[str, torch.Tensor]:
        """Get a single item's data."""
        if not 0 <= index < self._num_items:
            raise IndexError("Index out of range")
        return {
            "audio_features": self.data["audio_features"][index],
            "audio_len_after_cnn": self.data["audio_len_after_cnn"][index],
            "audio_token_num": self.data["audio_token_num"][index],
        }

    def get_processor_data(self) -> Mapping[str, object]:
        """No further HF processing needed for this pre-processed data."""
        print("DEBUG: InternVLAudioFeatureItems.get_processor_data() called (returning empty).") # Debug print
        return {}

    def get_passthrough_data(self) -> Mapping[str, object]:
        """Pass the structured data directly to the model kwargs, batched."""
        print("DEBUG: InternVLAudioFeatureItems.get_passthrough_data() called.") # Debug print
        # Stack tensors into batches expected by the model's forward pass
        try:
            # Ensure features are on the same device before stacking
            device = self.data["audio_features"][0].device if self._num_items > 0 else 'cpu'
            feature_list = [t.to(device) for t in self.data["audio_features"]]
            len_cnn_list = [t.to(device) for t in self.data["audio_len_after_cnn"]]
            token_num_list = [t.to(device) for t in self.data["audio_token_num"]]

            stacked_features = torch.stack(feature_list, dim=0)
            # These are likely scalar tensors, stacking adds a dimension [B]
            stacked_len_cnn = torch.stack(len_cnn_list, dim=0)
            stacked_token_num = torch.stack(token_num_list, dim=0)
            print(f"DEBUG: Passthrough shapes: feats={stacked_features.shape}, len={stacked_len_cnn.shape}, num={stacked_token_num.shape}")

        except RuntimeError as e:
            # This might happen if audio_features tensors have different lengths (C or T dimensions)
            print(f"ERROR: Could not stack tensors in get_passthrough_data: {e}.")
            print("       Make sure all audio_features tensors have the same shape.")
            # You might need padding logic *before* this step if lengths differ.
            # For now, re-raise to highlight the issue.
            raise ValueError("Failed to batch pre-processed audio data. Check tensor shapes.") from e

        # These keys MUST match the kwargs expected by InternVLChatAudioModel.forward
        return {
            "audio_features": stacked_features,
            "audio_len_after_cnn": stacked_len_cnn,
            "audio_token_num": stacked_token_num,
        }

class AudioEmbeddingItems(EmbeddingItems):

    def __init__(self, data: Union[torch.Tensor, list[torch.Tensor]]) -> None:
        super().__init__(data, "audio")


class ImageSize(NamedTuple):
    width: int
    height: int


class ImageProcessorItems(ProcessorBatchItems[HfImageItem]):

    def __init__(self, data: Sequence[HfImageItem]) -> None:
        super().__init__(data, "image")

    def get_image_size(self, item_idx: int) -> ImageSize:
        image = self.get(item_idx)

        if isinstance(image, Image):
            return ImageSize(*image.size)
        if isinstance(image, (np.ndarray, torch.Tensor)):
            _, h, w = image.shape
            return ImageSize(w, h)

        assert_never(image)


class ImageEmbeddingItems(EmbeddingItems):

    def __init__(self, data: Union[torch.Tensor, list[torch.Tensor]]) -> None:
        super().__init__(data, "image")


class VideoProcessorItems(ProcessorBatchItems[HfVideoItem]):

    def __init__(self, data: Sequence[HfVideoItem]) -> None:
        super().__init__(data, "video")

    def get_num_frames(self, item_idx: int) -> int:
        return len(self.get(item_idx))

    def get_frame_size(self, item_idx: int) -> ImageSize:
        image = self.get(item_idx)[0]  # Assume that the video isn't empty

        if isinstance(image, Image):
            return ImageSize(*image.size)
        if isinstance(image, (np.ndarray, torch.Tensor)):
            _, h, w = image.shape
            return ImageSize(w, h)

        assert_never(image)


class VideoEmbeddingItems(EmbeddingItems):

    def __init__(self, data: Union[torch.Tensor, list[torch.Tensor]]) -> None:
        super().__init__(data, "video")


_D = TypeVar("_D", bound=ModalityDataItems[Any, Any])


class MultiModalDataItems(UserDict[str, ModalityDataItems[Any, Any]]):
    """
    As :data:`~vllm.multimodal.inputs.MultiModalDataDict`, but normalized
    such that each entry corresponds to a list.
    """

    def get_count(self, modality: str, *, strict: bool = True) -> int:
        """
        Get the number of data items belonging to a modality.
        
        If `strict=False`, return `0` instead of raising :exc:`KeyError`
        even if the modality is not found.
        """
        if modality not in self:
            if strict:
                available_modalities = set(self.keys())
                raise KeyError(f"Modality {modality!r} not found. "
                               f"Available modalities: {available_modalities}")

            return 0

        return self[modality].get_count()

    def get_all_counts(self) -> Mapping[str, int]:
        """Get the number of items belonging to each modality."""
        return {m: items.get_count() for m, items in self.items()}

    def get_items(
        self,
        modality: str,
        typ: Union[type[_D], tuple[type[_D], ...]],
    ) -> _D:
        """
        Get the data items belonging to a modality,
        requiring that they belong to a certain type.
        """
        if modality not in self:
            available_modalities = set(self.keys())
            raise KeyError(f"Modality {modality!r} not found. "
                           f"Available modalities: {available_modalities}")

        items = self[modality]
        if not isinstance(items, typ):
            raise TypeError(f"Invalid type of data items for {modality=}. "
                            f"Expected type: {typ}, but "
                            f"found type: {type(items)}")

        return items  # type: ignore[return-value]


ModalityDataParser: TypeAlias = Callable[[ModalityData[Any]],
                                         ModalityDataItems[Any, Any]]


class MultiModalDataParser:
    """
    Parses :data:`~vllm.multimodal.inputs.MultiModalDataDict` into
    :class:`MultiModalDataItems`.

    Args:
        target_sr (float, optional): Enables automatic resampling of audio
            items to the model's expected sampling rate.
    """

    def __init__(self, *, target_sr: Optional[float] = None) -> None:
        super().__init__()

        self.target_sr = target_sr

    def _is_embeddings(
            self, data: object
    ) -> TypeGuard[Union[torch.Tensor, list[torch.Tensor]]]:
        if isinstance(data, torch.Tensor):
            return data.ndim == 3
        if is_list_of(data, torch.Tensor):
            return len(data) == 0 or data[0].ndim == 2

        return False

    def _get_audio_with_sr(
        self,
        audio: AudioItem,
    ) -> tuple[np.ndarray, Optional[float]]:
        if isinstance(audio, tuple):
            return audio
        if isinstance(audio, list):
            return np.array(audio), None
        if isinstance(audio, np.ndarray):
            return audio, None
        if isinstance(audio, torch.Tensor):
            return audio.numpy(), None

        assert_never(audio)


    def _parse_audio_data(
        self,
        data: ModalityData[AudioItem],
    ) -> ModalityDataItems[Any, Any]:
        """
        Parses various audio input formats into specific ModalityDataItems.

        Handles:
        1. Pre-processed dict: {"audio_features":..., "audio_len_after_cnn":..., "audio_token_num":...}
        2. Pre-computed embeddings: Tensor (B, Seq, D) or List[Tensor(Seq, D)]
        3. Raw audio: Single waveform, list of waveforms, or list of (waveform, sr) tuples.
        """
        # Case 1: Pre-processed Dictionary (from Dummy Builder or inference script)
        # Check if it's a dict and has the required keys
        if isinstance(data, dict) and \
           {"audio_features", "audio_len_after_cnn", "audio_token_num"}.issubset(data.keys()):
            print("DEBUG: _parse_audio_data using InternVLAudioFeatureItems (pre-processed dict)")

            # Ensure values are lists of tensors
            features = data["audio_features"]
            lens = data["audio_len_after_cnn"]
            tokens = data["audio_token_num"]

            # --- Standardize to lists of tensors ---
            # Handle case where input might be single tensor instead of list
            if not isinstance(features, list):
                 # If single tensor with >2 dims, assume first is batch dim
                 if isinstance(features, torch.Tensor) and features.ndim > 2:
                     features = [f for f in features]
                 else: # Wrap single item in list
                     features = [features]
            if not isinstance(lens, list):
                 if isinstance(lens, torch.Tensor) and lens.ndim > 0:
                     lens = [l for l in lens]
                 else: # Wrap single item in list
                     lens = [lens]
            if not isinstance(tokens, list):
                 if isinstance(tokens, torch.Tensor) and tokens.ndim > 0:
                     tokens = [t for t in tokens]
                 else: # Wrap single item in list
                     tokens = [tokens]

            # --- Ensure elements within lists are Tensors ---
            try:
                # Convert non-Tensor elements (like raw numbers if passed incorrectly)
                features_t = [f if isinstance(f, torch.Tensor) else torch.tensor(f) for f in features]
                lens_t = [l if isinstance(l, torch.Tensor) else torch.tensor(l, dtype=torch.long) for l in lens]
                tokens_t = [t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.long) for t in tokens]
            except Exception as e:
                 raise TypeError(f"Failed to convert elements in pre-processed audio dict to tensors: {e}")

            # --- Check for length consistency ---
            num_items = len(features_t)
            if not (num_items == len(lens_t) == len(tokens_t)):
                 raise ValueError(f"List lengths mismatch in pre-processed audio dictionary. "
                                  f"Got features={len(features_t)}, lens={len(lens_t)}, tokens={len(tokens_t)}")

            processed_data = {
                "audio_features": features_t,
                "audio_len_after_cnn": lens_t,
                "audio_token_num": tokens_t
            }
            # Ensure the InternVLAudioFeatureItems class definition is present in parse.py
            # from your previous code snippet.
            return InternVLAudioFeatureItems(processed_data, "audio")

        # Case 2: Pre-computed Embeddings
        if self._is_embeddings(data):
            print("DEBUG: _parse_audio_data using AudioEmbeddingItems (embeddings)")
            # Ensure AudioEmbeddingItems class is defined in parse.py
            return AudioEmbeddingItems(data)

        # Case 3: Raw Audio Waveform(s) -> Use AudioProcessorItems
        print("DEBUG: _parse_audio_data attempting raw audio processing for AudioProcessorItems")
        raw_audio_items_for_processor: List[HfAudioItem] = [] # Type expected by AudioProcessorItems

        # Subcase 3a: Single raw waveform (np.ndarray, list/tuple of floats, torch.Tensor ndim=1)
        if isinstance(data, (np.ndarray, torch.Tensor)) and data.ndim == 1:
            print("DEBUG: Raw audio - single ndarray/tensor waveform")
            waveform = data.numpy() if isinstance(data, torch.Tensor) else data
            raw_audio_items_for_processor.append(waveform.astype(np.float32)) # Ensure float32
        elif isinstance(data, (list, tuple)) and data and isinstance(data[0], (int, float)):
             print("DEBUG: Raw audio - single list/tuple waveform")
             raw_audio_items_for_processor.append(np.array(data, dtype=np.float32))

        # Subcase 3b: Batch of raw waveforms (np.ndarray/Tensor ndim > 1) - assume first dim is batch
        elif isinstance(data, (np.ndarray, torch.Tensor)) and data.ndim > 1:
             print(f"DEBUG: Raw audio - batch ndarray/tensor waveform (shape {data.shape})")
             waveforms = data.numpy() if isinstance(data, torch.Tensor) else data
             raw_audio_items_for_processor.extend([elem.astype(np.float32) for elem in waveforms])

        # Subcase 3c: List containing waveforms or (waveform, sr) tuples
        elif isinstance(data, list) and data:
             first_item = data[0]
             # List of (waveform, sr) tuples - Extract only waveform for AudioProcessorItems
             if isinstance(first_item, tuple) and len(first_item) == 2:
                  print("DEBUG: Raw audio - list of (waveform, sr) tuples (extracting waveforms)")
                  for item in data:
                      if isinstance(item, tuple) and len(item) == 2:
                          wf = item[0]
                          if isinstance(wf, torch.Tensor): wf = wf.numpy()
                          if isinstance(wf, (list, tuple)): wf = np.array(wf, dtype=np.float32)
                          if isinstance(wf, np.ndarray):
                              raw_audio_items_for_processor.append(wf.astype(np.float32))
                          else:
                              print(f"Warning: Skipping invalid waveform type in tuple: {type(wf)}")
                      else:
                           print(f"Warning: Skipping invalid item in list of tuples: {item}")
             # List of waveforms (arrays, tensors, lists of floats)
             elif isinstance(first_item, (np.ndarray, torch.Tensor, list, tuple)):
                  print("DEBUG: Raw audio - list of waveforms")
                  for item in data:
                      if isinstance(item, torch.Tensor):
                          raw_audio_items_for_processor.append(item.numpy().astype(np.float32))
                      elif isinstance(item, (list, tuple)) and item and isinstance(item[0], (int, float)):
                           raw_audio_items_for_processor.append(np.array(item, dtype=np.float32))
                      elif isinstance(item, np.ndarray):
                           raw_audio_items_for_processor.append(item.astype(np.float32))
                      else:
                           print(f"Warning: Skipping invalid item type in list of waveforms: {type(item)}")
             else:
                  raise TypeError(f"Unsupported format for list of audio data: first item type {type(first_item)}")
        else:
            # If data format is none of the above
            raise TypeError(f"Unsupported audio data format: {type(data)}")

        # If parsing resulted in raw items, pass them to AudioProcessorItems
        if raw_audio_items_for_processor:
             print(f"DEBUG: Parsed {len(raw_audio_items_for_processor)} raw audio items for AudioProcessorItems")
             # Ensure AudioProcessorItems class is defined in parse.py
             return AudioProcessorItems(raw_audio_items_for_processor)
        else:
            # This path shouldn't be reached if input validation is correct, but as a safeguard:
            raise ValueError("Could not parse any audio items from the provided data.")

    def _parse_image_data(
        self,
        data: ModalityData[ImageItem],
    ) -> ModalityDataItems[Any, Any]:
        if self._is_embeddings(data):
            return ImageEmbeddingItems(data)

        if (isinstance(data, Image)
                or isinstance(data,
                              (np.ndarray, torch.Tensor)) and data.ndim == 3):
            data_items = [data]
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            data_items = [elem for elem in data]
        else:
            data_items = data

        return ImageProcessorItems(data_items)

    def _parse_video_data(
        self,
        data: ModalityData[VideoItem],
    ) -> ModalityDataItems[Any, Any]:
        if self._is_embeddings(data):
            return VideoEmbeddingItems(data)

        if (is_list_of(data, Image)
                or isinstance(data,
                              (np.ndarray, torch.Tensor)) and data.ndim == 4):
            data_items = [data]
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            data_items = [elem for elem in data]
        else:
            data_items = data

        return VideoProcessorItems(data_items)

    def _get_subparsers(self) -> Mapping[str, ModalityDataParser]:
        return {
            "audio": self._parse_audio_data,
            "image": self._parse_image_data,
            "video": self._parse_video_data,
        }

    def parse_mm_data(self,
                      mm_data: MultiModalDataDict) -> MultiModalDataItems:
        subparsers = self._get_subparsers()

        mm_items = MultiModalDataItems()
        for k, v in mm_data.items():
            if k not in subparsers:
                raise ValueError(f"Unsupported modality: {k}")

            mm_items[k] = subparsers[k](v)

        return mm_items