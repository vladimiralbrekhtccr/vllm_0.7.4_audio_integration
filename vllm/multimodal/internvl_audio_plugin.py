# vllm/multimodal/internvl_audio_plugin.py
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Union

from vllm.inputs.registry import InputContext
from vllm.logger import init_logger
from vllm.utils import is_list_of

from .base import MultiModalPlugin
from .inputs import ModalityData, MultiModalKwargs

logger = init_logger(__name__)


# Define expected structure for the dictionary associated with the "audio" key
class AudioDataDict(Dict[str, Any]):
    audio_features: Union[np.ndarray, List[np.ndarray]]
    audio_len_after_cnn: Union[int, List[int]]
    audio_token_num: Union[int, List[int]]

class AudioPlugin(MultiModalPlugin):
    """
    Plugin for handling pre-processed audio features for InternVL-style models.
    Expects data structured as:
    {
        "audio": {
            "audio_features": [numpy_array_C_T, ...],
            "audio_len_after_cnn": [int, ...],
            "audio_token_num": [int, ...]
        }
    }
    """

    def get_data_key(self) -> str:
        """The key in `multi_modal_data` this plugin handles."""
        # ---> USE A UNIQUE KEY <---
        return "internvl_audio"

    def _default_input_mapper(
        self,
        ctx: InputContext,
        data: ModalityData[AudioDataDict], # Expects the inner dictionary
        **mm_processor_kwargs,
    ) -> MultiModalKwargs:
        """
        Processes the pre-computed audio features dictionary into tensors
        expected by the model's forward pass.
        """
        # Update log message
        logger.debug(f"AudioPlugin processing data of type: {type(data)} for key '{self.get_data_key()}'")

        if not isinstance(data, dict):
             # Update error message
             raise TypeError(f"AudioPlugin expected a dictionary for key '{self.get_data_key()}', but got {type(data)}")

        audio_dict: AudioDataDict = data

        # --- Validate and Extract Data ---
        required_keys = {"audio_features", "audio_len_after_cnn", "audio_token_num"}
        if not required_keys.issubset(audio_dict.keys()):
            # Update error message
            raise ValueError(f"Audio data dictionary for key '{self.get_data_key()}' is missing required keys. "
                             f"Expected {required_keys}, got {audio_dict.keys()}")

        raw_features = audio_dict["audio_features"]
        raw_len_cnn = audio_dict["audio_len_after_cnn"]
        raw_token_num = audio_dict["audio_token_num"]

        # --- Convert to Tensors (Handle lists for potential batching) ---
        # Convert features (expecting list of numpy arrays [C, T])
        if is_list_of(raw_features, np.ndarray):
            feature_tensors = [torch.from_numpy(f) for f in raw_features]
            # Assume features are already padded to same length (e.g., 3000)
            try:
                final_audio_features = torch.stack(feature_tensors, dim=0) # Stack to [B, C, T]
            except RuntimeError as e:
                 logger.error(f"Failed to stack audio features (likely different lengths): {e}")
                 raise ValueError("Audio features have different lengths and cannot be stacked. Plugin/Model needs padding logic.") from e
        else:
            raise TypeError(f"Expected audio_features to be a list of numpy.ndarray, got {type(raw_features)}")

        # Convert lengths/nums (expecting list of ints)
        if is_list_of(raw_len_cnn, int):
            final_audio_len_after_cnn = torch.tensor(raw_len_cnn, dtype=torch.long) # -> Tensor[B]
        else:
            raise TypeError(f"Expected audio_len_after_cnn to be a list of int, got {type(raw_len_cnn)}")

        if is_list_of(raw_token_num, int):
            final_audio_token_num = torch.tensor(raw_token_num, dtype=torch.long) # -> Tensor[B]
        else:
            raise TypeError(f"Expected audio_token_num to be a list of int, got {type(raw_token_num)}")

        logger.debug(f"AudioPlugin processed features shape: {final_audio_features.shape}")
        logger.debug(f"AudioPlugin processed len_cnn shape: {final_audio_len_after_cnn.shape}")
        logger.debug(f"AudioPlugin processed token_num shape: {final_audio_token_num.shape}")

        # --- Return processed data using keys the model expects ---
        # These keys MUST match the arguments expected by the model's forward pass
        # when handling multimodal inputs.
        return MultiModalKwargs({
            "audio_features": final_audio_features,
            "audio_len_after_cnn": final_audio_len_after_cnn,
            "audio_token_num": final_audio_token_num,
        })
class InternVLAudioDataItem:
    """
    Wraps the raw pre-processed audio data dictionary to satisfy
    the interface expected by BaseMultiModalProcessor._get_hf_mm_data.
    """
    def __init__(self, data: AudioDataDict):
        self.raw_data = data # Store the original dictionary

    def get_processor_data(self) -> Optional[Dict[str, Any]]:
        """
        Indicates data needing standard HF processing. Returns None as our
        data is pre-processed.
        """
        return None

    # Optional: Add a helper to retrieve the data if needed elsewhere,
    # though the main processor uses the original mm_data for the plugin mapper.
    def get_raw_data(self) -> AudioDataDict:
        return self.raw_data
    # Optional: Define max multimodal tokens if specific calculation is needed
    # def _default_max_multimodal_tokens(self, ctx: InputContext) -> int:
    #     # Could calculate based on audio_token_num if needed
    #     return super()._default_max_multimodal_tokens(ctx)