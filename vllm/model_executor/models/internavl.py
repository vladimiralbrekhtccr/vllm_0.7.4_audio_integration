# internavl.py

import os
from typing import List, Literal, Optional, Set, Tuple, TypedDict, Union
import numpy as np
import torch
import torch.nn as nn
from transformers import BatchFeature, PretrainedConfig, TensorType

from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargs, NestedTensors
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import BaseProcessingInfo, PromptReplacement, PromptUpdate, PromptUpdateDetails
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.tokenizer import AnyTokenizer
from importlib import import_module
from .utils import(merge_multimodal_embeddings)
from .internvl_whisper_processor import WhisperProcessor, load_audio

# Import the base InternVL components
from vllm.model_executor.models.internvl import *

# Define audio tokens
AUDIO_START = '<audio>'
AUDIO_END = '</audio>'
AUDIO_CONTEXT = '<AUDIO_CONTEXT>'

# Audio inputs type definition
class InternVLAudioInputs(TypedDict):
    type: Literal["audio_features"]
    data: torch.Tensor
    audio_len_after_cnn: List[int]
    audio_token_num: List[int]

# Audio processing info
class InternVLAudioProcessor(BaseInternVLProcessor):
    # Implement required abstract methods from BaseInternVLProcessor
    @property
    def image_token_id(self) -> int:
        return self.tokenizer.get_vocab()[IMG_CONTEXT]

    def get_image_repl_features(
        self,
        feature_size: int,
        num_patches: Optional[int],
    ) -> str:
        return IMG_CONTEXT * feature_size

    def get_image_repl_full(
        self,
        feature_size: int,
        num_patches: Optional[int],
    ) -> str:
        features = self.get_image_repl_features(feature_size, num_patches)
        return IMG_START + features + IMG_END

    # Add your additional audio methods
    @property
    def audio_token_id(self) -> int:
        return self.tokenizer.get_vocab()[AUDIO_CONTEXT]

    def get_audio_repl_features(
        self,
        feature_size: int,
        num_tokens: Optional[int],
    ) -> str:
        # Simply return the requested number of tokens without validation
        return AUDIO_CONTEXT * feature_size

    def get_audio_repl_full(
        self,
        feature_size: int,
        num_tokens: Optional[int],
    ) -> str:
        features = self.get_audio_repl_features(feature_size, num_tokens)
        return AUDIO_START + features + AUDIO_END
    
    def __call__(
        self,
        text: Optional[Union[str, list[str]]] = None,
        images: Optional[Union[Image.Image, list[Image.Image]]] = None,
        audios: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        # Initialize result dictionary
        result = BatchFeature({})
        
        # Process text input
        if text is None:
            text = [""]
        if not isinstance(text, list):
            text = [text]
        
        # Process images if any
        image_processed = False
        if images is not None:
            image_result = super().__call__(
                text=text, 
                images=images,
                min_dynamic_patch=min_dynamic_patch,
                max_dynamic_patch=max_dynamic_patch,
                dynamic_image_size=dynamic_image_size,
                return_tensors=None
            )
            result.update(image_result)
            image_processed = True
        
        # Process audio inputs
        audio_processed = False
        if audios is not None:
            if not isinstance(audios, list):
                audios = [audios]

            audio_features = []
            audio_token_num = []
            audio_len_after_cnn = []

            for audio in audios:
                audio_input = self.process_audio(audio)
                
                input_features = audio_input["audio_values"]
                audio_len = audio_input["audio_len_after_cnn"]
                token_count = audio_input["audio_token_num"]

                audio_features.append(input_features)
                audio_len_after_cnn.append(audio_len)
                audio_token_num.append(token_count)
                audio_processed = True

            # Replace <audio> tokens in the text with correct audio token placeholders
            for i in range(len(text)):
                if "<audio>" in text[i]:
                    token_count = audio_token_num[i]
                    audio_repl = self.get_audio_repl_full(token_count, None)
                    text[i] = text[i].replace("<audio>", audio_repl, 1)

                    actual_tokens = text[i].split().count(AUDIO_CONTEXT)
                    print(f"[Debug] Sample {i}: Expected {token_count} AUDIO_CONTEXT tokens, inserted {actual_tokens}")
                    if actual_tokens != token_count:
                        print(f"[Warning] Mismatch in sample {i}: placeholders ({actual_tokens}) ≠ expected ({token_count})")



            if audio_features:
                result.update({
                    "audio_features": torch.stack(audio_features),
                    "audio_len_after_cnn": torch.tensor(audio_len_after_cnn),
                    "audio_token_num": torch.tensor(audio_token_num),
                })

        
        # If neither audio nor image was processed, just tokenize the text
        if not image_processed and not audio_processed:
            # Need to ensure input_ids field exists
            text_inputs = self.tokenizer(text)
            result.update(text_inputs)
        elif "input_ids" not in result:
            # Tokenize if input_ids not set by image processing
            text_inputs = self.tokenizer(text)
            result.update(text_inputs)
        print("\n[Processor] Final tokenized text:")
        print(text)
        input_ids = result["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            print("[Processor] input_ids shape:", input_ids.shape)
        else:
            print("[Processor] input_ids is not a tensor yet:", type(input_ids), "->", input_ids[:5])

        print("[Processor] input_ids:", result["input_ids"])
        print("[Processor] audio_token_id:", self.audio_token_id)

        if "audio_features" in result:
            print("[Processor] audio_features shape:", result["audio_features"].shape)
            print("[Processor] audio_len_after_cnn:", result["audio_len_after_cnn"])
            print("[Processor] audio_token_num:", result["audio_token_num"])
        
        # Add token IDs for both modalities if not already present
        if image_processed and "image_token_id" not in result:
            result["image_token_id"] = torch.tensor(self.image_token_id)
        
        if audio_processed and "audio_token_id" not in result:
            result["audio_token_id"] = torch.tensor(self.audio_token_id)
        
        # Apply return_tensors if specified
        if return_tensors is not None:
            result = BatchFeature(result, tensor_type=return_tensors)
        
        return result
    
    def process_audio(self, audio: np.ndarray) -> torch.Tensor:
        whisper_processor = WhisperProcessor.from_pretrained(os.environ["WHISPER_MODEL_PATH"])
        audio_features = load_audio(audio, whisper_processor)
        return audio_features



# Dummy inputs builder for audio
class InternVLAudioDummyInputsBuilder(InternVLDummyInputsBuilder):
    def get_dummy_processor_inputs(self, seq_len, mm_counts):
        inputs = super().get_dummy_processor_inputs(seq_len, mm_counts)

        num_audios = mm_counts.get("audio", 0)
        if num_audios > 0:
            # Create dummy audio: 5 seconds of random waveform at 16kHz
            sr = 16000
            duration = 30  # seconds
            tokens_per_audio = self.info.get_max_audio_tokens()

            # Update prompt text with <audio> placeholders
            inputs.prompt_text += " " + " ".join(["<audio>"] * tokens_per_audio * num_audios)

            # Generate dummy waveforms (not spectrograms!)
            inputs.mm_data["audio"] = [
                np.random.rand(sr * duration).astype(np.float32)
                for _ in range(num_audios)
            ]

        # Use HF processor to tokenize and extract features
        processor = self.info.get_hf_processor()
        dummy_batch = processor(
            text=inputs.prompt_text,
            images=inputs.mm_data.get("image", []),
            audios=inputs.mm_data.get("audio", [])
        )

        return inputs


# Audio multimodal processor
class InternVLAudioMultiModalProcessor(InternVLMultiModalProcessor):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
        )

        audio_token_id = self.info.get_hf_processor(**mm_kwargs).audio_token_id
        audio_data = mm_data.get("audio", [])
        assert isinstance(audio_data, list)

        processed_outputs["audio_token_id"] = torch.tensor(audio_token_id)
        
        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # Get the image config from parent
        config = super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs)
        
        # Add audio config
        audio_token_num = hf_inputs.get("audio_token_num", torch.empty(0))
        num_audios = len(audio_token_num)
        
        if num_audios > 0:
            config.update({
                "audio_features": MultiModalFieldConfig.batched("audio"),
                "audio_len_after_cnn": MultiModalFieldConfig.batched("audio"),
                "audio_token_num": MultiModalFieldConfig.batched("audio"),
                "audio_token_id": MultiModalFieldConfig.shared("audio", num_audios),
            })
        
        return config

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        # Get updates from parent class (for images)
        updates = list(super()._get_prompt_updates(
            mm_items, hf_processor_mm_kwargs, out_mm_kwargs))

        # Add audio updates
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        if "audio_token_num" in out_mm_kwargs:
            audio_token_num = out_mm_kwargs["audio_token_num"]
            assert isinstance(audio_token_num, torch.Tensor)
            audio_token_num = audio_token_num.tolist()
        else:
            audio_token_num = []

        def get_replacement_audio(item_idx: int):
            feature_size = audio_token_num[item_idx]
            
            return PromptUpdateDetails(
                full=hf_processor.get_audio_repl_full(feature_size, None),
                features=hf_processor.get_audio_repl_features(feature_size, None),
            )

        updates.append(
            PromptReplacement(
                modality="audio",
                target="<audio>",
                replacement=get_replacement_audio,
            )
        )
        
        return updates

class InternVLAudioProcessingInfo(BaseInternVLProcessingInfo):
    def get_hf_processor(
        self,
        *,
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        **kwargs: object,
    ) -> InternVLAudioProcessor:
        if min_dynamic_patch is not None:
            kwargs["min_dynamic_patch"] = min_dynamic_patch
        if max_dynamic_patch is not None:
            kwargs["max_dynamic_patch"] = max_dynamic_patch
        if dynamic_image_size is not None:
            kwargs["dynamic_image_size"] = dynamic_image_size

        return self.ctx.init_processor(
            InternVLAudioProcessor,
            config=self.get_hf_config(),
            tokenizer=self.get_tokenizer(),
            **kwargs,
        )

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        # Override to support both image and audio
        return {"image": None, "audio": None}
    
    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        # Override to include both image and audio tokens
        return {
            "image": self.get_max_image_tokens(),
            "audio": self.get_max_audio_tokens()
        }
    
    def get_max_audio_tokens(self) -> int:
        # Implement logic to determine maximum audio tokens
        # This could be based on a fixed maximum or calculated from your audio model
        return 750  # Set an appropriate value for your model


@MULTIMODAL_REGISTRY.register_processor(
    InternVLAudioMultiModalProcessor,
    info=InternVLAudioProcessingInfo,
    dummy_inputs=InternVLAudioDummyInputsBuilder)
class InternVLChatAudioModel(nn.Module, SupportsMultiModal, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # Ensure audio support is initialized
        _ensure_audio_support()
        
        super().__init__()
        
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self._patch_quant_config(config, quant_config)
        
        # Initialize vision model like InternVLChatModel does
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.num_image_token = int(
            (image_size // patch_size)**2 * (config.downsample_ratio**2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version

        self.llm_arch_name = config.text_config.architectures[0]
        self.is_mono = self.llm_arch_name == 'InternLM2VEForCausalLM'
        self.vision_model = self._init_vision_model(
            config,
            quant_config=quant_config,
            is_mono=self.is_mono,
            prefix=maybe_prefix(prefix, "vision_model"),
        )

        # Initialize audio model
        self.audio_model = self._init_audio_model(
            config,
            quant_config=quant_config,
            is_mono=self.is_mono,
            prefix=maybe_prefix(prefix, "audio_model"),
        )
        
        # Language model
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        # Projection layers
        self.mlp1 = self._init_mlp1(config)  # For vision
        self.mlp2 = self._init_mlp2(config)  # For audio
        
        # Token IDs and masks
        self.img_context_token_id = None
        self.audio_context_token_id = None
        self.visual_token_mask = None  # Keep this
        
        # Required for compatibility
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    def _patch_quant_config(self, config: PretrainedConfig,
                            quant_config: QuantizationConfig):
        # the awq models from OpenGVLab missing modules_to_not_convert
        # patch the quant_config to add modules_to_not_convert back
        if isinstance(quant_config, AWQConfig):
            text_config = config.text_config
            llm_quant_config = getattr(text_config, "quantization_config",
                                       None)
            if (not quant_config.modules_to_not_convert) and \
                (llm_quant_config is not None):
                quant_config.modules_to_not_convert.append("vision_model")
                quant_config.modules_to_not_convert.append("audio_model") # CRITICAL: Also exclude audio_model
    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return get_sampler()

    def _init_vision_model(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        *,
        is_mono: bool,
        prefix: str,
    ):
        if not is_mono:
            vision_feature_layer = config.select_layer
            if vision_feature_layer < 0:
                num_hidden_layers = config.vision_config.num_hidden_layers \
                    + vision_feature_layer + 1
            else:
                num_hidden_layers = vision_feature_layer + 1

            return InternVisionModel(
                config.vision_config,
                quant_config=quant_config,
                num_hidden_layers_override=num_hidden_layers,
                prefix=prefix,
            )
        else:
            return InternVisionPatchModel(config.vision_config)

    def _init_audio_model(self, config: PretrainedConfig,
                          quant_config: Optional[QuantizationConfig],
                          *,
                          is_mono: bool,
                          prefix: str):
        # Initialize your audio model (Whisper based)
        from vllm.model_executor.models.internavl_whisper import AudioWhisperModel
        return AudioWhisperModel(config.audio_config)
    
    def _init_mlp1(self, config: PretrainedConfig) -> nn.Sequential:
        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.text_config.hidden_size

        return nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio)**2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio)**2,
                      llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def _init_mlp2(self, config: PretrainedConfig) -> nn.Sequential:
        audio_hidden_size = config.audio_config.hidden_size
        llm_hidden_size = config.text_config.hidden_size

        return nn.Sequential(
            nn.LayerNorm(audio_hidden_size),
            nn.Linear(audio_hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )
    
    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            pass
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vit_embeds = self.vision_model(pixel_values=pixel_values)
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1]**0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds,
                                        scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1,
                                        vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:

        h = w = self.config.vision_config.image_size
        expected_dims = (3, h, w)

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape)

            if actual_dims != expected_dims:
                expected_expr = str(expected_dims)
                raise ValueError(
                    "The expected shape of pixel values per image per batch "
                    f" per patch is {expected_expr}. "
                    f"You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[InternVLImageInputs]:
        pixel_values_flat = kwargs.pop("pixel_values_flat", None)
        image_num_patches = kwargs.pop("image_num_patches", None)
        image_embeds = kwargs.pop("image_embeds", None)
        #print("pixel_values_flat: ", pixel_values_flat)
        print("image_num_patches: ", image_num_patches)
        print("image_embeds: ", image_embeds)
        if pixel_values_flat is None and image_embeds is None:
            return None

        if image_embeds is not None:
            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")

            return InternVLImageEmbeddingInputs(
                type="image_embeds",
                data=flatten_bn(image_embeds),
            )

        image_token_id = kwargs["image_token_id"]
        assert isinstance(image_token_id, torch.Tensor)
        self.img_context_token_id = image_token_id.flatten().unique().item()
        print(f"INSIDE parse_and_validate_image_input: img_context_token_id: {self.img_context_token_id}")

        if pixel_values_flat is not None:
            if not isinstance(pixel_values_flat, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values_flat)}")

            assert isinstance(image_num_patches, (torch.Tensor, list))

            return InternVLImagePixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(
                    flatten_bn(pixel_values_flat, concat=True)),
                patches_per_image=flatten_bn(image_num_patches,
                                             concat=True).tolist())

        raise AssertionError("This line should be unreachable.")

    def _process_image_input(
        self,
        image_input: InternVLImageInputs,
    ) -> tuple[torch.Tensor, ...]:
        if image_input["type"] == "image_embeds":
            return image_input["data"]

        assert self.vision_model is not None

        image_embeds = self.extract_feature(image_input["data"])

        patches_per_image = image_input["patches_per_image"]

        # Only one image in the current batch
        if len(patches_per_image) == 1:
            image_embeds = image_embeds.view(
                -1, self.config.text_config.hidden_size).unsqueeze(0)
            return image_embeds

        # NOTE: Image embeddings are split into separate tensors for each image
        # by the size of each embedding.
        feature_size = image_embeds.shape[1]
        image_embeds = image_embeds.view(-1,
                                         self.config.text_config.hidden_size)
        image_feature_sizes = [
            num_patches * feature_size for num_patches in patches_per_image
        ]
        image_embeds = image_embeds.split(image_feature_sizes)
        return image_embeds

    def _set_visual_token_mask(self, input_ids: torch.Tensor) -> None:
        if self.is_mono:
            self.visual_token_mask = (
                input_ids == self.img_context_token_id).reshape(-1, 1)
        else:
            self.visual_token_mask = None


    def extract_audio_feature(self, audio_values, audio_len_after_cnn):
        print("EXTRACT AUDIO FEATURE")
        print("--------------------------------")
        
        # 🧠 Force to [B, C, T]
        while audio_values.dim() > 3:
            audio_values = audio_values.squeeze(1)
        
        print("audio_values after squeeze:", audio_values.shape)

        max_len_in_batch = int(torch.max(audio_len_after_cnn).item())
        padding_mask = torch.ones([audio_values.size(0), max_len_in_batch], dtype=audio_values.dtype, device=audio_values.device)
        for i in range(len(audio_values)):
            padding_mask[i, :audio_len_after_cnn[i].item()] = 0

        audio_values = audio_values.to(dtype=self.audio_model.dtype)

        print("goes to audio model")
        print("padding_mask", padding_mask.shape)
        print("audio_values", audio_values.shape)
        print("audio_len_after_cnn", audio_len_after_cnn)

        last_hidden_state = self.audio_model(
            input_features=audio_values,
            attention_mask=padding_mask,
            audio_len_after_cnn=audio_len_after_cnn
        )

        audio_embeds = self.mlp2(last_hidden_state)
        print("audio_embeds", audio_embeds.shape)
        return audio_embeds


    
    def _parse_and_validate_audio_input(self, **kwargs) -> Optional[InternVLAudioInputs]:
        audio_features = kwargs.get("audio_features", None)
        audio_len_after_cnn = kwargs.get("audio_len_after_cnn", None)
        audio_token_num = kwargs.get("audio_token_num", None)
        
        if audio_features is None:
            return None
        
        # Get the audio token ID
        audio_token_id = kwargs.get("audio_token_id", None)
        if audio_token_id is not None:
            if isinstance(audio_token_id, torch.Tensor):
                self.audio_context_token_id = audio_token_id.flatten().unique()[0].item()
            else:
                self.audio_context_token_id = audio_token_id

        
        return InternVLAudioInputs(
            type="audio_features",
            data=audio_features,
            audio_len_after_cnn=audio_len_after_cnn,
            audio_token_num=audio_token_num,
        )

    def _process_audio_input(self, audio_input: InternVLAudioInputs) -> Union[List[torch.Tensor], torch.Tensor]:
        audio_values = audio_input["data"]
        audio_len_after_cnn = audio_input["audio_len_after_cnn"]
        audio_token_num = audio_input["audio_token_num"]

        audio_embeds = self.extract_audio_feature(audio_values, audio_len_after_cnn)  # [B, T, D]

        # Convert token counts to Python ints
        token_nums = [n.item() if isinstance(n, torch.Tensor) else int(n) for n in audio_token_num]

        if audio_embeds.ndim == 3 and audio_embeds.shape[0] > 1:
            return [audio_embeds[i, :token_nums[i]] for i in range(len(token_nums))]
        else:
            return audio_embeds[0, :token_nums[0]]


    
    def get_multimodal_embeddings(self, **kwargs) -> Optional[tuple[NestedTensors, str]]:
        # First try to get image input
        image_input = self._parse_and_validate_image_input(**kwargs)
        
        # Then try to get audio input 
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        
        if image_input is not None:
            print("[Model] Got image input, but skipping it for audio-only testing.")
            return self._process_image_input(image_input), "image"
        elif audio_input is not None:
            print("[Model] Processing audio input for embedding.")
            audio_embeds = self._process_audio_input(audio_input)
            print("[Model] Audio embeddings shape:", audio_embeds[0].shape if isinstance(audio_embeds, list) else audio_embeds.shape)
            return audio_embeds, "audio"
        
        # Return a tuple of (None, None) instead of just None
        return None, None
            
    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
        multimodal_embeddings_text: Optional[str] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        print("input_ids.shape", input_ids.shape)
        if isinstance(multimodal_embeddings, list):
            print("audio embeddings:", [e.shape for e in multimodal_embeddings])
        elif isinstance(multimodal_embeddings, torch.Tensor):
            print("audio embeddings:", multimodal_embeddings.shape)
        else:
            print("audio embeddings:", type(multimodal_embeddings))
        if self.audio_context_token_id is not None:
            audio_mask = input_ids == self.audio_context_token_id
            print("num AUDIO_CONTEXT tokens in input_ids:", audio_mask.sum().item())
        if multimodal_embeddings is not None and multimodal_embeddings_text == "image":
            assert self.img_context_token_id is not None
            self._set_visual_token_mask(input_ids)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.img_context_token_id)
        elif multimodal_embeddings is not None and multimodal_embeddings_text == "audio":
            assert self.audio_context_token_id is not None
            token_mask = (input_ids == self.audio_context_token_id)
            print("[Embedding] Audio context token ID:", self.audio_context_token_id)
            # print("[Embedding] Token mask for audio_context_token_id:", token_mask.nonzero())
            print("[Embedding] Num tokens to replace:", token_mask.sum().item())

            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.audio_context_token_id)
        return inputs_embeds

    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[SamplerOutput, IntermediateTensors]:

        if intermediate_tensors is not None:
            input_ids = None
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            multimodal_embeddings, multimodal_embeddings_text = self.get_multimodal_embeddings(**kwargs)
            # Only proceed with embeddings if they're not None
            if multimodal_embeddings is not None:
                inputs_embeds = self.get_input_embeddings(input_ids, multimodal_embeddings, multimodal_embeddings_text)
                input_ids = None
            else:
                inputs_embeds = self.language_model.get_input_embeddings(input_ids)
                input_ids = None

        forward_kwargs = {
            "input_ids": input_ids,
            "positions": positions,
            "intermediate_tensors": intermediate_tensors,
            "inputs_embeds": inputs_embeds,
        }

        # Only required if the model is mono-architecture
        if self.visual_token_mask is not None:
            forward_kwargs.update(
                {"visual_token_mask": self.visual_token_mask})
            self.visual_token_mask = None

        hidden_states = self.language_model.model(**forward_kwargs)
        return hidden_states

    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states, sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.language_model.sample(logits, sampling_metadata)
    
    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

# Initialize audio support function
def _ensure_audio_support():
    """Ensure audio support is initialized"""
    try:
        # Lazily import the audio parsers module
        audio_parsers = import_module('vllm.multimodal.internavl_audio_parsers')
        # Call the registration function
        if hasattr(audio_parsers, 'register_audio_parsers'):
            audio_parsers.register_audio_parsers()
    except ImportError:
        # Audio support not available, but that's okay
        pass