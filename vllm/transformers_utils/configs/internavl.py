from transformers.configuration_utils import PretrainedConfig
from vllm.transformers_utils.configs.whisper_audio import WhisperConfig

class InternVLChatAudioConfig(PretrainedConfig):
    model_type = "internvl_chat_audio"
    is_composition = True
    
    def __init__(
            self,
            vision_config=None,
            audio_config=None,
            llm_config=None,
            use_backbone_lora=0,
            use_llm_lora=0,
            select_layer=-1,
            force_image_size=None,
            downsample_ratio=0.5,
            template=None,
            dynamic_image_size=False,
            use_thumbnail=False,
            ps_version='v1',
            min_dynamic_patch=1,
            max_dynamic_patch=6,
            **kwargs):
        super().__init__(**kwargs)

        # Initialize configs with empty dicts if None
        if vision_config is None:
            vision_config = {}
        if llm_config is None:
            llm_config = {}
        if audio_config is None:
            audio_config = {}

        # Create the needed configs
        self.vision_config = PretrainedConfig(**vision_config)
        self.text_config = PretrainedConfig(**llm_config)
        self.audio_config = WhisperConfig(**audio_config)
        
        # Set InternVL specific parameters
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.ps_version = ps_version
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch

    # def to_dict(self):
    #     """
    #     Serializes this instance to a Python dictionary.
    #     """
    #     output = super().to_dict()
    #     output['vision_config'] = self.vision_config.to_dict()
    #     output['text_config'] = self.text_config.to_dict()
    #     output['audio_config'] = self.audio_config.to_dict()
    #     return output