from .model_nanotabvlm import *
from typing import Optional, Tuple, List
from torch import nn
from .vision_model.repvit.repvit import repvit_m1_5
from .vision_model.repvit.processor import get_processor


warnings.filterwarnings('ignore')


class RepVitVisionProj(nn.Module):
    def __init__(self, ve_hidden_size=512, hidden_size=512):
        from .vision_model.repvit.repvit import LayerNorm2d
        super().__init__()
        self.ve_hidden_size = ve_hidden_size
        self.hidden_size = hidden_size
        self.vision_proj = nn.Sequential(
            nn.Conv2d(
                ve_hidden_size,
                hidden_size,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(hidden_size),
            nn.Conv2d(
                hidden_size,
                hidden_size,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(hidden_size),
        )

    def forward(self, image_encoders):
        vision_projs = []
        bs, num, c, im_h, im_w = image_encoders.shape
        for i in range(num):
            vision_proj = self.vision_proj(image_encoders[:, i, :, :, :])
            vision_proj = vision_proj.view(bs, self.hidden_size, -1)
            vision_proj = vision_proj.permute(0, 2, 1).contiguous()
            vision_projs.append(vision_proj)
        stack_dim = 1 if bs > 1 else 0
        vision_projs = torch.stack(vision_projs, dim=stack_dim)
        return vision_projs


# 继承自语言模型
class NanoTabVLMV2(NanoTabVLM):
    config_class = NanoTabVLMConfig

    def __init__(self, params: NanoTabVLMConfig = None, **kwargs):
        super().__init__(params)
        if not params:
            params = NanoTabVLMConfig()
        self.params = params
        self.vision_encoder = repvit_m1_5()
        self.processor = get_processor(**kwargs)
        self.vision_proj = RepVitVisionProj(hidden_size=params.hidden_size)

    @staticmethod
    def image2tensor(image, processor):
        if image.mode in ['RGBA', 'LA']:
            image = image.convert('RGB')
        inputs = processor(image)
        return inputs

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                pixel_values: Optional[torch.FloatTensor] = None,
                **args):
        batch_size, seq_length = input_ids.shape

        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        if pixel_values is not None and start_pos == 0:
            if len(pixel_values.shape) == 6:
                pixel_values = pixel_values.squeeze(2)
            bs, num, c, im_h, im_w = pixel_values.shape
            stack_dim = 1 if bs > 1 else 0
            vision_tensors = torch.stack([
                self.vision_encoder(pixel_values[:, i, :, :, :])
                for i in range(num)
            ], dim=stack_dim)
            hidden_states = self.count_vision_proj(tokens=input_ids, h=hidden_states, vision_tensors=vision_tensors,
                                                   seqlen=input_ids.shape[1])
        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)

        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.model.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        self.OUT.__setitem__('last_hidden_state', hidden_states)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', presents)
        return self.OUT

