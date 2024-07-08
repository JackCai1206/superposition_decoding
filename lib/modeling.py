import torch
from torch import Tensor, nn
from jaxtyping import Float, Int
from transformers import BertConfig, PreTrainedModel, Cache, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.bert.modeling_bert import BertEncoder
from typing import List, Literal, Optional, Sequence, Tuple, Union, cast
from tqdm import tqdm
from threading import Thread

from .config import ScriptArguments

import functools

# TODO design choices are hardcoded for now, customize later
class MultiplexedModelConfig(PretrainedConfig):
    merge_layer: int = -1
    n_streams: int = 1
    freeze_transformer: bool = False
    layers_attribute: str = "layers"
    model_attribute: str = "model"
    hidden_size: int = None

class Multiplexer(nn.Module):
    def __init__(self, n_streams, d_model):
        super().__init__()
        self.n_streams = n_streams
        self.rand_gaussians = nn.ParameterList([nn.Parameter(torch.randn(d_model)) for _ in range(n_streams)])
        self.transformer = BertEncoder(BertConfig(hidden_size=d_model, num_hidden_layers=2, num_attention_heads=4))

    def forward(self, x: List[Tensor]):
        x = [xi * phi for xi, phi in zip(x, self.rand_gaussians)]
        x = torch.stack(x)
        N, B, L, D = x.shape
        x = x.swapaxes(0, 2).reshape(L * B, N, D) # combine seq_len and batch
        x = self.transformer(x, output_hidden_states=True).last_hidden_state
        x = x.reshape(L, B, N, D).swapaxes(0, 2).mean(dim=0)
        return x

class Demultiplexer(nn.Module):
    def __init__(self, n_streams, d_model):
        super().__init__()
        self.n_streams = n_streams
        dim_key = d_model // 4
        self.keys = nn.ParameterList([nn.Parameter(torch.randn(dim_key)) for _ in range(n_streams)])
        self.MLP = nn.Sequential(
            nn.Linear(d_model+dim_key, d_model+dim_key),
            nn.GELU(),
            nn.Linear(d_model+dim_key, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x):
        x = torch.stack([torch.cat([x.clone(), key.expand(x.shape[0], x.shape[1], -1)], dim=-1) for key in self.keys])
        x = self.MLP(x)
        return x

class MultiplexedModel(PreTrainedModel):
    def __init__(self, args: MultiplexedModelConfig, tf_lm_model: PreTrainedModel):
        super().__init__(args)
        self.config = args
        self.n_streams = args.n_streams
        self.tf_model = getattr(tf_lm_model, args.model_attribute)
        self.pre_merge_layers = getattr(self.tf_model, args.layers_attribute)[:args.merge_layer]
        self.post_merge_layers = getattr(self.tf_model, args.layers_attribute)[args.merge_layer:]
        self.lm_head = getattr(tf_lm_model, "lm_head")
        self.merge_layer = args.merge_layer
        self.multiplexer = Multiplexer(args.n_streams, args.hidden_size)
        self.demultiplexer = Demultiplexer(args.n_streams, args.hidden_size)
        self.loss_fn = nn.CrossEntropyLoss()
        if args.freeze_transformer:
            self.tf_model.requires_grad_(False)
    
    def forward(
        self, input_ids: torch.LongTensor = None,
        attention_mask: Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | List[torch.FloatTensor] | None = None, 
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None
    ) -> Tuple | CausalLMOutputWithPast:
        if past_key_values is not None:
            raise NotImplementedError("Past key-value caches are not yet supported.")
        
        # combine n_streams and batch dimensions
        # input_ids = input_ids.view(-1, *input_ids.shape[2:])
        
        # Individually process until merge layer
        setattr(self.tf_model, self.config.layers_attribute, self.pre_merge_layers)
        resids = [self.tf_model.forward(input_ids[i], attention_mask=attention_mask[i])[0] for i in range(self.n_streams)]

        resid_comb = self.multiplexer(resids)
        if attention_mask is not None:
            attention_mask_comb = attention_mask.prod(dim=0) # different streams may have different left padding lengths, but since we are combining the streams, we have to multiply the attention masks - Is there a better way to do this?
        else:
            attention_mask_comb = None

        # Process together after merge layer
        setattr(self.tf_model, self.config.layers_attribute, self.post_merge_layers)
        last_resid_comb = self.tf_model.forward(inputs_embeds=resid_comb, attention_mask=attention_mask_comb)

        last_resid = self.demultiplexer(last_resid_comb)

        logits = self.lm_head(last_resid[i])
        loss = None
        if labels is not None:
            losses = []
            for i in range(self.n_streams):
                # Shift so that tokens < n predict n
                shift_logits = logits[i][..., :-1, :].contiguous()
                shift_labels = labels[i][..., 1:].contiguous()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, self.tf_model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = self.loss_fn(shift_logits, shift_labels)
                losses.append(loss)
            loss = sum(losses) / self.n_streams

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
    
    # def generate(self, inputs: Tensor | None = None, generation_config: GenerationConfig | None = None, logits_processor: LogitsProcessorList | None = None, stopping_criteria: StoppingCriteriaList | None = None, prefix_allowed_tokens_fn: Callable[[int, Tensor], List[int]] | None = None, synced_gpus: bool | None = None, assistant_model: PreTrainedModel | None = None, streamer: BaseStreamer | None = None, negative_prompt_ids: Tensor | None = None, negative_prompt_attention_mask: Tensor | None = None, **kwargs) -> GenerateDecoderOnlyOutput | GenerateEncoderDecoderOutput | GenerateBeamDecoderOnlyOutput | GenerateBeamEncoderDecoderOutput | torch.LongTensor:
    #     output =  super().generate(inputs, generation_config, logits_processor, stopping_criteria, prefix_allowed_tokens_fn, synced_gpus, assistant_model, streamer, negative_prompt_ids, negative_prompt_attention_mask, **kwargs)

