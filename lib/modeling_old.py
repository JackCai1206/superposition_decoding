import torch
from torch import Tensor, nn
from jaxtyping import Float, Int
from transformer_lens.utilities import devices
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache
import transformer_lens.utils as utils
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
from typing import List, Literal, Optional, Sequence, Tuple, Union
from transformer_lens import HookedTransformer, HookedTransformerConfig
from tqdm import tqdm

from .config import ScriptArguments

class Multiplexer(nn.Module):
    def __init__(self, n_streams, d_model):
        super().__init__()
        self.n_streams = n_streams
        # self.phis = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_streams)])
        # for i, phi in enumerate(self.phis):
        #     # nn.init.orthogonal_(phi.weight)
        #     phi.weight.data = torch.diag(self.rand_gaussians[i]) # not trainable
        # rand_gaussians = torch.stack([torch.randn(d_model) for _ in range(n_streams)])
        # self.register_buffer('rand_gaussians', rand_gaussians)
        self.rand_gaussians = nn.ParameterList([nn.Parameter(torch.randn(d_model)) for _ in range(n_streams)])
        self.transformer = BertEncoder(BertConfig(hidden_size=d_model, num_hidden_layers=2, num_attention_heads=4))

    def forward(self, x: List[Tensor]):
        # assert len(x) == self.n_streams
        # x = [xx.to(self.phis[0].weight.data.dtype) for xx in x]
        # x = [phi(xi) for xi, phi in zip(x, self.phis)]
        x = [xi * phi for xi, phi in zip(x, self.rand_gaussians)]
        x = torch.stack(x)
        N, B, L, D = x.shape
        x = x.swapaxes(0, 2).reshape(L * B, N, D) # combine seq_len and batch
        x = self.transformer(x, output_hidden_states=True).last_hidden_state
        x = x.reshape(L, B, N, D).swapaxes(0, 2).mean(dim=0)
        return x
        # return x[0]

class Demultiplexer(nn.Module):
    def __init__(self, n_streams, d_model):
        super().__init__()
        self.n_streams = n_streams
        # self.phis = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_streams)])
        # for phi in self.phis:
            # nn.init.orthogonal_(phi.weight)
        dim_key = d_model // 4
        self.keys = nn.ParameterList([nn.Parameter(torch.randn(dim_key)) for _ in range(n_streams)])
        self.MLP = nn.Sequential(
            nn.Linear(d_model+dim_key, d_model+dim_key),
            nn.GELU(),
            nn.Linear(d_model+dim_key, d_model),
            nn.LayerNorm(d_model)
        )
        # self.MLP = nn.Identity(d_model, d_model)

    def forward(self, x):
        # x = x.to(self.phis[0].weight.data.dtype)
        # x = [phi(x) for phi in self.phis]
        x = [torch.cat([x.clone(), key.expand(x.shape[0], x.shape[1], -1)], dim=-1) for key in self.keys]
        # x = [x for key in self.keys]
        x = [self.MLP(xi) for xi in x]
        return x
        # return [x]

class MultiplexedModel(nn.Module):
    def __init__(self, model: HookedTransformer, args: ScriptArguments):
        super().__init__()
        self.n_streams = args.n_streams
        self.model = model
        self.merge_layer = args.merge_layer
        if args.freeze_transformer:
            model.requires_grad_(False)
        self.multiplexer = Multiplexer(args.n_streams, model.cfg.d_model)
        self.demultiplexer = Demultiplexer(args.n_streams, model.cfg.d_model)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask=None, labels=None, past_kv_caches=None, **kwargs):
        assert len(input_ids) == self.n_streams
        if past_kv_caches is not None:
            raise NotImplementedError("Past key-value caches are not yet supported.")
        # assert len(attention_mask) == self.n_streams

        # device = devices.get_device_for_block_index(0, self.model.cfg)
        input_ids = [input_id for input_id in input_ids]
        if attention_mask is not None:
            attention_mask = [mask for mask in attention_mask]
        else:
            attention_mask = [None for _ in input_ids]
        resids = [self.model.forward(input_ids[i], attention_mask=attention_mask[i], stop_at_layer=self.merge_layer, **kwargs) for i in range(self.n_streams)]

        resid_comb = self.multiplexer(resids)
        if attention_mask[0] is not None:
            attention_mask_comb = torch.stack(attention_mask).prod(dim=0)
        else:
            attention_mask_comb = None

        last_resid_comb = self.model.forward(resid_comb, attention_mask=attention_mask_comb, start_at_layer=self.merge_layer, stop_at_layer=self.model.cfg.n_layers, **kwargs)

        last_resid = self.demultiplexer(last_resid_comb)

        if labels is not None:
            logits_and_losses = [self.model.forward(last_resid[i], tokens=labels[i], start_at_layer=self.model.cfg.n_layers, return_type='both') for i in range(self.n_streams)]
            logits = [logits_and_losses[i][0] for i in range(self.n_streams)]
            losses = [logits_and_losses[i][1] for i in range(self.n_streams)]
            loss = sum(losses) / self.n_streams
        else:
            logits = [self.model.forward(last_resid[i], start_at_layer=self.model.cfg.n_layers, return_type='logits') for i in range(self.n_streams)]
            loss = None

        return (
            loss, 
            logits,
            labels
        )
    
    @torch.inference_mode()
    def generate(
        self,
        input: Tuple[Float[torch.Tensor, "batch pos"]] = "",
        max_new_tokens: int = 10,
        stop_at_eos: bool = True,
        eos_token_id: Optional[int] = None,
        do_sample: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: float = 1.0,
        freq_penalty: float = 0.0,
        use_past_kv_cache: bool = True,
        prepend_bos: Optional[bool] = True,
        padding_side: Optional[Literal["left", "right"]] = "right",
        verbose: bool = True,
    ) -> Tuple[Int[torch.Tensor, "batch pos_plus_new_tokens"]]:
        tokens = input
        if torch.is_tensor(tokens):
            assert tokens.ndim == 3 and tokens.shape[0] == self.n_streams
            tokens = tokens.unbind(0)
        else:
            assert len(tokens) == self.n_streams
        assert isinstance(tokens[0], torch.Tensor)
        batch_size, ctx_length = tokens[0].shape
        device = devices.get_device_for_block_index(0, self.model.cfg)
        tokens = [t.to(device) for t in tokens]
        if use_past_kv_cache:
            past_kv_caches = [HookedTransformerKeyValueCache.init_cache(
                self.model.cfg, self.model.cfg.device, batch_size
            ) for _ in range(self.n_streams)]
        else:
            past_kv_cache = None

        stop_tokens: List[int] = []
        eos_token_for_padding = 0
        assert self.model.tokenizer is not None
        if stop_at_eos:
            tokenizer_has_eos_token = (
                self.model.tokenizer is not None and self.model.tokenizer.eos_token_id is not None
            )
            if eos_token_id is None:
                assert (
                    tokenizer_has_eos_token
                ), "Must pass a eos_token_id if stop_at_eos is True and tokenizer is None or has no eos_token_id"

                eos_token_id = self.model.tokenizer.eos_token_id

            if isinstance(eos_token_id, int):
                stop_tokens = [eos_token_id]
                eos_token_for_padding = eos_token_id
            else:
                # eos_token_id is a Sequence (e.g. list or tuple)
                stop_tokens = eos_token_id
                eos_token_for_padding = (
                    self.model.tokenizer.eos_token_id if tokenizer_has_eos_token else eos_token_id[0]
                )

        # An array to track which sequences in the batch have finished.
        finished_sequences = [torch.zeros(batch_size, dtype=torch.bool, device=self.model.cfg.device) for _ in range(self.n_streams)]

        # Currently nothing in HookedTransformer changes with eval, but this is here in case
        # that changes in the future.
        self.eval()
        for index in tqdm.tqdm(range(max_new_tokens), disable=not verbose):
            # While generating, we keep generating logits, throw away all but the final logits,
            # and then use those logits to sample from the distribution We keep adding the
            # sampled tokens to the end of tokens.
            if use_past_kv_cache:
                # We just take the final tokens, as a [batch, 1] tensor
                if index > 0:
                    logits = self.forward(
                        [t[:, -1:] for t in tokens],
                        return_type="logits",
                        prepend_bos=prepend_bos,
                        padding_side=padding_side,
                        past_kv_caches=past_kv_caches,
                    )[1]
                else:
                    logits = self.forward(
                        tokens,
                        return_type="logits",
                        prepend_bos=prepend_bos,
                        padding_side=padding_side,
                        past_kv_caches=past_kv_caches,
                    )[1]
            else:
                # We input the entire sequence, as a [batch, pos] tensor, since we aren't using
                # the cache.
                logits = self.forward(
                    tokens,
                    return_type="logits",
                    prepend_bos=prepend_bos,
                    padding_side=padding_side,
                )[1]
            final_logits = [l[:, -1, :] for l in logits]

            if do_sample:
                sampled_tokens = [
                    utils.sample_logits(
                        fl,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        freq_penalty=freq_penalty,
                        tokens=tokens,
                    ).to(devices.get_device_for_block_index(0, self.model.cfg))
                for fl in final_logits]
            else:
                sampled_tokens = [
                    fl.argmax(-1).to(
                        devices.get_device_for_block_index(0, self.model.cfg)
                    )
                for fl in final_logits]

            if stop_at_eos:
                # For all unfinished sequences, add on the next token. If a sequence was
                # finished, throw away the generated token and add eos_token_for_padding
                # instead.
                for st, fs in zip(sampled_tokens, finished_sequences):
                    st[fs] = eos_token_for_padding
                for i, fs in enumerate(finished_sequences):
                    fs.logical_or_(
                        torch.isin(
                            sampled_tokens[i].to(self.model.cfg.device),
                            torch.tensor(stop_tokens).to(self.model.cfg.device),
                        )
                    )

            tokens = [torch.cat([tokens[i], sampled_tokens[i].unsqueeze(-1)], dim=-1) for i in range(self.n_streams)]

            if stop_at_eos and all([fs.all() for fs in finished_sequences]):
                break

        return tokens
