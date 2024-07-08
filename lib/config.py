from dataclasses import dataclass

@dataclass
class ScriptArguments:
    model_id: str = "meta-llama/Meta-Llama-3-8B"
    from_pretrained: bool = False
    random_weights: bool = False
    use_lora: bool = False
    num_proc: int = 24

    n_streams: int = 1
    merge_layer: int = -1
    freeze_transformer: bool = False
    layers_attribute: str = "layers"
    model_attribute: str = "model"
    
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    block_size: int = 512
