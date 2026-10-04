SEED_OPTIONS = (2, 4, 6)
DEFAULT_LAYER_CFG = dict(type="dense")
TARGET_LAYERS_FS = ("src_embed", "self_attn", "feed_forward")
TARGET_LAYERS_FT = ("self_attn", "feed_forward")

CONFIG_KEYS = {
    "src_embed": "embed_config_linear",
    "self_attn": "att_config_linear",
    "feed_forward": "ff_config_linear",
}
