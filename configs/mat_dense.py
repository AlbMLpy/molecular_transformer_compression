from .shared import SEED_OPTIONS, DEFAULT_LAYER_CFG


MODEL_CONFIG = dict(
    N=2,
    d_model=128, # 128 or 256 
    h=8,
    N_dense=2,
    lambda_attention=0.33, 
    lambda_distance=0.33,
    leaky_relu_slope=0.1,
    dense_output_nonlinearity='relu', 
    distance_matrix_kernel='exp', 
    dropout=0.0,
    aggregation_type='mean',
    n_generator_layers=1,
    embed_config_linear=DEFAULT_LAYER_CFG,
    att_config_linear=DEFAULT_LAYER_CFG,
    ff_config_linear=DEFAULT_LAYER_CFG,
)


def get_config_mat_dense():
    return dict(
        model_config=MODEL_CONFIG,
        seed_options=SEED_OPTIONS,
        batch_size=32, # 16 or 32
        epochs=100,
        lr=1e-3,
        patience=60,
    )
