from categorical_embedder.generators.outer.PandasGenerator import PandasGenerator
from categorical_embedder.embedders.vae.Trainer import Trainer
from categorical_embedder.embedders.core.training.Embedder import Embedder

# Here is an example embedding_info for categoricals (assuming the data is composed of only categorical columns).
# In the example, lstm encoder-decoder based vae is implemented (No NSP this time!).
# Holistic Categorical Embedding is implemented. Vocabulary is composed of the categories over all the
# categorical columns; also "unknown" categories are added for each categorical columns to handle missing and
# new/unseen values.

# For flexibility, we can build encoders and decoders from embedding info.
# Currently a few custom and keras layers are supported (Add new ones with one-liners in layer_creator.py)
# In this example, lstm-based encoder-decoder is implemented. However, since we are not using NSP,
# the embedding vector is repeated before inputing to the decoder.
# SelectorLayer can be used when previous layer returns more than 1 tensor and we need just one of them.
# hidden_length is for the hidden layer which we sample from (not exactly, see the code VAEEncoder)~
# In VAE, we have implicit losses. It should be set True when the loss is computed in call method (e.g. with add_loss).
# The loss is composed of reconstruction loss and discrete latent loss (we should weight them).
# We may also add a continuous counterpart for embedding.
# Discrete embedding is implemented as "concrete" sampling based on gumbel distribution.


# num_of_cols = 16
# vocab_size = 61
# embedding_length = 10
#
# experiment_info = {
#     "generator_info": {
#         "pass_count": None,
#         "use_remaining": False
#     },
#     "model_info": {
#         "hidden_length": 10,
#         "has_implicit_loss": True,
#         "optimizer": "adam",
#         "metrics": None,
#         "loss_info": None,
#         "inner_loss_info": {
#             "reconstruction_loss": {
#                 "type": "cross_entropy",
#                 "parameters": {}
#             },
#             "z_loss_weight": 0.0,
#             "c_loss_weight": 0.5,
#             "r_loss_weight": 0.5
#         },
#         "encoder_latent_info": {
#             "temperature": 0.95,
#             "continuous_latent_length": None,
#             "discrete_latent_length": embedding_length
#         },
#         "encoder_layer_info": [
#             {
#                 "type": "StackedRecurrentEncoder",
#                 "parameters": {
#                     "units": embedding_length,
#                     "num_of_layers": 1,
#                     "recurrent_type": "LSTM",
#                     "recurrent_parameters": {"activation": "selu", "recurrent_activation": "tanh"},
#                     "should_normalize": True
#                 }
#             },
#             {
#                 "type": "SelectorLayer",
#                 "parameters": {
#                     "index": 0
#                 }
#             }
#         ],
#         "decoder_layer_info": [
#             {
#                 "type": "RepeatVector",
#                 "parameters": {
#                     "n": num_of_cols
#                 }
#             },
#             {
#                 "type": "StackedRecurrentDecoder",
#                 "parameters": {
#                     "units": embedding_length,
#                     "num_of_layers": 1,
#                     "recurrent_type": "LSTM",
#                     "recurrent_parameters": {"activation": "selu", "recurrent_activation": "tanh"}
#                 }
#             },
#             {
#                 "type": "TDD",
#                 "parameters": {
#                     "units": vocab_size,
#                     "activation": "softmax"
#                 }
#             }
#         ],
#         "fit_parameters": {"epochs": 100}
#     },
#     "save_info": {
#         "main_model_path": ".../main_model.h5",
#         "embedder_model_path": ".../embedder_model.h5",
#         "main_model_artifacts_path": ".../main_model_artifacts.pkl",
#         "embedder_artifacts_path": ".../embedder_model_artifacts.pkl"
#     }
# }


def train_and_embed(cat_data, batch_size, num_of_categories, uniques, embedding_info):
    # cat_data: pandas dataframe representing a list of categorical columns.

    train_embedder(cat_data, batch_size, num_of_categories, uniques, embedding_info)
    embedder = Embedder(embedding_info["save_info"]["embedder_model_path"], embedding_info["save_info"]["embedder_artifacts_path"])
    embeddings = embed(embedder, cat_data)
    return embeddings


def train_embedder(cat_data, batch_size, num_of_categories, uniques, embedding_info):
    # cat_data: pandas dataframe representing a list of categorical columns.

    # Create a generator from the given pandas dataframe (We may create custom outer generators from any source).
    # All the embedders use InnerGenerator which is a definite interface between the outer generator and the embedders.
    outer_generator = PandasGenerator(cat_data, batch_size)
    trainer = Trainer(num_of_categories, uniques, outer_generator, embedding_info["generator_info"], embedding_info["model_info"], embedding_info["save_info"])
    embedder_model, main_model = trainer.train()
    return embedder_model, main_model


def embed(embedder, cat_data):
    # cat_data: pandas dataframe representing a list of categorical columns.

    embeddings = embedder.embed(cat_data.to_numpy())
    return embeddings
