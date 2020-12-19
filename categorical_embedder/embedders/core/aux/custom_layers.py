from categorical_embedder.embedders.core.seq2seq.EncoderDecoder import EncoderDecoder
from categorical_embedder.embedders.core.seq2seq.StackedRecurrentEncoder import StackedRecurrentEncoder
from categorical_embedder.embedders.core.seq2seq.StackedRecurrentDecoder import StackedRecurrentDecoder
from categorical_embedder.embedders.core.vae.concrete.VAE import VAE
from categorical_embedder.embedders.core.vae.concrete.VAEEncoder import VAEEncoder
from categorical_embedder.embedders.core.vae.concrete.VAEDecoder import VAEDecoder


def get_custom_layer_class(layer_name):
    return _custom_layer_mappings[layer_name]


_custom_layer_mappings = {
    "EncoderDecoder": EncoderDecoder,
    "StackedRecurrentEncoder": StackedRecurrentEncoder,
    "StackedRecurrentDecoder": StackedRecurrentDecoder,
    "VAE": VAE,
    "VAEEncoder": VAEEncoder,
    "VAEDecoder": VAEDecoder
}