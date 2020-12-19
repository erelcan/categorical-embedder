from categorical_embedder.embedders.core.aux.custom_layers import get_custom_layer_class
from categorical_embedder.embedders.core.aux.loss_factory import get_loss_function


def prepare_custom_objects(custom_object_info):
    custom_objects = {}
    custom_objects.update(_prepare_custom_layers(custom_object_info["layer_info"]))
    if not custom_object_info["has_implicit_loss"]:
        custom_objects.update(_prepare_custom_loss(custom_object_info["loss_info"]))
    return custom_objects


def _prepare_custom_layers(layer_info):
    custom_layers = {}
    for layer_name in layer_info:
        custom_layers[layer_name] = get_custom_layer_class(layer_name)
    return custom_layers


def _prepare_custom_loss(loss_info):
    return {"loss": get_loss_function(loss_info)}
