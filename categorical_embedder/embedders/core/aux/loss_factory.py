from keras import losses
from keras import backend as K


def get_loss_function(loss_info):
    if "discriminative" in loss_info:
        loss_dict = {}
        for name, info in loss_info.items():
            loss_dict[name] = _create_loss_function(info)
        return loss_dict
    else:
        return _create_loss_function(loss_info)


def _create_loss_function(loss_info):
    if loss_info is None:
        return None
    else:
        if "class_weights" in loss_info:
            return _class_weight_wrapper(loss_info["class_weights"], _loss_functions[loss_info["type"]](loss_info["parameters"]))
        else:
            return _loss_functions[loss_info["type"]](loss_info["parameters"])


def _get_cross_entropy_loss(from_logits=False, label_smoothing=0):
    return lambda y_true, y_pred: losses.categorical_crossentropy(y_true, y_pred, from_logits, label_smoothing)


def _get_binary_cross_entropy_loss(from_logits=False, label_smoothing=0):
    return lambda y_true, y_pred: losses.binary_crossentropy(y_true, y_pred, from_logits, label_smoothing)


def _get_sparse_categorical_crossentropy_loss(from_logits=False, axis=-1):
    return lambda y_true, y_pred: losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits, axis)


def _get_cosine_similarity_loss(axis=-1, squeeze=False):
    def inner_func(y_true, y_pred):
        if squeeze:
            return -losses.cosine_similarity(K.cast_to_floatx(K.squeeze(y_true, -1)), K.squeeze(y_pred, -1), axis=axis)
        else:
            return -losses.cosine_similarity(K.cast_to_floatx(y_true), y_pred, axis=axis)

    return inner_func


def _get_kl_divergence_loss():
    return lambda y_true, y_pred: losses.kl_divergence(y_true, y_pred)


def _get_log_cosh_loss():
    return lambda y_true, y_pred: losses.log_cosh(y_true, y_pred)


def _get_mean_absolute_percentage_error_loss():
    return lambda y_true, y_pred: losses.mean_absolute_percentage_error(y_true, y_pred)


def _get_mean_squared_logarithmic_error_loss():
    return lambda y_true, y_pred: losses.mean_squared_logarithmic_error(y_true, y_pred)


def _get_hinge_loss():
    return lambda y_true, y_pred: losses.hinge(y_true, y_pred)


def _get_jaccard_distance_loss(smooth=100):
    def jaccard_distance_loss(y_true, y_pred):
        y_true = K.cast_to_floatx(y_true)
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return (1 - jac) * smooth
    return jaccard_distance_loss


def _class_weight_wrapper(class_weights, loss_fn):
    # Designed for 1D output!!
    def inner_loss(y_true, y_pred):
        #weight_list = class_weights
        #fn = lambda x: weight_list[K.eval(x)]
        #weights_per_sample = K.map_fn(fn, K.cast(y_true, dtype="int64"))
        weights_per_sample = K.map_fn(lambda x: x * class_weights[1] + (1 - x) * class_weights[0], K.cast_to_floatx(y_true))
        loss_per_sample = loss_fn(K.expand_dims(y_true), K.expand_dims(y_pred))
        loss = K.mean(loss_per_sample * K.expand_dims(weights_per_sample))
        return loss

    return inner_loss


_loss_functions = {
    "cross_entropy": lambda parameters: _get_cross_entropy_loss(**parameters),
    "binary_cross_entropy": lambda parameters: _get_binary_cross_entropy_loss(**parameters),
    "sparse_cross_entropy": lambda parameters: _get_sparse_categorical_crossentropy_loss(**parameters),
    "negative_cosine_similarity": lambda parameters: _get_cosine_similarity_loss(**parameters),
    "kl_divergence": lambda parameters: _get_kl_divergence_loss(),
    "log_cosh": lambda parameters: _get_log_cosh_loss(),
    "MAPE": lambda parameters: _get_mean_absolute_percentage_error_loss(),
    "MSLE": lambda parameters: _get_mean_squared_logarithmic_error_loss(),
    "hinge": lambda parameters: _get_hinge_loss(),
    "jaccard_distance": lambda parameters: _get_jaccard_distance_loss(**parameters)
}