from keras import backend as K
from keras.layers import Layer, Dense, Lambda, Concatenate

from categorical_embedder.embedders.core.aux import layer_creator
from categorical_embedder.embedders.core.vae.concrete.concrete_utils import concrete_sampler, normal_sampler, compute_kl_discrete, compute_kl_normal


class VAEEncoder(Layer):
    def __init__(self, layer_info, hidden_length, continuous_latent_length=None, discrete_latent_length=None, temperature=0.1, z_loss_weight=0.5, c_loss_weight=0.5, **kwargs):
        super().__init__(**kwargs)

        if continuous_latent_length is None and discrete_latent_length is None:
            raise Exception("At least one of the latent length must be defined: [continuous_latent_length, discrete_latent_length]")

        self._layer_info = layer_info
        self._hidden_length = hidden_length
        self._continuous_latent_length = continuous_latent_length
        self._discrete_latent_length = discrete_latent_length
        self._temperature = temperature
        self._z_loss_weight = z_loss_weight
        self._c_loss_weight = c_loss_weight

        self._custom_layers = []
        for info in self._layer_info:
            self._custom_layers.append(layer_creator.create(info["type"], info["parameters"]))

        self._latent_layer = Dense(self._hidden_length, activation='relu')

        if self._continuous_latent_length is not None:
            self._z_mean_layer = Dense(self._continuous_latent_length)
            self._z_log_var_layer = Dense(self._continuous_latent_length)
            self._normal_sampling_layer = Lambda(normal_sampler())

        if self._discrete_latent_length is not None:
            self._Q_c_layer = Dense(self._discrete_latent_length, activation='softmax')
            self._concrete_sampling_layer = Lambda(concrete_sampler(self._temperature))

        if self._continuous_latent_length is not None and self._discrete_latent_length is not None:
            # Be careful about concat axis..
            self._concat_layer = Concatenate()

    def call(self, inputs, **kwargs):
        output = K.cast_to_floatx(inputs)
        for cur_layer in self._custom_layers:
            output = cur_layer(output)

        hidden = self._latent_layer(output)

        encoding = []
        if self._continuous_latent_length is not None:
            z_mean = self._z_mean_layer(hidden)
            z_log_var = self._z_log_var_layer(hidden)
            z = self._normal_sampling_layer((z_mean, z_log_var))
            encoding.append(z)
            self.add_loss(self._z_loss_weight * compute_kl_normal(z_mean, z_log_var))

        if self._discrete_latent_length is not None:
            alpha = self._Q_c_layer(hidden)
            c = self._concrete_sampling_layer(alpha, self._temperature)
            encoding.append(c)
            self.add_loss(self._c_loss_weight * compute_kl_discrete(alpha))

        if len(encoding) == 2:
            return self._concat_layer(encoding)
        else:
            return encoding[0]

    def compute_output_shape(self, input_shape):
        feature_dim = 0 if self._continuous_latent_length is None else self._continuous_latent_length
        feature_dim += 0 if self._discrete_latent_length is None else self._discrete_latent_length
        return input_shape[0], feature_dim

    def get_config(self):
        config = {
            "layer_info": self._layer_info,
            "hidden_length": self._hidden_length,
            "continuous_latent_length": self._continuous_latent_length,
            "discrete_latent_length": self._discrete_latent_length,
            "temperature": self._temperature,
            "z_loss_weight": self._z_loss_weight,
            "c_loss_weight": self._c_loss_weight
        }
        base_config = super().get_config()
        config.update(base_config)

        return config
