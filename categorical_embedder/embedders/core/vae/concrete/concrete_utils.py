from keras import backend as K


def compute_kl_normal(z_mean, z_log_var):
    # KL divergence between N(0,1) and N(z_mean, exp(z_log_var))
    kl_per_sample = 0.5 * (K.sum(K.square(z_mean) + K.exp(z_log_var) - 1 - z_log_var, axis=1))
    return K.mean(kl_per_sample)


def compute_kl_discrete(distributions):
    # KL divergence between a uniform distribution over distributions of the categories.
    # Re-consider whether you need a modified version of KL-div for ELBO!
    num_of_categories = K.int_shape(distributions)[1]
    dist_sum = K.sum(distributions * K.log(K.constant(1.0 / num_of_categories) + K.epsilon()), axis=1)
    dist_neg_entropy = K.sum(distributions * K.log(distributions + K.epsilon()), axis=1)
    return K.mean(dist_neg_entropy - dist_sum)


def sample_concrete(alpha, temperature):
    # Sample from a concrete distribution given alpha and temperature
    uniform = K.random_uniform(shape=K.shape(alpha))
    gumbel = - K.log(- K.log(uniform + K.epsilon()) + K.epsilon())
    logits = (K.log(alpha + K.epsilon()) + gumbel) / temperature
    return K.softmax(logits)


def sample_normal(z_mean, z_log_var):
    # Sample from a normal distribution with mean z_mean and variance z_log_var
    epsilon = K.random_normal(shape=K.shape(z_mean), mean=0, stddev=1)
    return z_mean + K.exp(z_log_var / 2) * epsilon


def concrete_sampler(temperature=0.1):
    return lambda alpha: sample_concrete(alpha, temperature)


def normal_sampler():
    return lambda z_tuple: sample_normal(z_tuple[0], z_tuple[1])
