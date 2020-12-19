# Categorical Embedder

It is not straightforward to handle categorical variables. There are *encoding schemes* [1] (one-hot encoding, label encoding, ordinal encoding etc.) and categorical *embedding techiques* [2, 3] to address this problem. 

This project aims to provide an umbrella for all such solutions. Moreover, we present a new technique called "Holistic Categorical Embedding".

## Holistic Categorical Embedding

Holistic Categorical Embedding approaches categorical embedding problem as a sequence to sequence learning task. Each categorical column represents a time-step whereas each available category (plus an additional "unknown category") represents an observation at a given time-step. All categories over all the categorical columns forms the vocabulary.

Notice that there is no "temporal" interaction between the columns in reality. Therefore, consider the problem as learning latent distributions where the trainsition matrix for the states is trivial but learning the observation probabilities is the main focus.

This approach paves the road for applying NLP advancements on the categorical embedding domain. We may also apply any auto-encoding solution as well as sequence to sequence learning solutions.

We expect to see the most efficiency when there are lots of categorical columns with small/moderate number of categories. In constrast, cases where there are many columns with a few categories but one column having a great number of categories may cause unecessary memory consumption and other techniques might be considered. Though, as long as there are "enough" data, the solution is applicable for all domains.


## Software Engineering Aspects

- Keeps a definite interface for generators (the so-called *inner-generator*) feeding the models.
  - Users may provide their custom generators (the so-called *outer-generator*) to integrate any data source.
  - Thus, we separate the ingestion logic from ML/DL architectures/models.
- Abstracts shared training and embedding tasks so that developers can focus only on the model architectures.
- Currently supports 2 main approaches:
  - Seq2seq learning
    - Re-usable and generic encoder-decoders allowing choice of recurrent cells, their parameters; the number of layers in encoder and decoder etc.
  - VAE learning
    - Enables both continuous and discrete/concerete latent embeddings.
    - Enables users to define encoder and decoder over a config file for high flexiblity.
- Provides discriminative wrappers for both approaches.
- Provides basic processors and layers in addition to the ones in Keras.
- Hopefully, this software engineering approach will yield a better maintainable and flexible solution.


## TO-DO
- Although informal tests are conducted, we need more formal tests.
- Please, be careful with loss usage. Do not use before analyzing the needs, tensor shapes and theory.
- Need to benchmark!
  - On small number of samples, it tends to overfit.
  - Any contribution is kindly welcomed for applying the solution on large number of samples.
- May add existing categorical embedding solutions.


# References

[1] [All about Categorical Variable Encoding](https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02)

[2] [Entity Embeddings of Categorical Variables](https://arxiv.org/abs/1604.06737)

[3] [CAT2VEC: LEARNING DISTRIBUTED REPRESENTATION OF MULTI-FIELD CATEGORICAL DATA](https://openreview.net/pdf?id=HyNxRZ9xg)