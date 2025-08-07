import optuna
from typing import Self, Dict, Any

from posnnet.hp_spaces.subspaces import ChoiceSpace, FloatSpace, IntegerSpace


class TemporalConvolutionalAttentiveNetworkWithBypassHpSpace:

    def __init__(
        self: Self,
        d_model_subspace: IntegerSpace=IntegerSpace(name="d_model", low=16, high=256, log2=True),
        n_head_subspace: IntegerSpace=IntegerSpace(name="n_head", low=1, high=16, log2=True),
        n_encoder_layers_subspace: IntegerSpace=IntegerSpace(name="n_encoder_layers", low=1, high=8),
        p_dropout_subspace: FloatSpace=FloatSpace(name="p_dropout", low=0.1, high=0.5, step=0.1)
    ) -> None:

        """
        Initiate the hyperparameters space for the TCANWB model.
        For a more complete hyperparameters descriptions, check the class docstring in posnnet.models.temporal_convolutional_attentive_network_with_bypass.

        Args:
            - d_model_subspace (posnnet.hp_spaces.subspaces.IntegerSpace): The subspace for the dimension of the model (embedding dimension).
                                                                           Default: IntegerSpace(name="d_model", low=16, high=256, log2=True) = [low, 2 * low, 4 * low, ..., high].
            - n_head_subspace (posnnet.hp_spaces.subspaces.IntegerSpace): The subspace for the number of parallel heads in the multi-head attention.
                                                                          Default: IntegerSpace(name="n_head", low=1, high=16, log2=True) = [low, 2 * low, 4 * low, ..., high].
                                                                                        Default: IntegerSpace(name="ratio_hidden_size_ff", low=1, high=4, log2=True) = [low, 2 * low, 4 * low, ..., high].
            - n_encoder_layers_subspace (posnnet.hp_spaces.subspaces.IntegerSpace): The subspace for the number of encoder layers.
                                                                                    Default: IntegerSpace(name="n_encoder_layers", low=1, high=8) = [low, low + 1, low + 2, ..., high].
            - p_dropout_subspace (posnnet.hp_spaces.subspaces.FloatSpace): The subspace for the probability of dropping neurons during training.
                                                                           Default: FloatSpace(name="p_dropout", low=0.1, high=0.5, step=0.1) = [0.1, 0.2, 0.3, 0.4, 0.5].
        """

        # Memorize subspaces.
        self.d_model_subspace=d_model_subspace
        self.n_head_subspace=n_head_subspace
        self.n_encoder_layers_subspace=n_encoder_layers_subspace
        self.p_dropout_subspace=p_dropout_subspace

    
    def sample_model_hp(
        self: Self,
        trial: optuna.trial.Trial,
        model_params: Dict[str, Any],
        **kwargs: Any
    ) -> Dict[str, Any]:

        """
        Sample a set of hyperparameters.

        Args:
            - trial (optuna.trial.Trial): The Trial object of the actual trial.
            - model_params (Dict[str, Any]): A dictionnary inside which the hyperparameters will be added.

        Returns:
            - model_params(Dict[str, Any]): The input dictionnary with the following new keys: 'd_model', 'n_head',
                                            'n_encoder_layers' and 'p_dropout'.
        """

        # Sample the dimension of the model (embedding dimension).
        d_model = self.d_model_subspace.sample_value(trial=trial)

        # Sample the number of parallel heads in the multi-head attention.
        # NB : The maximum number of heads is constraint to the half of the dimension of the model
        # so every head will receive at least a tensor where the last dimension size is >= 2.
        n_head = self.n_head_subspace.sample_value(trial=trial, constraint_high=d_model // 2)

        # Sample the number of encoders in the model.
        n_encoder_layers = self.n_encoder_layers_subspace.sample_value(trial=trial)
        
        # Sample the dropout (dropping a neuron during training) probability for every neurons.
        p_dropout = self.p_dropout_subspace.sample_value(trial=trial)

        # Add the hyperparameters to the dedicated dictionnary.
        model_params.update({
            "d_model": d_model,
            "n_head": n_head,
            "n_encoder_layers": n_encoder_layers,
            "p_dropout": p_dropout
        })

        return model_params