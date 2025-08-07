import optuna
from typing import Self, Dict, Any, Union

from posnnet.hp_spaces.convolutional_long_short_term_memory_transformer_frequency_aware_with_bypass_hp_space import ConvolutionalLongShortTermMemoryTransformerFrequencyAwareWithBypassHpSpace
from posnnet.hp_spaces.convolutional_long_short_term_memory_transformer_with_bypass_hp_space import ConvolutionalLongShortTermMemoryTransformerWithBypassHpSpace
from posnnet.hp_spaces.spatio_temporal_transformer_frequency_aware_with_bypass_hp_space import SpatioTemporalTransformerFrequencyAwareWithBypassHpSpace
from posnnet.hp_spaces.spatio_temporal_transformer_with_bypass_hp_space import SpatioTemporalTransformerWithBypassHpSpace
from posnnet.hp_spaces.temporal_convolutional_attentive_network_with_bypass_hp_space import TemporalConvolutionalAttentiveNetworkWithBypassHpSpace
from posnnet.hp_spaces.training_hp_space import TrainingHpSpace

class HpSpace:

    def __init__(
        self: Self,
        model_name: str,
        use_adversarial: Union[str, None],
        min_len_seq: int
    ) -> None:

        """
        Initiate an instance of HpSpace that can sample hyperparameters for models and training.

        Args:
            - model_name (str): The name of the model to use (can be either 'CLSTMTFAWB', 'CLSTMWB', 'STTFAWB', 'STTWB' or 'TCANWB').
            - use_adversarial (Union[str, None]): Indicate if the training use adversarial example (True) or not (False).
            - min_len_seq (int): The minimal length of the sequence provided to the model after frequency division.
        """

        # Memorize parameters.
        self.min_len_seq = min_len_seq

        # Create a mapping between model name and hp space
        mapping_model_name_to_model_class = {
            "CLSTMTFAWB": ConvolutionalLongShortTermMemoryTransformerFrequencyAwareWithBypassHpSpace(),
            "CLSTMTWB": ConvolutionalLongShortTermMemoryTransformerWithBypassHpSpace(),
            "STTFAWB": SpatioTemporalTransformerFrequencyAwareWithBypassHpSpace(),
            "STTWB": SpatioTemporalTransformerWithBypassHpSpace(),
            "TCANWB": TemporalConvolutionalAttentiveNetworkWithBypassHpSpace(),
        }

        self.__model_hp_space = mapping_model_name_to_model_class.get(model_name)

        self.__training_hp_space = TrainingHpSpace(use_adversarial=use_adversarial)

    def sample_model_hp(
        self: Self,
        trial: optuna.trial.Trial,
        model_params: Dict[str, Any]
    ) -> Dict[str, Any]:

        """
        Sample a set of hyperparameters for the model.

        Args:
            - trial (optuna.trial.Trial): The Trial object of the actual trial.
            - model_params (Dict[str, Any]): A dictionnary inside which the hyperparameters will be added.

        Returns:
            - model_params(Dict[str, Any]): The input dictionnary with the model hyperparameters.
        """

        model_params = self.__model_hp_space.sample_model_hp(trial=trial, model_params=model_params, min_len_seq=self.min_len_seq)

        return model_params

    def sample_training_hp(
        self: Self,
        trial: optuna.trial.Trial,
        training_params: Dict[str, Any]
    ) -> Dict[str, Any]:

        """
        Sample a set of hyperparameters for the training.

        Args:
            - trial (optuna.trial.Trial): The Trial object of the actual trial.
            - training_params (Dict[str, Any]): A dictionnary inside which the hyperparameters will be added.

        Returns:
            - training_params(Dict[str, Any]): The input dictionnary with the training hyperparameters.
        """

        training_params = self.__training_hp_space.sample_training_hp(trial=trial, training_params=training_params)

        return training_params