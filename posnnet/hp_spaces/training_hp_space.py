import optuna
from typing import Self, Dict, Any, Union

from posnnet.hp_spaces.subspaces import ChoiceSpace, FloatSpace, IntegerSpace


class TrainingHpSpace:

    def __init__(
        self: Self,
        use_adversarial: Union[str, None],
        scaling_type_subspace: ChoiceSpace=ChoiceSpace(name="scaling_type", choices=["normalization", "standardization"]),
        alpha_subspace: FloatSpace=FloatSpace(name="alpha", low=0.5, high=1.0),
        batch_size_subspace: IntegerSpace=IntegerSpace(name="batch_size", low=2, high=64, log2=True),
        learning_rate_subspace: FloatSpace=FloatSpace(name="learning_rate", low=1e-5, high=1e-3, log=True),
        weight_decay_subspace: FloatSpace=FloatSpace(name="weight_decay", low=1e-3, high=1e-1, log=True),
        beta_subspace: FloatSpace=FloatSpace(name="epsilon", low=1e-2, high=1.0, log=True),
        epsilon_subspace: FloatSpace=FloatSpace(name="lambda", low=1e-2, high=1.0, log=True)
    ) -> None:

        """
        Initiate the hyperparameters space for the training.

        Args:
            - use_adversarial (Union[str, None]): Indicate if the training use adversarial example (True) or not (False).
            - scaling_type_subspace (posnnet.hp_spaces.subspaces.ChoiceSpace): The subspace for the scaling type of the data.
                                                                               Default: ChoiceSpace(name="scaling_type", choice=["normalization", "standardization"]) = ['normalization', 'standardization'].
            - alpha_subspace (posnnet.hp_spaces.subspaces.FloatSpace): The subspace for the weight of the loss for the masked part of the sequence.
                                                                       Default: FloatSpace(name="alpha", low=0.5, high=1.0) = [0.5; 1].
            - batch_size_subspace (posnnet.hp_spaces.subspaces.IntegerSpace): The subspace for the size of the batch.
                                                                              Default: IntegerSpace(name="batch_size", low=2, high=64, log2=True) = [2, 4, 8, 16, 32, 64].
            - learning_rate_subspace (posnnet.hp_spaces.subspaces.FloatSpace): The subspace for the learning rate of the AdamW optimizer.
                                                                               Default: FloatSpace(name="learning_rate", low=1e-5, high=1e-3, log=True) = [1e-5; 1e-3].
            - weight_decay_subspace (posnnet.hp_spaces.subspaces.FloatSpace): The subspace for the weight decay of the AdamW optimizer.
                                                                              Default: FloatSpace(name="weight_decay", low=1e-3, high=1e-1, log=True) = [1e-3; 1e-1].
            - beta_subspace (posnnet.hp_spaces.subspaces.FloatSpace): The subspace for the weight of the adversarial loss.
                                                                      Default: FloatSpace(name="epsilon", low=1e-2, high=1.0, log=True) = [1e-2; 1.0].
            - epsilon_subspace (posnnet.hp_spaces.subspaces.FloatSpace): The subspace for the weight of the gradient during adversarial example computation.
                                                                         Default: FloatSpace(name="epsilon", low=1e-2, high=1.0, log=True) = [1e-2; 1.0].
        """

        self.use_adversarial = use_adversarial

        # Memorize subspaces.
        self.scaling_type_subspace = scaling_type_subspace
        self.alpha_subspace = alpha_subspace
        self.batch_size_subspace = batch_size_subspace
        self.learning_rate_subspace = learning_rate_subspace
        self.weight_decay_subspace = weight_decay_subspace

        if self.use_adversarial is not None:
            self.beta_subspace = beta_subspace
            self.epsilon_subspace = epsilon_subspace

    def sample_training_hp(
        self: Self,
        trial: optuna.trial.Trial,
        training_params: Dict[str, Any]
    ) -> Dict[str, Any]:

        """
        Sample a set of hyperparameters.

        Args:
            - trial (optuna.trial.Trial): The Trial object of the actual trial.
            - training_params (Dict[str, Any]): A dictionnary inside which the hyperparameters will be added.

        Returns:
            - training_params(Dict[str, Any]): The input dictionnary with the following new keys: 'scaling_type',
                                               'alpha', 'batch_size', 'learning_rate', 'weight_decay' and if 
                                               use_adversarial is not None, 'beta' and 'epsilon'.
        """

        # Sample the scaling type of the data.
        scaling_type = self.scaling_type_subspace.sample_value(trial=trial)

        # Sample the weight of the loss of the masked part of the sequence.
        alpha = self.alpha_subspace.sample_value(trial=trial)

        # Sample the size of the batchs during training.
        batch_size = self.batch_size_subspace.sample_value(trial=trial) 

        # Sample the learning rate of the AdamW optimizer.
        learning_rate = self.learning_rate_subspace.sample_value(trial=trial)

        # Sample the weight decay of the AdamW optimizer.
        weight_decay = self.weight_decay_subspace.sample_value(trial=trial)

        # Add the hyperparameters to the dedicated dictionnary.
        training_params.update({
            "scaling_type": scaling_type,
            "alpha": alpha,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay
        })

        # If the training use adversarial example, hanlde beta and epsilon
        if self.use_adversarial is not None:

            # Sample the weight of the loss of adversarial example.
            beta = self.beta_subspace.sample_value(trial=trial)

            # Sample the weight of the gradient during adversarial example computation.
            epsilon = self.epsilon_subspace.sample_value(trial=trial)

            # Add the hyperparameters to the dedicated dictionnary.
            training_params.update({
                "beta": beta,
                "epsilon": epsilon
            })

        return training_params