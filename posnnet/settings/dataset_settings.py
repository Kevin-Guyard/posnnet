import pathlib
from typing import Self, Union, List, Tuple, Any

from posnnet.setter_guards import (
    setter_typeguard,
    setter_list_of_unique_strictly_positive_integer_guard,
    setter_string_choice_or_strictly_positive_integer_guard
)
from posnnet.settings.base_settings import BaseSettings


class DatasetSettings(BaseSettings):

    def __init__(
        self: Self
    ) -> None:

        """
        Initialize the dataset settings.

        Settings:
            - n_duplications_per_eval_session (Union[int, str]): The number of duplications per evaluation session. If an integer is provided (strictly positive), every session is duplicated the same amount of time.
                                                                If 'light', 'normal' or 'extensive' is provided, the framework compute an optimized number of duplication different for every session.
            - sessions_id_training (List[int]): The list of the id of the sessions that will be used for the training dataset. Id have to be strictly positive integer.
            - sessions_id_validation (List[int]): The list of the id of the sessions that will be used for the validation dataset. Id have to be strictly positive integer.
            - sessions_id_evaluation (List[int]): The list of the id of the sessions that will be used for the evaluation dataset. Id have to be strictly positive integer.
        """

        self.n_duplications_per_eval_session = "normal"
        self.sessions_id_training = []
        self.sessions_id_validation = []
        self.sessions_id_evaluation = []

    @property
    def n_duplications_per_eval_session(
        self: Self
    ) -> Union[str, int]:

        return self._n_duplications_per_eval_session

    @n_duplications_per_eval_session.setter
    @setter_typeguard
    @setter_string_choice_or_strictly_positive_integer_guard(allowed_choices=["light", "normal", "extensive"])
    def n_duplications_per_eval_session(
        self: Self,
        n_duplications_per_eval_session: Union[str, int]
    ) -> None:

        self._n_duplications_per_eval_session = n_duplications_per_eval_session

    @property
    def sessions_id_training(
        self: Self
    ) -> List[int]:

        return self._sessions_id_training

    @sessions_id_training.setter
    @setter_typeguard
    @setter_list_of_unique_strictly_positive_integer_guard
    def sessions_id_training(
        self: Self,
        sessions_id_training: List[int]
    ) -> None:

        self._sessions_id_training = sessions_id_training

    @property
    def sessions_id_validation(
        self: Self
    ) -> List[int]:

        return self._sessions_id_validation

    @sessions_id_validation.setter
    @setter_typeguard
    @setter_list_of_unique_strictly_positive_integer_guard
    def sessions_id_validation(
        self: Self,
        sessions_id_validation: List[int]
    ) -> None:

        self._sessions_id_validation = sessions_id_validation

    @property
    def sessions_id_evaluation(
        self: Self
    ) -> List[int]:

        return self._sessions_id_evaluation

    @sessions_id_evaluation.setter
    @setter_typeguard
    @setter_list_of_unique_strictly_positive_integer_guard
    def sessions_id_evaluation(
        self: Self,
        sessions_id_evaluation: List[int]
    ) -> None:

        self._sessions_id_evaluation = sessions_id_evaluation

    def validate_settings(
        self: Self,
        path_raw_data: pathlib.Path,
        **kwargs: Any
    ) -> Tuple[List[str], List[str]]:

        """
        Validate the value of the different setting parameters.

        Args:
            - path_raw_data (pathlib.Path): The path where raw data are stored.

        Returns:
            - warnings (List[str]): A list of warnings on the different setting parameter values.
            - errors (List[str]): A list of errors on the different setting parameter values.
        """

        warnings, errors = [], []

        # If the number of duplications per evaluation session is an integer, generate a warning.
        if isinstance(self.n_duplications_per_eval_session, int):
            warnings.append(f"You have provided a fixed number of duplications per evaluation session ({self.n_duplications_per_eval_session:d}). It is recommended to use an automatic mode ('light', 'normal', 'extensive'). Ignore this message if you are sure to want a fixed number of duplications.")

        # If the number of duplications per evaluation session is 'extensive', generate a warning.
        if self.n_duplications_per_eval_session == "extensive":
            warnings.append("You have provided 'extensive' as number of duplications per evaluation session. Extensive mode will generate a maximum of duplications per evaluation session to compensate a small evaluation dataset. However, it will still underperform a case with more data. If you have the possibility to increase the size of the evaluation dataset, it is recommended to collect more data. Otherwise, ignore this message.")

        # If the number of duplications per evaluation session is 'light', generate a warning.
        if self.n_duplications_per_eval_session == "light":
            warnings.append("You have provided 'light' as number of duplications per evaluation session. Light mode will generate only a few duplications per evaluation session. Be sure to have a large evaluation dataset. If you are not sure, consider using 'normal' mode. Otherwise, ignore this message.")

        # If not training session is provided, generate an error.
        if len(self.sessions_id_training) == 0:
            errors.append("You do not have provided any session for the training dataset.")

        # If not validation session is provided, generate an error.
        if len(self.sessions_id_validation) == 0:
            errors.append("You do not have provided any session for the validation dataset.")

        # If not evaluation session is provided, generate an error.
        if len(self.sessions_id_evaluation) == 0:
            errors.append("You do not have provided any session for the evaluation dataset.")

        # If a training session is not present on disk, generate an error.
        for session_id in self.sessions_id_training:
            if not pathlib.Path(path_raw_data, f"session_{session_id:d}.csv").exists():
                errors.append(f"You provided the session id {session_id:d} as part of the training dataset. However, the session cannot be found inside the raw data folder ({str(path_raw_data):s}). Either you forgot to include it or the file name is not correct (the name should be 'session_{session_id:d}.csv'.")

        # If a validation session is not present on disk, generate an error.
        for session_id in self.sessions_id_validation:
            if not pathlib.Path(path_raw_data, f"session_{session_id:d}.csv").exists():
                errors.append(f"You provided the session id {session_id:d} as part of the validation dataset. However, the session cannot be found inside the raw data folder ({str(path_raw_data):s}). Either you forgot to include it or the file name is not correct (the name should be 'session_{session_id:d}.csv'.")

        # If a evaluation session is not present on disk, generate an error.
        for session_id in self.sessions_id_evaluation:
            if not pathlib.Path(path_raw_data, f"session_{session_id:d}.csv").exists():
                errors.append(f"You provided the session id {session_id:d} as part of the evaluation dataset. However, the session cannot be found inside the raw data folder ({str(path_raw_data):s}). Either you forgot to include it or the file name is not correct (the name should be 'session_{session_id:d}.csv'.")

        # If a session is present in both training and validation dataset, generate an error.
        if set(self.sessions_id_training).intersection(set(self.sessions_id_validation)):
            errors.append(f"The following sessions are presents in both the training and the validation datasets: [{', '.join([str(item) for item in set(self.sessions_id_training).intersection(set(self.sessions_id_validation))])}]. A session cannot be used in more than one dataset. Please ensure that every session is present only in one dataset.")

        # If a session is present in both training and evaluation dataset, generate an error.
        if set(self.sessions_id_training).intersection(set(self.sessions_id_evaluation)):
            errors.append(f"The following sessions are presents in both the training and the evaluation datasets: [{', '.join([str(item) for item in set(self.sessions_id_training).intersection(set(self.sessions_id_evaluation))])}]. A session cannot be used in more than one dataset. Please ensure that every session is present only in one dataset.")

        # If a session is present in both validation and evaluation dataset, generate an error.
        if set(self.sessions_id_validation).intersection(set(self.sessions_id_evaluation)):
            errors.append(f"The following sessions are presents in both the validation and the evaluation datasets: [{', '.join([str(item) for item in set(self.sessions_id_validation).intersection(set(self.sessions_id_evaluation))])}]. A session cannot be used in more than one dataset. Please ensure that every session is present only in one dataset.")
                
        return warnings, errors