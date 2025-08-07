from functools import wraps
import pathlib
import shutil
from typing import Self, Callable, Any


class ProjectLifeCycleManager:

    SETTINGS_DEFINITION_STAGE = 0
    DATA_PREPROCESSING_STAGE = 1
    SCALER_FIT_STAGE = 2
    MODELS_TUNING_STAGE = 3
    AVERAGING_CONFIGURATION_STAGE = 4
    PRODUCTION_STAGE = 5

    MAPPING_STAGE_TO_NAME = {
        0: "SETTINGS_DEFINITION_STAGE",
        1: "DATA_PREPROCESSING_STAGE",
        2: "SCALER_FIT_STAGE",
        3: "MODELS_TUNING_STAGE",
        4: "AVERAGING_CONFIGURATION_STAGE",
        5: "PRODUCTION_STAGE"
    }

    def __init__(
        self: Self
    ) -> None:

        """
        Manage the life cycle of the project by storing its stage and providing method guards for Project methods that allow
        the call of specific methods only during specific project stage.
        """

        self.__stage = ProjectLifeCycleManager.SETTINGS_DEFINITION_STAGE

    def move_to_next_stage(
        self: Self
    ) -> None:

        """
        Move to the next stage of the life cycle.
        """

        self.__stage += 1

    def __move_from_data_preprocessing_to_settings_definition_stage(
        self: Self
    ) -> None:

        """
        Move the actual stage of the life cycle from DATA_PREPROCESSING_STAGE to SETTINGS_DEFINITION_STAGE
        """

        self.__stage = ProjectLifeCycleManager.SETTINGS_DEFINITION_STAGE

    def __move_from_scaler_fit_to_data_preprocessing_stage(
        self: Self,
        path_preprocessed_training_data: pathlib.Path,
        path_preprocessed_validation_data: pathlib.Path,
        path_preprocessed_evaluation_data: pathlib.Path
    ) -> None:

        """
        Move the actual stage of the life cycle from SCALER_FIT_STAGE to DATA_PREPROCESSING_STAGE

        Args:
            - path_preprocessed_training_data (pathlib.Path): The path where are stored the preprocessed training data.
            - path_preprocessed_validation_data (pathlib.Path): The path where are stored the preprocessed validation data.
            - path_preprocessed_evaluation_data (pathlib.Path): The path where are stored the preprocessed evaluation data.
        """

        # Remove training, validation and evaluation data.
        shutil.rmtree(path_preprocessed_training_data)
        shutil.rmtree(path_preprocessed_validation_data)
        shutil.rmtree(path_preprocessed_evaluation_data)

        # Recreate empty folders.
        path_preprocessed_training_data.mkdir(exist_ok=False)
        path_preprocessed_validation_data.mkdir(exist_ok=False)
        path_preprocessed_evaluation_data.mkdir(exist_ok=False)

        self.__stage = ProjectLifeCycleManager.DATA_PREPROCESSING_STAGE

    def __move_from_models_tuning_to_scaler_fit_stage(
        self: Self,
        path_scalers: pathlib.Path,
        path_tuning_results: pathlib.Path,
        path_evaluation_results: pathlib.Path,
        path_studies: pathlib.Path,
        path_temp: pathlib.Path
    ) -> None:

        """
        Move the actual stage of the life cycle from MODELS_TUNING_STAGE to SCALER_FIT_STAGE

        Args:
            - path_scalers (pathlib.Path): The path where are stored the scalers.
            - path_tuning_results (pathlib.Path): The path where are stored the tuning results.
            - path_evaluation_results (pathlib.Path): The path where are stored the evaluation results.
            - path_studies (pathlib.Path): The path where are stored the tuning studies.
            - path_temp (pathlib.Path): The path where are stored the temporary files.
        """

        # Remove scalers and relax points from disk.
        pathlib.Path(path_scalers, "scalers.pkl").unlink()
        pathlib.Path(path_scalers, "relax_points.pkl").unlink()

        # Remove tuning and evaluation results, studies and temporary files.
        shutil.rmtree(path_tuning_results)
        shutil.rmtree(path_evaluation_results)
        shutil.rmtree(path_studies)
        shutil.rmtree(path_temp)

        # Recreate empty folders.
        path_tuning_results.mkdir(exist_ok=False)
        path_evaluation_results.mkdir(exist_ok=False)
        path_studies.mkdir(exist_ok=False)
        path_temp.mkdir(exist_ok=False)

        self.__stage = ProjectLifeCycleManager.SCALER_FIT_STAGE

    def __move_from_averaging_configuration_stage_to_models_tuning_stage(
        self: Self
    ) -> None:

        """
        Move the actual stage of the life cycle from AVERAGING_CONFIGURATION_STAGE to MODELS_TUNING_STAGE.
        """

        self.__stage = ProjectLifeCycleManager.MODELS_TUNING_STAGE

    def __move_from_production_stage_to_averaging_configuration_stage(
        self: Self,
        path_evaluation_results: pathlib.Path
    ) -> None:

        # Iterate over cases.
        for case_folder in path_evaluation_results.iterdir():

            # Ensure the folder is a case evaluation results folder, otherwise skip this folder.
            if not case_folder.name in ["beginning", "within", "end"]:
                continue

            # Iterate over subcases.
            for subcase_folder in case_folder.iterdir():

                # Ensure the folder is a subcase evaluation results folder, otherwise skip this folder.
                if not " to " in subcase_folder.name:
                    continue

                # Iterate over evaluation results files.
                for evaluation_results_file in subcase_folder.iterdir():

                    # Remove averaging evaluation results.
                    if "averaging" in evaluation_results_file.name:
                        evaluation_results_file.unlink()

        self.__stage = ProjectLifeCycleManager.AVERAGING_CONFIGURATION_STAGE

    def move_to_previous_stage(
        self: Self,
        desired_stage: int,
        path_preprocessed_training_data: pathlib.Path,
        path_preprocessed_validation_data: pathlib.Path,
        path_preprocessed_evaluation_data: pathlib.Path,
        path_scalers: pathlib.Path,
        path_tuning_results: pathlib.Path,
        path_evaluation_results: pathlib.Path,
        path_studies: pathlib.Path,
        path_temp: pathlib.Path,
        force_move: bool=False,
    ) -> None:

        """
        Move back the actual stage of the life cycle to the desired stage.

        Args:
            - desired_stage (int): The desired stage to which come back :
                * ProjectLifeCycleManager.SETTINGS_DEFINITION_STAGE
                * ProjectLifeCycleManager.DATA_PREPROCESSING_STAGE
                * ProjectLifeCycleManager.SCALER_FIT_STAGE
                * ProjectLifeCycleManager.MODELS_TUNING_STAGE
            - path_preprocessed_training_data (pathlib.Path): The path where are stored the preprocessed training data.
            - path_preprocessed_validation_data (pathlib.Path): The path where are stored the preprocessed validation data.
            - path_preprocessed_evaluation_data (pathlib.Path): The path where are stored the preprocessed evaluation data.
            - path_scalers (pathlib.Path): The path where are stored the scalers.
            - path_tuning_results (pathlib.Path): The path where are stored the tuning results.
            - path_evaluation_results (pathlib.Path): The path where are stored the evaluation results.
            - path_studies (pathlib.Path): The path where are stored the tuning studies.
            - path_temp (pathlib.Path): The path where are stored the temporary files.
            - force_move (bool): Either to force the mouvement to a previous stage (True) or raise a warning a do not take action (False) in case of destructive stage transition. Default = False.
        """

        # No action, the desired stage is the actual stage.
        if self.__stage == desired_stage:
            print("You are trying to move to a stage that is the current stage. No action has been taken.")
            return None

        # No action, the desired stage is a next stage.
        if self.__stage < desired_stage:
            print("You are trying to move to a next stage. This function aims only to move back to previous stage. No action has been taken.")
            return None

        # Force move = False ==> No destructive action.
        if force_move == False:

            # Move from PRODUCTION_STAGE to AVERAGING_CONFIGURATION_STAGE is not done (destructive).
            if self.__stage == ProjectLifeCycleManager.PRODUCTION_STAGE and desired_stage <= ProjectLifeCycleManager.AVERAGING_CONFIGURATION_STAGE:
                print("You are trying to move from PRODUCTION_STAGE to AVERAGING_CONFIGURATION_STAGE. This action is a destructive action. You will lose the averaging configuration fitted. Because 'force_move=False', this action has not been performed. If you are sure about realizing this action, call back the method with 'force_move=True'.", end="\n\n")

            # Move from AVERAGING_CONFIGURATION_STAGE to MODELS_TUNING_STAGE is done (non destructive).
            if self.__stage == ProjectLifeCycleManager.AVERAGING_CONFIGURATION_STAGE and desired_stage < ProjectLifeCycleManager.AVERAGING_CONFIGURATION_STAGE:
                self.__move_from_averaging_configuration_stage_to_models_tuning_stage()
                print("You have performed a stage move from AVERAGING_CONFIGURATION_STAGE to MODELS_TUNING_STAGE.", end="\n\n")

            # Move from MODELS_TUNING_STAGE to SCALER_FIT_STAGE is not done (destructive).
            if self.__stage > ProjectLifeCycleManager.SCALER_FIT_STAGE and desired_stage <= ProjectLifeCycleManager.SCALER_FIT_STAGE:
                print("You are trying to move from MODELS_TUNING_STAGE to SCALER_FIT_STAGE. This action is a destructive action. You will lose the scalers that have been already fitted, the tuned configurations and their evaluation results. Because 'force_move=False', this action has not been performed. If you are sure about realizing this action, call back the method with 'force_move=True'.", end="\n\n")

            # Move from SCALER_FIT_STAGE to DATA_PREPROCESSING_STAGE is not done (destructive).
            if self.__stage > ProjectLifeCycleManager.DATA_PREPROCESSING_STAGE and desired_stage <= ProjectLifeCycleManager.DATA_PREPROCESSING_STAGE:
                print("You are trying to move from SCALER_FIT_STAGE to DATA_PREPROCESSING_STAGE. This action is a destructive action. You will lose the preprocessed data (not the raw data). Because 'force_move=False', this action has not been performed. If you are sure about realizing this action, call back the method with 'force_move=True'.", end="\n\n")

            # Move from DATA_PREPROCESSING_STAGE to SETTINGS_DEFINITION_STAGE is done (non destructive).
            if self.__stage == ProjectLifeCycleManager.DATA_PREPROCESSING_STAGE and desired_stage <= ProjectLifeCycleManager.SETTINGS_DEFINITION_STAGE:
                self.__move_from_data_preprocessing_to_settings_definition_stage()
                print("You have performed a stage move from DATA_PREPROCESSING_STAGE to SETTINGS_DEFINITION_STAGE.", end="\n\n")

        # Force move = True ==> Every mouvement is performed.
        else:

            # Move from PRODUCTION_STAGE to AVERAGING_CONFIGURATION_STAGE is done (destructive).
            if self.__stage == ProjectLifeCycleManager.PRODUCTION_STAGE and desired_stage <= ProjectLifeCycleManager.AVERAGING_CONFIGURATION_STAGE:
                self.__move_from_production_stage_to_averaging_configuration_stage(
                    path_evaluation_results=path_evaluation_results
                )
                print("You have performed a stage move from PRODUCTION_STAGE to AVERAGING_CONFIGURATION_STAGE. This is a destructive action. Averaging configuration data have been deleted. You have to run agin the averaging configuration.", end="\n\n")

            # Move from AVERAGING_CONFIGURATION_STAGE to MODELS_TUNING_STAGE is done (non destructive).
            if self.__stage == ProjectLifeCycleManager.AVERAGING_CONFIGURATION_STAGE and desired_stage <= ProjectLifeCycleManager.MODELS_TUNING_STAGE:
                self.__move_from_averaging_configuration_stage_to_models_tuning_stage()
                print("You have performed a stage move from AVERAGING_CONFIGURATION_STAGE to MODELS_TUNING_STAGE", end="\n\n")

            # Move from MODELS_TUNING_STAGE to SCALER_FIT_STAGE is done (destructive).
            if self.__stage == ProjectLifeCycleManager.MODELS_TUNING_STAGE and desired_stage <= ProjectLifeCycleManager.SCALER_FIT_STAGE:
                self.__move_from_models_tuning_to_scaler_fit_stage(
                    path_scalers=path_scalers,
                    path_tuning_results=path_tuning_results,
                    path_evaluation_results=path_evaluation_results,
                    path_studies=path_studies,
                    path_temp=path_temp
                )
                print("You have performed a stage move from MODELS_TUNING_STAGE to SCALER_FIT_STAGE. This is a destructive action. Scalers that have been already fitted, the tuned configurations and their evaluations results have been deleted. You will have to fit the scalers again.", end="\n\n")

            # Move from SCALER_FIT_STAGE to DATA_PREPROCESSING_STAGE is done (destructive).
            if self.__stage == ProjectLifeCycleManager.SCALER_FIT_STAGE and desired_stage <= ProjectLifeCycleManager.DATA_PREPROCESSING_STAGE:
                self.__move_from_scaler_fit_to_data_preprocessing_stage(
                    path_preprocessed_training_data=path_preprocessed_training_data,
                    path_preprocessed_validation_data=path_preprocessed_validation_data,
                    path_preprocessed_evaluation_data=path_preprocessed_evaluation_data
                )
                print("You have performed a stage move from SCALER_FIT_STAGE to DATA_PREPROCESSING_STAGE. This action is a destructive action. The data that have been already preprocessed have been deleted (not the raw data). You will have to preprocess again the raw data.", end="\n\n")

            # Move from DATA_PREPROCESSING_STAGE to SETTINGS_DEFINITION_STAGE is done (non destructive).
            if self.__stage == ProjectLifeCycleManager.DATA_PREPROCESSING_STAGE and desired_stage <= ProjectLifeCycleManager.SETTINGS_DEFINITION_STAGE:
                self.__move_from_data_preprocessing_to_settings_definition_stage()
                print("You have performed a stage move from DATA_PREPROCESSING_STAGE to SETTINGS_DEFINITION_STAGE.", end="\n\n")

    def settings_definition_stage_guard(
        project_method: Callable
    ) -> Callable:

        """
        Add a guard to the project method that ensures the method is run only if the project stage is 'Settings definition stage'.
        If yes, the method is called normally. Otherwise, the method call is ignored and a message is displayed to inform the user.
    
        Args:
            - project_method (Callable): The method of Project for which add the guard.
    
        Returns:
            - project_method_settings_definition_stage_guarded (Callable): The method of Project guarded.
        """

        @wraps(project_method)
        def project_method_settings_definition_stage_guarded(
            self: Self,
            *args: Any,
            **kwargs: Any
        ) -> None:

            if self._life_cycle_manager.__stage == ProjectLifeCycleManager.SETTINGS_DEFINITION_STAGE:
                project_method(self, *args, **kwargs)
            else:
                print(f"You are trying to use a project method that can only be called when the project stage is SETTINGS_DEFINITION_STAGE. The actual stage of your project is {ProjectLifeCycleManager.MAPPING_STAGE_TO_NAME[self._life_cycle_manager.__stage]:s}. You have to get back to the SETTINGS_DEFINITION_STAGE to use this method. To do this, use the project method 'Project.move_to_previous_stage' with argument 'desired_stage=ProjectLifeCycleManager.SETTINGS_DEFINITION_STAGE'.")

        return project_method_settings_definition_stage_guarded

    def data_preprocessing_stage_guard(
        project_method: Callable
    ) -> Callable:

        """
        Add a guard to the project method that ensures the method is run only if the project stage is 'Data preprocessing stage'.
        If yes, the method is called normally. Otherwise, the method call is ignored and a message is displayed to inform the user.
    
        Args:
            - project_method (Callable): The method of Project for which add the guard.
    
        Returns:
            - project_method_data_preprocessing_stage_guarded (Callable): The method of Project guarded.
        """

        @wraps(project_method)
        def project_method_data_preprocessing_stage_guarded(
            self: Self,
            *args: Any,
            **kwargs: Any
        ) -> None:

            if self._life_cycle_manager.__stage == ProjectLifeCycleManager.DATA_PREPROCESSING_STAGE:
                project_method(self, *args, **kwargs)
            elif self._life_cycle_manager.__stage < ProjectLifeCycleManager.DATA_PREPROCESSING_STAGE:
                print(f"You are trying to use a project method that can only be called when the project stage is DATA_PREPROCESSING_STAGE. The actual stage of your project is {ProjectLifeCycleManager.MAPPING_STAGE_TO_NAME[self._life_cycle_manager.__stage]:s}. You have to perform the preceding stages ({', '.join([ProjectLifeCycleManager.MAPPING_STAGE_TO_NAME[stage] for stage in range(self._life_cycle_manager.__stage, ProjectLifeCycleManager.DATA_PREPROCESSING_STAGE)])}) before being able to call this method.")
            else:
                print(f"You are trying to use a project method that can only be called when the project stage is DATA_PREPROCESSING_STAGE. The actual stage of your project is {ProjectLifeCycleManager.MAPPING_STAGE_TO_NAME[self._life_cycle_manager.__stage]:s}. You have to get back to the DATA_PREPROCESSING_STAGE to use this method. To do this, use the project method 'Project.move_to_previous_stage' with argument 'desired_stage=ProjectLifeCycleManager.DATA_PREPROCESSING_STAGE'.")

        return project_method_data_preprocessing_stage_guarded

    def scaler_fit_stage_guard(
        project_method: Callable
    ) -> Callable:

        """
        Add a guard to the project method that ensures the method is run only if the project stage is 'Scaler fit stage'.
        If yes, the method is called normally. Otherwise, the method call is ignored and a message is displayed to inform the user.
    
        Args:
            - project_method (Callable): The method of Project for which add the guard.
    
        Returns:
            - project_method_scaler_fit_stage_guarded (Callable): The method of Project guarded.
        """

        @wraps(project_method)
        def project_method_scaler_fit_stage_guarded(
            self: Self,
            *args: Any,
            **kwargs: Any
        ) -> None:

            if self._life_cycle_manager.__stage == ProjectLifeCycleManager.SCALER_FIT_STAGE:
                project_method(self, *args, **kwargs)
            elif self._life_cycle_manager.__stage < ProjectLifeCycleManager.SCALER_FIT_STAGE:
                print(f"You are trying to use a project method that can only be called when the project stage is SCALER_FIT_STAGE. The actual stage of your project is {ProjectLifeCycleManager.MAPPING_STAGE_TO_NAME[self._life_cycle_manager.__stage]:s}. You have to perform the preceding stages ({', '.join([ProjectLifeCycleManager.MAPPING_STAGE_TO_NAME[stage] for stage in range(self._life_cycle_manager.__stage, ProjectLifeCycleManager.SCALER_FIT_STAGE)])}) before being able to call this method.")
            else:
                print(f"You are trying to use a project method that can only be called when the project stage is SCALER_FIT_STAGE. The actual stage of your project is {ProjectLifeCycleManager.MAPPING_STAGE_TO_NAME[self._life_cycle_manager.__stage]:s}. You have to get back to the SCALER_FIT_STAGE to use this method. To do this, use the project method 'Project.move_to_previous_stage' with argument 'desired_stage=ProjectLifeCycleManager.SCALER_FIT_STAGE'.")

        return project_method_scaler_fit_stage_guarded

    def models_tuning_stage_guard(
        project_method: Callable
    ) -> Callable:

        """
        Add a guard to the project method that ensures the method is run only if the project stage is 'Models tuning stage'.
        If yes, the method is called normally. Otherwise, the method call is ignored and a message is displayed to inform the user.
    
        Args:
            - project_method (Callable): The method of Project for which add the guard.
    
        Returns:
            - project_method_models_tuning_stage_guarded (Callable): The method of Project guarded.
        """

        @wraps(project_method)
        def project_method_models_tuning_stage_guarded(
            self: Self,
            *args: Any,
            **kwargs: Any
        ) -> None:

            if self._life_cycle_manager.__stage == ProjectLifeCycleManager.MODELS_TUNING_STAGE:
                project_method(self, *args, **kwargs)
            elif self._life_cycle_manager.__stage < ProjectLifeCycleManager.MODELS_TUNING_STAGE:
                print(f"You are trying to use a project method that can only be called when the project stage is MODELS_TUNING_STAGE. The actual stage of your project is {ProjectLifeCycleManager.MAPPING_STAGE_TO_NAME[self._life_cycle_manager.__stage]:s}. You have to perform the preceding stages ({', '.join([ProjectLifeCycleManager.MAPPING_STAGE_TO_NAME[stage] for stage in range(self._life_cycle_manager.__stage, ProjectLifeCycleManager.MODELS_TUNING_STAGE)])}) before being able to call this method.")
            else:
                print(f"You are trying to use a project method that can only be called when the project stage is MODELS_TUNING_STAGE. The actual stage of your project is {ProjectLifeCycleManager.MAPPING_STAGE_TO_NAME[self._life_cycle_manager.__stage]:s}. You have to get back to the MODELS_TUNING_STAGE to use this method. To do this, use the project method 'Project.move_to_previous_stage' with argument 'desired_stage=ProjectLifeCycleManager.MODELS_TUNING_STAGE'.")

        return project_method_models_tuning_stage_guarded

    def averaging_configuration_stage_guard(
        project_method: Callable
    ) -> Callable:

        """
        Add a guard to the project method that ensures the method is run only if the project stage is "Averaging configuration stage".
        If yes, the method is called normally. Otherwise, the method call is ignored and a message is displayed to inform the user.

        Args:
            - project_method (Callable): The method of Project for which add the guard.
    
        Returns:
            - project_method_averaging_configuration_stage_guarded (Callable): The method of Project guarded.
        """

        @wraps(project_method)
        def project_method_averaging_configuration_stage_guarded(
            self: Self,
            *args: Any,
            **kwargs: Any
        ) -> None:

            if self._life_cycle_manager.__stage == ProjectLifeCycleManager.AVERAGING_CONFIGURATION_STAGE:
                project_method(self, *args, **kwargs)
            elif self._life_cycle_manager.__stage < ProjectLifeCycleManager.AVERAGING_CONFIGURATION_STAGE:
                print(f"You are trying to use a project method that can only be called when the project stage is AVERAGING_CONFIGURATION_STAGE. The actual stage of your project is {ProjectLifeCycleManager.MAPPING_STAGE_TO_NAME[self._life_cycle_manager.__stage]:s}. You have to perform the preceding stages ({', '.join([ProjectLifeCycleManager.MAPPING_STAGE_TO_NAME[stage] for stage in range(self._life_cycle_manager.__stage, ProjectLifeCycleManager.AVERAGING_CONFIGURATION_STAGE)])}) before being able to call this method.")
            else:
                print(f"You are trying to use a project method that can only be called when the project stage is AVERAGING_CONFIGURATION_STAGE. The actual stage of your project is {ProjectLifeCycleManager.MAPPING_STAGE_TO_NAME[self._life_cycle_manager.__stage]:s}. You have to get back to the AVERAGING_CONFIGURATION_STAGE to use this method. To do this, use the project method 'Project.move_to_previous_stage' with argument 'desired_stage=ProjectLifeCycleManager.AVERAGING_CONFIGURATION_STAGE'.")

        return project_method_averaging_configuration_stage_guarded