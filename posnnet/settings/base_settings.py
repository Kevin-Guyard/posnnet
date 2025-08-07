from typing import Self, Any


class BaseSettings:

    """
    Base class for settings.
    """

    def print_settings(
        self: Self,
    ) -> None:

        """
        Print the actual settings.
        """

        for param_name, param_value in vars(self).items():
            print(f"The parameter '{param_name[1:]:s}' is set to {param_value}")

    def set_settings(
        self: Self,
        **kwargs: Any
    ) -> None:

        """
        Set the settings.

        Args:
            - kwargs (Any): The settings to modifiate

        Raises:
            - Exception: If any parameter in kwargs does not exist in the settings class.
        """

        # Check that the provided parameters name exists.
        if not all([
            f"_{param_name}" in vars(self).keys()
            for param_name in kwargs.keys()
        ]):
            raise Exception("You are trying to set a setting that does not exist.")

        # Iterate over the provided parameters to set them into the setting class.
        for param_name, param_value in kwargs.items():
            try:
                setattr(self, param_name, param_value)
            except Exception as exception:
                print(f" - {exception} The value of the parameter has been ignored. The actual value is {vars(self)[f"_{param_name}"]}.", end="\n\n")