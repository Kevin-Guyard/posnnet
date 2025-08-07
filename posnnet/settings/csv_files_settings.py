from typing import Self, Tuple, List, Any

from posnnet.setter_guards import (
    setter_typeguard, 
    setter_no_empty_string_guard
)
from posnnet.settings.base_settings import BaseSettings


class CsvFilesSettings(BaseSettings):

    def __init__(
        self: Self
    ) -> None:

        """
        Initialize the CSV files settings.

        Settings:
            - csv_sep (str): The character used to delimit columns in the CSV files (default = ',').
            - csv_encoding (str): The encoding used to encode the data in the CSV files (default = 'utf-8').
        """

        self.csv_sep = ","
        self.csv_encoding = "utf-8"

    @property
    def csv_sep(
        self: Self
    ) -> str:

        return self._csv_sep

    @csv_sep.setter
    @setter_typeguard
    @setter_no_empty_string_guard
    def csv_sep(
        self: Self,
        csv_sep: str
    ) -> None:

        self._csv_sep = csv_sep

    @property
    def csv_encoding(
        self: Self
    ) -> str:

        return self._csv_encoding

    @csv_encoding.setter
    @setter_typeguard
    @setter_no_empty_string_guard
    def csv_encoding(
        self: Self,
        csv_encoding: str
    ) -> None:

        self._csv_encoding = csv_encoding

    def validate_settings(
        self: Self,
        **kwargs: Any
    ) -> Tuple[List[str], List[str]]:

        """
        Validate the value of the different setting parameters.

        Returns:
            - warnings (List[str]): A list of warnings on the different setting parameter values.
            - errors (List[str]): A list of errors on the different setting parameter values.
        """

        warnings, errors = [], []

        # If the csv separator is not one of the usual ones, generate a warning.
        if self._csv_sep not in [',', ';', '\t', '|', ' ', ':']:
            warnings.append(f"The csv separator provided ('{self.csv_sep:s}') seems to be a unusual separator. Usual separator are: [',', ';', '\t', '|', ' ', ':']. Ignore this message if you are sure about the provided separator.")

        # If the csv encoding format is not one of the usual ones, generate a warning.
        if self._csv_encoding not in ['utf-8', 'utf-8-sig', 'latin1', 'iso-8859-1', 'cp1252', 'utf-16']:
            warnings.append(f"The csv encoding format provided ('{self.csv_encoding:s}') seems to be a unusual encoding format. Usual format are: ['utf-8', 'utf-8-sig', 'latin1', 'iso-8859-1', 'cp1252', 'utf-16']. Ignore this message if you are sure about the provided format.")

        return warnings, errors