import pandas as pd
import pathlib
import streamlit as st
from typing import Self, Dict, List, Any, Tuple

from posnnet.dashboard.dashboard_data_collector import DashboardDataCollector


class AveragingPage:

    def __init__(
        self: Self,
        path_evaluation_results: pathlib.Path,
    ) -> None:

        """
        Handle the averaging results page of the dashboard.
        
        Args:
            - path_evaluation_results (pathlib.Path): The path where are stored the evaluation results.
        """

        # Collect the averaging results.
        averaging_dashboard_data = DashboardDataCollector.collect_averaging_data(
            path_evaluation_results=path_evaluation_results
        )

        # Create a dataframe for every subcase.
        self.dfs_averaging_dashboard = {
            case: {
                subcase: pd.DataFrame(data=averaging_dashboard_data[case][subcase])
                for subcase in averaging_dashboard_data[case].keys()
            }
            for case in averaging_dashboard_data.keys()
        }

        # Collect the cases available.
        self.case_available = list(averaging_dashboard_data.keys())

        # Collect the subcases available for each case.
        self.subcase_available = {
            case: list(averaging_dashboard_data[case].keys())
            for case in self.case_available
        }

    def is_available(
        self: Self
    ) -> bool:

        """
        Indicate if the averaging page is available. The averaging page is available if at least one model has been evaluated.

        Returns:
            - averaging_page_available (bool): Indicate if the averaging page is available.
        """

        if len(self.dfs_averaging_dashboard) > 0:
            averaging_page_available = True
        else:
            averaging_page_available = False

        return averaging_page_available

    def __case_selection_section(
        self: Self
    ) -> Tuple[str, str]:

        """
        Display the case selection section of the averaging page.

        Returns:
            - selected_case (str): The case selected by the user.
            - selected_subcase (str): The gsubcase selected by the user.
        """

        # Formatting 1/3 by items.
        section_formatting = st.columns([1, 1, 1])

        # Case selection box.
        with section_formatting[0]:
            selected_case = st.selectbox(
                label="Select the case of GPS outage:", 
                options=self.case_available
            )

        # Subcase selection box.
        with section_formatting[1]:
            selected_subcase = st.selectbox(
                label="Select the GPS outage duration subcase:", 
                options=self.subcase_available[selected_case]
            )

        return selected_case, selected_subcase

    def __call__(
        self: Self
    ) -> None:

        """
        Display the averaging page on the dashboard.
        """

        # Header of the page.
        st.header(body="Averaging dashboard")

        # Display the case selection section. Collect the choice of the user.
        selected_case, selected_subcase = self.__case_selection_section()

        st.markdown(
            "<hr style='border:1px solid #ddd;'/>",
            unsafe_allow_html=True
        )

        # Get the dataframe of the subcase.
        df_averaging_dashboard = self.dfs_averaging_dashboard[selected_case][selected_subcase]

        st.markdown(
            "<hr style='border:1px solid #ddd;'/>",
            unsafe_allow_html=True
        )

        # Subheader for results part.
        st.subheader(body="Results")

        # Display the results table.
        st.dataframe(df_averaging_dashboard, use_container_width=False)