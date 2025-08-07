import pandas as pd
import pathlib
import streamlit as st
from typing import TypeVar, Type, Dict, List, Any

from posnnet.dashboard.averaging_page import AveragingPage
from posnnet.dashboard.evaluation_page import EvaluationPage
from posnnet.dashboard.tuning_page import TuningPage


class DashboardManager:

    @classmethod
    def display(
        cls: Type[TypeVar("DashboardManager")],
        path_studies: pathlib.Path,
        path_tuning_results: pathlib.Path,
        path_evaluation_results: pathlib.Path
    ) -> None:

        """
        Display the dashboard.

        Args:
            - path_studies (pathlib.Path): The path where are stored the study object of the tuning.
            - path_tuning_results (pathlib.Path): The path where are stored the tuning results.
            - path_evaluation_results (pathlib.Path): The path where are stored the evaluation results.
        """

        tuning_page = TuningPage(
            path_studies=path_studies,
            path_tuning_results=path_tuning_results
        )

        evaluation_page = EvaluationPage(
            path_evaluation_results=path_evaluation_results
        )

        averaging_page = AveragingPage(
            path_evaluation_results=path_evaluation_results
        )

        # Collect the available pages.
        pages = []
        
        if tuning_page.is_available():
            pages += ["Tuning results"]
        if evaluation_page.is_available():
            pages += ["Evaluation results"]
        if averaging_page.is_available():
            pages += ["Averaging results"]

        # If their is at least one page available, display a page menu and the selected pages.
        if len(pages) > 0:
            
            page = st.sidebar.radio(label="Go to", options=pages)
    
            if page == "Tuning results":
                tuning_page()
            elif page == "Evaluation results":
                evaluation_page()
            elif page == "Averaging results":
                averaging_page()

        # Otherwise, display a message to indicate that their is not any page to display.
        else:

            st.header(body="No results found.")
            st.text("The dashboard renders the tuning and evaluation results. You have first to perform tuning and evaluation steps to have results displayed here.")