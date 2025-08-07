import pandas as pd
import pathlib
import streamlit as st
from typing import Self, Dict, List, Any, Tuple

from posnnet.dashboard.dashboard_data_collector import DashboardDataCollector


class EvaluationPage:

    def __init__(
        self: Self,
        path_evaluation_results: pathlib.Path,
    ) -> None:

        """
        Handle the evaluation results page of the dashboard.
        
        Args:
            - path_evaluation_results (pathlib.Path): The path where are stored the evaluation results.
        """

        # Collect the evaluation results.
        evaluation_dashboard_data = DashboardDataCollector.collect_evaluation_data(
            path_evaluation_results=path_evaluation_results
        )

        # Create a dataframe for every subcase.
        self.dfs_evaluation_dashboard = {
            case: {
                subcase: pd.DataFrame(data=evaluation_dashboard_data[case][subcase])
                for subcase in evaluation_dashboard_data[case].keys()
            }
            for case in evaluation_dashboard_data.keys()
        }

        # Collect the cases available.
        self.case_available = list(evaluation_dashboard_data.keys())

        # Collect the subcases available for each case.
        self.subcase_available = {
            case: list(evaluation_dashboard_data[case].keys())
            for case in self.case_available
        }

    def is_available(
        self: Self
    ) -> bool:

        """
        Indicate if the evaluation page is available. The evaluation page is available if at least one model has been evaluated.

        Returns:
            - evaluation_page_available (bool): Indicate if the evaluation page is available.
        """

        if len(self.dfs_evaluation_dashboard) > 0:
            evaluation_page_available = True
        else:
            evaluation_page_available = False

        return evaluation_page_available

    def __case_selection_section(
        self: Self
    ) -> Tuple[str, str]:

        """
        Display the case selection section of the evaluation page.

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

    def __reset_configurations_filters(
        self: Self
    ) -> None:

        """
        Reset the default value of the multiselect of the configurations filtering.
        """

        st.session_state.filter_key_version += 1

    def __configurations_filtering_section(
        self: Self,
        label_mapping: Dict[str, str],
        available_options: Dict[str, List[Any]]
    ) -> Dict[str, List[Any]]:

        """
        Display the configurations filtering section.

        Args:
            - label_mapping (Dict[str, str]): A dictionnary that map input key to label to display.
            - available_options (Dict[str, List[Any]]): A dictionnary that map input key to available choices.

        Returns:
            - configurations_filters_selection (Dict[str, List[Any]]): A dictionnary with the user selection.
        """

        # Formatting 1/3 for subheader and button.
        subheader_formatting = st.columns([1, 1, 1])

        # Formatting 1/3 for the filter selection input.
        filters_formatting = st.columns([1, 1, 1])

        # Subheader of the section.
        with subheader_formatting[0]:
            
            st.subheader(body="Filter the configurations")

            # Display the reset button if their is at least one available filter.
            if any([len(available_options[key]) > 1 for key in available_options.keys()]):
                
                st.button(
                    label="Reset configurations filters", 
                    on_click=self.__reset_configurations_filters
                )
        
        # Initialize a counter for columns formatting.
        columns_counter = 0
        # Intialize a dict to store user selection.
        configurations_filters_selection = {}

        if "filter_key_version" not in st.session_state:
            st.session_state.filter_key_version = 0

        # Iterate over filters.
        for key in label_mapping.keys():

            # Multi choices selection.
            with filters_formatting[columns_counter % 3]:

                # If their is more than one available option for this filter, display a mutli selection box.
                if len(available_options[key]) > 1:
                    configurations_filters_selection[key] = st.multiselect(
                        label=label_mapping[key], 
                        options=available_options[key], 
                        default=available_options[key], 
                        key=f"{key}_{st.session_state.filter_key_version:d}"
                    )
                    columns_counter += 1
                # Otherwise, no selection.
                else:
                    configurations_filters_selection[key] = available_options[key]
                    
        return configurations_filters_selection

    def __configurations_filtering_stat_section(
        self: Self,
        available_options: Dict[str, List[Any]],
        n_configurations_filtered: int
    ) -> None:

        """
        Display the configurations filtering stat section.

        Args:
            - available_options (Dict[str, List[Any]]): A dictionnary that map input key to available choices.
            - n_configurations_filtered (int): The number of configurations removed by the filtering selection.
        """

        # If their is no option (only one option for every criterion), display a message to inform that no selection is available.
        if all([
            len(available_options[key]) <= 1
            for key in available_options.keys()
        ]):
            st.text(body="No filter available.")
        # Otherwise, display the number of masked configurations based on the user filters selection.
        else:
            st.text(body=f"{n_configurations_filtered:d} {'configurations are' if n_configurations_filtered > 1 else 'configuration is':} masked using these filters.")

    def __reset_results_filters(
        self: Self,
    ) -> None:

        """
        Reset the default value of the multiselect of the configurations filtering.
        """
        
        st.session_state.max_ave = None
        st.session_state.max_rmve = None
        st.session_state.max_stdve = None
        st.session_state.max_ate = None
        st.session_state.max_rmte = None
        st.session_state.max_stdte = None
        st.session_state.max_rde = None
        st.session_state.max_rftte = None
        st.session_state.max_ste = None
        st.session_state.max_tte = None
        st.session_state.max_rte = None

    def __result_filtering_section(
        self: Self
    ) -> Dict[str, List[Any]]:

        """
        Display the results filtering section.

        Returns:
            - results_filters_selection (Dict[str, List[Any]]): A dictionnary with the user selection.
        """

        # Formatting 1/3 for subheader and button.
        subheader_formatting = st.columns([1, 1, 1])

        # Formatting 1/3 for the filter selection input.
        filters_formatting = st.columns([1, 1, 1])

        # Create a dictionnary for filters parameters mapping.
        results_filters = {
            "max_ave": (filters_formatting[0], "AVE lower than (in m/s):", 0.0, "%0.4f", 1e-4),
            "max_rmve": (filters_formatting[0], "RMVE lower than (%):", 0.0, "%0.2f", 1e-2),
            "max_stdve": (filters_formatting[0], "STDVE lower than (in m/s):", 0.0, "%0.4f", 1e-4),
            "max_ate": (filters_formatting[1], "ATE lower than (in m):", 0.0, "%0.2f", 1e-2),
            "max_rmte": (filters_formatting[1], "RMTE lower than (%):", 0.0, "%0.2f", 1e-2),
            "max_stdte": (filters_formatting[1], "STDTE lower than (in m):", 0.0, "%0.2f", 1e-2),
            "max_rde": (filters_formatting[1], "RDE lower than:", 0.0, "%0.4f", 1e-4),
            "max_rftte": (filters_formatting[2], "RFTTE lower than (%):", 0.0, "%0.2f", 1e-2),
            "max_ste": (filters_formatting[2], "STE lower than (%)", 0.0, "%0.2f", 1e-2),
            "max_tte": (filters_formatting[2], "TTE lower than (in m):", 0.0, "%0.2f", 1e-2),
            "max_rte": (filters_formatting[2], "RTE lower than (in degree):", 0.0, "%0.1f", 1e-1),
        }
        
        # Subheader of the section.
        with subheader_formatting[0]:
            st.subheader(body="Filter the results")
            st.button(
                label="Reset results filters", 
                on_click=self.__reset_results_filters
            )

        # Intialize a dict to store user selection.
        results_filters_selection = {}

        # Iterate over the filters.
        for key, (column_formatting, label, min_value, format_input, step) in results_filters.items():

            # Display the filter input and collect the user value.
            with column_formatting:
                results_filters_selection[key] = st.number_input(label=label, min_value=min_value, value=None, format=format_input, step=step, key=key)

        return results_filters_selection

    def __results_filtering_stat_section(
        self: Self,
        n_results_filtered: int
    ) -> None:

        """
        Display the results filtering stat section.

        Args:
            - n_results_filtered (int): The number of results removed by the filtering selection.
        """

        st.text(body=f"{n_results_filtered:d} {'results are' if n_results_filtered > 1 else 'result is':} masked using these filters.")

    def __call__(
        self: Self
    ) -> None:

        """
        Display the evaluation page on the dashboard.
        """

        # Header of the page.
        st.header(body="Evaluation dashboard")

        # Display the case selection section. Collect the choice of the user.
        selected_case, selected_subcase = self.__case_selection_section()

        st.markdown(
            "<hr style='border:1px solid #ddd;'/>",
            unsafe_allow_html=True
        )

        # Get the dataframe of the subcase.
        df_evaluation_dashboard = self.dfs_evaluation_dashboard[selected_case][selected_subcase]

        n_initial_results = len(df_evaluation_dashboard)

        # Create a mapping of labels inside multi selection input for configurations filtering.
        label_mapping = {
            "training_types": "Training type:",
            "gps_outage_duration_capacities": "GPS outage duration capacity:",
            "model_names": "Model name:",
            "coeffs_frequency_division": "Coefficient frequency division:",
            "adversarial_examples": "Adversarial example:"
        }

        # Create a mapping of available options inside multi selection input for configurations filtering.
        available_options = {
            "training_types": df_evaluation_dashboard["Training type"].unique(),
            "gps_outage_duration_capacities": df_evaluation_dashboard["GPS outage duration capacity"].unique(),
            "model_names": df_evaluation_dashboard["Model name"].unique(),
            "coeffs_frequency_division": sorted(df_evaluation_dashboard["Coeff freq div"].unique()),
            "adversarial_examples": df_evaluation_dashboard["Adversarial example"].unique()
        }

        # Display the configurations filtering section. Collect the filtering choices of the user.
        configurations_filters_selection = self.__configurations_filtering_section(
            label_mapping=label_mapping,
            available_options=available_options
        )

        # Filter the configurations.
        df_evaluation_dashboard = df_evaluation_dashboard[df_evaluation_dashboard["Training type"].isin(values=configurations_filters_selection["training_types"])]
        df_evaluation_dashboard = df_evaluation_dashboard[df_evaluation_dashboard["GPS outage duration capacity"].isin(values=configurations_filters_selection["gps_outage_duration_capacities"])]
        df_evaluation_dashboard = df_evaluation_dashboard[df_evaluation_dashboard["Model name"].isin(values=configurations_filters_selection["model_names"])]
        df_evaluation_dashboard = df_evaluation_dashboard[df_evaluation_dashboard["Coeff freq div"].isin(values=configurations_filters_selection["coeffs_frequency_division"])]
        df_evaluation_dashboard = df_evaluation_dashboard[df_evaluation_dashboard["Adversarial example"].isin(values=configurations_filters_selection["adversarial_examples"])]

        n_configurations_filtered = n_initial_results - len(df_evaluation_dashboard)

        # Display the configurations filtering stat section.
        self.__configurations_filtering_stat_section(
            available_options=available_options,
            n_configurations_filtered=n_configurations_filtered
        )

        st.markdown(
            "<hr style='border:1px solid #ddd;'/>",
            unsafe_allow_html=True
        )

        results_filters_selection = self.__result_filtering_section()

        # Filter the results.
        if results_filters_selection["max_ave"] is not None:
            df_evaluation_dashboard = df_evaluation_dashboard[df_evaluation_dashboard["AVE (in m/s)"] < results_filters_selection["max_ave"]]
        if results_filters_selection["max_rmve"] is not None:
            df_evaluation_dashboard = df_evaluation_dashboard[df_evaluation_dashboard["RMVE (%)"] < results_filters_selection["max_rmve"]]
        if results_filters_selection["max_stdve"] is not None:
            df_evaluation_dashboard = df_evaluation_dashboard[df_evaluation_dashboard["STDVE (in m/s)"] < results_filters_selection["max_stdve"]]
        if results_filters_selection["max_ate"] is not None:
            df_evaluation_dashboard = df_evaluation_dashboard[df_evaluation_dashboard["ATE (in m)"] < results_filters_selection["max_ate"]]
        if results_filters_selection["max_rmte"] is not None:
            df_evaluation_dashboard = df_evaluation_dashboard[df_evaluation_dashboard["RMTE (%)"] < results_filters_selection["max_rmte"]]
        if results_filters_selection["max_stdte"] is not None:
            df_evaluation_dashboard = df_evaluation_dashboard[df_evaluation_dashboard["STDTE (in m)"] < results_filters_selection["max_stdte"]]
        if results_filters_selection["max_rde"] is not None:
            df_evaluation_dashboard = df_evaluation_dashboard[df_evaluation_dashboard["RDE"] < results_filters_selection["max_rde"]]
        if results_filters_selection["max_rftte"] is not None:
            df_evaluation_dashboard = df_evaluation_dashboard[df_evaluation_dashboard["RFTTE (%)"] < results_filters_selection["max_rftte"]]
        if results_filters_selection["max_ste"] is not None:
            df_evaluation_dashboard = df_evaluation_dashboard[df_evaluation_dashboard["STE (%)"] < results_filters_selection["max_ste"]]
        if results_filters_selection["max_tte"] is not None:
            df_evaluation_dashboard = df_evaluation_dashboard[df_evaluation_dashboard["TTE (in m)"] < results_filters_selection["max_tte"]]
        if results_filters_selection["max_rte"] is not None:
            df_evaluation_dashboard = df_evaluation_dashboard[df_evaluation_dashboard["RTE (in deg)"] < results_filters_selection["max_rte"]]

        n_results_filtered = n_initial_results - n_configurations_filtered - len(df_evaluation_dashboard)

        self.__results_filtering_stat_section(
            n_results_filtered=n_results_filtered
        )

        st.markdown(
            "<hr style='border:1px solid #ddd;'/>",
            unsafe_allow_html=True
        )

        # Subheader for results part.
        st.subheader(body="Results")

        # Display the results table.
        st.dataframe(df_evaluation_dashboard, use_container_width=False)