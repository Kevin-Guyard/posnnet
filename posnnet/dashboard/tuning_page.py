import pandas as pd
import pathlib
import streamlit as st
from typing import Self, Dict, List, Any, Tuple

from posnnet.dashboard.dashboard_data_collector import DashboardDataCollector


class TuningPage:

    def __init__(
        self: Self,
        path_studies: pathlib.Path,
        path_tuning_results: pathlib.Path
    ) -> None:

        """
        Handle the tuning results page of the dashboard.
        
        Args:
            - path_studies (pathlib.Path): The path where are stored the study object of the tuning.
            - path_tuning_results (pathlib.Path): The path where are stored the tuning results.
        """

        # Collect the tuning results.
        tuning_dashboard_data = DashboardDataCollector.collect_tuning_data(
            path_studies=path_studies,
            path_tuning_results=path_tuning_results
        )

        # Create a dataframe for every subcase.
        self.dfs_tuning_dashboard = {
            training_type: {
                gps_outage_duration: pd.DataFrame(data=tuning_dashboard_data[training_type][gps_outage_duration])
                for gps_outage_duration in tuning_dashboard_data[training_type].keys()
            }
            for training_type in tuning_dashboard_data.keys()
        }

        # Collect the cases available.
        self.training_type_available = list(tuning_dashboard_data.keys())

        # Collect the subcases available for each case.
        self.gps_outage_durations_available = {
            training_type: list(tuning_dashboard_data[training_type].keys())
            for training_type in self.training_type_available
        }

    def is_available(
        self: Self
    ) -> bool:

        """
        Indicate if the tuning page is available. The tuning page is available if at least one model has been tuned.

        Returns:
            - tuning_page_available (bool): Indicate if the tuning page is available.
        """

        if len(self.dfs_tuning_dashboard) > 0:
            tuning_page_available = True
        else:
            tuning_page_available = False

        return tuning_page_available

    def __case_selection_section(
        self: Self
    ) -> Tuple[str, str]:

        """
        Display the case selection section of the tuning page.

        Returns:
            - selected_training_type (str): The training type case selected by the user.
            - selected_gps_outage_duration (str): The gps outage duration subcase selected by the user.
        """

        # Formatting 1/3 by items.
        section_formatting = st.columns([1, 1, 1])

        # Case selection box.
        with section_formatting[0]:
            selected_training_type = st.selectbox(
                label="Select the training type:", 
                options=self.training_type_available
            )

        # Subcase selection box.
        with section_formatting[1]:
            selected_gps_outage_duration = st.selectbox(
                label="Select the GPS outage duration capacity of the models:", 
                options=self.gps_outage_durations_available[selected_training_type]
            )

        return selected_training_type, selected_gps_outage_duration

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
        
        st.session_state.max_velocity_loss = None
        st.session_state.max_ate = None

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
            "max_velocity_loss": (filters_formatting[0], "Velocity loss lower than:", 0.0, "%0.6f", 1e-6), # 10.0
            "max_ate": (filters_formatting[1], "ATE lower than (in m):", 0.0, "%0.2f", 1e-2) #100.0
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
        Display the tuning page on the dashboard.
        """

        # Header of the page.
        st.header(body="Tuning dashboard")

        # Display the case selection section. Collect the choice of the user.
        selected_training_type, selected_gps_outage_duration = self.__case_selection_section()

        st.markdown(
            "<hr style='border:1px solid #ddd;'/>",
            unsafe_allow_html=True
        )

        # Get the dataframe of the subcase.
        df_tuning_dashboard = self.dfs_tuning_dashboard[selected_training_type][selected_gps_outage_duration]

        n_initial_results = len(df_tuning_dashboard)

        # Create a mapping of labels inside multi selection input for configurations filtering.
        label_mapping = {
            "model_names": "Model name:",
            "coeffs_frequency_division": "Coefficient frequency division:",
            "adversarial_examples": "Adversarial example:"
        }

        # Create a mapping of available options inside multi selection input for configurations filtering.
        available_options = {
            "model_names": df_tuning_dashboard["Model name"].unique(),
            "coeffs_frequency_division": sorted(df_tuning_dashboard["Coeff freq div"].unique()),
            "adversarial_examples": df_tuning_dashboard["Adversarial example"].unique()
        }

        # Display the configurations filtering section. Collect the filtering choices of the user.
        configurations_filters_selection = self.__configurations_filtering_section(
            label_mapping=label_mapping,
            available_options=available_options
        )

        # Filter the configurations.
        df_tuning_dashboard = df_tuning_dashboard[df_tuning_dashboard["Model name"].isin(values=configurations_filters_selection["model_names"])]
        df_tuning_dashboard = df_tuning_dashboard[df_tuning_dashboard["Coeff freq div"].isin(values=configurations_filters_selection["coeffs_frequency_division"])]
        df_tuning_dashboard = df_tuning_dashboard[df_tuning_dashboard["Adversarial example"].isin(values=configurations_filters_selection["adversarial_examples"])]

        n_configurations_filtered = n_initial_results - len(df_tuning_dashboard)

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
        if results_filters_selection["max_velocity_loss"] is not None:
            df_tuning_dashboard = df_tuning_dashboard[df_tuning_dashboard["Velocity loss"] < results_filters_selection["max_velocity_loss"]]
        if results_filters_selection["max_ate"] is not None:
            df_tuning_dashboard = df_tuning_dashboard[df_tuning_dashboard["ATE (in m)"] < results_filters_selection["max_ate"]]

        n_results_filtered = n_initial_results - n_configurations_filtered - len(df_tuning_dashboard)

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
        st.dataframe(df_tuning_dashboard, use_container_width=False)