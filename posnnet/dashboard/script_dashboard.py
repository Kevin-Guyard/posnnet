import streamlit as st

st.set_page_config(
    page_title="POSNNET dashboard",
    layout="wide"
)

import sys, pathlib

# Walk two levels up to your project root
project_root = pathlib.Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

    
from posnnet.dashboard.dashboard_manager import DashboardManager


if __name__ == "__main__":

    # Collect the argument and cast them to pathlib.Path
    path_studies, path_tuning_results, path_evaluation_results = map(pathlib.Path, sys.argv[1:4])

    # Start the dashboard
    DashboardManager.display(
        path_studies=path_studies,
        path_tuning_results=path_tuning_results,
        path_evaluation_results=path_evaluation_results
    )