import streamlit as st
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from utils.db_manager import (
    render_history_view, save_uploaded_data, save_analysis_results,
    time_execution, get_current_user
)
from utils.database import init_database

def render_history_page():
    """
    Render the history page showing past uploads and analyses
    """
    st.header("Analysis History", divider=True)
    
    # Initialize database connection
    db_connection = init_database()
    if not db_connection:
        st.error("Database connection could not be established. Some features may not work properly.")
        return
    
    # Get current user
    user = get_current_user()
    if not user:
        st.error("User account could not be created or retrieved. Please restart the application.")
        return
    
    st.info(f"Viewing history for user: {user['username']}")
    
    # Render the history view
    render_history_view()
    
    # Database management section (for admin/developer use)
    with st.expander("Database Management", expanded=False):
        st.warning("This section is for application administrators only.")
        
        # Test database connection
        if st.button("Test Database Connection"):
            try:
                session = db_connection.get_session()
                from sqlalchemy import text
                session.execute(text("SELECT 1"))
                session.close()
                st.success("Database connection successful!")
            except Exception as e:
                st.error(f"Database connection failed: {str(e)}")
        
        # Initialize database tables
        if st.button("Initialize Database Tables"):
            try:
                db_connection.create_tables()
                st.success("Database tables created successfully!")
            except Exception as e:
                st.error(f"Failed to create database tables: {str(e)}")

def record_file_upload(df: pd.DataFrame, filename: str, profile: Dict[str, Any]) -> Optional[int]:
    """
    Record a file upload to the database
    
    Parameters:
    df - The uploaded dataframe
    filename - The original filename
    profile - The data profile information
    
    Returns:
    The upload ID if successful, None otherwise
    """
    return save_uploaded_data(df, filename, profile)

def record_analysis(
    data_upload_id: int,
    analysis_type: str,
    analysis_method: str,
    parameters: Dict[str, Any],
    results: Dict[str, Any],
    execution_time: float,
    visualizations: List[Dict[str, Any]] = None
) -> Optional[int]:
    """
    Record an analysis to the database
    
    Parameters:
    data_upload_id - The ID of the related data upload
    analysis_type - Type of analysis (e.g., 'exploratory', 'regression')
    analysis_method - Method used (e.g., 'linear_regression')
    parameters - Dictionary of analysis parameters
    results - Dictionary of analysis results
    execution_time - Time taken to execute analysis in seconds
    visualizations - List of visualization metadata dictionaries
    
    Returns:
    The analysis ID if successful, None otherwise
    """
    return save_analysis_results(
        data_upload_id=data_upload_id,
        analysis_type=analysis_type,
        analysis_method=analysis_method,
        parameters=parameters,
        results=results,
        execution_time=execution_time,
        visualizations=visualizations
    )