import streamlit as st
import pandas as pd
import hashlib
import json
import os
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from utils.database import (
    init_database, get_or_create_user, save_data_upload, save_analysis,
    save_visualization, get_user_uploads, get_user_analyses, get_analysis_visualizations,
    User, DataUpload, Analysis, Visualization
)

# Default username for demo purposes
DEFAULT_USERNAME = "default_user"

def get_current_user() -> dict:
    """Get or create the current user and return as a dict"""
    # In a real app, this would use authentication
    # For now, we'll use a default user
    if 'current_user' not in st.session_state:
        db_connection = init_database()
        if not db_connection:
            st.error("Database connection failed")
            return None
        session = db_connection.get_session()
        try:
            user = get_or_create_user(DEFAULT_USERNAME)
            if user:
                # Copy user info to a dict
                user_dict = {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'created_at': user.created_at
                }
                st.session_state['current_user'] = user_dict
                return user_dict
            return None
        finally:
            session.close()
    return st.session_state['current_user']

def compute_file_hash(df: pd.DataFrame) -> str:
    """
    Compute a hash of the dataframe to detect duplicate uploads
    
    Parameters:
    df - Pandas dataframe to hash
    
    Returns:
    String hash representation
    """
    # Convert dataframe to string and hash it
    df_str = df.to_string()
    return hashlib.sha256(df_str.encode()).hexdigest()

def save_uploaded_data(df: pd.DataFrame, original_filename: str, profile: Dict[str, Any]) -> Optional[int]:
    """
    Save uploaded data information to database
    
    Parameters:
    df - The uploaded dataframe
    original_filename - The original filename of the uploaded file
    profile - The data profile information
    
    Returns:
    The data upload ID if successful, None otherwise
    """
    try:
        # Get current user
        user = get_current_user()
        if not user:
            st.error("User not found. Please reload the application.")
            return None
        
        # Compute file hash to detect duplicates
        file_hash = compute_file_hash(df)
        
        # Get file size (approx)
        file_size = df.memory_usage(deep=True).sum()
        
        # Save to database
        upload = save_data_upload(
            user_id=int(user['id']),
            filename=f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            original_filename=original_filename,
            file_size=int(file_size),
            row_count=len(df),
            column_count=len(df.columns),
            file_hash=file_hash,
            metadata=profile
        )
        
        if upload:
            return upload.id
        return None
    except Exception as e:
        st.error(f"Error saving uploaded data: {str(e)}")
        return None

def save_analysis_results(
    data_upload_id: int,
    analysis_type: str,
    analysis_method: str,
    parameters: Dict[str, Any],
    results: Dict[str, Any],
    execution_time: float,
    visualizations: List[Dict[str, Any]] = None
) -> Optional[int]:
    """
    Save analysis results to database
    
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
    try:
        # Get current user
        user = get_current_user()
        if not user:
            st.error("User not found. Please reload the application.")
            return None
        
        # Save analysis
        analysis = save_analysis(
            user_id=int(user['id']),
            data_upload_id=data_upload_id,
            analysis_type=analysis_type,
            analysis_method=analysis_method,
            parameters=parameters,
            results=results,
            execution_time=execution_time
        )
        
        if not analysis:
            return None
        
        # Save visualizations if provided
        if visualizations:
            for viz in visualizations:
                save_visualization(
                    analysis_id=int(analysis.id),
                    visualization_type=viz.get('type', 'unknown'),
                    title=viz.get('title', ''),
                    description=viz.get('description', ''),
                    configuration=viz.get('configuration', {}),
                    image_data=viz.get('image_data', '')
                )
        
        return analysis.id
    except Exception as e:
        st.error(f"Error saving analysis results: {str(e)}")
        return None

def get_user_history() -> Tuple[List[DataUpload], List[Analysis]]:
    """
    Get the current user's upload and analysis history
    
    Returns:
    Tuple of (uploads, analyses) lists
    """
    # Get current user
    user = get_current_user()
    if not user:
        return [], []
    
    # Get user's uploads and analyses
    uploads = get_user_uploads(user['id'])
    analyses = get_user_analyses(user['id'])
    
    return uploads, analyses

def render_history_view():
    """
    Render a view of the user's history
    """
    st.subheader("Your Analysis History")
    
    # Get user history
    uploads, analyses = get_user_history()
    
    # Display uploads
    if uploads:
        st.write("### Data Uploads")
        
        upload_data = []
        for upload in uploads:
            upload_data.append({
                "ID": upload.id,
                "Filename": upload.original_filename,
                "Date": upload.upload_date.strftime("%Y-%m-%d %H:%M"),
                "Rows": upload.row_count,
                "Columns": upload.column_count,
                "Size (KB)": f"{upload.file_size / 1024:.2f}"
            })
        
        st.dataframe(pd.DataFrame(upload_data), use_container_width=True)
    else:
        st.info("No data uploads found. Upload a CSV file to get started.")
    
    # Display analyses
    if analyses:
        st.write("### Analysis Results")
        
        analysis_data = []
        for analysis in analyses:
            analysis_data.append({
                "ID": analysis.id,
                "Type": analysis.analysis_type,
                "Method": analysis.analysis_method,
                "Date": analysis.created_at.strftime("%Y-%m-%d %H:%M"),
                "Status": "✅ Success" if analysis.is_successful else "❌ Failed",
                "Runtime (s)": f"{analysis.execution_time:.2f}"
            })
        
        st.dataframe(pd.DataFrame(analysis_data), use_container_width=True)
        
        # Allow user to select an analysis to view details
        if analysis_data:
            selected_analysis_id = st.selectbox(
                "Select an analysis to view details",
                options=[a["ID"] for a in analysis_data],
                format_func=lambda x: f"Analysis #{x} - {next((a['Type'] + ' / ' + a['Method']) for a in analysis_data if a['ID'] == x)}"
            )
            
            if selected_analysis_id:
                display_analysis_details(selected_analysis_id)
    else:
        st.info("No analyses found. Complete an analysis to see results here.")

def display_analysis_details(analysis_id: int):
    """
    Display details for a specific analysis
    
    Parameters:
    analysis_id - The ID of the analysis to display
    """
    try:
        # Get database connection
        db_connection = init_database()
        if not db_connection:
            st.error("Database connection not available")
            return
        
        session = db_connection.get_session()
        try:
            # Get analysis
            analysis = session.query(Analysis).filter(Analysis.id == analysis_id).first()
            
            if not analysis:
                st.error(f"Analysis #{analysis_id} not found")
                return
            
            # Display analysis details
            st.write(f"### Analysis #{analysis_id} Details")
            
            # Basic info
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Analysis Type", analysis.analysis_type)
                st.metric("Created", analysis.created_at.strftime("%Y-%m-%d %H:%M"))
            
            with col2:
                st.metric("Method", analysis.analysis_method)
                st.metric("Runtime", f"{analysis.execution_time:.2f} seconds")
            
            # Parameters
            st.write("#### Parameters")
            parameters = json.loads(analysis.parameters)
            st.json(parameters)
            
            # Results
            st.write("#### Results")
            results = json.loads(analysis.results)
            st.json(results)
            
            # Get visualizations
            visualizations = get_analysis_visualizations(analysis_id)
            
            if visualizations:
                st.write("#### Visualizations")
                for viz in visualizations:
                    st.write(f"**{viz.title}**")
                    st.write(viz.description)
                    
                    # If there's image data, display it
                    if viz.image_data:
                        st.image(viz.image_data)
        finally:
            session.close()
    except Exception as e:
        st.error(f"Error displaying analysis details: {str(e)}")

def time_execution(func, *args, **kwargs):
    """
    Time the execution of a function
    
    Parameters:
    func - The function to time
    *args, **kwargs - Arguments to pass to the function
    
    Returns:
    Tuple of (result, execution_time)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    return result, end_time - start_time