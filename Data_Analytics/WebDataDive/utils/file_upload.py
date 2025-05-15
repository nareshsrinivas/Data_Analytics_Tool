import pandas as pd
import numpy as np
import streamlit as st
import io
from typing import Tuple, Dict, Any, List

def validate_csv(file) -> Tuple[bool, str, pd.DataFrame]:
    """
    Validates a CSV file and returns a tuple containing:
    (is_valid, message, dataframe)
    
    Parameters:
    file - The uploaded file object
    
    Returns:
    Tuple containing validation status, message, and dataframe (empty if invalid)
    """
    if file is None:
        return False, "No file uploaded", pd.DataFrame()
    
    try:
        # Read the first chunk to validate
        df = pd.read_csv(file)
        
        # Check if dataframe is empty
        if df.empty:
            return False, "The uploaded CSV file is empty", pd.DataFrame()
        
        # Basic validation passed
        return True, "File successfully validated", df
    
    except pd.errors.EmptyDataError:
        return False, "The uploaded CSV file is empty", pd.DataFrame()
    except pd.errors.ParserError:
        return False, "Unable to parse the CSV file. Please check the format", pd.DataFrame()
    except Exception as e:
        return False, f"An error occurred: {str(e)}", pd.DataFrame()

def get_data_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a basic profile of the dataset
    
    Parameters:
    df - Pandas dataframe to analyze
    
    Returns:
    Dictionary containing profile information
    """
    if df.empty:
        return {}
    
    # Get basic statistics
    profile = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
    }
    
    # Add numeric column statistics if any exist
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        profile["numeric_columns"] = numeric_columns
        profile["numeric_stats"] = df[numeric_columns].describe().to_dict()
    
    # Add categorical column statistics if any exist
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_columns:
        profile["categorical_columns"] = categorical_columns
        profile["categorical_stats"] = {
            col: {
                "unique_values": df[col].nunique(),
                "top_values": df[col].value_counts().head(5).to_dict() if df[col].nunique() < 100 else {}
            } for col in categorical_columns
        }
    
    return profile

def render_file_upload() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Renders the file upload section and returns the uploaded dataframe
    and its profile information
    
    Returns:
    Tuple with dataframe and profile dictionary
    """
    st.header("1. File Upload", divider=True)
    
    # Upload file box with detailed instructions
    uploaded_file = st.file_uploader(
        "Upload your CSV data file",
        type=["csv"],
        help="Please upload a CSV file containing your data. The file should have headers in the first row."
    )
    
    # Initialize the session state for dataframe if it doesn't exist
    if 'dataframe' not in st.session_state:
        st.session_state.dataframe = None
        st.session_state.data_profile = {}
    
    if uploaded_file is not None:
        # Validate the uploaded file
        is_valid, message, df = validate_csv(uploaded_file)
        
        if is_valid:
            st.success(message)
            
            # Generate the data profile
            profile = get_data_profile(df)
            
            # Store in session state
            st.session_state.dataframe = df
            st.session_state.data_profile = profile
            st.session_state.uploaded_filename = uploaded_file.name
            
            # Display file information
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Display basic statistics
            st.subheader("File Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", profile["rows"])
                st.metric("Columns", profile["columns"])
            
            with col2:
                # Calculate total missing values
                total_missing = sum(profile["missing_values"].values())
                missing_percentage = total_missing / (profile["rows"] * profile["columns"]) * 100
                st.metric("Missing Values", f"{total_missing} ({missing_percentage:.2f}%)")
                st.metric("Data Size", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")
            
            # Display column information
            st.subheader("Column Information")
            col_info = []
            for col in df.columns:
                col_type = profile["dtypes"][col]
                missing = profile["missing_values"][col]
                missing_pct = profile["missing_percentage"][col]
                col_info.append({
                    "Column": col,
                    "Type": col_type,
                    "Missing Values": missing,
                    "Missing (%)": f"{missing_pct:.2f}%"
                })
            
            st.dataframe(pd.DataFrame(col_info), use_container_width=True)
            
            return df, profile
        else:
            st.error(message)
            st.session_state.dataframe = None
            st.session_state.data_profile = {}
            return pd.DataFrame(), {}
    
    # If we have a dataframe in the session state, display it
    elif st.session_state.dataframe is not None:
        df = st.session_state.dataframe
        profile = st.session_state.data_profile
        
        # Display file information
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Display basic statistics
        st.subheader("File Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", profile["rows"])
            st.metric("Columns", profile["columns"])
        
        with col2:
            # Calculate total missing values
            total_missing = sum(profile["missing_values"].values())
            missing_percentage = total_missing / (profile["rows"] * profile["columns"]) * 100
            st.metric("Missing Values", f"{total_missing} ({missing_percentage:.2f}%)")
            st.metric("Data Size", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        # Display column information
        st.subheader("Column Information")
        col_info = []
        for col in df.columns:
            col_type = profile["dtypes"][col]
            missing = profile["missing_values"][col]
            missing_pct = profile["missing_percentage"][col]
            col_info.append({
                "Column": col,
                "Type": col_type,
                "Missing Values": missing,
                "Missing (%)": f"{missing_pct:.2f}%"
            })
        
        st.dataframe(pd.DataFrame(col_info), use_container_width=True)
        
        return df, profile
    else:
        # Display upload instructions when no file is uploaded
        st.info("Please upload a CSV file to get started with the analysis pipeline.")
        
        # Example of expected CSV format
        st.subheader("Expected CSV Format Example:")
        example_data = {
            "age": [25, 30, 35, 40],
            "income": [50000, 60000, None, 80000],
            "education": ["Bachelor", "Master", "PhD", "Bachelor"],
            "satisfaction": [7, 8, 9, 6]
        }
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df, use_container_width=True)
        
        st.markdown("""
        **CSV File Requirements:**
        - First row should contain column headers
        - Columns should be separated by commas
        - Numeric values should use period as decimal separator
        - Missing values can be empty or represented as NA, N/A, NULL, etc.
        """)
        
        return pd.DataFrame(), {}
