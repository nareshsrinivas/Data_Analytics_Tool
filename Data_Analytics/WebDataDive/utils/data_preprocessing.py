import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple, Any
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

def render_problem_definition(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Renders the problem definition interface and returns the selected options
    
    Parameters:
    df - The dataset as a pandas DataFrame
    
    Returns:
    Dictionary containing problem definition selections
    """
    st.header("2. Problem Definition", divider=True)
    
    if df.empty:
        st.warning("Please upload a dataset first to define your analysis problem")
        return {}
    
    # Initialize session state for problem definition if not exists
    if 'problem_definition' not in st.session_state:
        st.session_state.problem_definition = {
            'analysis_goal': 'Exploratory Analysis',
            'target_variable': None,
            'analysis_method': 'Descriptive Statistics',
            'features': []
        }
    
    problem_def = st.session_state.problem_definition
    
    # Analysis goal selection
    st.subheader("Analysis Goal")
    analysis_goal = st.selectbox(
        "What is your primary analysis goal?",
        options=[
            "Exploratory Analysis", 
            "Correlation Analysis", 
            "Regression Analysis",
            "Classification Analysis",
            "Clustering",
            "Time Series Analysis"
        ],
        index=0 if problem_def['analysis_goal'] not in ["Exploratory Analysis", "Correlation Analysis", 
                                                       "Regression Analysis", "Classification Analysis",
                                                       "Clustering", "Time Series Analysis"] else
              ["Exploratory Analysis", "Correlation Analysis", "Regression Analysis", 
               "Classification Analysis", "Clustering", "Time Series Analysis"].index(problem_def['analysis_goal']),
        help="Select the primary goal of your data analysis"
    )
    problem_def['analysis_goal'] = analysis_goal
    
    # Target variable selection if applicable
    if analysis_goal in ["Regression Analysis", "Classification Analysis"]:
        st.subheader("Target Variable")
        
        # For regression, suggest numeric columns
        if analysis_goal == "Regression Analysis":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                target_variable = st.selectbox(
                    "Select the target variable to predict",
                    options=["None"] + numeric_cols,
                    index=0 if problem_def['target_variable'] is None else 
                          (numeric_cols.index(problem_def['target_variable']) + 1 if problem_def['target_variable'] in numeric_cols else 0),
                    help="This is the variable you want to predict or explain"
                )
                problem_def['target_variable'] = None if target_variable == "None" else target_variable
            else:
                st.warning("No numeric columns available for regression analysis")
                problem_def['target_variable'] = None
        
        # For classification, suggest categorical columns
        elif analysis_goal == "Classification Analysis":
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            # Also include numeric columns with few unique values that might be categorical
            for col in df.select_dtypes(include=[np.number]).columns:
                if df[col].nunique() < 10:  # Heuristic for potential categorical variables
                    cat_cols.append(col)
            
            if cat_cols:
                target_variable = st.selectbox(
                    "Select the target variable to classify",
                    options=["None"] + cat_cols,
                    index=0 if problem_def['target_variable'] is None else 
                          (cat_cols.index(problem_def['target_variable']) + 1 if problem_def['target_variable'] in cat_cols else 0),
                    help="This is the categorical variable you want to predict"
                )
                problem_def['target_variable'] = None if target_variable == "None" else target_variable
            else:
                st.warning("No suitable categorical columns available for classification analysis")
                problem_def['target_variable'] = None
    else:
        problem_def['target_variable'] = None
    
    # Analysis method selection
    st.subheader("Analysis Method")
    
    # Different methods based on analysis goal
    if analysis_goal == "Exploratory Analysis":
        analysis_method = st.selectbox(
            "Select analysis method",
            options=["Descriptive Statistics", "Data Distribution Analysis", "Outlier Detection"],
            index=0 if problem_def['analysis_method'] not in ["Descriptive Statistics", "Data Distribution Analysis", "Outlier Detection"] else
                  ["Descriptive Statistics", "Data Distribution Analysis", "Outlier Detection"].index(problem_def['analysis_method']),
            help="Method to explore and understand your data"
        )
    elif analysis_goal == "Correlation Analysis":
        analysis_method = st.selectbox(
            "Select analysis method",
            options=["Pearson Correlation", "Spearman Correlation", "Feature Importance"],
            index=0 if problem_def['analysis_method'] not in ["Pearson Correlation", "Spearman Correlation", "Feature Importance"] else
                  ["Pearson Correlation", "Spearman Correlation", "Feature Importance"].index(problem_def['analysis_method']),
            help="Method to assess relationships between variables"
        )
    elif analysis_goal == "Regression Analysis":
        analysis_method = st.selectbox(
            "Select analysis method",
            options=["Linear Regression", "Multiple Regression"],
            index=0 if problem_def['analysis_method'] not in ["Linear Regression", "Multiple Regression"] else
                  ["Linear Regression", "Multiple Regression"].index(problem_def['analysis_method']),
            help="Method for regression analysis"
        )
    elif analysis_goal == "Classification Analysis":
        analysis_method = st.selectbox(
            "Select analysis method",
            options=["Logistic Regression"],
            index=0,
            help="Method for classification analysis"
        )
    elif analysis_goal == "Clustering":
        analysis_method = st.selectbox(
            "Select analysis method",
            options=["K-Means Clustering"],
            index=0,
            help="Method for clustering analysis"
        )
    elif analysis_goal == "Time Series Analysis":
        analysis_method = st.selectbox(
            "Select analysis method",
            options=["Trend Analysis", "Seasonality Analysis"],
            index=0 if problem_def['analysis_method'] not in ["Trend Analysis", "Seasonality Analysis"] else
                  ["Trend Analysis", "Seasonality Analysis"].index(problem_def['analysis_method']),
            help="Method for time series analysis"
        )
    
    problem_def['analysis_method'] = analysis_method
    
    # Feature selection
    st.subheader("Feature Selection")
    
    # Display columns based on the analysis goal and method
    if analysis_goal in ["Exploratory Analysis", "Correlation Analysis"]:
        all_columns = df.columns.tolist()
        
        # Allow multiselect for features
        selected_features = st.multiselect(
            "Select features to include in the analysis",
            options=all_columns,
            default=problem_def['features'] if problem_def['features'] else all_columns[:min(5, len(all_columns))],
            help="Select the columns you want to analyze"
        )
        
        problem_def['features'] = selected_features
        
    elif analysis_goal in ["Regression Analysis", "Classification Analysis"]:
        if problem_def['target_variable']:
            # Exclude the target variable from potential features
            feature_options = [col for col in df.columns if col != problem_def['target_variable']]
            
            # Allow multiselect for features
            selected_features = st.multiselect(
                "Select features to include in the analysis",
                options=feature_options,
                default=problem_def['features'] if problem_def['features'] else feature_options[:min(5, len(feature_options))],
                help="Select the predictor variables for your model"
            )
            
            problem_def['features'] = selected_features
        else:
            st.warning("Please select a target variable first")
            problem_def['features'] = []
    
    elif analysis_goal == "Clustering":
        # For clustering, we typically want numeric features
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_features:
            selected_features = st.multiselect(
                "Select numeric features for clustering",
                options=numeric_features,
                default=problem_def['features'] if problem_def['features'] else numeric_features[:min(3, len(numeric_features))],
                help="Select the numeric columns to use for clustering"
            )
            
            problem_def['features'] = selected_features
        else:
            st.warning("No numeric features available for clustering")
            problem_def['features'] = []
    
    elif analysis_goal == "Time Series Analysis":
        # For time series, we need a date/time column and a numeric value column
        
        # Try to identify potential datetime columns
        datetime_cols = []
        for col in df.columns:
            # Check if column name suggests it's a date
            if any(term in col.lower() for term in ['date', 'time', 'day', 'month', 'year']):
                datetime_cols.append(col)
            # Try to convert to datetime to check if it works
            elif df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col], errors='raise')
                    datetime_cols.append(col)
                except:
                    pass
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if datetime_cols and numeric_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                time_col = st.selectbox(
                    "Select the time/date column",
                    options=datetime_cols,
                    index=0,
                    help="Column containing time/date information"
                )
            
            with col2:
                value_col = st.selectbox(
                    "Select the value column to analyze over time",
                    options=numeric_cols,
                    index=0,
                    help="Numeric column to analyze for trends and patterns over time"
                )
            
            problem_def['features'] = [time_col, value_col]
        else:
            if not datetime_cols:
                st.warning("No datetime columns identified for time series analysis")
            if not numeric_cols:
                st.warning("No numeric columns available for time series analysis")
            problem_def['features'] = []
    
    # Save to session state
    st.session_state.problem_definition = problem_def
    
    return problem_def

def render_data_preprocessing(df: pd.DataFrame, problem_def: Dict[str, Any]) -> pd.DataFrame:
    """
    Renders the data preprocessing interface and returns the processed dataframe
    
    Parameters:
    df - The original dataset
    problem_def - The problem definition dictionary
    
    Returns:
    Processed dataframe
    """
    st.header("3. Data Preprocessing", divider=True)
    
    if df.empty:
        st.warning("Please upload a dataset first to preprocess your data")
        return df
    
    # Initialize preprocessing state if not exists
    if 'preprocessing_options' not in st.session_state:
        st.session_state.preprocessing_options = {
            'handle_missing': 'remove_rows',
            'handle_outliers': 'none',
            'scaling_method': 'none',
            'encoding_method': 'none',
            'feature_selection': problem_def.get('features', []),
            'processed_df': None
        }
    
    preprocessing = st.session_state.preprocessing_options
    
    # Create a working copy of the dataframe
    working_df = df.copy()
    
    # Feature filtering based on problem definition
    if problem_def and 'features' in problem_def and problem_def['features']:
        # Include target variable if it exists
        if problem_def.get('target_variable'):
            selected_columns = problem_def['features'] + [problem_def['target_variable']]
            selected_columns = list(set(selected_columns))  # Remove duplicates
        else:
            selected_columns = problem_def['features']
        
        # Filter dataframe to selected columns
        if selected_columns:
            working_df = working_df[selected_columns]
    
    st.subheader("Data Overview")
    # Create tabs for different preprocessing steps
    missing_tab, outliers_tab, scaling_tab, encoding_tab = st.tabs(
        ["Handle Missing Values", "Outlier Treatment", "Scaling/Normalization", "Categorical Encoding"]
    )
    
    # Handle missing values
    with missing_tab:
        st.subheader("Missing Values Treatment")
        
        # Show missing value summary
        missing_summary = working_df.isnull().sum()
        if missing_summary.sum() > 0:
            missing_df = pd.DataFrame({
                'Column': missing_summary.index,
                'Missing Values': missing_summary.values,
                'Percentage': (missing_summary.values / len(working_df) * 100).round(2)
            })
            missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values('Missing Values', ascending=False)
            
            st.dataframe(missing_df, use_container_width=True)
            
            # Missing value handling options
            handle_missing = st.radio(
                "How to handle missing values?",
                options=["Remove rows with missing values", 
                         "Remove columns with high missing rate", 
                         "Fill missing with mean/mode",
                         "Fill missing with median",
                         "Fill missing with constant value",
                         "Keep missing values"],
                index=0 if preprocessing['handle_missing'] not in ["remove_rows", "remove_columns", "fill_mean_mode", 
                                                                  "fill_median", "fill_constant", "keep"] else
                      ["remove_rows", "remove_columns", "fill_mean_mode", 
                       "fill_median", "fill_constant", "keep"].index(preprocessing['handle_missing']),
                horizontal=True,
                help="Select a method to handle missing values in your dataset"
            )
            
            # Map radio button selection to state value
            option_mapping = {
                "Remove rows with missing values": "remove_rows",
                "Remove columns with high missing rate": "remove_columns",
                "Fill missing with mean/mode": "fill_mean_mode",
                "Fill missing with median": "fill_median",
                "Fill missing with constant value": "fill_constant",
                "Keep missing values": "keep"
            }
            preprocessing['handle_missing'] = option_mapping[handle_missing]
            
            # Additional options based on selection
            if preprocessing['handle_missing'] == "remove_columns":
                threshold = st.slider(
                    "Remove columns with missing values above (%)",
                    min_value=0,
                    max_value=100,
                    value=50,
                    help="Columns with missing percentage above this threshold will be removed"
                )
                preprocessing['missing_threshold'] = threshold
            
            elif preprocessing['handle_missing'] == "fill_constant":
                st.text_input(
                    "Constant value to fill",
                    value="0",
                    key="missing_fill_value",
                    help="Value to use for filling missing data"
                )
                preprocessing['missing_fill_value'] = st.session_state.missing_fill_value
        else:
            st.success("No missing values found in the selected columns")
            preprocessing['handle_missing'] = "none"
    
    # Handle outliers
    with outliers_tab:
        st.subheader("Outlier Detection and Treatment")
        
        # Only apply to numeric columns
        numeric_cols = working_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            # Outlier detection method
            handle_outliers = st.radio(
                "How to handle outliers?",
                options=["Do not handle outliers", 
                         "Remove outliers (Z-score)", 
                         "Remove outliers (IQR)",
                         "Cap outliers (Z-score)",
                         "Cap outliers (IQR)"],
                index=0 if preprocessing['handle_outliers'] not in ["none", "remove_zscore", "remove_iqr", 
                                                                  "cap_zscore", "cap_iqr"] else
                      ["none", "remove_zscore", "remove_iqr", 
                       "cap_zscore", "cap_iqr"].index(preprocessing['handle_outliers']),
                horizontal=True,
                help="Select a method to handle outliers in numeric columns"
            )
            
            # Map radio button selection to state value
            option_mapping = {
                "Do not handle outliers": "none",
                "Remove outliers (Z-score)": "remove_zscore",
                "Remove outliers (IQR)": "remove_iqr",
                "Cap outliers (Z-score)": "cap_zscore",
                "Cap outliers (IQR)": "cap_iqr"
            }
            preprocessing['handle_outliers'] = option_mapping[handle_outliers]
            
            # Additional options based on selection
            if preprocessing['handle_outliers'] in ["remove_zscore", "cap_zscore"]:
                z_threshold = st.slider(
                    "Z-score threshold",
                    min_value=1.0,
                    max_value=5.0,
                    value=3.0,
                    step=0.1,
                    help="Values above this threshold (in standard deviations) will be considered outliers"
                )
                preprocessing['z_threshold'] = z_threshold
            
            elif preprocessing['handle_outliers'] in ["remove_iqr", "cap_iqr"]:
                iqr_multiplier = st.slider(
                    "IQR multiplier",
                    min_value=1.0,
                    max_value=3.0,
                    value=1.5,
                    step=0.1,
                    help="Multiplier for IQR to determine outlier boundaries"
                )
                preprocessing['iqr_multiplier'] = iqr_multiplier
            
            # Visualization of potential outliers
            if preprocessing['handle_outliers'] != "none":
                st.subheader("Outlier Visualization")
                
                selected_col = st.selectbox(
                    "Select column to visualize outliers",
                    options=numeric_cols
                )
                
                if selected_col:
                    col_data = working_df[selected_col].dropna()
                    
                    if preprocessing['handle_outliers'] in ["remove_zscore", "cap_zscore"]:
                        # Z-score outlier detection
                        z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                        threshold = preprocessing.get('z_threshold', 3.0)
                        outliers = col_data[z_scores > threshold]
                        outlier_percentage = len(outliers) / len(col_data) * 100
                        
                        st.metric(
                            "Potential Outliers",
                            f"{len(outliers)} ({outlier_percentage:.2f}%)"
                        )
                        
                        if not outliers.empty:
                            import plotly.express as px
                            fig = px.box(col_data, title=f"Boxplot of {selected_col} with Z-score Outliers")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif preprocessing['handle_outliers'] in ["remove_iqr", "cap_iqr"]:
                        # IQR outlier detection
                        Q1 = col_data.quantile(0.25)
                        Q3 = col_data.quantile(0.75)
                        IQR = Q3 - Q1
                        multiplier = preprocessing.get('iqr_multiplier', 1.5)
                        
                        lower_bound = Q1 - multiplier * IQR
                        upper_bound = Q3 + multiplier * IQR
                        
                        outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                        outlier_percentage = len(outliers) / len(col_data) * 100
                        
                        st.metric(
                            "Potential Outliers",
                            f"{len(outliers)} ({outlier_percentage:.2f}%)"
                        )
                        
                        if not outliers.empty:
                            import plotly.express as px
                            fig = px.box(col_data, title=f"Boxplot of {selected_col} with IQR Outliers")
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns available for outlier detection")
            preprocessing['handle_outliers'] = "none"
    
    # Data scaling
    with scaling_tab:
        st.subheader("Data Scaling / Normalization")
        
        # Only apply to numeric columns
        numeric_cols = working_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            scaling_method = st.radio(
                "Select scaling method",
                options=["No scaling", 
                         "StandardScaler (Z-score normalization)", 
                         "MinMaxScaler (0-1 scaling)"],
                index=0 if preprocessing['scaling_method'] not in ["none", "standard", "minmax"] else
                      ["none", "standard", "minmax"].index(preprocessing['scaling_method']),
                horizontal=True,
                help="Select a method to scale numeric features"
            )
            
            # Map radio button selection to state value
            option_mapping = {
                "No scaling": "none",
                "StandardScaler (Z-score normalization)": "standard",
                "MinMaxScaler (0-1 scaling)": "minmax"
            }
            preprocessing['scaling_method'] = option_mapping[scaling_method]
            
            if preprocessing['scaling_method'] != "none":
                # Select columns to scale
                scale_cols = st.multiselect(
                    "Select numeric columns to scale",
                    options=numeric_cols,
                    default=numeric_cols,
                    help="Choose which numeric columns to apply scaling to"
                )
                preprocessing['scale_columns'] = scale_cols
        else:
            st.info("No numeric columns available for scaling")
            preprocessing['scaling_method'] = "none"
    
    # Categorical encoding
    with encoding_tab:
        st.subheader("Categorical Variable Encoding")
        
        # Only apply to categorical columns
        cat_cols = working_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if cat_cols:
            encoding_method = st.radio(
                "Select encoding method",
                options=["No encoding", 
                         "One-Hot Encoding", 
                         "Label Encoding"],
                index=0 if preprocessing['encoding_method'] not in ["none", "onehot", "label"] else
                      ["none", "onehot", "label"].index(preprocessing['encoding_method']),
                horizontal=True,
                help="Select a method to encode categorical variables"
            )
            
            # Map radio button selection to state value
            option_mapping = {
                "No encoding": "none",
                "One-Hot Encoding": "onehot",
                "Label Encoding": "label"
            }
            preprocessing['encoding_method'] = option_mapping[encoding_method]
            
            if preprocessing['encoding_method'] != "none":
                # Select columns to encode
                encode_cols = st.multiselect(
                    "Select categorical columns to encode",
                    options=cat_cols,
                    default=cat_cols,
                    help="Choose which categorical columns to apply encoding to"
                )
                preprocessing['encode_columns'] = encode_cols
        else:
            st.info("No categorical columns available for encoding")
            preprocessing['encoding_method'] = "none"
    
    # Apply preprocessing
    if st.button("Apply Preprocessing", type="primary"):
        with st.spinner("Preprocessing data..."):
            processed_df = working_df.copy()
            
            # Handle missing values
            if preprocessing['handle_missing'] == "remove_rows":
                original_rows = len(processed_df)
                processed_df = processed_df.dropna()
                rows_removed = original_rows - len(processed_df)
                if rows_removed > 0:
                    st.info(f"Removed {rows_removed} rows with missing values")
            
            elif preprocessing['handle_missing'] == "remove_columns":
                threshold = preprocessing.get('missing_threshold', 50)
                threshold_decimal = threshold / 100
                original_cols = processed_df.columns.tolist()
                processed_df = processed_df.loc[:, processed_df.isnull().mean() < threshold_decimal]
                cols_removed = set(original_cols) - set(processed_df.columns.tolist())
                if cols_removed:
                    st.info(f"Removed {len(cols_removed)} columns with high missing rate: {', '.join(cols_removed)}")
            
            elif preprocessing['handle_missing'] == "fill_mean_mode":
                for column in processed_df.columns:
                    if processed_df[column].dtype.kind in 'ifc':  # numeric
                        processed_df[column].fillna(processed_df[column].mean(), inplace=True)
                    else:  # categorical
                        processed_df[column].fillna(processed_df[column].mode()[0], inplace=True)
                st.info("Filled missing values with mean/mode for each column")
            
            elif preprocessing['handle_missing'] == "fill_median":
                for column in processed_df.columns:
                    if processed_df[column].dtype.kind in 'ifc':  # numeric
                        processed_df[column].fillna(processed_df[column].median(), inplace=True)
                    else:  # categorical
                        processed_df[column].fillna(processed_df[column].mode()[0], inplace=True)
                st.info("Filled missing values with median for numeric columns and mode for categorical")
            
            elif preprocessing['handle_missing'] == "fill_constant":
                fill_value = preprocessing.get('missing_fill_value', "0")
                try:
                    # Try to convert to numeric if possible
                    numeric_fill = float(fill_value)
                    processed_df = processed_df.fillna(numeric_fill)
                except ValueError:
                    # Use as string otherwise
                    processed_df = processed_df.fillna(fill_value)
                st.info(f"Filled missing values with constant: {fill_value}")
            
            # Handle outliers
            if preprocessing['handle_outliers'] == "remove_zscore":
                numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                original_rows = len(processed_df)
                threshold = preprocessing.get('z_threshold', 3.0)
                
                for col in numeric_cols:
                    z_scores = np.abs((processed_df[col] - processed_df[col].mean()) / processed_df[col].std())
                    processed_df = processed_df[z_scores <= threshold]
                
                rows_removed = original_rows - len(processed_df)
                if rows_removed > 0:
                    st.info(f"Removed {rows_removed} rows with outliers (Z-score method)")
            
            elif preprocessing['handle_outliers'] == "remove_iqr":
                numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                original_rows = len(processed_df)
                multiplier = preprocessing.get('iqr_multiplier', 1.5)
                
                for col in numeric_cols:
                    Q1 = processed_df[col].quantile(0.25)
                    Q3 = processed_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - multiplier * IQR
                    upper_bound = Q3 + multiplier * IQR
                    
                    processed_df = processed_df[
                        (processed_df[col] >= lower_bound) & 
                        (processed_df[col] <= upper_bound)
                    ]
                
                rows_removed = original_rows - len(processed_df)
                if rows_removed > 0:
                    st.info(f"Removed {rows_removed} rows with outliers (IQR method)")
            
            elif preprocessing['handle_outliers'] == "cap_zscore":
                numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                threshold = preprocessing.get('z_threshold', 3.0)
                
                for col in numeric_cols:
                    mean = processed_df[col].mean()
                    std = processed_df[col].std()
                    
                    # Calculate upper and lower bounds
                    upper_bound = mean + threshold * std
                    lower_bound = mean - threshold * std
                    
                    # Cap outliers
                    processed_df[col] = processed_df[col].clip(lower=lower_bound, upper=upper_bound)
                
                st.info(f"Capped outliers using Z-score method (threshold: {threshold})")
            
            elif preprocessing['handle_outliers'] == "cap_iqr":
                numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                multiplier = preprocessing.get('iqr_multiplier', 1.5)
                
                for col in numeric_cols:
                    Q1 = processed_df[col].quantile(0.25)
                    Q3 = processed_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # Calculate upper and lower bounds
                    upper_bound = Q3 + multiplier * IQR
                    lower_bound = Q1 - multiplier * IQR
                    
                    # Cap outliers
                    processed_df[col] = processed_df[col].clip(lower=lower_bound, upper=upper_bound)
                
                st.info(f"Capped outliers using IQR method (multiplier: {multiplier})")
            
            # Apply scaling
            if preprocessing['scaling_method'] != "none":
                scale_columns = preprocessing.get('scale_columns', [])
                
                if scale_columns:
                    if preprocessing['scaling_method'] == "standard":
                        scaler = StandardScaler()
                        processed_df[scale_columns] = scaler.fit_transform(processed_df[scale_columns])
                        st.info(f"Applied StandardScaler to {len(scale_columns)} columns")
                    
                    elif preprocessing['scaling_method'] == "minmax":
                        scaler = MinMaxScaler()
                        processed_df[scale_columns] = scaler.fit_transform(processed_df[scale_columns])
                        st.info(f"Applied MinMaxScaler to {len(scale_columns)} columns")
            
            # Apply categorical encoding
            if preprocessing['encoding_method'] != "none":
                encode_columns = preprocessing.get('encode_columns', [])
                
                if encode_columns:
                    if preprocessing['encoding_method'] == "onehot":
                        # One-hot encoding
                        processed_df = pd.get_dummies(processed_df, columns=encode_columns, drop_first=True)
                        st.info(f"Applied one-hot encoding to {len(encode_columns)} columns")
                    
                    elif preprocessing['encoding_method'] == "label":
                        # Label encoding
                        for col in encode_columns:
                            processed_df[col] = pd.factorize(processed_df[col])[0]
                        st.info(f"Applied label encoding to {len(encode_columns)} columns")
            
            # Store the processed dataframe in session state
            preprocessing['processed_df'] = processed_df
            st.session_state.preprocessing_options = preprocessing
            
            # Show results
            st.subheader("Preprocessing Results")
            st.write(f"Original shape: {working_df.shape}")
            st.write(f"Processed shape: {processed_df.shape}")
            st.dataframe(processed_df.head(10), use_container_width=True)
    
    # If we have a processed dataframe, return it, otherwise return the working dataframe
    if preprocessing.get('processed_df') is not None:
        return preprocessing['processed_df']
    else:
        return working_df
