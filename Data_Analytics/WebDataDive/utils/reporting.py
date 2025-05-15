import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
from typing import Dict, List, Any, Tuple
from datetime import datetime
from assets.stock_photos import display_header_image

def render_reporting(original_df: pd.DataFrame, processed_df: pd.DataFrame, 
                     problem_def: Dict[str, Any], analysis_results: Dict[str, Any],
                     visualization_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Renders the reporting section and generates a comprehensive report
    
    Parameters:
    original_df - The original uploaded dataset
    processed_df - The processed dataset
    problem_def - The problem definition dictionary
    analysis_results - The analysis results dictionary
    visualization_results - The visualization results dictionary
    
    Returns:
    Dictionary containing report data
    """
    st.header("6. Reporting", divider=True)
    
    # Display a relevant image at the top
    display_header_image("business analytics interface")
    
    # Initialize results dictionary
    report_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sections": []
    }
    
    # Create tabs for different reporting views
    report_tabs = st.tabs(["Summary Report", "Detailed Report", "Export Options"])
    
    # First tab: Summary Report
    with report_tabs[0]:
        # Generate and display summary report
        summary_data = generate_summary_report(original_df, processed_df, problem_def, analysis_results)
        report_data["sections"].append({"type": "summary", "data": summary_data})
    
    # Second tab: Detailed Report
    with report_tabs[1]:
        # Generate and display detailed report
        detailed_data = generate_detailed_report(original_df, processed_df, problem_def, analysis_results, visualization_results)
        report_data["sections"].append({"type": "detailed", "data": detailed_data})
    
    # Third tab: Export Options
    with report_tabs[2]:
        st.subheader("Export Options")
        st.write("Choose what to include in your export and select the export format.")
        
        # Checkboxes for selecting what to include in the export
        col1, col2 = st.columns(2)
        
        with col1:
            include_original = st.checkbox("Include Original Data", value=True)
            include_processed = st.checkbox("Include Processed Data", value=True)
            include_summary = st.checkbox("Include Summary Report", value=True)
        
        with col2:
            include_analysis = st.checkbox("Include Analysis Results", value=True)
            include_visualizations = st.checkbox("Include Visualization Descriptions", value=True)
            include_detailed = st.checkbox("Include Detailed Report", value=True)
        
        export_format = st.radio(
            "Export Format",
            options=["CSV", "Excel", "HTML", "JSON"],
            horizontal=True
        )
        
        # Store export selections
        report_data["export_settings"] = {
            "include_original": include_original,
            "include_processed": include_processed,
            "include_summary": include_summary,
            "include_analysis": include_analysis,
            "include_visualizations": include_visualizations,
            "include_detailed": include_detailed,
            "export_format": export_format
        }
    
    # Store the full report information
    return report_data

def generate_summary_report(original_df: pd.DataFrame, processed_df: pd.DataFrame, 
                            problem_def: Dict[str, Any], analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a summary report of the analysis"""
    summary_data = {}
    
    st.subheader("Analysis Summary Report")
    
    # Dataset summary
    st.write("### Dataset Summary")
    
    cols = st.columns(2)
    with cols[0]:
        st.metric("Original Dataset Size", f"{original_df.shape[0]} rows × {original_df.shape[1]} columns")
    with cols[1]:
        st.metric("Processed Dataset Size", f"{processed_df.shape[0]} rows × {processed_df.shape[1]} columns")
    
    # Data transformation metrics
    rows_diff = processed_df.shape[0] - original_df.shape[0]
    cols_diff = processed_df.shape[1] - original_df.shape[1]
    
    cols = st.columns(2)
    with cols[0]:
        st.metric("Rows Change", f"{rows_diff} ({rows_diff/original_df.shape[0]*100:.1f}%)" if original_df.shape[0] > 0 else "N/A",
                 delta=rows_diff)
    with cols[1]:
        st.metric("Columns Change", f"{cols_diff} ({cols_diff/original_df.shape[1]*100:.1f}%)" if original_df.shape[1] > 0 else "N/A",
                 delta=cols_diff)
    
    # Problem definition summary
    st.write("### Analysis Goal")
    
    analysis_goal = problem_def.get('analysis_goal', 'Not specified')
    analysis_method = problem_def.get('analysis_method', 'Not specified')
    target_variable = problem_def.get('target_variable', 'None')
    features = problem_def.get('features', [])
    
    cols = st.columns(2)
    with cols[0]:
        st.metric("Analysis Goal", analysis_goal)
        st.metric("Selected Features", f"{len(features)} features")
    with cols[1]:
        st.metric("Analysis Method", analysis_method)
        if target_variable:
            st.metric("Target Variable", target_variable)
    
    # Key findings summary
    st.write("### Key Findings")
    
    # Different summaries based on analysis type
    analysis_type = analysis_results.get('type', '')
    
    if analysis_type == "exploratory":
        method = analysis_results.get('method', '')
        
        if method == "Descriptive Statistics":
            if 'stats' in analysis_results:
                # Summary of numeric statistics
                stats = pd.DataFrame(analysis_results['stats'])
                
                st.write("**Numeric Variable Statistics:**")
                
                # Create a summary of min, max, mean for each numeric column
                stats_summary = []
                for col in stats.columns:
                    if 'mean' in stats[col] and 'min' in stats[col] and 'max' in stats[col]:
                        stats_summary.append({
                            'Variable': col,
                            'Mean': f"{stats[col]['mean']:.2f}",
                            'Min': f"{stats[col]['min']:.2f}",
                            'Max': f"{stats[col]['max']:.2f}",
                            'Std Dev': f"{stats[col]['std']:.2f}" if 'std' in stats[col] else "N/A"
                        })
                
                if stats_summary:
                    st.dataframe(pd.DataFrame(stats_summary), use_container_width=True)
        
        elif method == "Data Distribution Analysis":
            # Find the distribution analysis results
            for key, value in analysis_results.items():
                if key.startswith('distribution_'):
                    var_name = key.replace('distribution_', '')
                    
                    st.write(f"**Distribution Analysis for {var_name}:**")
                    
                    if 'p_value' in value:
                        p_value = value['p_value']
                        if p_value < 0.05:
                            st.info(f"The distribution is significantly non-normal (p={p_value:.4f}).")
                        else:
                            st.success(f"The distribution appears to be normal (p={p_value:.4f}).")
        
        elif method == "Outlier Detection":
            # Find the outlier detection results
            for key, value in analysis_results.items():
                if key.startswith('outliers_'):
                    var_name = key.replace('outliers_', '')
                    
                    st.write(f"**Outlier Analysis for {var_name}:**")
                    
                    z_count = value.get('z_score_count', 0)
                    z_pct = value.get('z_score_percentage', 0)
                    iqr_count = value.get('iqr_count', 0)
                    iqr_pct = value.get('iqr_percentage', 0)
                    
                    cols = st.columns(2)
                    with cols[0]:
                        st.metric("Z-score Outliers", f"{z_count} ({z_pct:.2f}%)")
                    with cols[1]:
                        st.metric("IQR Outliers", f"{iqr_count} ({iqr_pct:.2f}%)")
    
    elif analysis_type == "correlation":
        method = analysis_results.get('method', '')
        
        if method in ["Pearson Correlation", "Spearman Correlation"]:
            if 'correlation_matrix' in analysis_results:
                corr_matrix = pd.DataFrame(analysis_results['correlation_matrix'])
                
                # Get the strongest correlations
                strongest_corrs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        corr_val = corr_matrix.iloc[i, j]
                        strongest_corrs.append((col1, col2, corr_val))
                
                # Sort by absolute correlation value
                strongest_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
                
                # Display top 5 strongest correlations
                st.write("**Top 5 Strongest Correlations:**")
                
                if strongest_corrs:
                    corr_data = []
                    for col1, col2, corr_val in strongest_corrs[:5]:
                        corr_data.append({
                            'Variable 1': col1,
                            'Variable 2': col2,
                            'Correlation': f"{corr_val:.4f}",
                            'Strength': 'Strong' if abs(corr_val) > 0.7 else 'Moderate' if abs(corr_val) > 0.3 else 'Weak',
                            'Direction': 'Positive' if corr_val > 0 else 'Negative'
                        })
                    
                    st.dataframe(pd.DataFrame(corr_data), use_container_width=True)
    
    elif analysis_type == "regression":
        method = analysis_results.get('method', '')
        
        if method in ["Linear Regression", "Multiple Regression"]:
            if 'metrics' in analysis_results:
                metrics = analysis_results['metrics']
                
                cols = st.columns(2)
                with cols[0]:
                    st.metric("R² (Training)", f"{metrics.get('train_r2', 0):.4f}")
                    st.metric("RMSE (Training)", f"{metrics.get('train_rmse', 0):.4f}")
                with cols[1]:
                    st.metric("R² (Testing)", f"{metrics.get('test_r2', 0):.4f}")
                    st.metric("RMSE (Testing)", f"{metrics.get('test_rmse', 0):.4f}")
                
                # Model quality assessment
                r2 = metrics.get('test_r2', 0)
                if r2 > 0.7:
                    st.success(f"The model explains {r2*100:.1f}% of the variance in the target variable.")
                elif r2 > 0.3:
                    st.info(f"The model explains {r2*100:.1f}% of the variance in the target variable.")
                else:
                    st.warning(f"The model only explains {r2*100:.1f}% of the variance in the target variable.")
            
            # Top features by importance
            if 'model_coefficients' in analysis_results:
                coef_data = pd.DataFrame(analysis_results['model_coefficients'])
                
                if 'Feature' in coef_data and ('Coefficient' in coef_data or 'Importance' in coef_data):
                    st.write("**Top 5 Most Important Features:**")
                    
                    # Sort by importance or absolute coefficient
                    if 'Importance' in coef_data:
                        sorted_coef = coef_data.sort_values('Importance', ascending=False).head(5)
                        value_col = 'Importance'
                    elif 'Absolute Impact' in coef_data:
                        sorted_coef = coef_data.sort_values('Absolute Impact', ascending=False).head(5)
                        value_col = 'Coefficient'
                    else:
                        sorted_coef = coef_data.sort_values('Coefficient', key=abs, ascending=False).head(5)
                        value_col = 'Coefficient'
                    
                    # Display top features
                    feature_data = []
                    for _, row in sorted_coef.iterrows():
                        feature_data.append({
                            'Feature': row['Feature'],
                            'Impact': f"{row[value_col]:.4f}"
                        })
                    
                    st.dataframe(pd.DataFrame(feature_data), use_container_width=True)
    
    elif analysis_type == "classification":
        method = analysis_results.get('method', '')
        
        if method == "Logistic Regression":
            if 'metrics' in analysis_results:
                metrics = analysis_results['metrics']
                
                cols = st.columns(2)
                with cols[0]:
                    st.metric("Accuracy (Training)", f"{metrics.get('train_accuracy', 0):.4f}")
                with cols[1]:
                    st.metric("Accuracy (Testing)", f"{metrics.get('test_accuracy', 0):.4f}")
                
                # If we have a confusion matrix, calculate derived metrics
                if 'confusion_matrix' in analysis_results and len(analysis_results['confusion_matrix']) == 2:
                    cm = analysis_results['confusion_matrix']
                    tn, fp = cm[0][0], cm[0][1]
                    fn, tp = cm[1][0], cm[1][1]
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Precision", f"{precision:.4f}")
                    with cols[1]:
                        st.metric("Recall", f"{recall:.4f}")
                    with cols[2]:
                        st.metric("F1 Score", f"{f1:.4f}")
            
            # Top features by importance
            if 'model_coefficients' in analysis_results:
                coef_data = pd.DataFrame(analysis_results['model_coefficients'])
                
                if 'Feature' in coef_data and 'Coefficient' in coef_data:
                    st.write("**Top 5 Most Important Features:**")
                    
                    # Sort by absolute coefficient
                    if 'Absolute Impact' in coef_data:
                        sorted_coef = coef_data.sort_values('Absolute Impact', ascending=False).head(5)
                    else:
                        sorted_coef = coef_data.sort_values('Coefficient', key=abs, ascending=False).head(5)
                    
                    # Display top features
                    feature_data = []
                    for _, row in sorted_coef.iterrows():
                        feature_data.append({
                            'Feature': row['Feature'],
                            'Coefficient': f"{row['Coefficient']:.4f}"
                        })
                    
                    st.dataframe(pd.DataFrame(feature_data), use_container_width=True)
    
    elif analysis_type == "clustering":
        method = analysis_results.get('method', '')
        
        if method == "K-Means Clustering":
            if 'selected_k' in analysis_results:
                st.metric("Number of Clusters", analysis_results['selected_k'])
            
            if 'silhouette_score' in analysis_results:
                silhouette = analysis_results['silhouette_score']
                
                st.metric("Silhouette Score", f"{silhouette:.4f}")
                
                # Interpretation
                if silhouette > 0.7:
                    st.success("The clusters are well separated and dense.")
                elif silhouette > 0.5:
                    st.info("The clusters have reasonable separation and density.")
                elif silhouette > 0:
                    st.warning("The clusters have some overlap or are not very dense.")
                else:
                    st.error("The clustering may not be appropriate for this data.")
            
            # Cluster sizes
            if 'cluster_counts' in analysis_results:
                st.write("**Cluster Sizes:**")
                
                cluster_sizes = []
                for cluster, count in analysis_results['cluster_counts'].items():
                    cluster_sizes.append({
                        'Cluster': cluster,
                        'Size': count
                    })
                
                st.dataframe(pd.DataFrame(cluster_sizes), use_container_width=True)
    
    elif analysis_type == "time_series":
        method = analysis_results.get('method', '')
        
        if method == "Trend Analysis":
            if 'trend_statistics' in analysis_results:
                stats = analysis_results['trend_statistics']
                
                cols = st.columns(2)
                with cols[0]:
                    st.metric("Slope", f"{stats.get('slope', 0):.6f}")
                    st.metric("R²", f"{stats.get('r_squared', 0):.4f}")
                with cols[1]:
                    st.metric("P-value", f"{stats.get('p_value', 0):.6f}")
                    st.metric("Standard Error", f"{stats.get('std_err', 0):.6f}")
                
                # Trend interpretation
                slope = stats.get('slope', 0)
                p_value = stats.get('p_value', 1)
                
                if p_value < 0.05:
                    if slope > 0:
                        st.success(f"Significant upward trend detected (p={p_value:.6f})")
                    else:
                        st.warning(f"Significant downward trend detected (p={p_value:.6f})")
                else:
                    st.info(f"No significant trend detected (p={p_value:.6f})")
        
        elif method == "Seasonality Analysis":
            if 'seasonality' in analysis_results:
                seasonality = analysis_results['seasonality']
                
                st.metric("Seasonal Period", seasonality.get('period', 'N/A'))
                
                if 'seasonal_strength' in seasonality and seasonality['seasonal_strength'] is not None:
                    strength = seasonality['seasonal_strength']
                    
                    st.metric("Seasonal Strength", f"{strength:.4f}")
                    
                    # Interpretation
                    if strength > 0.6:
                        st.success(f"Strong seasonality detected")
                    elif strength > 0.3:
                        st.info(f"Moderate seasonality detected")
                    else:
                        st.warning(f"Weak seasonality detected")
    
    # Store summary data
    summary_data = {
        "dataset_summary": {
            "original_size": original_df.shape,
            "processed_size": processed_df.shape,
            "rows_change": rows_diff,
            "cols_change": cols_diff
        },
        "problem_definition": {
            "analysis_goal": analysis_goal,
            "analysis_method": analysis_method,
            "target_variable": target_variable,
            "features_count": len(features)
        },
        "analysis_type": analysis_type,
        "key_metrics": extract_key_metrics(analysis_results)
    }
    
    return summary_data

def generate_detailed_report(original_df: pd.DataFrame, processed_df: pd.DataFrame, 
                            problem_def: Dict[str, Any], analysis_results: Dict[str, Any],
                            visualization_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a detailed report of the analysis process and results"""
    detailed_data = {}
    
    st.subheader("Detailed Analysis Report")
    
    # Create sections for the detailed report
    sections = [
        "Data Overview",
        "Problem Definition",
        "Data Preprocessing",
        "Analysis Methodology",
        "Results & Findings",
        "Visualizations",
        "Conclusions & Recommendations"
    ]
    
    # Set up tabs for each section
    detailed_tabs = st.tabs(sections)
    
    # 1. Data Overview
    with detailed_tabs[0]:
        data_overview = {}
        
        st.write("### Data Overview")
        
        # Original dataset information
        st.write("#### Original Dataset")
        
        st.dataframe(original_df.head(5), use_container_width=True)
        
        # Summary statistics
        st.write("**Summary Statistics:**")
        st.dataframe(original_df.describe(), use_container_width=True)
        
        # Column information
        col_info = []
        for col in original_df.columns:
            dtype = str(original_df[col].dtype)
            missing = original_df[col].isna().sum()
            missing_pct = missing / len(original_df) * 100
            
            col_info.append({
                "Column": col,
                "Type": dtype,
                "Missing Values": missing,
                "Missing (%)": f"{missing_pct:.2f}%",
                "Unique Values": original_df[col].nunique()
            })
        
        st.write("**Column Information:**")
        st.dataframe(pd.DataFrame(col_info), use_container_width=True)
        
        # Store data overview information
        data_overview = {
            "original_shape": original_df.shape,
            "processed_shape": processed_df.shape,
            "column_info": col_info
        }
        
        detailed_data["data_overview"] = data_overview
    
    # 2. Problem Definition
    with detailed_tabs[1]:
        st.write("### Problem Definition")
        
        # Display problem definition details
        analysis_goal = problem_def.get('analysis_goal', 'Not specified')
        analysis_method = problem_def.get('analysis_method', 'Not specified')
        target_variable = problem_def.get('target_variable', 'None')
        features = problem_def.get('features', [])
        
        st.write(f"**Analysis Goal:** {analysis_goal}")
        st.write(f"**Analysis Method:** {analysis_method}")
        
        if target_variable:
            st.write(f"**Target Variable:** {target_variable}")
        
        st.write("**Selected Features:**")
        if features:
            st.write(", ".join(features))
        else:
            st.write("No features selected")
        
        # Store problem definition information
        detailed_data["problem_definition"] = problem_def
    
    # 3. Data Preprocessing
    with detailed_tabs[2]:
        preprocessing_details = {}
        
        st.write("### Data Preprocessing")
        
        # Compare original and processed data
        st.write("#### Data Transformation")
        
        cols = st.columns(2)
        with cols[0]:
            st.metric("Original Rows", original_df.shape[0])
            st.metric("Original Columns", original_df.shape[1])
        with cols[1]:
            st.metric("Processed Rows", processed_df.shape[0])
            st.metric("Processed Columns", processed_df.shape[1])
        
        # Calculate changes in shape
        rows_diff = processed_df.shape[0] - original_df.shape[0]
        cols_diff = processed_df.shape[1] - original_df.shape[1]
        
        # Describe the changes
        if rows_diff != 0 or cols_diff != 0:
            st.write("#### Changes Applied")
            
            if rows_diff < 0:
                st.write(f"- Removed {abs(rows_diff)} rows ({abs(rows_diff)/original_df.shape[0]*100:.1f}% of original data)")
            elif rows_diff > 0:
                st.write(f"- Added {rows_diff} rows (e.g., through data augmentation)")
            
            if cols_diff < 0:
                st.write(f"- Removed {abs(cols_diff)} columns")
            elif cols_diff > 0:
                st.write(f"- Added {cols_diff} columns (e.g., through feature engineering or encoding)")
        
        # Processed data preview
        st.write("#### Processed Dataset Preview")
        st.dataframe(processed_df.head(5), use_container_width=True)
        
        # Check for preprocessing options in session state
        if 'preprocessing_options' in st.session_state:
            preproc_options = st.session_state.preprocessing_options
            
            st.write("#### Preprocessing Steps Applied")
            
            # Missing values handling
            missing_method = preproc_options.get('handle_missing', 'none')
            if missing_method != 'none':
                methods = {
                    'remove_rows': "Removed rows with missing values",
                    'remove_columns': f"Removed columns with missing values above {preproc_options.get('missing_threshold', 50)}%",
                    'fill_mean_mode': "Filled missing values with mean for numeric columns and mode for categorical columns",
                    'fill_median': "Filled missing values with median for numeric columns and mode for categorical columns",
                    'fill_constant': f"Filled missing values with constant: {preproc_options.get('missing_fill_value', '0')}",
                    'keep': "Kept missing values"
                }
                
                st.write(f"**Missing Values:** {methods.get(missing_method, missing_method)}")
            
            # Outlier handling
            outlier_method = preproc_options.get('handle_outliers', 'none')
            if outlier_method != 'none':
                methods = {
                    'remove_zscore': f"Removed outliers using Z-score method (threshold: {preproc_options.get('z_threshold', 3.0)})",
                    'remove_iqr': f"Removed outliers using IQR method (multiplier: {preproc_options.get('iqr_multiplier', 1.5)})",
                    'cap_zscore': f"Capped outliers using Z-score method (threshold: {preproc_options.get('z_threshold', 3.0)})",
                    'cap_iqr': f"Capped outliers using IQR method (multiplier: {preproc_options.get('iqr_multiplier', 1.5)})"
                }
                
                st.write(f"**Outliers:** {methods.get(outlier_method, outlier_method)}")
            
            # Scaling
            scaling_method = preproc_options.get('scaling_method', 'none')
            if scaling_method != 'none':
                methods = {
                    'standard': "StandardScaler (Z-score normalization)",
                    'minmax': "MinMaxScaler (0-1 scaling)"
                }
                
                scale_columns = preproc_options.get('scale_columns', [])
                
                st.write(f"**Scaling:** {methods.get(scaling_method, scaling_method)} applied to {len(scale_columns)} columns")
            
            # Encoding
            encoding_method = preproc_options.get('encoding_method', 'none')
            if encoding_method != 'none':
                methods = {
                    'onehot': "One-Hot Encoding",
                    'label': "Label Encoding"
                }
                
                encode_columns = preproc_options.get('encode_columns', [])
                
                st.write(f"**Categorical Encoding:** {methods.get(encoding_method, encoding_method)} applied to {len(encode_columns)} columns")
            
            # Store preprocessing information
            preprocessing_details = {
                "missing_values": missing_method,
                "outliers": outlier_method,
                "scaling": scaling_method,
                "encoding": encoding_method
            }
        
        detailed_data["preprocessing"] = preprocessing_details
    
    # 4. Analysis Methodology
    with detailed_tabs[3]:
        st.write("### Analysis Methodology")
        
        # Different descriptions based on analysis type
        analysis_type = analysis_results.get('type', '')
        method = analysis_results.get('method', '')
        
        if analysis_type == "exploratory":
            if method == "Descriptive Statistics":
                st.write("""
                **Methodology: Descriptive Statistical Analysis**
                
                Descriptive statistics were calculated for all numeric variables, including:
                - Central tendency measures (mean, median)
                - Dispersion measures (standard deviation, range)
                - Distribution characteristics (skewness, kurtosis)
                
                For categorical variables, frequency distributions and proportions were analyzed.
                """)
            
            elif method == "Data Distribution Analysis":
                st.write("""
                **Methodology: Distribution Analysis**
                
                The distribution of key variables was analyzed through:
                - Histograms and density plots to visualize the shape of distributions
                - Q-Q plots to assess normality
                - Shapiro-Wilk tests to quantitatively assess normality
                - Box plots to identify the central tendency and spread
                """)
            
            elif method == "Outlier Detection":
                st.write("""
                **Methodology: Outlier Detection**
                
                Two complementary methods were used to identify outliers:
                1. **Z-score method**: Data points with absolute Z-scores greater than 3 were flagged as outliers
                2. **IQR method**: Data points falling below Q1 - 1.5×IQR or above Q3 + 1.5×IQR were flagged as outliers
                
                Outlier identification helps understand extreme values that might influence analysis results.
                """)
        
        elif analysis_type == "correlation":
            if method == "Pearson Correlation":
                st.write("""
                **Methodology: Pearson Correlation Analysis**
                
                Pearson correlation coefficients were calculated to measure the linear relationship between pairs of variables. This method:
                - Quantifies the strength and direction of linear relationships
                - Ranges from -1 (perfect negative correlation) to +1 (perfect positive correlation)
                - Assumes variables are normally distributed and have a linear relationship
                
                A correlation matrix was generated to visualize all pairwise correlations simultaneously.
                """)
            
            elif method == "Spearman Correlation":
                st.write("""
                **Methodology: Spearman Rank Correlation Analysis**
                
                Spearman's rank correlation coefficients were calculated to measure the monotonic relationship between pairs of variables. This method:
                - Assesses whether variables increase or decrease together, regardless of linearity
                - Is more robust to outliers and non-normal distributions
                - Does not require assumptions about the distribution of the data
                
                A correlation matrix was generated to visualize all pairwise rank correlations simultaneously.
                """)
            
            elif method == "Feature Importance":
                st.write("""
                **Methodology: Feature Importance Analysis**
                
                A linear regression model was used to determine feature importance, where:
                - The target variable was regressed against predictor variables
                - Standardized coefficients were calculated to enable direct comparison
                - The absolute magnitude of coefficients indicates relative importance
                - The sign of coefficients indicates the direction of influence
                
                This analysis helps identify which features have the strongest relationship with the target variable.
                """)
        
        elif analysis_type == "regression":
            if method in ["Linear Regression", "Multiple Regression"]:
                st.write(f"""
                **Methodology: {method}**
                
                A regression model was built with the following steps:
                1. The dataset was split into training (80%) and testing (20%) sets
                2. A linear regression model was fitted on the training data
                3. Model performance was assessed on both training and testing data
                4. Performance metrics calculated included:
                   - R² (coefficient of determination)
                   - RMSE (root mean squared error)
                   - Coefficient significance (p-values)
                
                Regression coefficients were examined to understand the relationship between each predictor and the target variable.
                """)
        
        elif analysis_type == "classification":
            if method == "Logistic Regression":
                st.write("""
                **Methodology: Logistic Regression Classification**
                
                A logistic regression model was built with the following steps:
                1. The dataset was split into training (80%) and testing (20%) sets
                2. A logistic regression model was fitted on the training data
                3. Model performance was assessed on both training and testing data
                4. Performance metrics calculated included:
                   - Accuracy
                   - Precision, Recall, and F1-score (for binary classification)
                   - Confusion matrix
                
                Model coefficients were examined to understand the influence of each predictor on the target class.
                """)
        
        elif analysis_type == "clustering":
            if method == "K-Means Clustering":
                k = analysis_results.get('selected_k', 'N/A')
                
                st.write(f"""
                **Methodology: K-Means Clustering**
                
                K-Means clustering was performed with k={k} clusters:
                1. Data was preprocessed (scaled) to ensure equal influence of features
                2. The optimal number of clusters was determined using the elbow method
                3. K-Means algorithm was applied with multiple random initializations
                4. Cluster assignments were analyzed in terms of:
                   - Cluster sizes and distribution
                   - Cluster centroids and characteristics
                   - Silhouette score to assess cluster quality
                
                This unsupervised learning approach helps identify natural groupings in the data.
                """)
        
        elif analysis_type == "time_series":
            if method == "Trend Analysis":
                st.write("""
                **Methodology: Time Series Trend Analysis**
                
                The time series was analyzed for trends using:
                1. Visual examination of the time series plot
                2. Linear regression to quantify the trend:
                   - Slope coefficient indicates the direction and magnitude of the trend
                   - R² value measures how well the linear trend fits the data
                   - P-value determines the statistical significance of the trend
                3. Rolling averages to smooth short-term fluctuations and highlight long-term trends
                
                This analysis helps understand the underlying direction of the time series.
                """)
            
            elif method == "Seasonality Analysis":
                st.write("""
                **Methodology: Time Series Seasonality Analysis**
                
                The time series was analyzed for seasonal patterns using:
                1. Visual examination of the time series plot
                2. Identification of the seasonal period (frequency)
                3. Decomposition of the time series into:
                   - Trend component
                   - Seasonal component
                   - Residual component
                4. Calculation of seasonal strength to quantify the magnitude of seasonality
                5. Visualization of the seasonal pattern across multiple cycles
                
                This analysis helps identify and quantify recurring patterns in the time series.
                """)
        
        # Store methodology information
        detailed_data["methodology"] = {
            "analysis_type": analysis_type,
            "method": method
        }
    
    # 5. Results & Findings
    with detailed_tabs[4]:
        results_data = {}
        
        st.write("### Results & Findings")
        
        # Different results based on analysis type
        analysis_type = analysis_results.get('type', '')
        method = analysis_results.get('method', '')
        
        if analysis_type == "exploratory":
            # Already covered in summary, but can add more details here
            st.write("Refer to the Summary Report for key exploratory findings.")
            
            # Additional details can be added here
            if 'stats' in analysis_results:
                st.write("**Detailed Statistics Available:**")
                st.write("The analysis produced comprehensive descriptive statistics for all numeric variables.")
        
        elif analysis_type == "correlation":
            if 'correlation_matrix' in analysis_results:
                st.write("**Key Correlations:**")
                
                corr_matrix = pd.DataFrame(analysis_results['correlation_matrix'])
                
                # Get all correlations in a flat format
                all_corrs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        corr_val = corr_matrix.iloc[i, j]
                        
                        # Classify correlation strength
                        abs_corr = abs(corr_val)
                        if abs_corr < 0.3:
                            strength = "Weak"
                        elif abs_corr < 0.7:
                            strength = "Moderate"
                        else:
                            strength = "Strong"
                        
                        # Classify direction
                        direction = "Positive" if corr_val > 0 else "Negative"
                        
                        all_corrs.append({
                            'Variable 1': col1,
                            'Variable 2': col2,
                            'Correlation': corr_val,
                            'Strength': strength,
                            'Direction': direction
                        })
                
                # Sort by absolute correlation
                all_corrs.sort(key=lambda x: abs(x['Correlation']), reverse=True)
                
                # Create a DataFrame for display
                corr_df = pd.DataFrame(all_corrs)
                corr_df['Correlation'] = corr_df['Correlation'].apply(lambda x: f"{x:.4f}")
                
                st.dataframe(corr_df, use_container_width=True)
                
                # Summarize findings
                strong_positive = len([c for c in all_corrs if c['Strength'] == 'Strong' and c['Direction'] == 'Positive'])
                strong_negative = len([c for c in all_corrs if c['Strength'] == 'Strong' and c['Direction'] == 'Negative'])
                moderate = len([c for c in all_corrs if c['Strength'] == 'Moderate'])
                
                st.write(f"**Summary of Relationships:**")
                st.write(f"- Strong positive correlations: {strong_positive}")
                st.write(f"- Strong negative correlations: {strong_negative}")
                st.write(f"- Moderate correlations: {moderate}")
                
                # Store correlation results
                results_data["correlations"] = {
                    "strong_positive": strong_positive,
                    "strong_negative": strong_negative,
                    "moderate": moderate,
                    "top_correlations": all_corrs[:5]
                }
        
        elif analysis_type == "regression":
            if 'model_coefficients' in analysis_results and 'metrics' in analysis_results:
                st.write("**Regression Model Results:**")
                
                # Model metrics
                metrics = analysis_results['metrics']
                
                st.write(f"**Model Fit:**")
                st.write(f"- R² (training): {metrics.get('train_r2', 0):.4f}")
                st.write(f"- R² (testing): {metrics.get('test_r2', 0):.4f}")
                st.write(f"- RMSE (training): {metrics.get('train_rmse', 0):.4f}")
                st.write(f"- RMSE (testing): {metrics.get('test_rmse', 0):.4f}")
                
                # Interpret R²
                r2 = metrics.get('test_r2', 0)
                if r2 > 0.7:
                    st.write(f"The model explains {r2*100:.1f}% of the variance in the target variable, indicating a good fit.")
                elif r2 > 0.3:
                    st.write(f"The model explains {r2*100:.1f}% of the variance in the target variable, indicating a moderate fit.")
                else:
                    st.write(f"The model explains only {r2*100:.1f}% of the variance in the target variable, indicating a poor fit.")
                
                # Check for overfitting
                r2_diff = metrics.get('train_r2', 0) - metrics.get('test_r2', 0)
                if r2_diff > 0.1:
                    st.write(f"The difference between training and testing R² is {r2_diff:.4f}, suggesting potential overfitting.")
                else:
                    st.write(f"The model generalizes well to unseen data (R² difference: {r2_diff:.4f}).")
                
                # Model coefficients
                coef_data = pd.DataFrame(analysis_results['model_coefficients'])
                
                if 'Feature' in coef_data and 'Coefficient' in coef_data:
                    st.write("**Coefficient Interpretation:**")
                    
                    # Sort by absolute coefficient
                    if 'Absolute Impact' in coef_data:
                        sorted_coef = coef_data.sort_values('Absolute Impact', ascending=False)
                    else:
                        sorted_coef = coef_data.sort_values('Coefficient', key=abs, ascending=False)
                    
                    # Create a clean DataFrame for display
                    display_coef = sorted_coef[['Feature', 'Coefficient']].copy()
                    display_coef['Coefficient'] = display_coef['Coefficient'].apply(lambda x: f"{x:.4f}")
                    
                    st.dataframe(display_coef, use_container_width=True)
                    
                    # Interpret top positive and negative coefficients
                    if len(coef_data) > 0:
                        top_positive = sorted_coef[sorted_coef['Coefficient'] > 0].head(1)
                        top_negative = sorted_coef[sorted_coef['Coefficient'] < 0].head(1)
                        
                        if not top_positive.empty:
                            feature = top_positive['Feature'].iloc[0]
                            coef = top_positive['Coefficient'].iloc[0]
                            st.write(f"- **{feature}** has the strongest positive effect: For each unit increase, the target is expected to increase by {coef:.4f} units, holding other variables constant.")
                        
                        if not top_negative.empty:
                            feature = top_negative['Feature'].iloc[0]
                            coef = top_negative['Coefficient'].iloc[0]
                            st.write(f"- **{feature}** has the strongest negative effect: For each unit increase, the target is expected to decrease by {abs(coef):.4f} units, holding other variables constant.")
                
                # Store regression results
                results_data["regression"] = {
                    "metrics": metrics,
                    "r2_interpretation": "good" if r2 > 0.7 else "moderate" if r2 > 0.3 else "poor",
                    "overfitting": r2_diff > 0.1
                }
        
        elif analysis_type == "classification":
            if 'metrics' in analysis_results:
                st.write("**Classification Model Results:**")
                
                # Model metrics
                metrics = analysis_results['metrics']
                
                st.write(f"**Model Performance:**")
                st.write(f"- Accuracy (training): {metrics.get('train_accuracy', 0):.4f}")
                st.write(f"- Accuracy (testing): {metrics.get('test_accuracy', 0):.4f}")
                
                # Confusion matrix interpretation
                if 'confusion_matrix' in analysis_results and 'classes' in analysis_results:
                    cm = analysis_results['confusion_matrix']
                    classes = analysis_results['classes']
                    
                    st.write("**Confusion Matrix Interpretation:**")
                    
                    if len(cm) == 2 and len(cm[0]) == 2:  # Binary classification
                        tn, fp = cm[0][0], cm[0][1]
                        fn, tp = cm[1][0], cm[1][1]
                        
                        total = tn + fp + fn + tp
                        
                        accuracy = (tp + tn) / total if total > 0 else 0
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        
                        st.write(f"- True Negatives: {tn} ({tn/total*100:.1f}% of total)")
                        st.write(f"- False Positives: {fp} ({fp/total*100:.1f}% of total)")
                        st.write(f"- False Negatives: {fn} ({fn/total*100:.1f}% of total)")
                        st.write(f"- True Positives: {tp} ({tp/total*100:.1f}% of total)")
                        
                        st.write(f"**Derived Metrics:**")
                        st.write(f"- Precision: {precision:.4f} (ability to avoid false positives)")
                        st.write(f"- Recall: {recall:.4f} (ability to find all positives)")
                        st.write(f"- F1 Score: {f1:.4f} (harmonic mean of precision and recall)")
                        
                        # ROC AUC
                        if 'roc_auc' in analysis_results:
                            auc = analysis_results['roc_auc']
                            st.write(f"- AUC-ROC: {auc:.4f}")
                            
                            if auc > 0.9:
                                st.write("  The model has excellent discriminative ability.")
                            elif auc > 0.8:
                                st.write("  The model has good discriminative ability.")
                            elif auc > 0.7:
                                st.write("  The model has fair discriminative ability.")
                            else:
                                st.write("  The model has poor discriminative ability.")
                    
                    # Store classification results
                    results_data["classification"] = {
                        "accuracy": metrics.get('test_accuracy', 0),
                        "precision": precision if 'precision' in locals() else None,
                        "recall": recall if 'recall' in locals() else None,
                        "f1": f1 if 'f1' in locals() else None,
                        "auc": analysis_results.get('roc_auc')
                    }
        
        elif analysis_type == "clustering":
            if 'cluster_counts' in analysis_results:
                st.write("**Clustering Results:**")
                
                # Number of clusters
                k = analysis_results.get('selected_k', 'N/A')
                st.write(f"**Number of clusters identified:** {k}")
                
                # Cluster sizes
                cluster_counts = analysis_results['cluster_counts']
                
                st.write("**Cluster Sizes:**")
                
                cluster_sizes = []
                total = sum(cluster_counts.values())
                
                for cluster, count in cluster_counts.items():
                    percentage = count / total * 100
                    cluster_sizes.append({
                        'Cluster': cluster,
                        'Size': count,
                        'Percentage': f"{percentage:.1f}%"
                    })
                
                st.dataframe(pd.DataFrame(cluster_sizes), use_container_width=True)
                
                # Cluster characteristics
                if 'cluster_profiles' in analysis_results:
                    st.write("**Cluster Characteristics:**")
                    
                    cluster_profiles = pd.DataFrame(analysis_results['cluster_profiles'])
                    
                    # Interpret cluster profiles
                    st.write("Each cluster can be characterized by the following feature values:")
                    st.dataframe(cluster_profiles, use_container_width=True)
                    
                    # Provide interpretation for each cluster
                    for cluster in cluster_counts.keys():
                        if str(cluster) in cluster_profiles.columns:
                            # Get top 3 highest and lowest features for this cluster
                            cluster_col = str(cluster)
                            sorted_features = cluster_profiles[cluster_col].sort_values(ascending=False)
                            
                            top_high = sorted_features.head(3)
                            top_low = sorted_features.tail(3)
                            
                            st.write(f"**Cluster {cluster} is characterized by:**")
                            
                            high_features = ", ".join([f"high {idx}" for idx in top_high.index])
                            low_features = ", ".join([f"low {idx}" for idx in top_low.index])
                            
                            st.write(f"- High values: {high_features}")
                            st.write(f"- Low values: {low_features}")
                
                # Silhouette score interpretation
                if 'silhouette_score' in analysis_results:
                    silhouette = analysis_results['silhouette_score']
                    
                    st.write(f"**Clustering Quality:**")
                    st.write(f"- Silhouette Score: {silhouette:.4f}")
                    
                    if silhouette > 0.7:
                        st.write("  The clusters are well separated and dense, indicating a very good clustering.")
                    elif silhouette > 0.5:
                        st.write("  The clusters have reasonable separation and density, indicating a good clustering.")
                    elif silhouette > 0:
                        st.write("  The clusters have some overlap or are not very dense, indicating a fair clustering.")
                    else:
                        st.write("  The clustering may not be appropriate for this data, as clusters overlap significantly.")
                
                # Store clustering results
                results_data["clustering"] = {
                    "k": k,
                    "silhouette": analysis_results.get('silhouette_score'),
                    "largest_cluster": max(cluster_counts.values()) if cluster_counts else 0,
                    "smallest_cluster": min(cluster_counts.values()) if cluster_counts else 0
                }
        
        elif analysis_type == "time_series":
            if method == "Trend Analysis":
                if 'trend_statistics' in analysis_results:
                    st.write("**Time Series Trend Results:**")
                    
                    stats = analysis_results['trend_statistics']
                    
                    slope = stats.get('slope', 0)
                    p_value = stats.get('p_value', 1)
                    r_squared = stats.get('r_squared', 0)
                    
                    st.write(f"**Trend Direction and Magnitude:**")
                    st.write(f"- Slope: {slope:.6f}")
                    st.write(f"- R²: {r_squared:.4f}")
                    st.write(f"- P-value: {p_value:.6f}")
                    
                    # Interpret trend
                    if p_value < 0.05:
                        if slope > 0:
                            st.write(f"A significant **upward trend** was detected (p={p_value:.6f}).")
                            st.write(f"On average, the variable increases by {slope:.6f} units per time unit.")
                        else:
                            st.write(f"A significant **downward trend** was detected (p={p_value:.6f}).")
                            st.write(f"On average, the variable decreases by {abs(slope):.6f} units per time unit.")
                        
                        if r_squared > 0.7:
                            st.write(f"The linear trend explains {r_squared*100:.1f}% of the variance in the data, indicating a very consistent trend.")
                        elif r_squared > 0.3:
                            st.write(f"The linear trend explains {r_squared*100:.1f}% of the variance in the data, indicating a moderately consistent trend.")
                        else:
                            st.write(f"The linear trend explains only {r_squared*100:.1f}% of the variance in the data, indicating a weak trend with high variability.")
                    else:
                        st.write(f"No statistically significant trend was detected (p={p_value:.6f}).")
                        st.write("The time series appears to be stationary in terms of its level.")
                
                # Rolling average results
                if 'rolling_statistics' in analysis_results:
                    rolling_stats = analysis_results['rolling_statistics']
                    
                    window_size = rolling_stats.get('window_size', 'N/A')
                    
                    st.write(f"**Rolling Average Analysis (window size: {window_size}):**")
                    
                    # Compare original vs smoothed standard deviation
                    original_std = rolling_stats.get('original_std', 0)
                    smoothed_std = rolling_stats.get('smoothed_std', 0)
                    
                    if original_std > 0:
                        reduction = (original_std - smoothed_std) / original_std * 100
                        st.write(f"- Original standard deviation: {original_std:.4f}")
                        st.write(f"- Smoothed standard deviation: {smoothed_std:.4f}")
                        st.write(f"- Noise reduction: {reduction:.1f}%")
                        
                        if reduction > 50:
                            st.write("The time series contains significant noise that was smoothed by the rolling average.")
                        elif reduction > 20:
                            st.write("The time series contains moderate noise that was partially smoothed by the rolling average.")
                        else:
                            st.write("The time series contains relatively little noise, as evidenced by the small reduction in variability after smoothing.")
                
                # Store trend analysis results
                results_data["time_series_trend"] = {
                    "slope": stats.get('slope', 0) if 'stats' in locals() else None,
                    "p_value": stats.get('p_value', 1) if 'stats' in locals() else None,
                    "r_squared": stats.get('r_squared', 0) if 'stats' in locals() else None,
                    "significant": p_value < 0.05 if 'p_value' in locals() else None,
                    "direction": "upward" if 'slope' in locals() and slope > 0 else "downward" if 'slope' in locals() else None
                }
            
            elif method == "Seasonality Analysis":
                if 'seasonality' in analysis_results:
                    st.write("**Time Series Seasonality Results:**")
                    
                    seasonality = analysis_results['seasonality']
                    
                    period = seasonality.get('period', 'N/A')
                    strength = seasonality.get('seasonal_strength')
                    
                    st.write(f"**Seasonality Characteristics:**")
                    st.write(f"- Seasonal Period: {period}")
                    
                    if strength is not None:
                        st.write(f"- Seasonal Strength: {strength:.4f}")
                        
                        # Interpret strength
                        if strength > 0.6:
                            st.write("The time series exhibits **strong seasonality**, indicating that the seasonal pattern accounts for a large portion of the variance.")
                        elif strength > 0.3:
                            st.write("The time series exhibits **moderate seasonality**, with a noticeable but not dominant seasonal pattern.")
                        else:
                            st.write("The time series exhibits **weak seasonality**, with the seasonal pattern accounting for only a small portion of the variance.")
                    
                    # Seasonal pattern
                    if 'seasonal_pattern' in seasonality:
                        st.write("**Seasonal Pattern:**")
                        st.write("The seasonal pattern shows how the variable typically varies within each cycle:")
                        
                        # Convert to DataFrame for better display
                        pattern = pd.DataFrame({
                            'Position': list(seasonality['seasonal_pattern'].keys()),
                            'Value': list(seasonality['seasonal_pattern'].values())
                        })
                        
                        # Convert position to numeric if needed
                        pattern['Position'] = pd.to_numeric(pattern['Position'], errors='coerce')
                        
                        # Sort by position
                        pattern = pattern.sort_values('Position')
                        
                        st.dataframe(pattern, use_container_width=True)
                        
                        # Identify peak and trough
                        if not pattern.empty:
                            peak_row = pattern.loc[pattern['Value'].idxmax()]
                            trough_row = pattern.loc[pattern['Value'].idxmin()]
                            
                            st.write(f"- Peak at position {peak_row['Position']}: {peak_row['Value']:.4f}")
                            st.write(f"- Trough at position {trough_row['Position']}: {trough_row['Value']:.4f}")
                
                # Store seasonality analysis results
                results_data["time_series_seasonality"] = {
                    "period": period if 'period' in locals() else None,
                    "strength": strength if 'strength' in locals() else None,
                    "strength_category": "strong" if 'strength' in locals() and strength > 0.6 else 
                                        "moderate" if 'strength' in locals() and strength > 0.3 else 
                                        "weak" if 'strength' in locals() else None
                }
        
        detailed_data["results"] = results_data
    
    # 6. Visualizations
    with detailed_tabs[5]:
        st.write("### Key Visualizations")
        
        # Mention that visualizations are interactive in the app
        st.info("The interactive visualizations are available in the Visualization section of the application. This report includes descriptions of key visualizations.")
        
        # Different visualization descriptions based on analysis type
        analysis_type = analysis_results.get('type', '')
        method = analysis_results.get('method', '')
        
        if analysis_type == "exploratory":
            if method == "Descriptive Statistics":
                st.write("""
                **Key Visualizations:**
                
                1. **Box Plots**: Show the distribution of numeric variables, including median, quartiles, and potential outliers.
                
                2. **Distribution Comparisons**: Violin plots that combine box plots with kernel density estimates to show the full distribution.
                
                3. **Correlation Heatmap**: Visualizes the correlation matrix between numeric variables, with color intensity representing correlation strength.
                
                4. **Bar Charts**: Display the frequency of categorical variables, helping identify the most common categories.
                
                5. **Pie Charts**: Show the proportion of each category in categorical variables, useful for understanding the relative importance of each category.
                """)
            
            elif method == "Data Distribution Analysis":
                st.write("""
                **Key Visualizations:**
                
                1. **Histograms with KDE**: Show the distribution of variables with both histogram bars and a kernel density estimate line.
                
                2. **Box Plots**: Display the median, quartiles, and potential outliers of numeric variables.
                
                3. **Q-Q Plots**: Compare the distribution of the data against a theoretical normal distribution, helping assess normality.
                
                4. **Scatter Matrix**: Show pairwise relationships between multiple variables, useful for identifying correlations and patterns.
                """)
            
            elif method == "Outlier Detection":
                st.write("""
                **Key Visualizations:**
                
                1. **Box Plots**: Highlight potential outliers as points beyond the whiskers.
                
                2. **Histograms with Outlier Markers**: Show the distribution with Z-score and IQR outliers marked separately.
                
                3. **Scatter Plots**: Display the relationship between variables with outliers highlighted.
                """)
        
        elif analysis_type == "correlation":
            if method in ["Pearson Correlation", "Spearman Correlation"]:
                st.write(f"""
                **Key Visualizations:**
                
                1. **Correlation Heatmap**: Color-coded matrix showing the {method} coefficients between all pairs of variables.
                
                2. **Scatter Plot Matrix**: Grid of scatter plots showing pairwise relationships between selected variables.
                
                3. **Interactive Correlation Explorer**: Focused scatter plots of specific variable pairs with trendlines, allowing detailed examination of key relationships.
                """)
            
            elif method == "Feature Importance":
                st.write("""
                **Key Visualizations:**
                
                1. **Feature Importance Bar Chart**: Horizontal bar chart showing the relative importance of each feature.
                
                2. **Feature Coefficient Bar Chart**: Shows the direction and magnitude of each feature's influence.
                
                3. **Scatter Plots**: Show the relationship between top features and the target variable.
                
                4. **3D Visualization**: Three-dimensional plot showing the relationship between the top two features and the target variable.
                """)
        
        elif analysis_type == "regression":
            if method in ["Linear Regression", "Multiple Regression"]:
                st.write("""
                **Key Visualizations:**
                
                1. **Coefficient Bar Chart**: Shows the magnitude and direction of each feature's influence on the target variable.
                
                2. **R² Gauge Chart**: Visualizes the model's explanatory power.
                
                3. **Training vs Testing Metrics**: Bar chart comparing model performance on training and testing data.
                
                4. **Actual vs Predicted Plot**: Scatter plot showing how well predicted values match actual values.
                
                5. **Residual Plot**: Shows the difference between actual and predicted values, helping assess model assumptions.
                """)
        
        elif analysis_type == "classification":
            if method == "Logistic Regression":
                st.write("""
                **Key Visualizations:**
                
                1. **Confusion Matrix Heatmap**: Shows the counts of true positives, false positives, true negatives, and false negatives.
                
                2. **Feature Importance Bar Chart**: Displays the influence of each feature on the classification decision.
                
                3. **ROC Curve**: Shows the trade-off between true positive rate and false positive rate at different classification thresholds.
                
                4. **AUC Gauge Chart**: Visualizes the area under the ROC curve as a measure of model quality.
                
                5. **Class Distribution Pie Chart**: Shows the proportion of each class in the target variable.
                """)
        
        elif analysis_type == "clustering":
            if method == "K-Means Clustering":
                st.write("""
                **Key Visualizations:**
                
                1. **Cluster Scatter Plot**: Shows the distribution of data points in feature space, colored by cluster assignment.
                
                2. **Cluster Size Distribution**: Bar chart showing the number of points in each cluster.
                
                3. **Cluster Profile Heatmap**: Shows the average value of each feature within each cluster.
                
                4. **Radar Chart**: Displays the normalized feature values for each cluster, helping visualize cluster characteristics.
                
                5. **Silhouette Score Gauge**: Visualizes the quality of the clustering.
                
                6. **Elbow Method Plot**: Shows inertia values for different numbers of clusters, helping validate the chosen k value.
                """)
        
        elif analysis_type == "time_series":
            if method == "Trend Analysis":
                st.write("""
                **Key Visualizations:**
                
                1. **Time Series Plot**: Shows the variable values over time.
                
                2. **Linear Trend Line**: Overlays a fitted trend line on the time series plot.
                
                3. **Rolling Average Plot**: Shows the original time series with a smoothed version overlaid.
                """)
            
            elif method == "Seasonality Analysis":
                st.write("""
                **Key Visualizations:**
                
                1. **Time Series Plot**: Shows the variable values over time.
                
                2. **Seasonal Pattern Plot**: Shows the average pattern within each seasonal cycle.
                
                3. **Seasonal Strength Gauge**: Visualizes the strength of the seasonal component.
                
                4. **Time Series Decomposition**: Separates the time series into trend, seasonal, and residual components.
                """)
        
        # Custom visualizations
        if 'custom_visualizations' in visualization_results:
            custom_viz = visualization_results['custom_visualizations']
            
            if 'chart_type' in custom_viz:
                st.write(f"""
                **Custom Visualization: {custom_viz['chart_type']}**
                
                A custom {custom_viz['chart_type'].lower()} was created to explore specific aspects of the data. This visualization allows for interactive exploration of relationships and patterns in the dataset.
                """)
        
        # Advanced visualizations
        if 'advanced_visualizations' in visualization_results:
            advanced_viz = visualization_results['advanced_visualizations']
            
            if 'advanced_viz_type' in advanced_viz:
                st.write(f"""
                **Advanced Visualization: {advanced_viz['advanced_viz_type']}**
                
                An advanced {advanced_viz['advanced_viz_type'].lower()} was created to provide deeper insights into the data structure and relationships. This sophisticated visualization technique helps uncover complex patterns that might not be visible in standard plots.
                """)
    
    # 7. Conclusions & Recommendations
    with detailed_tabs[6]:
        st.write("### Conclusions & Recommendations")
        
        # Different conclusions based on analysis type
        analysis_type = analysis_results.get('type', '')
        method = analysis_results.get('method', '')
        
        if analysis_type == "exploratory":
            st.write("""
            **Conclusions:**
            
            The exploratory data analysis has provided valuable insights into the structure and characteristics of the dataset. Key patterns and distributions have been identified, which can guide further analysis and decision making.
            
            **Recommendations:**
            
            1. **Focus on Key Variables**: Prioritize variables that showed significant variation or unusual distributions for further analysis.
            
            2. **Address Data Quality Issues**: Consider strategies for handling missing values and outliers identified during the analysis.
            
            3. **Consider Transformations**: For variables with skewed distributions, consider applying transformations to normalize them before proceeding with more advanced analyses.
            
            4. **Explore Relationships**: Investigate the correlations identified to understand underlying mechanisms and potential causal relationships.
            """)
        
        elif analysis_type == "correlation":
            st.write("""
            **Conclusions:**
            
            The correlation analysis has revealed important relationships between variables in the dataset. These relationships provide insights into how variables influence each other and which variables are most closely connected.
            
            **Recommendations:**
            
            1. **Focus on Strong Correlations**: Prioritize the strongest correlations for further investigation, as these represent the most significant relationships in the data.
            
            2. **Consider Causality**: While correlation does not imply causation, the identified relationships can guide causal hypothesis formation for further research.
            
            3. **Address Multicollinearity**: For predictive modeling, be aware of highly correlated predictors that might cause multicollinearity issues.
            
            4. **Feature Selection**: Use correlation results to inform feature selection for machine learning models, potentially removing redundant features.
            """)
        
        elif analysis_type == "regression":
            # Get R² for recommendations
            r2 = analysis_results.get('metrics', {}).get('test_r2', 0)
            
            st.write(f"""
            **Conclusions:**
            
            The regression analysis has provided a model that explains relationships between the predictors and the target variable. The model {'has good predictive power' if r2 > 0.7 else 'has moderate predictive power' if r2 > 0.3 else 'has limited predictive power'}, and has identified the most influential factors affecting the target variable.
            
            **Recommendations:**
            
            1. **Focus on Key Predictors**: Prioritize the variables with the largest coefficients for business decisions and further investigation.
            
            2. **Model Improvement**: {'Consider the model sufficient for prediction purposes.' if r2 > 0.7 else 'Consider adding more relevant predictors or using more complex models to improve performance.' if r2 > 0.3 else 'Consider alternative modeling approaches or additional data collection to improve predictive power.'}
            
            3. **Validate Assumptions**: Ensure that regression assumptions are met by examining residual plots and other diagnostics.
            
            4. **Consider Interactions**: Explore potential interaction effects between predictors that might improve model performance.
            """)
        
        elif analysis_type == "classification":
            # Get accuracy for recommendations
            accuracy = analysis_results.get('metrics', {}).get('test_accuracy', 0)
            
            st.write(f"""
            **Conclusions:**
            
            The classification analysis has produced a model that predicts the target variable with {'high' if accuracy > 0.8 else 'moderate' if accuracy > 0.6 else 'limited'} accuracy. The model has identified the key factors that influence the classification outcomes.
            
            **Recommendations:**
            
            1. **Focus on Key Predictors**: Prioritize the variables with the largest coefficients for decision making and further investigation.
            
            2. **Model Deployment**: {'The model appears robust and could be deployed for prediction purposes.' if accuracy > 0.8 else 'Consider improving the model before deployment for critical applications.' if accuracy > 0.6 else 'Significant improvements are needed before the model can be reliably deployed.'}
            
            3. **Class Imbalance**: Check for and address any class imbalance issues that might affect model performance.
            
            4. **Consider Alternative Models**: {'The current model performs well, but comparing with other algorithms could yield further improvements.' if accuracy > 0.8 else 'Explore other classification algorithms that might better capture the patterns in the data.' if accuracy > 0.6 else 'Try different classification algorithms that might be more suitable for this dataset.'}
            """)
        
        elif analysis_type == "clustering":
            # Get silhouette score for recommendations
            silhouette = analysis_results.get('silhouette_score', 0)
            
            st.write(f"""
            **Conclusions:**
            
            The clustering analysis has identified natural groupings in the data, revealing distinct segments with specific characteristics. The clustering {'is of high quality' if silhouette > 0.7 else 'is of moderate quality' if silhouette > 0.5 else 'shows some overlap between clusters' if silhouette > 0 else 'has significant overlap between clusters'}, suggesting that the identified segments are {'well-separated' if silhouette > 0.7 else 'reasonably distinct' if silhouette > 0.5 else 'somewhat distinct' if silhouette > 0 else 'not well-separated'}.
            
            **Recommendations:**
            
            1. **Segment-Specific Strategies**: Develop tailored approaches for each identified cluster based on their unique characteristics.
            
            2. **Cluster Validation**: {'The clusters appear robust and can be used with confidence.' if silhouette > 0.7 else 'Consider validating the clusters with domain expertise before making critical decisions.' if silhouette > 0.5 else 'Use caution when interpreting these clusters and validate with additional methods.' if silhouette > 0 else 'Reconsider the clustering approach, as the current results show poor separation.'}
            
            3. **Feature Importance**: Focus on the features that most strongly differentiate between clusters for decision making.
            
            4. **Consider Alternative Clustering**: {'The current approach works well, but comparing with other clustering methods could provide additional insights.' if silhouette > 0.7 else 'Explore other clustering algorithms or different distance metrics that might better separate the data.' if silhouette > 0.5 else 'Try different clustering approaches that might better capture the natural groupings in this dataset.' if silhouette > 0 else 'Try entirely different clustering algorithms, as K-means may not be suitable for this data structure.'}
            """)
        
        elif analysis_type == "time_series":
            if method == "Trend Analysis":
                # Get p-value and R² for recommendations
                p_value = analysis_results.get('trend_statistics', {}).get('p_value', 1)
                r_squared = analysis_results.get('trend_statistics', {}).get('r_squared', 0)
                slope = analysis_results.get('trend_statistics', {}).get('slope', 0)
                
                trend_direction = "upward" if slope > 0 else "downward"
                
                st.write(f"""
                **Conclusions:**
                
                The time series analysis has {'revealed a significant ' + trend_direction + ' trend' if p_value < 0.05 else 'not found a significant trend'} in the data. {'This trend explains a ' + ('large' if r_squared > 0.7 else 'moderate' if r_squared > 0.3 else 'small') + ' portion of the variability in the data.' if p_value < 0.05 else 'The data appears to be relatively stable over time without a clear directional trend.'}
                
                **Recommendations:**
                
                1. **{'Trend-Based Planning' if p_value < 0.05 else 'Stability-Based Planning'}**: {'Incorporate the identified trend into forecasting and planning processes.' if p_value < 0.05 else 'Plan based on the relative stability of the variable, focusing on mean levels rather than trends.'}
                
                2. **{'Monitor Trend Changes' if p_value < 0.05 else 'Continue Monitoring'}**: {'Regularly update the trend analysis to detect any changes in the trend direction or magnitude.' if p_value < 0.05 else 'Continue monitoring the time series to detect any emerging trends.'}
                
                3. **{'Consider Growth Rate' if p_value < 0.05 and slope > 0 else 'Consider Decline Rate' if p_value < 0.05 and slope < 0 else 'Consider Variability'}**: {'The growth rate of approximately ' + f"{abs(slope):.6f}" + ' units per time period should be factored into projections.' if p_value < 0.05 and slope > 0 else 'The decline rate of approximately ' + f"{abs(slope):.6f}" + ' units per time period should be factored into projections.' if p_value < 0.05 and slope < 0 else 'Focus on understanding the sources of variability in the time series.'}
                
                4. **Explore Causal Factors**: Investigate potential factors that might be {'driving the observed trend.' if p_value < 0.05 else 'causing fluctuations in the time series.'}
                """)
            
            elif method == "Seasonality Analysis":
                # Get seasonal strength for recommendations
                strength = analysis_results.get('seasonality', {}).get('seasonal_strength')
                period = analysis_results.get('seasonality', {}).get('period', 'the identified')
                
                st.write(f"""
                **Conclusions:**
                
                The seasonality analysis has {'revealed strong seasonal patterns' if strength > 0.6 else 'identified moderate seasonal patterns' if strength > 0.3 else 'found weak seasonal patterns' if strength is not None else 'examined seasonal patterns'} with a period of {period} units. {'These seasonal effects account for a significant portion of the variability in the data.' if strength > 0.6 else 'These seasonal effects contribute moderately to the variability in the data.' if strength > 0.3 else 'These seasonal effects account for only a small portion of the variability in the data.' if strength is not None else ''}
                
                **Recommendations:**
                
                1. **Seasonal Planning**: Adjust operations and planning to account for the {'strong' if strength > 0.6 else 'moderate' if strength > 0.3 else 'weak' if strength is not None else 'identified'} seasonal patterns, particularly around peak and trough periods.
                
                2. **Seasonal Adjustments**: When analyzing trends or comparing performance over time, apply seasonal adjustments to isolate the underlying patterns.
                
                3. **Resource Allocation**: Allocate resources to match the seasonal demand patterns, increasing capacity during peak periods and potentially reducing it during troughs.
                
                4. **{'Leverage Predictability' if strength > 0.6 else 'Enhance Predictability' if strength > 0.3 else 'Consider Other Factors' if strength is not None else 'Further Investigation'}**: {'The strong seasonality makes the time series relatively predictable, which can be leveraged for forecasting.' if strength > 0.6 else 'The moderate seasonality provides some predictability, but other factors should also be considered in forecasting.' if strength > 0.3 else 'Given the weak seasonality, focus on other factors that might better explain the variability in the data.' if strength is not None else 'Further investigate the factors driving the time patterns in the data.'}
                """)
        
        # General recommendations for all analysis types
        st.write("""
        **General Recommendations:**
        
        1. **Data Collection**: Continue collecting high-quality data to support ongoing analysis and monitoring.
        
        2. **Documentation**: Document the insights and methodologies used in this analysis for future reference.
        
        3. **Implementation Plan**: Develop a clear implementation plan for acting on the insights gained from this analysis.
        
        4. **Feedback Loop**: Establish a feedback loop to measure the impact of any actions taken based on this analysis.
        
        5. **Regular Updates**: Update the analysis periodically to incorporate new data and refine the findings.
        """)
    
    return detailed_data

def extract_key_metrics(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from analysis results for the summary report"""
    metrics = {}
    
    analysis_type = analysis_results.get('type', '')
    
    if analysis_type == "correlation":
        if 'correlation_matrix' in analysis_results:
            # Find the max absolute correlation (excluding self-correlations)
            corr_matrix = pd.DataFrame(analysis_results['correlation_matrix'])
            np.fill_diagonal(corr_matrix.values, 0)  # Exclude diagonal (self-correlations)
            max_corr = corr_matrix.abs().max().max()
            metrics['max_correlation'] = float(max_corr) if not pd.isna(max_corr) else 0
    
    elif analysis_type == "regression":
        if 'metrics' in analysis_results:
            metrics.update(analysis_results['metrics'])
    
    elif analysis_type == "classification":
        if 'metrics' in analysis_results:
            metrics.update(analysis_results['metrics'])
        
        if 'roc_auc' in analysis_results:
            metrics['auc'] = analysis_results['roc_auc']
    
    elif analysis_type == "clustering":
        if 'silhouette_score' in analysis_results:
            metrics['silhouette'] = analysis_results['silhouette_score']
        
        if 'selected_k' in analysis_results:
            metrics['k'] = analysis_results['selected_k']
    
    elif analysis_type == "time_series":
        method = analysis_results.get('method', '')
        
        if method == "Trend Analysis" and 'trend_statistics' in analysis_results:
            metrics.update(analysis_results['trend_statistics'])
        
        elif method == "Seasonality Analysis" and 'seasonality' in analysis_results:
            seasonality = analysis_results['seasonality']
            if 'seasonal_strength' in seasonality:
                metrics['seasonal_strength'] = seasonality['seasonal_strength']
            if 'period' in seasonality:
                metrics['period'] = seasonality['period']
    
    return metrics

def export_results(original_df: pd.DataFrame, processed_df: pd.DataFrame, 
                  problem_def: Dict[str, Any], analysis_results: Dict[str, Any], 
                  visualization_results: Dict[str, Any]) -> None:
    """Generate and provide download options for the analysis results"""
    st.subheader("Export Data and Results")
    
    # Check if we're in the export options tab
    if 'export_settings' not in st.session_state:
        st.session_state.export_settings = {
            "include_original": True,
            "include_processed": True,
            "include_summary": True,
            "include_analysis": True,
            "include_visualizations": True,
            "include_detailed": True,
            "export_format": "CSV"
        }
    
    export_settings = st.session_state.export_settings
    
    # Generate exports based on selected format
    export_format = export_settings["export_format"]
    
    if export_format == "CSV":
        # Generate CSV exports
        if export_settings["include_original"]:
            csv_original = original_df.to_csv(index=False)
            st.download_button(
                label="Download Original Data (CSV)",
                data=csv_original,
                file_name="original_data.csv",
                mime="text/csv"
            )
        
        if export_settings["include_processed"]:
            csv_processed = processed_df.to_csv(index=False)
            st.download_button(
                label="Download Processed Data (CSV)",
                data=csv_processed,
                file_name="processed_data.csv",
                mime="text/csv"
            )
        
        if export_settings["include_summary"] or export_settings["include_analysis"] or export_settings["include_detailed"]:
            # Create a summary report CSV
            report_rows = []
            
            # Add timestamp
            report_rows.append(["Report Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            report_rows.append([])
            
            # Problem definition
            if export_settings["include_summary"]:
                report_rows.append(["ANALYSIS SUMMARY", ""])
                report_rows.append(["Analysis Goal", problem_def.get('analysis_goal', 'Not specified')])
                report_rows.append(["Analysis Method", problem_def.get('analysis_method', 'Not specified')])
                report_rows.append(["Target Variable", problem_def.get('target_variable', 'None')])
                report_rows.append(["Selected Features", ", ".join(problem_def.get('features', []))])
                report_rows.append([])
            
            # Analysis results
            if export_settings["include_analysis"]:
                report_rows.append(["ANALYSIS RESULTS", ""])
                
                analysis_type = analysis_results.get('type', '')
                
                if analysis_type == "regression":
                    if 'metrics' in analysis_results:
                        metrics = analysis_results['metrics']
                        report_rows.append(["R² (Training)", f"{metrics.get('train_r2', 0):.4f}"])
                        report_rows.append(["R² (Testing)", f"{metrics.get('test_r2', 0):.4f}"])
                        report_rows.append(["RMSE (Training)", f"{metrics.get('train_rmse', 0):.4f}"])
                        report_rows.append(["RMSE (Testing)", f"{metrics.get('test_rmse', 0):.4f}"])
                
                elif analysis_type == "classification":
                    if 'metrics' in analysis_results:
                        metrics = analysis_results['metrics']
                        report_rows.append(["Accuracy (Training)", f"{metrics.get('train_accuracy', 0):.4f}"])
                        report_rows.append(["Accuracy (Testing)", f"{metrics.get('test_accuracy', 0):.4f}"])
                
                elif analysis_type == "clustering":
                    if 'silhouette_score' in analysis_results:
                        report_rows.append(["Silhouette Score", f"{analysis_results['silhouette_score']:.4f}"])
                    
                    if 'selected_k' in analysis_results:
                        report_rows.append(["Number of Clusters", str(analysis_results['selected_k'])])
                
                elif analysis_type == "time_series":
                    method = analysis_results.get('method', '')
                    
                    if method == "Trend Analysis" and 'trend_statistics' in analysis_results:
                        stats = analysis_results['trend_statistics']
                        report_rows.append(["Slope", f"{stats.get('slope', 0):.6f}"])
                        report_rows.append(["R²", f"{stats.get('r_squared', 0):.4f}"])
                        report_rows.append(["P-value", f"{stats.get('p_value', 0):.6f}"])
                    
                    elif method == "Seasonality Analysis" and 'seasonality' in analysis_results:
                        seasonality = analysis_results['seasonality']
                        report_rows.append(["Seasonal Period", str(seasonality.get('period', 'N/A'))])
                        if 'seasonal_strength' in seasonality and seasonality['seasonal_strength'] is not None:
                            report_rows.append(["Seasonal Strength", f"{seasonality['seasonal_strength']:.4f}"])
            
            # Create a DataFrame and convert to CSV
            report_df = pd.DataFrame(report_rows)
            csv_report = report_df.to_csv(index=False, header=False)
            
            st.download_button(
                label="Download Analysis Report (CSV)",
                data=csv_report,
                file_name="analysis_report.csv",
                mime="text/csv"
            )
    
    elif export_format == "Excel":
        # Create Excel file in memory
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Add sheets based on user selection
            if export_settings["include_original"]:
                original_df.to_excel(writer, sheet_name="Original Data", index=False)
            
            if export_settings["include_processed"]:
                processed_df.to_excel(writer, sheet_name="Processed Data", index=False)
            
            if export_settings["include_summary"]:
                # Create summary sheet
                summary_data = [
                    ["Report Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    [],
                    ["ANALYSIS SUMMARY", ""],
                    ["Analysis Goal", problem_def.get('analysis_goal', 'Not specified')],
                    ["Analysis Method", problem_def.get('analysis_method', 'Not specified')],
                    ["Target Variable", problem_def.get('target_variable', 'None')],
                    ["Selected Features", ", ".join(problem_def.get('features', []))],
                    []
                ]
                
                pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary Report", index=False, header=False)
            
            if export_settings["include_analysis"]:
                # Create analysis results sheet
                analysis_type = analysis_results.get('type', '')
                
                analysis_data = [
                    ["ANALYSIS RESULTS", ""],
                    ["Analysis Type", analysis_type.capitalize()],
                    ["Analysis Method", analysis_results.get('method', 'Not specified')],
                    []
                ]
                
                if analysis_type == "regression":
                    if 'metrics' in analysis_results:
                        metrics = analysis_results['metrics']
                        analysis_data.extend([
                            ["R² (Training)", f"{metrics.get('train_r2', 0):.4f}"],
                            ["R² (Testing)", f"{metrics.get('test_r2', 0):.4f}"],
                            ["RMSE (Training)", f"{metrics.get('train_rmse', 0):.4f}"],
                            ["RMSE (Testing)", f"{metrics.get('test_rmse', 0):.4f}"]
                        ])
                
                elif analysis_type == "classification":
                    if 'metrics' in analysis_results:
                        metrics = analysis_results['metrics']
                        analysis_data.extend([
                            ["Accuracy (Training)", f"{metrics.get('train_accuracy', 0):.4f}"],
                            ["Accuracy (Testing)", f"{metrics.get('test_accuracy', 0):.4f}"]
                        ])
                
                pd.DataFrame(analysis_data).to_excel(writer, sheet_name="Analysis Results", index=False, header=False)
                
                # Add model coefficients if available
                if 'model_coefficients' in analysis_results:
                    coef_data = pd.DataFrame(analysis_results['model_coefficients'])
                    if not coef_data.empty:
                        coef_data.to_excel(writer, sheet_name="Model Coefficients", index=False)
            
            if export_settings["include_detailed"]:
                # Create detailed report sheet
                detailed_data = [
                    ["DETAILED REPORT", ""],
                    ["Report Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    [],
                    ["Original Dataset Shape", f"{original_df.shape[0]} rows × {original_df.shape[1]} columns"],
                    ["Processed Dataset Shape", f"{processed_df.shape[0]} rows × {processed_df.shape[1]} columns"],
                    []
                ]
                
                pd.DataFrame(detailed_data).to_excel(writer, sheet_name="Detailed Report", index=False, header=False)
        
        # Offer the Excel file for download
        excel_data = output.getvalue()
        
        st.download_button(
            label="Download Full Report (Excel)",
            data=excel_data,
            file_name="data_analysis_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    elif export_format == "HTML":
        # Create HTML report
        html_parts = []
        
        # Start HTML document
        html_parts.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
                h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
                h2 { color: #3498db; margin-top: 30px; }
                h3 { color: #2980b9; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { padding: 12px 15px; border-bottom: 1px solid #ddd; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:hover { background-color: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; }
                .section { margin-bottom: 30px; }
                .metric { font-weight: bold; margin-right: 10px; }
                .value { color: #2c3e50; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Data Analysis Report</h1>
                <p>Report Generated: %s</p>
        """ % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Summary section
        if export_settings["include_summary"]:
            html_parts.append("""
                <div class="section">
                    <h2>Analysis Summary</h2>
                    <p><span class="metric">Analysis Goal:</span> <span class="value">%s</span></p>
                    <p><span class="metric">Analysis Method:</span> <span class="value">%s</span></p>
                    <p><span class="metric">Target Variable:</span> <span class="value">%s</span></p>
                    <p><span class="metric">Selected Features:</span> <span class="value">%s</span></p>
                </div>
            """ % (
                problem_def.get('analysis_goal', 'Not specified'),
                problem_def.get('analysis_method', 'Not specified'),
                problem_def.get('target_variable', 'None'),
                ", ".join(problem_def.get('features', []))
            ))
        
        # Data section
        if export_settings["include_original"] or export_settings["include_processed"]:
            html_parts.append("""
                <div class="section">
                    <h2>Dataset Information</h2>
            """)
            
            if export_settings["include_original"]:
                html_parts.append("""
                    <h3>Original Dataset</h3>
                    <p>Shape: %d rows × %d columns</p>
                    <h4>Preview:</h4>
                    %s
                """ % (
                    original_df.shape[0],
                    original_df.shape[1],
                    original_df.head(5).to_html(index=False)
                ))
            
            if export_settings["include_processed"]:
                html_parts.append("""
                    <h3>Processed Dataset</h3>
                    <p>Shape: %d rows × %d columns</p>
                    <h4>Preview:</h4>
                    %s
                """ % (
                    processed_df.shape[0],
                    processed_df.shape[1],
                    processed_df.head(5).to_html(index=False)
                ))
            
            html_parts.append("""
                </div>
            """)
        
        # Analysis results
        if export_settings["include_analysis"]:
            analysis_type = analysis_results.get('type', '')
            method = analysis_results.get('method', '')
            
            html_parts.append("""
                <div class="section">
                    <h2>Analysis Results</h2>
                    <p><span class="metric">Analysis Type:</span> <span class="value">%s</span></p>
                    <p><span class="metric">Analysis Method:</span> <span class="value">%s</span></p>
            """ % (analysis_type.capitalize(), method))
            
            if analysis_type == "regression" and 'metrics' in analysis_results:
                metrics = analysis_results['metrics']
                html_parts.append("""
                    <h3>Regression Metrics</h3>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Training</th>
                            <th>Testing</th>
                        </tr>
                        <tr>
                            <td>R²</td>
                            <td>%.4f</td>
                            <td>%.4f</td>
                        </tr>
                        <tr>
                            <td>RMSE</td>
                            <td>%.4f</td>
                            <td>%.4f</td>
                        </tr>
                    </table>
                """ % (
                    metrics.get('train_r2', 0),
                    metrics.get('test_r2', 0),
                    metrics.get('train_rmse', 0),
                    metrics.get('test_rmse', 0)
                ))
                
                if 'model_coefficients' in analysis_results:
                    coef_data = pd.DataFrame(analysis_results['model_coefficients'])
                    if not coef_data.empty:
                        html_parts.append("""
                            <h3>Model Coefficients</h3>
                            %s
                        """ % coef_data.to_html(index=False))
            
            elif analysis_type == "classification" and 'metrics' in analysis_results:
                metrics = analysis_results['metrics']
                html_parts.append("""
                    <h3>Classification Metrics</h3>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Training</th>
                            <th>Testing</th>
                        </tr>
                        <tr>
                            <td>Accuracy</td>
                            <td>%.4f</td>
                            <td>%.4f</td>
                        </tr>
                    </table>
                """ % (
                    metrics.get('train_accuracy', 0),
                    metrics.get('test_accuracy', 0)
                ))
            
            elif analysis_type == "clustering":
                html_parts.append("""
                    <h3>Clustering Results</h3>
                """)
                
                if 'selected_k' in analysis_results:
                    html_parts.append("""
                        <p><span class="metric">Number of Clusters:</span> <span class="value">%d</span></p>
                    """ % analysis_results['selected_k'])
                
                if 'silhouette_score' in analysis_results:
                    html_parts.append("""
                        <p><span class="metric">Silhouette Score:</span> <span class="value">%.4f</span></p>
                    """ % analysis_results['silhouette_score'])
                
                if 'cluster_counts' in analysis_results:
                    cluster_counts = analysis_results['cluster_counts']
                    html_parts.append("""
                        <h4>Cluster Sizes</h4>
                        <table>
                            <tr>
                                <th>Cluster</th>
                                <th>Size</th>
                            </tr>
                    """)
                    
                    for cluster, count in cluster_counts.items():
                        html_parts.append("""
                            <tr>
                                <td>%s</td>
                                <td>%d</td>
                            </tr>
                        """ % (cluster, count))
                    
                    html_parts.append("""
                        </table>
                    """)
            
            elif analysis_type == "time_series":
                if method == "Trend Analysis" and 'trend_statistics' in analysis_results:
                    stats = analysis_results['trend_statistics']
                    html_parts.append("""
                        <h3>Trend Analysis Results</h3>
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                            <tr>
                                <td>Slope</td>
                                <td>%.6f</td>
                            </tr>
                            <tr>
                                <td>R²</td>
                                <td>%.4f</td>
                            </tr>
                            <tr>
                                <td>P-value</td>
                                <td>%.6f</td>
                            </tr>
                        </table>
                    """ % (
                        stats.get('slope', 0),
                        stats.get('r_squared', 0),
                        stats.get('p_value', 0)
                    ))
                
                elif method == "Seasonality Analysis" and 'seasonality' in analysis_results:
                    seasonality = analysis_results['seasonality']
                    html_parts.append("""
                        <h3>Seasonality Analysis Results</h3>
                        <p><span class="metric">Seasonal Period:</span> <span class="value">%s</span></p>
                    """ % seasonality.get('period', 'N/A'))
                    
                    if 'seasonal_strength' in seasonality and seasonality['seasonal_strength'] is not None:
                        html_parts.append("""
                            <p><span class="metric">Seasonal Strength:</span> <span class="value">%.4f</span></p>
                        """ % seasonality['seasonal_strength'])
            
            html_parts.append("""
                </div>
            """)
        
        # Conclusions
        if export_settings["include_detailed"]:
            html_parts.append("""
                <div class="section">
                    <h2>Conclusions & Recommendations</h2>
                    <p>The analysis has provided valuable insights into the data structure and relationships. 
                    Please refer to the interactive application for detailed interpretations and visualizations.</p>
                </div>
            """)
        
        # Close HTML document
        html_parts.append("""
            </div>
        </body>
        </html>
        """)
        
        # Combine HTML parts
        html_report = "".join(html_parts)
        
        # Offer HTML file for download
        st.download_button(
            label="Download Report (HTML)",
            data=html_report,
            file_name="data_analysis_report.html",
            mime="text/html"
        )
    
    elif export_format == "JSON":
        # Create a JSON report
        json_report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "problem_definition": problem_def,
            "analysis_results": analysis_results
        }
        
        # Add original data if requested
        if export_settings["include_original"]:
            json_report["original_data"] = original_df.to_dict(orient="records")
        
        # Add processed data if requested
        if export_settings["include_processed"]:
            json_report["processed_data"] = processed_df.to_dict(orient="records")
        
        # Convert to JSON string
        import json
        json_str = json.dumps(json_report, indent=2)
        
        # Offer JSON file for download
        st.download_button(
            label="Download Report (JSON)",
            data=json_str,
            file_name="data_analysis_report.json",
            mime="application/json"
        )
    
    # Add additional information
    st.markdown("""
    ### Next Steps
    
    After downloading your report, you can:
    
    1. **Share the analysis** with colleagues or stakeholders
    2. **Import the data** into other tools for further analysis
    3. **Document your findings** in presentations or reports
    4. **Implement actions** based on the insights gained
    """)

