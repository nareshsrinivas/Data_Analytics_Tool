import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from assets.stock_photos import display_header_image

def render_visualization(df: pd.DataFrame, problem_def: Dict[str, Any], 
                        analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Renders the visualization section and returns the visualization results
    
    Parameters:
    df - The dataset as a pandas DataFrame
    problem_def - The problem definition dictionary
    analysis_results - The analysis results dictionary
    
    Returns:
    Dictionary containing visualization results
    """
    st.header("5. Visualization", divider=True)
    
    # Display a relevant image at the top
    display_header_image("data visualization charts")
    
    if df.empty:
        st.warning("Please upload and analyze your dataset first")
        return {}
    
    if not analysis_results:
        st.warning("Please complete the analysis step first")
        return {}
    
    # Initialize visualization results if not exists
    if 'visualization_results' not in st.session_state:
        st.session_state.visualization_results = {}
    
    visualization_results = st.session_state.visualization_results
    
    # Get analysis goal and method from problem definition
    goal = problem_def.get('analysis_goal', 'Exploratory Analysis')
    method = problem_def.get('analysis_method', 'Descriptive Statistics')
    target = problem_def.get('target_variable')
    features = problem_def.get('features', [])
    
    # Create tabs for different types of visualizations
    vis_tabs = st.tabs(["Key Insights", "Custom Visualizations", "Advanced Plots"])
    
    # First tab: Key Insights (Based on Analysis Results)
    with vis_tabs[0]:
        st.subheader("Key Insights from Analysis")
        
        # The visualizations here will depend on the type of analysis performed
        analysis_type = analysis_results.get('type', '')
        
        if analysis_type == "exploratory":
            visualize_exploratory_results(df, analysis_results, features)
        
        elif analysis_type == "correlation":
            visualize_correlation_results(df, analysis_results, features)
        
        elif analysis_type == "regression":
            visualize_regression_results(df, analysis_results, target, features)
        
        elif analysis_type == "classification":
            visualize_classification_results(df, analysis_results, target, features)
        
        elif analysis_type == "clustering":
            visualize_clustering_results(df, analysis_results, features)
        
        elif analysis_type == "time_series":
            visualize_time_series_results(df, analysis_results, features)
        
        else:
            st.info("No specific insights to visualize for the current analysis")
    
    # Second tab: Custom Visualizations
    with vis_tabs[1]:
        st.subheader("Custom Visualizations")
        
        # Let users create custom plots
        custom_viz_results = create_custom_visualizations(df, features, target)
        
        # Add custom visualization results to the output
        if custom_viz_results:
            visualization_results['custom_visualizations'] = custom_viz_results
    
    # Third tab: Advanced Plots
    with vis_tabs[2]:
        st.subheader("Advanced Visualizations")
        
        # Create more complex/specialized visualizations
        advanced_viz_results = create_advanced_visualizations(df, problem_def)
        
        # Add advanced visualization results to the output
        if advanced_viz_results:
            visualization_results['advanced_visualizations'] = advanced_viz_results
    
    # Store visualization results in session state
    st.session_state.visualization_results = visualization_results
    
    return visualization_results

def visualize_exploratory_results(df: pd.DataFrame, 
                                 analysis_results: Dict[str, Any], 
                                 features: List[str]):
    """Visualize results from exploratory analysis"""
    method = analysis_results.get('method', '')
    
    if method == "Descriptive Statistics":
        st.subheader("Statistical Summary Visualization")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            # Let user select columns to visualize
            selected_cols = st.multiselect(
                "Select columns to visualize",
                options=numeric_cols,
                default=numeric_cols[:min(4, len(numeric_cols))]
            )
            
            if selected_cols:
                # Create a box plot for selected columns
                fig = px.box(
                    df,
                    y=selected_cols,
                    title="Box Plot of Selected Features",
                    points="all",  # Show all points
                    notched=True  # Show confidence interval for median
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Create a violin plot for distribution comparison
                fig = px.violin(
                    df,
                    y=selected_cols,
                    title="Distribution Comparison of Selected Features",
                    box=True,  # Show box plot inside violin
                    points="all"  # Show all points
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Create a correlation heatmap
                if len(selected_cols) > 1:
                    corr_matrix = df[selected_cols].corr()
                    fig = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        color_continuous_scale='RdBu_r',
                        title="Correlation Heatmap"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Categorical columns visualization
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if cat_cols:
            st.subheader("Categorical Data Visualization")
            
            selected_cat_col = st.selectbox(
                "Select a categorical column to visualize",
                options=cat_cols
            )
            
            if selected_cat_col:
                # Count the categories and sort by frequency
                value_counts = df[selected_cat_col].value_counts().reset_index()
                value_counts.columns = [selected_cat_col, 'Count']
                
                # Limit to top 10 categories if there are too many
                if len(value_counts) > 10:
                    value_counts = value_counts.head(10)
                    title_suffix = " (Top 10)"
                else:
                    title_suffix = ""
                
                # Create a bar chart
                fig = px.bar(
                    value_counts,
                    x=selected_cat_col,
                    y='Count',
                    title=f"Frequency of {selected_cat_col} Categories{title_suffix}",
                    color='Count',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Create a pie chart
                fig = px.pie(
                    value_counts,
                    names=selected_cat_col,
                    values='Count',
                    title=f"Proportion of {selected_cat_col} Categories{title_suffix}"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    elif method == "Data Distribution Analysis":
        st.subheader("Distribution Analysis Visualization")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            # Let user select a column to visualize distribution
            selected_col = st.selectbox(
                "Select a column to visualize distribution",
                options=numeric_cols
            )
            
            if selected_col:
                # Combined histogram and KDE plot
                fig = px.histogram(
                    df,
                    x=selected_col,
                    marginal="box",
                    title=f"Distribution of {selected_col}",
                    histnorm="probability density",
                    nbins=30
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Q-Q plot
                fig = go.Figure()
                
                # Calculate the quantiles
                data = df[selected_col].dropna()
                
                # Theoretical quantiles from standard normal distribution
                theoretical_quantiles = np.sort(stats.norm.ppf(np.linspace(0.01, 0.99, len(data))))
                
                # Actual quantiles from the data
                actual_quantiles = np.sort(data)
                
                # Plot the Q-Q plot
                fig.add_trace(go.Scatter(
                    x=theoretical_quantiles,
                    y=actual_quantiles,
                    mode='markers',
                    marker=dict(color='blue'),
                    name='Q-Q Plot'
                ))
                
                # Add reference line
                min_val = min(np.min(theoretical_quantiles), np.min(actual_quantiles))
                max_val = max(np.max(theoretical_quantiles), np.max(actual_quantiles))
                
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Reference Line'
                ))
                
                fig.update_layout(
                    title=f"Q-Q Plot for {selected_col}",
                    xaxis_title="Theoretical Quantiles",
                    yaxis_title="Sample Quantiles"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    elif method == "Outlier Detection":
        st.subheader("Outlier Visualization")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            # Let user select columns to visualize outliers
            selected_cols = st.multiselect(
                "Select columns to visualize outliers",
                options=numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))]
            )
            
            if selected_cols:
                # Create a box plot for outlier detection
                fig = px.box(
                    df,
                    y=selected_cols,
                    title="Box Plot for Outlier Detection",
                    points="all"  # Show all points
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Let user select a column for detailed outlier analysis
                selected_col = st.selectbox(
                    "Select a column for detailed outlier analysis",
                    options=selected_cols
                )
                
                if selected_col:
                    # Calculate Z-scores
                    data = df[selected_col].dropna()
                    z_scores = np.abs((data - data.mean()) / data.std())
                    
                    # Identify outliers (Z-score > 3)
                    outliers_z = data[z_scores > 3]
                    
                    # Calculate IQR
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # Identify outliers (outside 1.5*IQR)
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers_iqr = data[(data < lower_bound) | (data > upper_bound)]
                    
                    # Create histogram with marked outliers
                    fig = go.Figure()
                    
                    # Add histogram of all data
                    fig.add_trace(go.Histogram(
                        x=data,
                        name=f"{selected_col} Values",
                        marker=dict(color='blue', opacity=0.7),
                        nbinsx=30
                    ))
                    
                    # Add scatter points for Z-score outliers
                    if not outliers_z.empty:
                        fig.add_trace(go.Scatter(
                            x=outliers_z,
                            y=[0] * len(outliers_z),  # Place at the bottom
                            mode='markers',
                            marker=dict(color='red', size=10, symbol='x'),
                            name='Z-score Outliers'
                        ))
                    
                    # Add scatter points for IQR outliers
                    if not outliers_iqr.empty:
                        fig.add_trace(go.Scatter(
                            x=outliers_iqr,
                            y=[0] * len(outliers_iqr),  # Place at the bottom
                            mode='markers',
                            marker=dict(color='green', size=10, symbol='circle'),
                            name='IQR Outliers'
                        ))
                    
                    # Add vertical lines for IQR bounds
                    fig.add_vline(x=lower_bound, line_dash="dash", line_color="green", name="Lower Bound")
                    fig.add_vline(x=upper_bound, line_dash="dash", line_color="green", name="Upper Bound")
                    
                    fig.update_layout(
                        title=f"Distribution of {selected_col} with Outliers",
                        xaxis_title=selected_col,
                        yaxis_title="Frequency",
                        bargap=0.01
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display summary of outliers
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Z-score Outliers",
                            f"{len(outliers_z)} ({len(outliers_z)/len(data)*100:.2f}%)"
                        )
                    
                    with col2:
                        st.metric(
                            "IQR Outliers",
                            f"{len(outliers_iqr)} ({len(outliers_iqr)/len(data)*100:.2f}%)"
                        )

def visualize_correlation_results(df: pd.DataFrame, 
                                 analysis_results: Dict[str, Any], 
                                 features: List[str]):
    """Visualize results from correlation analysis"""
    method = analysis_results.get('method', '')
    
    if method in ["Pearson Correlation", "Spearman Correlation"]:
        st.subheader(f"{method} Visualization")
        
        # Check if correlation matrix is available
        if 'correlation_matrix' in analysis_results:
            corr_matrix = pd.DataFrame(analysis_results['correlation_matrix'])
            
            # Create a heatmap with plotly
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale='RdBu_r',
                title=f"{method} Heatmap",
                zmin=-1, zmax=1  # Fixed range for correlation
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                st.subheader("Pair-wise Correlation Visualization")
                
                # Let user select columns
                selected_cols = st.multiselect(
                    "Select columns for pair-wise visualization",
                    options=numeric_cols,
                    default=numeric_cols[:min(3, len(numeric_cols))]
                )
                
                if len(selected_cols) >= 2:
                    # Create scatter plot matrix
                    fig = px.scatter_matrix(
                        df,
                        dimensions=selected_cols,
                        title="Scatter Plot Matrix",
                        color_discrete_sequence=['blue']
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Interactive correlation explorer
            st.subheader("Interactive Correlation Explorer")
            
            # Get the upper triangle of the correlation matrix to find strongest correlations
            corr_matrix_np = np.triu(corr_matrix.values, k=1)
            
            # Only keep the top 10 correlations (by absolute value)
            sorted_indices = np.argsort(np.abs(corr_matrix_np), axis=None)[::-1][:10]
            row_indices, col_indices = np.unravel_index(sorted_indices, corr_matrix.shape)
            
            # Create a list of correlation pairs
            corr_pairs = []
            for i, (row, col) in enumerate(zip(row_indices, col_indices)):
                if row < corr_matrix.shape[0] and col < corr_matrix.shape[1]:
                    var1 = corr_matrix.index[row]
                    var2 = corr_matrix.columns[col]
                    corr_value = corr_matrix.iloc[row, col]
                    
                    # Only include actual pairs (not self-correlations)
                    if var1 != var2 and not np.isnan(corr_value):
                        pair_name = f"{var1} — {var2} ({corr_value:.3f})"
                        corr_pairs.append((pair_name, var1, var2, corr_value))
            
            if corr_pairs:
                # Display the correlation pairs as options
                pair_options = [pair[0] for pair in corr_pairs]
                selected_pair = st.selectbox(
                    "Select a correlation pair to visualize",
                    options=pair_options
                )
                
                # Find the selected pair
                selected_idx = pair_options.index(selected_pair)
                var1, var2, corr_value = corr_pairs[selected_idx][1:]
                
                # Create scatter plot with trendline
                fig = px.scatter(
                    df,
                    x=var1,
                    y=var2,
                    trendline="ols",
                    title=f"Correlation between {var1} and {var2} ({method}: {corr_value:.3f})"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display the correlation value and interpretation
                st.subheader("Correlation Interpretation")
                
                # Determine correlation strength
                abs_corr = abs(corr_value)
                if abs_corr < 0.3:
                    strength = "weak"
                elif abs_corr < 0.7:
                    strength = "moderate"
                else:
                    strength = "strong"
                
                # Determine direction
                direction = "positive" if corr_value > 0 else "negative"
                
                st.info(f"There is a {strength} {direction} correlation between {var1} and {var2}.")
                
                # Additional interpretation
                if corr_value > 0:
                    st.write(f"As {var1} increases, {var2} tends to increase as well.")
                else:
                    st.write(f"As {var1} increases, {var2} tends to decrease.")
    
    elif method == "Feature Importance":
        st.subheader("Feature Importance Visualization")
        
        # Check if feature importance data is available
        if 'feature_importance' in analysis_results:
            # Convert to DataFrame
            importance_data = pd.DataFrame(analysis_results['feature_importance'])
            
            if 'Feature' in importance_data and 'Importance' in importance_data:
                # Sort by importance
                importance_data = importance_data.sort_values('Importance', ascending=False)
                
                # Create horizontal bar chart
                fig = px.bar(
                    importance_data,
                    y='Feature',
                    x='Importance',
                    orientation='h',
                    title="Feature Importance",
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                
                st.plotly_chart(fig, use_container_width=True)
                
                # If coefficients are available, show them
                if 'Coefficient' in importance_data:
                    fig = px.bar(
                        importance_data,
                        y='Feature',
                        x='Coefficient',
                        orientation='h',
                        title="Feature Coefficients (with direction)",
                        color='Coefficient',
                        color_continuous_scale='RdBu_r'
                    )
                    
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show target variable
                if 'target_variable' in analysis_results:
                    target = analysis_results['target_variable']
                    
                    # Get the top 2 features
                    top_features = importance_data['Feature'].head(2).tolist()
                    
                    if len(top_features) >= 1:
                        st.subheader(f"Top Feature vs. Target Variable")
                        
                        fig = px.scatter(
                            df,
                            x=top_features[0],
                            y=target,
                            trendline="ols",
                            title=f"Relationship between {top_features[0]} and {target}"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    if len(top_features) >= 2:
                        st.subheader(f"3D Visualization of Top Features")
                        
                        fig = px.scatter_3d(
                            df,
                            x=top_features[0],
                            y=top_features[1],
                            z=target,
                            title=f"3D Relationship between Top Features and {target}",
                            opacity=0.7
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)

def visualize_regression_results(df: pd.DataFrame, 
                                analysis_results: Dict[str, Any], 
                                target: str, 
                                features: List[str]):
    """Visualize results from regression analysis"""
    method = analysis_results.get('method', '')
    
    if method in ["Linear Regression", "Multiple Regression"]:
        st.subheader(f"{method} Visualization")
        
        # Check if model coefficients are available
        if 'model_coefficients' in analysis_results:
            # Convert to DataFrame
            coef_data = pd.DataFrame(analysis_results['model_coefficients'])
            
            if 'Feature' in coef_data and 'Coefficient' in coef_data:
                # Sort by absolute impact
                if 'Absolute Impact' in coef_data:
                    coef_data = coef_data.sort_values('Absolute Impact', ascending=False)
                else:
                    coef_data = coef_data.sort_values('Coefficient', key=abs, ascending=False)
                
                # Create coefficient visualization
                fig = px.bar(
                    coef_data,
                    y='Feature',
                    x='Coefficient',
                    orientation='h',
                    title="Regression Coefficients",
                    color='Coefficient',
                    color_continuous_scale='RdBu_r'
                )
                
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                
                st.plotly_chart(fig, use_container_width=True)
                
                # If standardized coefficients are available, show them
                if 'Standardized Coefficient' in coef_data:
                    fig = px.bar(
                        coef_data,
                        y='Feature',
                        x='Standardized Coefficient',
                        orientation='h',
                        title="Standardized Coefficients (Feature Importance)",
                        color='Standardized Coefficient',
                        color_continuous_scale='RdBu_r'
                    )
                    
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Metrics visualization
        if 'metrics' in analysis_results:
            metrics = analysis_results['metrics']
            
            # Create a gauge chart for R²
            if 'test_r2' in metrics:
                r2 = metrics['test_r2']
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=r2,
                    title={'text': "R² (Test Set)"},
                    gauge={
                        'axis': {'range': [0, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.3], 'color': "red"},
                            {'range': [0.3, 0.7], 'color': "orange"},
                            {'range': [0.7, 1], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': r2
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Create a comparison chart for train vs test metrics
            train_r2 = metrics.get('train_r2', 0)
            test_r2 = metrics.get('test_r2', 0)
            train_rmse = metrics.get('train_rmse', 0)
            test_rmse = metrics.get('test_rmse', 0)
            
            comparison_data = pd.DataFrame({
                'Metric': ['R²', 'RMSE'],
                'Training': [train_r2, train_rmse],
                'Testing': [test_r2, test_rmse]
            })
            
            fig = px.bar(
                comparison_data,
                x='Metric',
                y=['Training', 'Testing'],
                barmode='group',
                title="Model Performance Comparison",
                color_discrete_sequence=['blue', 'red']
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation of results
            st.subheader("Model Performance Interpretation")
            
            if test_r2 > 0.7:
                st.success(f"The model explains {test_r2*100:.1f}% of the variance in the target variable.")
            elif test_r2 > 0.3:
                st.info(f"The model explains {test_r2*100:.1f}% of the variance in the target variable.")
            else:
                st.warning(f"The model only explains {test_r2*100:.1f}% of the variance in the target variable.")
            
            # Check for overfitting
            if train_r2 - test_r2 > 0.1:
                st.warning(f"The model may be overfitting (training R² is {(train_r2-test_r2)*100:.1f}% higher than testing R²).")
            else:
                st.success("The model does not show signs of significant overfitting.")
        
        # Feature exploration
        if target and features:
            st.subheader("Feature Exploration")
            
            # Let user select a feature to explore its relationship with target
            feature_options = [f for f in features if f != target]
            
            if feature_options:
                selected_feature = st.selectbox(
                    "Select a feature to explore its relationship with the target",
                    options=feature_options
                )
                
                if selected_feature:
                    # Create scatter plot with trendline
                    fig = px.scatter(
                        df,
                        x=selected_feature,
                        y=target,
                        trendline="ols",
                        title=f"Relationship between {selected_feature} and {target}"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate correlation
                    corr = df[[selected_feature, target]].corr().iloc[0, 1]
                    
                    st.metric("Correlation", f"{corr:.4f}")
                    
                    # Create residual plot for this feature
                    # Simple linear regression for this specific feature
                    from sklearn.linear_model import LinearRegression
                    
                    X = df[[selected_feature]].dropna()
                    y = df[target].loc[X.index]
                    
                    if not X.empty and not y.empty:
                        model = LinearRegression()
                        model.fit(X, y)
                        
                        y_pred = model.predict(X)
                        residuals = y - y_pred
                        
                        # Create residual plot
                        residual_df = pd.DataFrame({
                            'Predicted': y_pred,
                            'Residuals': residuals
                        })
                        
                        fig = px.scatter(
                            residual_df,
                            x='Predicted',
                            y='Residuals',
                            title=f"Residual Plot for {selected_feature}",
                        )
                        
                        fig.add_shape(
                            type="line",
                            x0=min(y_pred),
                            y0=0,
                            x1=max(y_pred),
                            y1=0,
                            line=dict(color="red", dash="dash")
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)

def visualize_classification_results(df: pd.DataFrame, 
                                    analysis_results: Dict[str, Any], 
                                    target: str, 
                                    features: List[str]):
    """Visualize results from classification analysis"""
    method = analysis_results.get('method', '')
    
    if method == "Logistic Regression":
        st.subheader("Classification Results Visualization")
        
        # Visualize class distribution
        if target:
            # Get target variable distribution
            target_counts = df[target].value_counts().reset_index()
            target_counts.columns = [target, 'Count']
            
            # Create pie chart of class distribution
            fig = px.pie(
                target_counts,
                names=target,
                values='Count',
                title=f"Class Distribution for {target}"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrix visualization
        if 'confusion_matrix' in analysis_results and 'classes' in analysis_results:
            cm = analysis_results['confusion_matrix']
            classes = analysis_results['classes']
            
            # Create heatmap
            fig = px.imshow(
                cm,
                x=classes,
                y=classes,
                text_auto=True,
                color_continuous_scale='Blues',
                title="Confusion Matrix"
            )
            
            fig.update_layout(
                xaxis_title="Predicted",
                yaxis_title="Actual"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate derived metrics
            if len(cm) == 2 and len(cm[0]) == 2:  # Binary classification
                tn, fp = cm[0][0], cm[0][1]
                fn, tp = cm[1][0], cm[1][1]
                
                accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # Display metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Accuracy", f"{accuracy:.4f}")
                    st.metric("Precision", f"{precision:.4f}")
                
                with col2:
                    st.metric("Recall", f"{recall:.4f}")
                    st.metric("F1 Score", f"{f1:.4f}")
                
                # ROC curve
                if 'roc_auc' in analysis_results:
                    auc = analysis_results['roc_auc']
                    
                    # Create AUC gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=auc,
                        title={'text': "AUC-ROC"},
                        gauge={
                            'axis': {'range': [0, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 0.7], 'color': "red"},
                                {'range': [0.7, 0.9], 'color': "orange"},
                                {'range': [0.9, 1], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': auc
                            }
                        }
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance visualization
        if 'model_coefficients' in analysis_results:
            # Convert to DataFrame
            coef_data = pd.DataFrame(analysis_results['model_coefficients'])
            
            if 'Feature' in coef_data and 'Coefficient' in coef_data:
                # Sort by absolute impact
                if 'Absolute Impact' in coef_data:
                    coef_data = coef_data.sort_values('Absolute Impact', ascending=False)
                else:
                    coef_data = coef_data.sort_values('Coefficient', key=abs, ascending=False)
                
                # Create coefficient visualization
                fig = px.bar(
                    coef_data,
                    y='Feature',
                    x='Coefficient',
                    orientation='h',
                    title="Feature Importance (Logistic Regression Coefficients)",
                    color='Coefficient',
                    color_continuous_scale='RdBu_r'
                )
                
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Feature exploration
        if target and features:
            st.subheader("Feature Exploration")
            
            # Let user select a feature to explore its relationship with target
            feature_options = [f for f in features if f != target]
            
            if feature_options:
                selected_feature = st.selectbox(
                    "Select a feature to explore its relationship with the target",
                    options=feature_options
                )
                
                if selected_feature:
                    # Create box plot
                    fig = px.box(
                        df,
                        x=target,
                        y=selected_feature,
                        title=f"Distribution of {selected_feature} by {target} Class",
                        color=target
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create violin plot
                    fig = px.violin(
                        df,
                        x=target,
                        y=selected_feature,
                        title=f"Distribution of {selected_feature} by {target} Class",
                        color=target,
                        box=True,
                        points="all"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

def visualize_clustering_results(df: pd.DataFrame, 
                               analysis_results: Dict[str, Any], 
                               features: List[str]):
    """Visualize results from clustering analysis"""
    method = analysis_results.get('method', '')
    
    if method == "K-Means Clustering":
        st.subheader("Clustering Results Visualization")
        
        # Display number of clusters
        if 'selected_k' in analysis_results:
            k = analysis_results['selected_k']
            st.subheader(f"K-Means Clustering (k={k})")
        
        # Cluster distribution
        if 'cluster_counts' in analysis_results:
            cluster_counts = analysis_results['cluster_counts']
            
            # Convert to DataFrame for visualization
            cluster_df = pd.DataFrame({
                'Cluster': list(cluster_counts.keys()),
                'Count': list(cluster_counts.values())
            })
            
            # Create bar chart
            fig = px.bar(
                cluster_df,
                x='Cluster',
                y='Count',
                title="Cluster Size Distribution",
                color='Count',
                color_continuous_scale='Viridis'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create pie chart
            fig = px.pie(
                cluster_df,
                names='Cluster',
                values='Count',
                title="Cluster Proportion"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Cluster profiles visualization
        if 'cluster_profiles' in analysis_results:
            cluster_profiles = pd.DataFrame(analysis_results['cluster_profiles'])
            
            # Create heatmap
            fig = px.imshow(
                cluster_profiles,
                color_continuous_scale='RdBu_r',
                title="Feature Values by Cluster",
                labels={'x': 'Cluster', 'y': 'Feature', 'color': 'Value'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create radar chart for cluster profiles
            if not cluster_profiles.empty:
                # Transpose if necessary (features should be columns)
                if 'Cluster' in cluster_profiles.columns:
                    radar_df = cluster_profiles
                else:
                    radar_df = cluster_profiles.T.reset_index()
                    radar_df.columns = ['Feature'] + [f'Cluster {i}' for i in range(len(radar_df.columns)-1)]
                
                # Normalize the values for radar chart
                feature_cols = [col for col in radar_df.columns if col != 'Feature']
                
                for col in feature_cols:
                    min_val = radar_df[col].min()
                    max_val = radar_df[col].max()
                    if max_val > min_val:
                        radar_df[col] = (radar_df[col] - min_val) / (max_val - min_val)
                
                # Create radar chart
                fig = go.Figure()
                
                for i, cluster in enumerate(feature_cols):
                    fig.add_trace(go.Scatterpolar(
                        r=radar_df[cluster],
                        theta=radar_df['Feature'],
                        fill='toself',
                        name=cluster
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    title="Cluster Profiles (Normalized)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Silhouette score visualization
        if 'silhouette_score' in analysis_results:
            silhouette = analysis_results['silhouette_score']
            
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=silhouette,
                title={'text': "Silhouette Score"},
                gauge={
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-1, 0], 'color': "red"},
                        {'range': [0, 0.5], 'color': "orange"},
                        {'range': [0.5, 1], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': silhouette
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation of silhouette score
            if silhouette > 0.7:
                st.success("The clusters are well separated and dense.")
            elif silhouette > 0.5:
                st.info("The clusters have reasonable separation and density.")
            elif silhouette > 0:
                st.warning("The clusters have some overlap or are not very dense.")
            else:
                st.error("The clustering may not be appropriate for this data.")
        
        # Elbow method visualization
        if 'inertia' in analysis_results:
            inertia = analysis_results['inertia']
            
            # Convert to DataFrame for visualization
            elbow_df = pd.DataFrame({
                'k': list(inertia.keys()),
                'Inertia': list(inertia.values())
            })
            
            # Create line chart
            fig = px.line(
                elbow_df,
                x='k',
                y='Inertia',
                markers=True,
                title="Elbow Method for Optimal k"
            )
            
            selected_k = analysis_results.get('selected_k')
            if selected_k:
                # Add vertical line for selected k
                fig.add_vline(
                    x=selected_k,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Selected k={selected_k}",
                    annotation_position="top right"
                )
            
            st.plotly_chart(fig, use_container_width=True)

def visualize_time_series_results(df: pd.DataFrame, 
                                 analysis_results: Dict[str, Any], 
                                 features: List[str]):
    """Visualize results from time series analysis"""
    method = analysis_results.get('method', '')
    
    if method == "Trend Analysis":
        st.subheader("Time Series Trend Visualization")
        
        # Display time series
        if len(features) >= 2:
            time_col = features[0]
            value_col = features[1]
            
            # Sort by time column
            ts_df = df[[time_col, value_col]].sort_values(time_col)
            
            # Create time series plot
            fig = px.line(
                ts_df,
                x=time_col,
                y=value_col,
                title=f"Time Series: {value_col} over Time"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add trend statistics
            if 'trend_statistics' in analysis_results:
                stats = analysis_results['trend_statistics']
                
                # Create table of statistics
                stats_df = pd.DataFrame({
                    'Metric': ['Slope', 'Intercept', 'R²', 'P-value'],
                    'Value': [
                        f"{stats['slope']:.6f}",
                        f"{stats['intercept']:.4f}",
                        f"{stats['r_squared']:.4f}",
                        f"{stats['p_value']:.6f}"
                    ]
                })
                
                st.table(stats_df)
                
                # Interpretation
                slope = stats['slope']
                p_value = stats['p_value']
                r_squared = stats['r_squared']
                
                if p_value < 0.05:
                    if slope > 0:
                        st.success(f"Significant upward trend detected (p={p_value:.6f})")
                    else:
                        st.warning(f"Significant downward trend detected (p={p_value:.6f})")
                        
                    if r_squared > 0.7:
                        st.info(f"The trend is very consistent (R²={r_squared:.4f})")
                    elif r_squared > 0.3:
                        st.info(f"The trend shows moderate consistency (R²={r_squared:.4f})")
                    else:
                        st.info(f"The trend shows weak consistency (R²={r_squared:.4f})")
                else:
                    st.error(f"No significant trend detected (p={p_value:.6f})")
            
            # Rolling average visualization
            if 'rolling_statistics' in analysis_results:
                rolling_stats = analysis_results['rolling_statistics']
                window_size = rolling_stats.get('window_size', 7)
                
                # Calculate rolling average
                ts_df['rolling_avg'] = ts_df[value_col].rolling(window=window_size).mean()
                
                # Create plot with rolling average
                fig = px.line(
                    ts_df,
                    x=time_col,
                    y=[value_col, 'rolling_avg'],
                    title=f"Time Series with {window_size}-point Rolling Average"
                )
                
                fig.update_layout(legend=dict(
                    title="",
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ))
                
                st.plotly_chart(fig, use_container_width=True)
    
    elif method == "Seasonality Analysis":
        st.subheader("Time Series Seasonality Visualization")
        
        # Display time series
        if len(features) >= 2:
            time_col = features[0]
            value_col = features[1]
            
            # Sort by time column
            ts_df = df[[time_col, value_col]].sort_values(time_col)
            
            # Create time series plot
            fig = px.line(
                ts_df,
                x=time_col,
                y=value_col,
                title=f"Time Series: {value_col} over Time"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Seasonal pattern visualization
            if 'seasonality' in analysis_results:
                seasonality = analysis_results['seasonality']
                period = seasonality.get('period')
                
                if period:
                    st.subheader(f"Seasonal Pattern (Period: {period})")
                    
                    if 'seasonal_pattern' in seasonality:
                        # Convert to DataFrame
                        pattern = pd.DataFrame({
                            'Position': list(seasonality['seasonal_pattern'].keys()),
                            'Value': list(seasonality['seasonal_pattern'].values())
                        })
                        
                        # Convert position to numeric if needed
                        pattern['Position'] = pd.to_numeric(pattern['Position'], errors='coerce')
                        
                        # Sort by position
                        pattern = pattern.sort_values('Position')
                        
                        # Create line chart
                        fig = px.line(
                            pattern,
                            x='Position',
                            y='Value',
                            title=f"Seasonal Pattern (Period={period})",
                            markers=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Seasonal strength
                    if 'seasonal_strength' in seasonality:
                        strength = seasonality['seasonal_strength']
                        
                        if strength is not None:
                            # Create gauge chart
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=strength,
                                title={'text': "Seasonal Strength"},
                                gauge={
                                    'axis': {'range': [0, 1]},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [0, 0.3], 'color': "red"},
                                        {'range': [0.3, 0.6], 'color': "orange"},
                                        {'range': [0.6, 1], 'color': "green"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "black", 'width': 4},
                                        'thickness': 0.75,
                                        'value': strength
                                    }
                                }
                            ))
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Interpretation
                            if strength > 0.6:
                                st.success(f"Strong seasonality detected (strength={strength:.4f})")
                            elif strength > 0.3:
                                st.info(f"Moderate seasonality detected (strength={strength:.4f})")
                            else:
                                st.warning(f"Weak seasonality detected (strength={strength:.4f})")

def create_custom_visualizations(df: pd.DataFrame, features: List[str], target: str = None) -> Dict[str, Any]:
    """Create custom visualizations based on user selections"""
    custom_viz_results = {}
    
    # Get column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = []
    
    # Try to identify datetime columns
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]':
            datetime_cols.append(col)
        elif df[col].dtype == 'object':
            # Try to convert to datetime
            try:
                # Specify format to avoid warnings if possible
                # Try common formats first
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']:
                    try:
                        pd.to_datetime(df[col], format=fmt, errors='raise')
                        datetime_cols.append(col)
                        break
                    except:
                        continue
                
                # If no format matched, try without specifying format but suppress warnings
                if col not in datetime_cols:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pd.to_datetime(df[col], errors='raise')
                    datetime_cols.append(col)
            except:
                pass
    
    # Chart type selection
    chart_type = st.selectbox(
        "Select Chart Type",
        options=["Scatter Plot", "Bar Chart", "Line Chart", "Histogram", 
                "Box Plot", "Violin Plot", "Pie Chart", "Heatmap",
                "3D Scatter Plot", "Bubble Chart"],
        key="custom_viz_chart_type"
    )
    
    # Store the chart type
    custom_viz_results['chart_type'] = chart_type
    
    # Different options based on chart type
    if chart_type == "Scatter Plot":
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox("X-axis", options=numeric_cols, index=min(0, len(numeric_cols)-1), key="x_axis_select")
        
        with col2:
            y_col = st.selectbox("Y-axis", options=numeric_cols, index=min(1, len(numeric_cols)-1), key="y_axis_select")
        
        color_col = st.selectbox("Color by (optional)", options=["None"] + df.columns.tolist())
        color_col = None if color_col == "None" else color_col
        
        # Create the plot
        if x_col and y_col:
            title = f"Scatter Plot: {y_col} vs {x_col}"
            if color_col:
                title += f", colored by {color_col}"
                
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_col,
                title=title,
                opacity=0.7
            )
            
            # Add trendline if both axes are numeric
            if x_col in numeric_cols and y_col in numeric_cols and color_col is None:
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    trendline="ols",
                    title=title,
                    opacity=0.7
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Store parameters
            custom_viz_results['params'] = {
                'x_col': x_col,
                'y_col': y_col,
                'color_col': color_col
            }
    
    elif chart_type == "Bar Chart":
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox("X-axis (Categories)", options=df.columns.tolist())
        
        with col2:
            y_col = st.selectbox("Y-axis (Values)", options=["Count"] + numeric_cols)
        
        color_col = st.selectbox("Color by (optional)", options=["None"] + df.columns.tolist())
        color_col = None if color_col == "None" else color_col
        
        # Get aggregation method if not using count
        if y_col != "Count":
            agg_method = st.selectbox(
                "Aggregation Method",
                options=["sum", "mean", "median", "min", "max"]
            )
        else:
            agg_method = "count"
        
        # Create the plot
        if x_col:
            if y_col == "Count":
                # Count the categories
                count_df = df[x_col].value_counts().reset_index()
                count_df.columns = [x_col, 'Count']
                
                # Limit to top 20 categories if there are too many
                if len(count_df) > 20:
                    count_df = count_df.head(20)
                    title = f"Bar Chart: Top 20 {x_col} by Count"
                else:
                    title = f"Bar Chart: {x_col} by Count"
                
                fig = px.bar(
                    count_df,
                    x=x_col,
                    y='Count',
                    color=color_col,
                    title=title
                )
            else:
                # Group by x_col and calculate aggregation
                if agg_method == "sum":
                    agg_df = df.groupby(x_col)[y_col].sum().reset_index()
                elif agg_method == "mean":
                    agg_df = df.groupby(x_col)[y_col].mean().reset_index()
                elif agg_method == "median":
                    agg_df = df.groupby(x_col)[y_col].median().reset_index()
                elif agg_method == "min":
                    agg_df = df.groupby(x_col)[y_col].min().reset_index()
                elif agg_method == "max":
                    agg_df = df.groupby(x_col)[y_col].max().reset_index()
                
                # Limit to top 20 categories if there are too many
                if len(agg_df) > 20:
                    agg_df = agg_df.sort_values(y_col, ascending=False).head(20)
                    title = f"Bar Chart: Top 20 {x_col} by {y_col} ({agg_method})"
                else:
                    title = f"Bar Chart: {x_col} by {y_col} ({agg_method})"
                
                fig = px.bar(
                    agg_df,
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    title=title
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Store parameters
            custom_viz_results['params'] = {
                'x_col': x_col,
                'y_col': y_col,
                'color_col': color_col,
                'agg_method': agg_method
            }
    
    elif chart_type == "Line Chart":
        col1, col2 = st.columns(2)
        
        # For line charts, we typically want a time/sequence on the x-axis
        with col1:
            x_options = datetime_cols + numeric_cols if datetime_cols else numeric_cols
            x_col = st.selectbox("X-axis", options=x_options, index=0)
        
        with col2:
            y_col = st.selectbox("Y-axis", options=numeric_cols, index=min(0, len(numeric_cols)-1))
        
        color_col = st.selectbox("Color by (optional)", options=["None"] + categorical_cols)
        color_col = None if color_col == "None" else color_col
        
        # Create the plot
        if x_col and y_col:
            # Sort by x_col for line chart
            plot_df = df.sort_values(x_col)
            
            title = f"Line Chart: {y_col} vs {x_col}"
            if color_col:
                title += f", grouped by {color_col}"
            
            fig = px.line(
                plot_df,
                x=x_col,
                y=y_col,
                color=color_col,
                title=title,
                markers=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Store parameters
            custom_viz_results['params'] = {
                'x_col': x_col,
                'y_col': y_col,
                'color_col': color_col
            }
    
    elif chart_type == "Histogram":
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox("Column", options=numeric_cols, index=0)
        
        with col2:
            n_bins = st.slider("Number of bins", min_value=5, max_value=100, value=30)
        
        color_col = st.selectbox("Color by (optional)", options=["None"] + categorical_cols)
        color_col = None if color_col == "None" else color_col
        
        # Create the plot
        if x_col:
            title = f"Histogram of {x_col}"
            if color_col:
                title += f", colored by {color_col}"
            
            fig = px.histogram(
                df,
                x=x_col,
                color=color_col,
                nbins=n_bins,
                title=title,
                opacity=0.7,
                marginal="box" if color_col is None else None
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Store parameters
            custom_viz_results['params'] = {
                'x_col': x_col,
                'n_bins': n_bins,
                'color_col': color_col
            }
    
    elif chart_type == "Box Plot":
        col1, col2 = st.columns(2)
        
        with col1:
            y_col = st.selectbox("Values", options=numeric_cols, index=0)
        
        with col2:
            x_col = st.selectbox("Categories (optional)", options=["None"] + categorical_cols)
            x_col = None if x_col == "None" else x_col
        
        color_col = st.selectbox("Color by (optional)", options=["None"] + categorical_cols)
        color_col = None if color_col == "None" else color_col
        
        # Create the plot
        if y_col:
            title = f"Box Plot of {y_col}"
            if x_col:
                title += f" by {x_col}"
            if color_col and color_col != x_col:
                title += f", colored by {color_col}"
            
            fig = px.box(
                df,
                y=y_col,
                x=x_col,
                color=color_col,
                title=title,
                points="all"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Store parameters
            custom_viz_results['params'] = {
                'y_col': y_col,
                'x_col': x_col,
                'color_col': color_col
            }
    
    elif chart_type == "Violin Plot":
        col1, col2 = st.columns(2)
        
        with col1:
            y_col = st.selectbox("Values", options=numeric_cols, index=0)
        
        with col2:
            x_col = st.selectbox("Categories (optional)", options=["None"] + categorical_cols)
            x_col = None if x_col == "None" else x_col
        
        color_col = st.selectbox("Color by (optional)", options=["None"] + categorical_cols)
        color_col = None if color_col == "None" else color_col
        
        # Create the plot
        if y_col:
            title = f"Violin Plot of {y_col}"
            if x_col:
                title += f" by {x_col}"
            if color_col and color_col != x_col:
                title += f", colored by {color_col}"
            
            fig = px.violin(
                df,
                y=y_col,
                x=x_col,
                color=color_col,
                title=title,
                box=True,
                points="all"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Store parameters
            custom_viz_results['params'] = {
                'y_col': y_col,
                'x_col': x_col,
                'color_col': color_col
            }
    
    elif chart_type == "Pie Chart":
        col1, col2 = st.columns(2)
        
        with col1:
            names_col = st.selectbox("Categories", options=categorical_cols + numeric_cols, index=0)
        
        with col2:
            values_col = st.selectbox("Values", options=["Count"] + numeric_cols)
        
        # Create the plot
        if names_col:
            if values_col == "Count":
                # Count the categories
                count_df = df[names_col].value_counts().reset_index()
                count_df.columns = [names_col, 'Count']
                
                # Limit to top 10 categories if there are too many
                if len(count_df) > 10:
                    count_df = count_df.head(10)
                    title = f"Pie Chart: Top 10 {names_col} by Count"
                else:
                    title = f"Pie Chart: {names_col} by Count"
                
                fig = px.pie(
                    count_df,
                    names=names_col,
                    values='Count',
                    title=title
                )
            else:
                # Group by names_col and sum values_col
                agg_df = df.groupby(names_col)[values_col].sum().reset_index()
                
                # Limit to top 10 categories if there are too many
                if len(agg_df) > 10:
                    agg_df = agg_df.sort_values(values_col, ascending=False).head(10)
                    title = f"Pie Chart: Top 10 {names_col} by {values_col}"
                else:
                    title = f"Pie Chart: {names_col} by {values_col}"
                
                fig = px.pie(
                    agg_df,
                    names=names_col,
                    values=values_col,
                    title=title
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Store parameters
            custom_viz_results['params'] = {
                'names_col': names_col,
                'values_col': values_col
            }
    
    elif chart_type == "Heatmap":
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox("X-axis", options=categorical_cols + numeric_cols, index=0)
        
        with col2:
            y_col = st.selectbox("Y-axis", options=categorical_cols + numeric_cols, index=min(1, len(categorical_cols + numeric_cols)-1))
        
        # Value column and aggregation method
        value_col = st.selectbox("Values", options=["Count"] + numeric_cols)
        
        if value_col != "Count":
            agg_method = st.selectbox(
                "Aggregation Method",
                options=["sum", "mean", "median", "min", "max"]
            )
        else:
            agg_method = "count"
        
        # Create the plot
        if x_col and y_col:
            # Create pivot table
            if value_col == "Count":
                pivot_df = pd.crosstab(df[y_col], df[x_col])
                title = f"Heatmap: Count of {y_col} vs {x_col}"
            else:
                if agg_method == "sum":
                    pivot_df = df.pivot_table(values=value_col, index=y_col, columns=x_col, aggfunc="sum")
                elif agg_method == "mean":
                    pivot_df = df.pivot_table(values=value_col, index=y_col, columns=x_col, aggfunc="mean")
                elif agg_method == "median":
                    pivot_df = df.pivot_table(values=value_col, index=y_col, columns=x_col, aggfunc="median")
                elif agg_method == "min":
                    pivot_df = df.pivot_table(values=value_col, index=y_col, columns=x_col, aggfunc="min")
                elif agg_method == "max":
                    pivot_df = df.pivot_table(values=value_col, index=y_col, columns=x_col, aggfunc="max")
                
                title = f"Heatmap: {value_col} ({agg_method}) of {y_col} vs {x_col}"
            
            # Limit size if there are too many categories
            if pivot_df.shape[0] > 20 or pivot_df.shape[1] > 20:
                if pivot_df.shape[0] > 20:
                    top_rows = pivot_df.sum(axis=1).sort_values(ascending=False).head(20).index
                    pivot_df = pivot_df.loc[top_rows]
                
                if pivot_df.shape[1] > 20:
                    top_cols = pivot_df.sum(axis=0).sort_values(ascending=False).head(20).index
                    pivot_df = pivot_df[top_cols]
                
                title += " (limited to top 20)"
            
            fig = px.imshow(
                pivot_df,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Viridis",
                title=title
            )
            
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title=y_col
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Store parameters
            custom_viz_results['params'] = {
                'x_col': x_col,
                'y_col': y_col,
                'value_col': value_col,
                'agg_method': agg_method
            }
    
    elif chart_type == "3D Scatter Plot":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_col = st.selectbox("X-axis", options=numeric_cols, index=min(0, len(numeric_cols)-1))
        
        with col2:
            y_col = st.selectbox("Y-axis", options=numeric_cols, index=min(1, len(numeric_cols)-1))
        
        with col3:
            z_col = st.selectbox("Z-axis", options=numeric_cols, index=min(2, len(numeric_cols)-1))
        
        color_col = st.selectbox("Color by (optional)", options=["None"] + df.columns.tolist())
        color_col = None if color_col == "None" else color_col
        
        # Create the plot
        if x_col and y_col and z_col:
            title = f"3D Scatter Plot: {x_col}, {y_col}, {z_col}"
            if color_col:
                title += f", colored by {color_col}"
            
            fig = px.scatter_3d(
                df,
                x=x_col,
                y=y_col,
                z=z_col,
                color=color_col,
                title=title,
                opacity=0.7
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Store parameters
            custom_viz_results['params'] = {
                'x_col': x_col,
                'y_col': y_col,
                'z_col': z_col,
                'color_col': color_col
            }
    
    elif chart_type == "Bubble Chart":
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox("X-axis", options=numeric_cols, index=min(0, len(numeric_cols)-1))
        
        with col2:
            y_col = st.selectbox("Y-axis", options=numeric_cols, index=min(1, len(numeric_cols)-1))
        
        size_col = st.selectbox("Size by", options=numeric_cols, index=min(2, len(numeric_cols)-1))
        
        color_col = st.selectbox("Color by (optional)", options=["None"] + df.columns.tolist())
        color_col = None if color_col == "None" else color_col
        
        # Create the plot
        if x_col and y_col and size_col:
            title = f"Bubble Chart: {y_col} vs {x_col}, size by {size_col}"
            if color_col:
                title += f", colored by {color_col}"
            
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                size=size_col,
                color=color_col,
                title=title,
                opacity=0.7
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Store parameters
            custom_viz_results['params'] = {
                'x_col': x_col,
                'y_col': y_col,
                'size_col': size_col,
                'color_col': color_col
            }
    
    return custom_viz_results

def create_advanced_visualizations(df: pd.DataFrame, problem_def: Dict[str, Any]) -> Dict[str, Any]:
    """Create advanced visualizations based on data and problem definition"""
    advanced_viz_results = {}
    
    # Get column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Advanced visualization types
    advanced_viz_type = st.selectbox(
        "Select Advanced Visualization Type",
        options=[
            "Correlation Network",
            "Parallel Coordinates Plot",
            "Sunburst Chart",
            "Treemap",
            "Density Heatmap",
            "Contour Plot"
        ],
        key="advanced_viz_type"
    )
    
    # Store the visualization type
    advanced_viz_results['advanced_viz_type'] = advanced_viz_type
    
    # Different visualizations based on type
    if advanced_viz_type == "Correlation Network":
        st.subheader("Correlation Network Visualization")
        
        # Let user select features
        selected_features = st.multiselect(
            "Select features for correlation network",
            options=numeric_cols,
            default=numeric_cols[:min(6, len(numeric_cols))]
        )
        
        if len(selected_features) >= 2:
            # Calculate correlation matrix
            corr_matrix = df[selected_features].corr()
            
            # Create network nodes
            nodes = []
            for i, feature in enumerate(selected_features):
                nodes.append({
                    'id': feature,
                    'label': feature,
                    'size': 20
                })
            
            # Create edges between nodes
            edges = []
            for i in range(len(selected_features)):
                for j in range(i+1, len(selected_features)):
                    feature1 = selected_features[i]
                    feature2 = selected_features[j]
                    
                    # Get the correlation value
                    corr_value = corr_matrix.loc[feature1, feature2]
                    
                    # Only include if correlation is significant
                    if abs(corr_value) > 0.3:
                        edges.append({
                            'source': feature1,
                            'target': feature2,
                            'weight': abs(corr_value),
                            'color': 'blue' if corr_value > 0 else 'red',
                            'value': abs(corr_value)
                        })
            
            # Create network visualization using plotly
            edge_x = []
            edge_y = []
            edge_colors = []
            edge_weights = []
            
            # Use a simple circular layout for nodes
            import math
            
            radius = 1
            node_positions = {}
            
            for i, node in enumerate(nodes):
                angle = 2 * math.pi * i / len(nodes)
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                node_positions[node['id']] = (x, y)
            
            for edge in edges:
                source_pos = node_positions[edge['source']]
                target_pos = node_positions[edge['target']]
                
                edge_x.extend([source_pos[0], target_pos[0], None])
                edge_y.extend([source_pos[1], target_pos[1], None])
                edge_colors.extend([edge['color'], edge['color'], None])
                edge_weights.extend([edge['weight'] * 5, edge['weight'] * 5, None])
            
            # Create a trace for the edges
            # Create individual traces for each edge to handle different colors
            edge_traces = []
            for i in range(0, len(edge_x), 3):
                # Each edge consists of 3 points (source, target, None)
                if i+2 < len(edge_x):  # Make sure we have a complete edge
                    # Get color and ensure it's a single string value, not a list
                    edge_color = edge_colors[i]
                    if isinstance(edge_color, list):
                        # If somehow we got a list, take the first non-None value
                        for color in edge_color:
                            if color is not None:
                                edge_color = color
                                break
                        # Fallback if all are None
                        if edge_color is None or isinstance(edge_color, list):
                            edge_color = 'gray'
                    
                    edge_traces.append(
                        go.Scatter(
                            x=edge_x[i:i+3], 
                            y=edge_y[i:i+3],
                            line=dict(width=edge_weights[i], color=edge_color),
                            hoverinfo='none',
                            mode='lines',
                            showlegend=False
                        )
                    )
            
            # Create a trace for the nodes
            node_x = [node_positions[node['id']][0] for node in nodes]
            node_y = [node_positions[node['id']][1] for node in nodes]
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=[node['label'] for node in nodes],
                textposition="top center",
                marker=dict(
                    size=15,
                    color='lightblue',
                    line=dict(width=2, color='black')
                )
            )
            
            # Create figure with all edge traces and the node trace
            fig = go.Figure(data=edge_traces + [node_trace],
                        layout=go.Layout(
                            title="Correlation Network",
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        )
                      )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add legend for edge colors
            st.info("🔵 Blue edges represent positive correlations, 🔴 red edges represent negative correlations. Edge thickness represents correlation strength.")
            
            # Store parameters
            advanced_viz_results['params'] = {
                'selected_features': selected_features,
                'node_count': len(nodes),
                'edge_count': len(edges)
            }
    
    elif advanced_viz_type == "Parallel Coordinates Plot":
        st.subheader("Parallel Coordinates Plot")
        
        # Get features to include
        selected_features = st.multiselect(
            "Select features to include",
            options=numeric_cols,
            default=numeric_cols[:min(5, len(numeric_cols))]
        )
        
        # Get column for coloring
        color_col = st.selectbox(
            "Color by (optional)",
            options=["None"] + df.columns.tolist(),
            key="color_column_select"
        )
        color_col = None if color_col == "None" else color_col
        
        if selected_features:
            # Create parallel coordinates plot
            title = "Parallel Coordinates Plot"
            if color_col:
                title += f" (colored by {color_col})"
            
            fig = px.parallel_coordinates(
                df,
                dimensions=selected_features,
                color=color_col,
                title=title
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Store parameters
            advanced_viz_results['params'] = {
                'selected_features': selected_features,
                'color_col': color_col
            }
    
    elif advanced_viz_type == "Sunburst Chart":
        st.subheader("Sunburst Chart")
        
        # Get hierarchical path columns
        path_cols = st.multiselect(
            "Select columns for hierarchy path",
            options=categorical_cols + numeric_cols,
            default=categorical_cols[:min(2, len(categorical_cols))]
        )
        
        # Get value column
        value_col = st.selectbox(
            "Values",
            options=["Count"] + numeric_cols
        )
        
        if path_cols:
            # Create sunburst chart
            if value_col == "Count":
                # Use count as value
                fig = px.sunburst(
                    df,
                    path=path_cols,
                    title="Sunburst Chart"
                )
            else:
                # Use specified value column
                fig = px.sunburst(
                    df,
                    path=path_cols,
                    values=value_col,
                    title=f"Sunburst Chart (values: {value_col})"
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Store parameters
            advanced_viz_results['params'] = {
                'path_cols': path_cols,
                'value_col': value_col
            }
    
    elif advanced_viz_type == "Treemap":
        st.subheader("Treemap")
        
        # Get hierarchical path columns
        path_cols = st.multiselect(
            "Select columns for hierarchy path",
            options=categorical_cols + numeric_cols,
            default=categorical_cols[:min(2, len(categorical_cols))]
        )
        
        # Get value column
        value_col = st.selectbox(
            "Values",
            options=["Count"] + numeric_cols
        )
        
        # Get column for coloring
        color_col = st.selectbox(
            "Color by (optional)",
            options=["None"] + df.columns.tolist(),
            key="color_column_select"
        )
        color_col = None if color_col == "None" else color_col
        
        if path_cols:
            # Create treemap
            title = "Treemap"
            if value_col != "Count":
                title += f" (values: {value_col})"
            if color_col:
                title += f" (colored by {color_col})"
            
            if value_col == "Count":
                # Use count as value
                fig = px.treemap(
                    df,
                    path=path_cols,
                    color=color_col,
                    title=title
                )
            else:
                # Use specified value column
                fig = px.treemap(
                    df,
                    path=path_cols,
                    values=value_col,
                    color=color_col,
                    title=title
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Store parameters
            advanced_viz_results['params'] = {
                'path_cols': path_cols,
                'value_col': value_col,
                'color_col': color_col
            }
    
    elif advanced_viz_type == "Density Heatmap":
        st.subheader("Density Heatmap")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox("X-axis", options=numeric_cols, index=min(0, len(numeric_cols)-1))
        
        with col2:
            y_col = st.selectbox("Y-axis", options=numeric_cols, index=min(1, len(numeric_cols)-1))
        
        # Get color column and NBins
        color_col = st.selectbox(
            "Color by (optional)",
            options=["None"] + categorical_cols
        )
        color_col = None if color_col == "None" else color_col
        
        nbinsx = st.slider("X-axis bins", min_value=10, max_value=100, value=30)
        nbinsy = st.slider("Y-axis bins", min_value=10, max_value=100, value=30)
        
        if x_col and y_col:
            # Create density heatmap
            title = f"Density Heatmap: {y_col} vs {x_col}"
            
            if color_col:
                # Create separate plots for each category
                categories = df[color_col].unique()
                
                # Limit to top 4 categories if there are too many
                if len(categories) > 4:
                    value_counts = df[color_col].value_counts()
                    categories = value_counts.index[:4]
                    st.info(f"Showing only the top 4 categories for {color_col}")
                
                # Create subplot grid
                n_cols = min(2, len(categories))
                n_rows = (len(categories) + 1) // 2
                
                fig = make_subplots(
                    rows=n_rows,
                    cols=n_cols,
                    subplot_titles=[f"{color_col}: {cat}" for cat in categories]
                )
                
                for i, category in enumerate(categories):
                    row = i // n_cols + 1
                    col = i % n_cols + 1
                    
                    # Filter data for this category
                    cat_df = df[df[color_col] == category]
                    
                    # Create 2D histogram
                    hist = go.Histogram2d(
                        x=cat_df[x_col],
                        y=cat_df[y_col],
                        colorscale='Viridis',
                        nbinsx=nbinsx,
                        nbinsy=nbinsy,
                        hoverinfo='skip'
                    )
                    
                    fig.add_trace(hist, row=row, col=col)
                
                fig.update_layout(
                    title=title,
                    height=300 * n_rows
                )
            else:
                # Create density heatmap for all data
                fig = px.density_heatmap(
                    df,
                    x=x_col,
                    y=y_col,
                    title=title,
                    nbinsx=nbinsx,
                    nbinsy=nbinsy
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Store parameters
            advanced_viz_results['params'] = {
                'x_col': x_col,
                'y_col': y_col,
                'color_col': color_col,
                'nbinsx': nbinsx,
                'nbinsy': nbinsy
            }
    
    elif advanced_viz_type == "Contour Plot":
        st.subheader("Contour Plot")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox("X-axis", options=numeric_cols, index=min(0, len(numeric_cols)-1))
        
        with col2:
            y_col = st.selectbox("Y-axis", options=numeric_cols, index=min(1, len(numeric_cols)-1))
        
        # Get Z value and NBins
        z_col = st.selectbox(
            "Z-axis (contour values)",
            options=["Density"] + numeric_cols
        )
        
        nbinsx = st.slider("X-axis bins", min_value=10, max_value=100, value=30)
        nbinsy = st.slider("Y-axis bins", min_value=10, max_value=100, value=30)
        
        if x_col and y_col:
            # Create contour plot
            if z_col == "Density":
                title = f"Density Contour Plot: {y_col} vs {x_col}"
                
                fig = px.density_contour(
                    df,
                    x=x_col,
                    y=y_col,
                    title=title,
                    nbinsx=nbinsx,
                    nbinsy=nbinsy
                )
            else:
                title = f"Contour Plot: {z_col} values for {y_col} vs {x_col}"
                
                fig = px.density_contour(
                    df,
                    x=x_col,
                    y=y_col,
                    z=z_col,
                    title=title,
                    nbinsx=nbinsx,
                    nbinsy=nbinsy
                )
            
            # Add scatter points
            fig.update_traces(contours_coloring="fill", contours_showlabels=True)
            
            # Add scatter points
            fig.add_trace(
                go.Scatter(
                    x=df[x_col],
                    y=df[y_col],
                    mode='markers',
                    marker=dict(
                        color='white',
                        size=4,
                        opacity=0.3
                    ),
                    showlegend=False
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Store parameters
            advanced_viz_results['params'] = {
                'x_col': x_col,
                'y_col': y_col,
                'z_col': z_col,
                'nbinsx': nbinsx,
                'nbinsy': nbinsy
            }
    
    return advanced_viz_results
