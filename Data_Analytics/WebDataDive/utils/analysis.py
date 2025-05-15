import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import statsmodels.api as sm
from scipy import stats
from typing import Dict, List, Tuple, Any

def render_analysis(df: pd.DataFrame, problem_def: Dict[str, Any]) -> Dict[str, Any]:
    """
    Renders the analysis section and returns the results
    
    Parameters:
    df - The dataset as a pandas DataFrame
    problem_def - The problem definition dictionary
    
    Returns:
    Dictionary containing analysis results
    """
    st.header("4. Analysis", divider=True)
    
    if df.empty:
        st.warning("Please upload and preprocess your dataset first")
        return {}
    
    # Initialize analysis results if not exists
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    
    # Get analysis goal and method from problem definition
    goal = problem_def.get('analysis_goal', 'Exploratory Analysis')
    method = problem_def.get('analysis_method', 'Descriptive Statistics')
    target = problem_def.get('target_variable')
    features = problem_def.get('features', [])
    
    # Display analysis settings
    st.subheader("Analysis Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Analysis Goal", goal)
    with col2:
        st.metric("Analysis Method", method)
    
    if target:
        st.metric("Target Variable", target)
    
    # Check if we have features to analyze
    if not features:
        st.warning("No features selected for analysis. Please select features in the Problem Definition section.")
        return {}
    
    # Run the analysis based on the goal and method
    if goal == "Exploratory Analysis":
        result = run_exploratory_analysis(df, method, features)
    elif goal == "Correlation Analysis":
        result = run_correlation_analysis(df, method, features)
    elif goal == "Regression Analysis":
        result = run_regression_analysis(df, method, target, features)
    elif goal == "Classification Analysis":
        result = run_classification_analysis(df, method, target, features)
    elif goal == "Clustering":
        result = run_clustering_analysis(df, method, features)
    elif goal == "Time Series Analysis":
        result = run_time_series_analysis(df, method, features)
    else:
        st.error(f"Unsupported analysis goal: {goal}")
        return {}
    
    # Store analysis results in session state
    st.session_state.analysis_results = result
    
    return result

def run_exploratory_analysis(df: pd.DataFrame, method: str, features: List[str]) -> Dict[str, Any]:
    """Run exploratory data analysis"""
    result = {"type": "exploratory", "method": method}
    
    # Filter to only include selected features
    df_subset = df[features].copy()
    
    if method == "Descriptive Statistics":
        st.subheader("Descriptive Statistics")
        
        # Get numeric columns
        numeric_cols = df_subset.select_dtypes(include=['number']).columns.tolist()
        
        if numeric_cols:
            try:
                # Calculate descriptive statistics for numeric columns only
                numeric_stats = df_subset[numeric_cols].describe().T
                
                if not numeric_stats.empty:
                    # Add additional statistics for numeric columns only
                    # Handle safely to avoid errors
                    try:
                        numeric_stats['skew'] = df_subset[numeric_cols].skew()
                    except Exception as e:
                        st.warning(f"Could not calculate skew: {str(e)}")
                        numeric_stats['skew'] = float('nan')
                        
                    try:
                        numeric_stats['kurtosis'] = df_subset[numeric_cols].kurtosis()
                    except Exception as e:
                        st.warning(f"Could not calculate kurtosis: {str(e)}")
                        numeric_stats['kurtosis'] = float('nan')
                    
                    st.dataframe(numeric_stats, use_container_width=True)
                    result["stats"] = numeric_stats.to_dict()
            except Exception as e:
                st.error(f"Error calculating numeric statistics: {str(e)}")
                result["stats"] = {}
        else:
            st.info("No numeric columns found in the selected features.")
        
        # For categorical columns
        cat_cols = df_subset.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            st.subheader("Categorical Statistics")
            
            cat_stats = {}
            for col in cat_cols:
                value_counts = df_subset[col].value_counts()
                unique_count = df_subset[col].nunique()
                
                st.write(f"**{col}**")
                st.write(f"- Unique values: {unique_count}")
                
                # Show top 10 most frequent values
                top_values = value_counts.head(10)
                st.bar_chart(top_values)
                
                cat_stats[col] = {
                    "unique_count": unique_count,
                    "top_values": value_counts.head(10).to_dict()
                }
            
            result["categorical_stats"] = cat_stats
    
    elif method == "Data Distribution Analysis":
        st.subheader("Data Distribution Analysis")
        
        # Analyze numeric columns
        numeric_cols = df_subset.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            # Allow user to select a column to visualize
            selected_col = st.selectbox(
                "Select column to visualize distribution",
                options=numeric_cols
            )
            
            if selected_col:
                # Create distribution plots
                fig = px.histogram(
                    df_subset, 
                    x=selected_col, 
                    marginal="box",
                    title=f"Distribution of {selected_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Normality test
                stat, p_value = stats.shapiro(df_subset[selected_col].dropna())
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Shapiro-Wilk Statistic", f"{stat:.4f}")
                with col2:
                    st.metric("p-value", f"{p_value:.4f}")
                
                if p_value < 0.05:
                    st.info("The distribution appears to be non-normal (p < 0.05)")
                else:
                    st.info("The distribution appears to be normal (p >= 0.05)")
                
                # QQ plot to visualize normality
                fig = px.scatter(
                    x=np.sort(stats.norm.ppf(np.linspace(0.01, 0.99, len(df_subset[selected_col].dropna())))),
                    y=np.sort(df_subset[selected_col].dropna()),
                    title=f"Q-Q Plot for {selected_col}"
                )
                fig.add_shape(
                    type='line',
                    x0=fig.data[0].x.min(),
                    y0=df_subset[selected_col].dropna().min(),
                    x1=fig.data[0].x.max(),
                    y1=df_subset[selected_col].dropna().max(),
                    line=dict(color='red', dash='dash')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                result[f"distribution_{selected_col}"] = {
                    "shapiro_stat": float(stat),
                    "p_value": float(p_value),
                    "mean": float(df_subset[selected_col].mean()),
                    "median": float(df_subset[selected_col].median()),
                    "std": float(df_subset[selected_col].std()),
                    "min": float(df_subset[selected_col].min()),
                    "max": float(df_subset[selected_col].max())
                }
            
            # Multivariate distribution
            if len(numeric_cols) >= 2:
                st.subheader("Multivariate Distribution")
                
                selected_cols = st.multiselect(
                    "Select columns for scatter plot",
                    options=numeric_cols,
                    default=numeric_cols[:2]
                )
                
                if len(selected_cols) >= 2:
                    fig = px.scatter_matrix(
                        df_subset[selected_cols],
                        dimensions=selected_cols,
                        title="Scatter Matrix"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    elif method == "Outlier Detection":
        st.subheader("Outlier Detection")
        
        # Analyze numeric columns
        numeric_cols = df_subset.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            # Allow user to select a column to analyze
            selected_col = st.selectbox(
                "Select column to detect outliers",
                options=numeric_cols
            )
            
            if selected_col:
                col_data = df_subset[selected_col].dropna()
                
                # Z-score method
                z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                outliers_z = col_data[z_scores > 3]
                
                # IQR method
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_iqr = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Z-score Outliers", f"{len(outliers_z)} ({len(outliers_z)/len(col_data)*100:.2f}%)")
                with col2:
                    st.metric("IQR Outliers", f"{len(outliers_iqr)} ({len(outliers_iqr)/len(col_data)*100:.2f}%)")
                
                # Plot the data with outliers
                fig = px.box(
                    col_data,
                    title=f"Box Plot of {selected_col} with Outliers"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show outlier values
                if not outliers_z.empty:
                    st.subheader("Z-score Outliers (|z| > 3)")
                    st.dataframe(outliers_z.sort_values(ascending=False), use_container_width=True)
                
                if not outliers_iqr.empty:
                    st.subheader("IQR Outliers")
                    st.dataframe(outliers_iqr.sort_values(ascending=False), use_container_width=True)
                
                result[f"outliers_{selected_col}"] = {
                    "z_score_count": len(outliers_z),
                    "z_score_percentage": float(len(outliers_z)/len(col_data)*100),
                    "iqr_count": len(outliers_iqr),
                    "iqr_percentage": float(len(outliers_iqr)/len(col_data)*100),
                    "lower_bound_iqr": float(lower_bound),
                    "upper_bound_iqr": float(upper_bound)
                }
    
    return result

def run_correlation_analysis(df: pd.DataFrame, method: str, features: List[str]) -> Dict[str, Any]:
    """Run correlation analysis"""
    result = {"type": "correlation", "method": method}
    
    # Filter to only include selected features
    df_subset = df[features].copy()
    
    # For correlation analysis, we need numeric columns
    numeric_df = df_subset.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        st.warning("No numeric columns available for correlation analysis")
        return result
    
    if method == "Pearson Correlation":
        st.subheader("Pearson Correlation Analysis")
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr(method='pearson')
        
        # Display correlation matrix
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title="Pearson Correlation Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display strongest correlations
        st.subheader("Strongest Correlations")
        
        # Get the upper triangle of the correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find the strongest correlations
        strongest_corr = upper.unstack().sort_values(ascending=False).dropna()
        
        if not strongest_corr.empty:
            # Display top correlations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top 10 Positive Correlations")
                positive_corr = strongest_corr[strongest_corr > 0].head(10)
                if not positive_corr.empty:
                    for (col_a, col_b), corr_value in positive_corr.items():
                        st.write(f"{col_a} — {col_b}: {corr_value:.4f}")
                else:
                    st.info("No positive correlations found")
            
            with col2:
                st.subheader("Top 10 Negative Correlations")
                negative_corr = strongest_corr[strongest_corr < 0].sort_values().head(10)
                if not negative_corr.empty:
                    for (col_a, col_b), corr_value in negative_corr.items():
                        st.write(f"{col_a} — {col_b}: {corr_value:.4f}")
                else:
                    st.info("No negative correlations found")
            
            # Scatter plot of strongest correlation
            if len(strongest_corr) > 0:
                st.subheader("Scatter Plot of Highest Correlation")
                top_pair = strongest_corr.index[0]
                col_a, col_b = top_pair
                
                fig = px.scatter(
                    numeric_df,
                    x=col_a,
                    y=col_b,
                    trendline="ols",
                    title=f"Correlation between {col_a} and {col_b} (r = {corr_matrix.loc[col_a, col_b]:.4f})"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        result["correlation_matrix"] = corr_matrix.to_dict()
    
    elif method == "Spearman Correlation":
        st.subheader("Spearman Rank Correlation Analysis")
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr(method='spearman')
        
        # Display correlation matrix
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title="Spearman Correlation Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display strongest correlations
        st.subheader("Strongest Correlations")
        
        # Get the upper triangle of the correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find the strongest correlations
        strongest_corr = upper.unstack().sort_values(ascending=False).dropna()
        
        if not strongest_corr.empty:
            # Display top correlations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top 10 Positive Correlations")
                positive_corr = strongest_corr[strongest_corr > 0].head(10)
                if not positive_corr.empty:
                    for (col_a, col_b), corr_value in positive_corr.items():
                        st.write(f"{col_a} — {col_b}: {corr_value:.4f}")
                else:
                    st.info("No positive correlations found")
            
            with col2:
                st.subheader("Top 10 Negative Correlations")
                negative_corr = strongest_corr[strongest_corr < 0].sort_values().head(10)
                if not negative_corr.empty:
                    for (col_a, col_b), corr_value in negative_corr.items():
                        st.write(f"{col_a} — {col_b}: {corr_value:.4f}")
                else:
                    st.info("No negative correlations found")
            
            # Scatter plot of strongest correlation
            if len(strongest_corr) > 0:
                st.subheader("Scatter Plot of Highest Correlation")
                top_pair = strongest_corr.index[0]
                col_a, col_b = top_pair
                
                fig = px.scatter(
                    numeric_df,
                    x=col_a,
                    y=col_b,
                    title=f"Correlation between {col_a} and {col_b} (rho = {corr_matrix.loc[col_a, col_b]:.4f})"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        result["correlation_matrix"] = corr_matrix.to_dict()
    
    elif method == "Feature Importance":
        st.subheader("Feature Importance Analysis")
        
        # We need at least one target variable for feature importance
        if len(numeric_df.columns) < 2:
            st.warning("Need at least 2 numeric columns for feature importance analysis")
            return result
        
        # Let user select the target variable
        target_col = st.selectbox(
            "Select target variable for feature importance",
            options=numeric_df.columns.tolist()
        )
        
        if target_col:
            # Separate features and target
            X = numeric_df.drop(columns=[target_col])
            y = numeric_df[target_col]
            
            # Run a simple linear regression to get feature importance
            model = LinearRegression()
            model.fit(X, y)
            
            # Calculate feature importance
            importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': np.abs(model.coef_),
                'Coefficient': model.coef_
            })
            importance = importance.sort_values('Importance', ascending=False)
            
            # Display feature importance
            st.dataframe(importance, use_container_width=True)
            
            # Plot feature importance
            fig = px.bar(
                importance,
                x='Feature',
                y='Importance',
                title=f"Feature Importance for {target_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot coefficients (with sign)
            fig = px.bar(
                importance,
                x='Feature',
                y='Coefficient',
                title=f"Feature Coefficients for {target_col}",
                color='Coefficient',
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            result["feature_importance"] = importance.to_dict()
            result["target_variable"] = target_col
    
    return result

def run_regression_analysis(df: pd.DataFrame, method: str, target: str, features: List[str]) -> Dict[str, Any]:
    """Run regression analysis"""
    result = {"type": "regression", "method": method}
    
    if not target:
        st.warning("No target variable selected for regression analysis")
        return result
    
    # Check if target is in the dataframe
    if target not in df.columns:
        st.error(f"Target variable '{target}' not found in the dataframe")
        return result
    
    # Filter to only include selected features and target
    feature_cols = [f for f in features if f != target]
    
    if not feature_cols:
        st.warning("No features selected for regression analysis")
        return result
    
    # Check if we have enough data
    if len(df) < len(feature_cols) + 1:
        st.warning("Not enough data points for the number of features")
        return result
    
    # Prepare data
    X = df[feature_cols].select_dtypes(include=[np.number])
    
    # If we lost features due to non-numeric types, warn the user
    if len(X.columns) < len(feature_cols):
        removed_cols = set(feature_cols) - set(X.columns)
        st.warning(f"Removed non-numeric features: {', '.join(removed_cols)}")
        feature_cols = X.columns.tolist()
    
    if X.empty:
        st.error("No numeric features available for regression analysis")
        return result
    
    y = df[target]
    
    # Check if target is numeric
    if not pd.api.types.is_numeric_dtype(y):
        st.error("Target variable must be numeric for regression analysis")
        return result
    
    if method == "Linear Regression":
        st.subheader("Linear Regression Analysis")
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Fit statsmodels for detailed statistics
        X_train_sm = sm.add_constant(X_train)
        sm_model = sm.OLS(y_train, X_train_sm).fit()
        
        # Display model summary
        st.subheader("Model Summary")
        st.text(sm_model.summary().as_text())
        
        # Display metrics
        st.subheader("Model Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Training R²", f"{train_r2:.4f}")
            st.metric("Training RMSE", f"{train_rmse:.4f}")
        
        with col2:
            st.metric("Testing R²", f"{test_r2:.4f}")
            st.metric("Testing RMSE", f"{test_rmse:.4f}")
        
        # Display coefficients
        st.subheader("Coefficients")
        
        coef_df = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': model.coef_,
            'Absolute Impact': np.abs(model.coef_)
        }).sort_values('Absolute Impact', ascending=False)
        
        st.dataframe(coef_df, use_container_width=True)
        
        # Plot coefficients
        fig = px.bar(
            coef_df,
            x='Feature',
            y='Coefficient',
            title="Regression Coefficients",
            color='Coefficient',
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Plot actual vs predicted
        fig = px.scatter(
            x=y_test,
            y=y_pred_test,
            labels={'x': 'Actual', 'y': 'Predicted'},
            title="Actual vs Predicted Values (Test Set)"
        )
        
        # Add a perfect prediction line
        min_val = min(y_test.min(), y_pred_test.min())
        max_val = max(y_test.max(), y_pred_test.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name="Perfect Prediction"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Residual plot
        residuals = y_test - y_pred_test
        
        fig = px.scatter(
            x=y_pred_test,
            y=residuals,
            labels={'x': 'Predicted', 'y': 'Residuals'},
            title="Residual Plot"
        )
        
        fig.add_shape(
            type="line",
            x0=y_pred_test.min(),
            y0=0,
            x1=y_pred_test.max(),
            y1=0,
            line=dict(color="red", dash="dash")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Store results
        result["model_coefficients"] = coef_df.to_dict()
        result["model_intercept"] = float(model.intercept_)
        result["metrics"] = {
            "train_r2": float(train_r2),
            "test_r2": float(test_r2),
            "train_rmse": float(train_rmse),
            "test_rmse": float(test_rmse)
        }
    
    elif method == "Multiple Regression":
        st.subheader("Multiple Regression Analysis")
        
        # This is basically the same as linear regression when using multiple features
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Fit statsmodels for detailed statistics
        X_train_sm = sm.add_constant(X_train)
        sm_model = sm.OLS(y_train, X_train_sm).fit()
        
        # Display model summary
        st.subheader("Model Summary")
        st.text(sm_model.summary().as_text())
        
        # Display metrics
        st.subheader("Model Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Training R²", f"{train_r2:.4f}")
            st.metric("Training RMSE", f"{train_rmse:.4f}")
        
        with col2:
            st.metric("Testing R²", f"{test_r2:.4f}")
            st.metric("Testing RMSE", f"{test_rmse:.4f}")
        
        # Calculate feature importance
        importance = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': model.coef_,
            'Standardized Coefficient': model.coef_ * X_train.std().values,
            'Absolute Impact': np.abs(model.coef_)
        }).sort_values('Absolute Impact', ascending=False)
        
        # Display coefficients
        st.subheader("Coefficients and Feature Importance")
        st.dataframe(importance, use_container_width=True)
        
        # Plot coefficients
        fig = px.bar(
            importance,
            x='Feature',
            y='Coefficient',
            title="Regression Coefficients",
            color='Coefficient',
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Standardized coefficients
        fig = px.bar(
            importance,
            x='Feature',
            y='Standardized Coefficient',
            title="Standardized Coefficients (Feature Importance)",
            color='Standardized Coefficient',
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Plot actual vs predicted
        fig = px.scatter(
            x=y_test,
            y=y_pred_test,
            labels={'x': 'Actual', 'y': 'Predicted'},
            title="Actual vs Predicted Values (Test Set)"
        )
        
        # Add a perfect prediction line
        min_val = min(y_test.min(), y_pred_test.min())
        max_val = max(y_test.max(), y_pred_test.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name="Perfect Prediction"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Residual plot
        residuals = y_test - y_pred_test
        
        fig = px.scatter(
            x=y_pred_test,
            y=residuals,
            labels={'x': 'Predicted', 'y': 'Residuals'},
            title="Residual Plot"
        )
        
        fig.add_shape(
            type="line",
            x0=y_pred_test.min(),
            y0=0,
            x1=y_pred_test.max(),
            y1=0,
            line=dict(color="red", dash="dash")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Store results
        result["model_coefficients"] = importance.to_dict()
        result["model_intercept"] = float(model.intercept_)
        result["metrics"] = {
            "train_r2": float(train_r2),
            "test_r2": float(test_r2),
            "train_rmse": float(train_rmse),
            "test_rmse": float(test_rmse)
        }
    
    return result

def run_classification_analysis(df: pd.DataFrame, method: str, target: str, features: List[str]) -> Dict[str, Any]:
    """Run classification analysis"""
    result = {"type": "classification", "method": method}
    
    if not target:
        st.warning("No target variable selected for classification analysis")
        return result
    
    # Check if target is in the dataframe
    if target not in df.columns:
        st.error(f"Target variable '{target}' not found in the dataframe")
        return result
    
    # Filter to only include selected features and target
    feature_cols = [f for f in features if f != target]
    
    if not feature_cols:
        st.warning("No features selected for classification analysis")
        return result
    
    # Check if we have enough data
    if len(df) < len(feature_cols) + 1:
        st.warning("Not enough data points for the number of features")
        return result
    
    # Prepare data
    X = df[feature_cols].select_dtypes(include=[np.number])
    
    # If we lost features due to non-numeric types, warn the user
    if len(X.columns) < len(feature_cols):
        removed_cols = set(feature_cols) - set(X.columns)
        st.warning(f"Removed non-numeric features: {', '.join(removed_cols)}")
        feature_cols = X.columns.tolist()
    
    if X.empty:
        st.error("No numeric features available for classification analysis")
        return result
    
    y = df[target]
    
    if method == "Logistic Regression":
        st.subheader("Logistic Regression Analysis")
        
        # Check if the target is binary
        unique_classes = y.unique()
        if len(unique_classes) > 10:
            st.error(f"Target variable has too many classes: {len(unique_classes)}. Consider a different analysis method.")
            return result
        
        # Display class distribution
        st.write("Target Class Distribution")
        class_counts = y.value_counts()
        fig = px.pie(
            values=class_counts.values,
            names=class_counts.index,
            title=f"Distribution of {target} Classes"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(unique_classes) > 1 else None)
        
        # Fit the model
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate accuracy
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Display metrics
        st.subheader("Model Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Training Accuracy", f"{train_accuracy:.4f}")
        
        with col2:
            st.metric("Testing Accuracy", f"{test_accuracy:.4f}")
        
        # Calculate feature importance
        if hasattr(model, 'coef_'):
            importance = pd.DataFrame({
                'Feature': X.columns,
                'Coefficient': model.coef_[0],
                'Absolute Impact': np.abs(model.coef_[0])
            }).sort_values('Absolute Impact', ascending=False)
            
            # Display coefficients
            st.subheader("Coefficients and Feature Importance")
            st.dataframe(importance, use_container_width=True)
            
            # Plot coefficients
            fig = px.bar(
                importance,
                x='Feature',
                y='Coefficient',
                title="Logistic Regression Coefficients",
                color='Coefficient',
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_test)
        class_names = model.classes_.tolist()
        
        fig = px.imshow(
            cm,
            x=class_names,
            y=class_names,
            text_auto=True,
            color_continuous_scale='Blues',
            title="Confusion Matrix"
        )
        
        fig.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # If binary classification, show ROC curve
        if len(unique_classes) == 2:
            from sklearn.metrics import roc_curve, auc
            
            # Get probability predictions
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            fig = px.line(
                x=fpr, y=tpr,
                title=f'ROC Curve (AUC = {roc_auc:.4f})',
                labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'}
            )
            
            # Add diagonal line
            fig.add_shape(
                type='line',
                line=dict(dash='dash', color='gray'),
                x0=0, y0=0,
                x1=1, y1=1
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Store AUC in results
            result["roc_auc"] = float(roc_auc)
        
        # Store results
        if hasattr(model, 'coef_'):
            result["model_coefficients"] = importance.to_dict()
        
        result["metrics"] = {
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy)
        }
        
        result["confusion_matrix"] = cm.tolist()
        result["classes"] = class_names
    
    return result

def run_clustering_analysis(df: pd.DataFrame, method: str, features: List[str]) -> Dict[str, Any]:
    """Run clustering analysis"""
    result = {"type": "clustering", "method": method}
    
    # Filter to only include selected features
    df_subset = df[features].copy()
    
    # For clustering, we need numeric columns
    numeric_df = df_subset.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        st.warning("No numeric columns available for clustering analysis")
        return result
    
    if method == "K-Means Clustering":
        st.subheader("K-Means Clustering Analysis")
        
        # Get user input for the number of clusters
        n_clusters = st.slider(
            "Number of clusters (k)",
            min_value=2,
            max_value=10,
            value=3,
            help="Select the number of clusters to identify in the data"
        )
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        # Ensure we have complete data
        data_for_clustering = numeric_df.dropna()
        
        if len(data_for_clustering) < 10:
            st.error("Not enough complete data points for clustering")
            return result
        
        # Fit the model
        kmeans.fit(data_for_clustering)
        
        # Add cluster labels to the data
        data_with_clusters = data_for_clustering.copy()
        data_with_clusters['Cluster'] = kmeans.labels_
        
        # Display cluster information
        st.subheader("Cluster Information")
        
        cluster_counts = data_with_clusters['Cluster'].value_counts().sort_index()
        
        for cluster_id, count in cluster_counts.items():
            percentage = count / len(data_with_clusters) * 100
            st.metric(f"Cluster {cluster_id}", f"{count} samples ({percentage:.2f}%)")
        
        # Visualize the clusters
        st.subheader("Cluster Visualization")
        
        if len(numeric_df.columns) >= 2:
            # Let user select 2 dimensions for visualization
            col1, col2 = st.columns(2)
            
            with col1:
                x_col = st.selectbox("X-axis feature", options=numeric_df.columns, index=0)
            
            with col2:
                y_col = st.selectbox("Y-axis feature", options=numeric_df.columns, index=min(1, len(numeric_df.columns)-1))
            
            # Create scatter plot
            fig = px.scatter(
                data_with_clusters,
                x=x_col,
                y=y_col,
                color='Cluster',
                color_continuous_scale='viridis',
                title=f"Clusters in {x_col} vs {y_col} space"
            )
            
            # Add cluster centers
            centers = kmeans.cluster_centers_
            center_df = pd.DataFrame(centers, columns=numeric_df.columns)
            
            for i in range(n_clusters):
                fig.add_scatter(
                    x=[center_df.loc[i, x_col]],
                    y=[center_df.loc[i, y_col]],
                    mode='markers',
                    marker=dict(color='red', size=15, symbol='x'),
                    name=f"Center {i}"
                )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Calculate cluster profiles
        st.subheader("Cluster Profiles")
        
        profiles = data_with_clusters.groupby('Cluster').mean().T
        
        # Create a heatmap of cluster profiles
        fig = px.imshow(
            profiles,
            color_continuous_scale='RdBu_r',
            title="Feature Values by Cluster",
            labels={'x': 'Cluster', 'y': 'Feature', 'color': 'Mean Value'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display cluster profiles numerically
        st.dataframe(profiles, use_container_width=True)
        
        # Calculate silhouette score
        from sklearn.metrics import silhouette_score
        
        silhouette_avg = silhouette_score(data_for_clustering, kmeans.labels_)
        st.metric("Silhouette Score", f"{silhouette_avg:.4f}", help="Measures how well clusters are separated. Higher is better (range: -1 to 1)")
        
        # Determine optimal number of clusters (elbow method)
        st.subheader("Optimal Number of Clusters")
        
        with st.spinner("Calculating inertia for different k values..."):
            inertia = []
            k_values = range(1, min(11, len(data_for_clustering) + 1))
            
            for k in k_values:
                if k == 1:
                    # For k=1, inertia is the sum of squared distances to the mean
                    mean_vector = data_for_clustering.mean().values.reshape(1, -1)
                    inertia.append(np.sum(np.square(data_for_clustering.values - mean_vector)))
                else:
                    kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans_model.fit(data_for_clustering)
                    inertia.append(kmeans_model.inertia_)
            
            # Plot elbow curve
            fig = px.line(
                x=list(k_values),
                y=inertia,
                markers=True,
                title="Elbow Method for Optimal k",
                labels={'x': 'Number of Clusters (k)', 'y': 'Inertia'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Store results
        result["cluster_counts"] = cluster_counts.to_dict()
        result["cluster_profiles"] = profiles.to_dict()
        result["silhouette_score"] = float(silhouette_avg)
        result["inertia"] = {k: float(i) for k, i in zip(k_values, inertia)}
        result["selected_k"] = n_clusters
    
    return result

def run_time_series_analysis(df: pd.DataFrame, method: str, features: List[str]) -> Dict[str, Any]:
    """Run time series analysis"""
    result = {"type": "time_series", "method": method}
    
    if len(features) < 2:
        st.warning("Need time column and value column for time series analysis")
        return result
    
    # Assume the first feature is the time column and the second is the value column
    time_col = features[0]
    value_col = features[1]
    
    # Convert time column to datetime if not already
    if df[time_col].dtype != 'datetime64[ns]':
        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except:
            st.error(f"Could not convert '{time_col}' to datetime format")
            return result
    
    # Sort by time column
    ts_df = df[[time_col, value_col]].sort_values(time_col)
    
    if method == "Trend Analysis":
        st.subheader("Time Series Trend Analysis")
        
        # Plot the time series
        fig = px.line(
            ts_df,
            x=time_col,
            y=value_col,
            title=f"Time Series: {value_col} over Time"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Check if the time series has at least 10 data points
        if len(ts_df) < 10:
            st.warning("Not enough data points for trend analysis")
            return result
        
        # Add a trend line
        trend_type = st.selectbox(
            "Trend Line Type",
            options=["Linear", "Rolling Average"],
            index=0
        )
        
        if trend_type == "Linear":
            # Add linear trend
            from scipy import stats
            
            # Convert to numeric for regression
            ts_df['time_num'] = (ts_df[time_col] - ts_df[time_col].min()).dt.total_seconds()
            
            # Fit a linear trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(ts_df['time_num'], ts_df[value_col])
            
            # Add trend line to plot
            ts_df['trend'] = intercept + slope * ts_df['time_num']
            
            fig = px.line(
                ts_df,
                x=time_col,
                y=[value_col, 'trend'],
                title=f"Time Series with Linear Trend (r²={r_value**2:.4f})"
            )
            
            fig.update_layout(legend=dict(
                title="",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display trend statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Slope", f"{slope:.6f}")
                st.metric("R²", f"{r_value**2:.4f}")
            
            with col2:
                st.metric("Intercept", f"{intercept:.4f}")
                st.metric("P-value", f"{p_value:.6f}")
            
            # Describe the trend
            if p_value < 0.05:
                if slope > 0:
                    trend_desc = f"Significant upward trend of approximately {slope:.6f} units per second"
                else:
                    trend_desc = f"Significant downward trend of approximately {abs(slope):.6f} units per second"
            else:
                trend_desc = "No significant trend detected (p > 0.05)"
            
            st.info(trend_desc)
            
            # Calculate time-based statistics
            result["trend_statistics"] = {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value**2),
                "p_value": float(p_value),
                "std_err": float(std_err)
            }
        
        elif trend_type == "Rolling Average":
            # Add rolling average
            window_size = st.slider(
                "Window Size",
                min_value=2,
                max_value=max(2, len(ts_df) // 2),
                value=min(7, max(2, len(ts_df) // 5))
            )
            
            # Calculate moving average
            ts_df['rolling_avg'] = ts_df[value_col].rolling(window=window_size).mean()
            
            fig = px.line(
                ts_df,
                x=time_col,
                y=[value_col, 'rolling_avg'],
                title=f"Time Series with {window_size}-point Rolling Average"
            )
            
            fig.update_layout(legend=dict(
                title="",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate statistics for original vs smoothed series
            original_std = ts_df[value_col].std()
            smoothed_std = ts_df['rolling_avg'].dropna().std()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Original Std Dev", f"{original_std:.4f}")
            
            with col2:
                st.metric("Smoothed Std Dev", f"{smoothed_std:.4f}", 
                         delta=f"{smoothed_std - original_std:.4f}")
            
            # Result storing
            result["rolling_statistics"] = {
                "window_size": window_size,
                "original_std": float(original_std),
                "smoothed_std": float(smoothed_std)
            }
    
    elif method == "Seasonality Analysis":
        st.subheader("Time Series Seasonality Analysis")
        
        # Plot the original time series
        fig = px.line(
            ts_df,
            x=time_col,
            y=value_col,
            title=f"Time Series: {value_col} over Time"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Check if we have enough data
        if len(ts_df) < 10:
            st.warning("Not enough data points for seasonality analysis")
            return result
        
        # Determine time frequency
        time_diffs = ts_df[time_col].diff().dropna()
        
        if time_diffs.empty:
            st.warning("Cannot determine time frequency")
            return result
        
        # Try to infer the frequency
        common_diff = time_diffs.mode()[0]
        
        # Convert to a human-readable string
        if common_diff.days >= 365:
            freq_str = f"{common_diff.days // 365} year(s)"
            period = 1
        elif common_diff.days >= 28:
            freq_str = f"{common_diff.days // 30} month(s)"
            period = 12
        elif common_diff.days >= 1:
            freq_str = f"{common_diff.days} day(s)"
            period = 7 if common_diff.days == 1 else 30 // common_diff.days
        elif common_diff.seconds >= 3600:
            freq_str = f"{common_diff.seconds // 3600} hour(s)"
            period = 24
        elif common_diff.seconds >= 60:
            freq_str = f"{common_diff.seconds // 60} minute(s)"
            period = 60
        else:
            freq_str = f"{common_diff.seconds} second(s)"
            period = 60
        
        st.write(f"Detected time frequency: {freq_str}")
        
        # Let the user select the seasonality period
        period = st.number_input(
            "Seasonality period (number of observations in one cycle)",
            min_value=2,
            max_value=len(ts_df) // 2,
            value=min(period, len(ts_df) // 2),
            help="Enter the number of observations that make up one complete cycle"
        )
        
        # Create seasonal plot if we have enough periods
        if len(ts_df) >= period * 2:
            # Add a cycle identifier
            ts_df['season_cycle'] = (np.arange(len(ts_df)) // period) + 1
            ts_df['season_position'] = np.arange(len(ts_df)) % period + 1
            
            # Calculate seasonal pattern
            seasonal_avg = ts_df.groupby('season_position')[value_col].mean()
            
            # Plot seasonal patterns
            fig = px.line(
                x=seasonal_avg.index,
                y=seasonal_avg.values,
                title=f"Seasonal Pattern (Period={period})",
                labels={'x': 'Position in Cycle', 'y': f'Average {value_col}'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Heatmap for multiple cycles
            if ts_df['season_cycle'].nunique() > 1:
                # Pivot the data
                pivot_df = ts_df.pivot(index='season_cycle', columns='season_position', values=value_col)
                
                fig = px.imshow(
                    pivot_df,
                    title=f"Seasonal Heatmap (Period={period})",
                    labels={'x': 'Position in Cycle', 'y': 'Cycle Number', 'color': value_col}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Calculate seasonal strength
            # Decompose the time series
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            try:
                decomposition = seasonal_decompose(ts_df[value_col], period=int(period), model='additive')
                
                # Plot decomposition
                fig = go.Figure()
                
                # Original
                fig.add_trace(go.Scatter(
                    x=ts_df[time_col],
                    y=decomposition.observed,
                    mode='lines',
                    name='Original'
                ))
                
                # Trend
                fig.add_trace(go.Scatter(
                    x=ts_df[time_col],
                    y=decomposition.trend,
                    mode='lines',
                    name='Trend'
                ))
                
                # Seasonal
                fig.add_trace(go.Scatter(
                    x=ts_df[time_col],
                    y=decomposition.seasonal,
                    mode='lines',
                    name='Seasonal'
                ))
                
                # Residual
                fig.add_trace(go.Scatter(
                    x=ts_df[time_col],
                    y=decomposition.resid,
                    mode='lines',
                    name='Residual'
                ))
                
                fig.update_layout(
                    title="Time Series Decomposition",
                    xaxis_title="Time",
                    yaxis_title="Value"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate seasonal strength
                seasonal_var = decomposition.seasonal.var()
                resid_var = decomposition.resid.var()
                
                if seasonal_var + resid_var > 0:
                    seasonal_strength = seasonal_var / (seasonal_var + resid_var)
                    st.metric("Seasonal Strength", f"{seasonal_strength:.4f}")
                    
                    if seasonal_strength > 0.6:
                        st.success("Strong seasonality detected")
                    elif seasonal_strength > 0.3:
                        st.info("Moderate seasonality detected")
                    else:
                        st.warning("Weak seasonality detected")
                else:
                    st.warning("Could not calculate seasonal strength")
                
                # Store results
                result["seasonality"] = {
                    "period": int(period),
                    "seasonal_pattern": seasonal_avg.to_dict(),
                    "seasonal_strength": float(seasonal_strength) if 'seasonal_strength' in locals() else None
                }
                
            except Exception as e:
                st.error(f"Could not decompose time series: {str(e)}")
        else:
            st.warning(f"Need at least {period * 2} data points for seasonal analysis with period={period}")
    
    return result
