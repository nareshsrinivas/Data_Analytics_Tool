import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import io
import base64
from datetime import datetime

# Import our utility modules
from utils.file_upload import render_file_upload
from utils.data_preprocessing import render_problem_definition, render_data_preprocessing
from utils.analysis import render_analysis
from utils.visualization import render_visualization
from utils.reporting import render_reporting, export_results
from utils.history import render_history_page, record_file_upload, record_analysis
from utils.database import init_database
from assets.stock_photos import display_header_image

# Configure the page
st.set_page_config(
    page_title="Data Analytics Pipeline",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
init_database()

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1  # Start at step 1: File Upload

def main():
    """Main function to run the data analytics pipeline application"""
    
    # Display header with image
    display_header_image("data analytics dashboard")
    
    # App title and description
    st.title("Data Analytics Pipeline")
    st.markdown("""
    Welcome to the Data Analytics Pipeline application. This tool allows you to upload your CSV data,
    define analysis goals, preprocess your data, perform various analyses, visualize insights, and 
    generate reports.
    
    Follow the step-by-step workflow to analyze your data effectively.
    """)
    
    # Create a sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        st.info("Progress through each step of the analytics pipeline by completing the current step.")
        
        # Display progress for pipeline steps (1-6)
        if st.session_state.step <= 6:
            progress_percentage = (st.session_state.step - 1) / 5 * 100
            st.progress(progress_percentage / 100)
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.step > 1 and st.session_state.step <= 7:
                if st.button("⬅️ Previous Step"):
                    st.session_state.step -= 1
                    st.rerun()
        
        with col2:
            if st.session_state.step < 6 and st.session_state.get('dataframe') is not None:
                if st.button("Next Step ➡️"):
                    st.session_state.step += 1
                    st.rerun()
        
        # Direct navigation
        steps = [
            "1. File Upload",
            "2. Problem Definition",
            "3. Data Preprocessing",
            "4. Analysis",
            "5. Visualization",
            "6. Reporting",
            "7. History & DB"
        ]
        
        selected_step = st.selectbox("Go to Step", options=steps, index=st.session_state.step - 1)
        new_step = steps.index(selected_step) + 1
        
        if new_step != st.session_state.step:
            # Only allow direct navigation if data is loaded (except for step 1 and 7)
            if new_step == 1 or new_step == 7 or st.session_state.get('dataframe') is not None:
                st.session_state.step = new_step
                st.rerun()
            else:
                st.warning("Please upload data first!")
                
        # Add a separate section for database/history access
        st.divider()
        if st.button("📊 View Analysis History", use_container_width=True):
            st.session_state.step = 7
            st.rerun()
                
        # Display help information in the sidebar
        with st.expander("Help & Information"):
            st.markdown("""
            ### How to use this application
            
            1. **File Upload**: Upload your CSV data file
            2. **Problem Definition**: Specify your analysis goals and target variables
            3. **Data Preprocessing**: Clean and prepare your data
            4. **Analysis**: Run analyses based on your goals
            5. **Visualization**: Explore visual representations of your data
            6. **Reporting**: Generate and download reports
            
            For more detailed instructions, click the help icons (❓) next to each section.
            """)
            
            # Add contact info or additional help resources
            st.divider()
            st.markdown("**Need help?** Refer to the documentation or contact support.")
    
    # Display different steps based on the current step in the session state
    if st.session_state.step == 1:
        # Step 1: File Upload
        df, profile = render_file_upload()
        
        # Call to action if no file has been uploaded yet
        if df.empty and not st.session_state.get('dataframe') is not None:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.info("👆 Start by uploading a CSV file above")
            with col2:
                display_header_image("business analytics interface", height=150)
        
        # Show the next step button separately for better visibility
        if not df.empty:
            # Store the upload in the database
            if 'upload_id' not in st.session_state:
                original_filename = st.session_state.get('uploaded_filename', 'uploaded_file.csv')
                upload_id = record_file_upload(df, original_filename, profile)
                if upload_id:
                    st.session_state['upload_id'] = upload_id
                    st.success(f"File uploaded and saved to database (ID: {upload_id})")
                
            st.success("File uploaded successfully! You can now proceed to the next step.")
            if st.button("Proceed to Problem Definition ➡️", type="primary"):
                st.session_state.step = 2
                st.rerun()
    
    elif st.session_state.step == 2:
        # Step 2: Problem Definition
        if st.session_state.get('dataframe') is not None:
            # Show data preview
            df = st.session_state.dataframe
            with st.expander("Data Preview", expanded=False):
                st.dataframe(df.head(5), use_container_width=True)
            
            # Render problem definition interface
            problem_def = render_problem_definition(df)
            
            # Store in session state
            st.session_state.problem_definition = problem_def
            
            # Show next step button
            if problem_def and 'features' in problem_def and problem_def['features']:
                st.success("Problem definition complete! You can now proceed to the next step.")
                if st.button("Proceed to Data Preprocessing ➡️", type="primary"):
                    st.session_state.step = 3
                    st.rerun()
            else:
                st.warning("Please select features to analyze before proceeding")
        else:
            st.error("No data available. Please upload a CSV file first.")
            if st.button("Go to File Upload"):
                st.session_state.step = 1
                st.rerun()
    
    elif st.session_state.step == 3:
        # Step 3: Data Preprocessing
        if st.session_state.get('dataframe') is not None:
            # Show problem definition summary
            problem_def = st.session_state.get('problem_definition', {})
            with st.expander("Problem Definition Summary", expanded=False):
                st.write(f"**Analysis Goal:** {problem_def.get('analysis_goal', 'Not defined')}")
                st.write(f"**Analysis Method:** {problem_def.get('analysis_method', 'Not defined')}")
                if problem_def.get('target_variable'):
                    st.write(f"**Target Variable:** {problem_def.get('target_variable')}")
                st.write(f"**Selected Features:** {', '.join(problem_def.get('features', []))}")
            
            # Render preprocessing interface
            df = st.session_state.dataframe
            processed_df = render_data_preprocessing(df, problem_def)
            
            # Store processed dataframe in session state
            st.session_state.processed_dataframe = processed_df
            
            # Show next step button
            if not processed_df.empty:
                if 'preprocessing_options' in st.session_state and st.session_state.preprocessing_options.get('processed_df') is not None:
                    st.success("Data preprocessing complete! You can now proceed to the next step.")
                    if st.button("Proceed to Analysis ➡️", type="primary"):
                        st.session_state.step = 4
                        st.rerun()
                else:
                    st.info("Apply preprocessing to continue")
            else:
                st.warning("Please process your data before continuing")
        else:
            st.error("No data available. Please upload a CSV file first.")
            if st.button("Go to File Upload"):
                st.session_state.step = 1
                st.rerun()
    
    elif st.session_state.step == 4:
        # Step 4: Analysis
        if st.session_state.get('processed_dataframe') is not None:
            # Show data preview
            df = st.session_state.processed_dataframe
            problem_def = st.session_state.get('problem_definition', {})
            
            with st.expander("Preprocessed Data Preview", expanded=False):
                st.dataframe(df.head(5), use_container_width=True)
            
            # Track analysis start time
            start_time = datetime.now()
            
            # Render analysis interface
            analysis_results = render_analysis(df, problem_def)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Store analysis results in session state
            st.session_state.analysis_results = analysis_results
            
            # Save analysis results to database
            if analysis_results and 'upload_id' in st.session_state:
                upload_id = st.session_state['upload_id']
                analysis_type = problem_def.get('analysis_goal', 'Unknown')
                analysis_method = problem_def.get('analysis_method', 'Unknown')
                
                # Save to database
                analysis_id = record_analysis(
                    data_upload_id=upload_id,
                    analysis_type=analysis_type,
                    analysis_method=analysis_method,
                    parameters=problem_def,
                    results=analysis_results,
                    execution_time=execution_time
                )
                
                if analysis_id:
                    st.session_state['analysis_id'] = analysis_id
                    st.info(f"Analysis results saved to database (ID: {analysis_id})")
            
            # Show next step button
            if analysis_results:
                st.success("Analysis complete! You can now proceed to visualization.")
                if st.button("Proceed to Visualization ➡️", type="primary"):
                    st.session_state.step = 5
                    st.rerun()
        else:
            st.error("No processed data available. Please preprocess your data first.")
            if st.button("Go to Data Preprocessing"):
                st.session_state.step = 3
                st.rerun()
    
    elif st.session_state.step == 5:
        # Step 5: Visualization
        if st.session_state.get('analysis_results') is not None:
            # Get the required data
            df = st.session_state.processed_dataframe
            problem_def = st.session_state.get('problem_definition', {})
            analysis_results = st.session_state.get('analysis_results', {})
            
            # Render visualization interface
            visualization_results = render_visualization(df, problem_def, analysis_results)
            
            # Store visualization results in session state
            st.session_state.visualization_results = visualization_results
            
            # Show next step button
            if visualization_results:
                st.success("Visualization complete! You can now proceed to the reporting step.")
                if st.button("Proceed to Reporting ➡️", type="primary"):
                    st.session_state.step = 6
                    st.rerun()
        else:
            st.error("No analysis results available. Please complete the analysis first.")
            if st.button("Go to Analysis"):
                st.session_state.step = 4
                st.rerun()
    
    elif st.session_state.step == 6:
        # Step 6: Reporting
        if st.session_state.get('analysis_results') is not None:
            # Get the required data
            df = st.session_state.processed_dataframe
            original_df = st.session_state.dataframe
            problem_def = st.session_state.get('problem_definition', {})
            analysis_results = st.session_state.get('analysis_results', {})
            visualization_results = st.session_state.get('visualization_results', {})
            
            # Render reporting interface
            report_data = render_reporting(original_df, df, problem_def, analysis_results, visualization_results)
            
            # Export functionality
            export_results(original_df, df, problem_def, analysis_results, visualization_results)
            
            # Show link to history view
            st.divider()
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write("Want to view your analysis history and database records?")
            with col2:
                if st.button("View History", type="primary"):
                    st.session_state.step = 7
                    st.rerun()
        else:
            st.error("No analysis results available. Please complete the analysis first.")
            if st.button("Go to Analysis"):
                st.session_state.step = 4
                st.rerun()
    
    elif st.session_state.step == 7:
        # Step 7: View History & Database
        render_history_page()

    # Footer
    st.divider()
    st.markdown("© 2023 Data Analytics Pipeline App | Build with Streamlit")

if __name__ == "__main__":
    main()
