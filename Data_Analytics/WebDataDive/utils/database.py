import os
import streamlit as st
import pandas as pd
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, joinedload
import sqlalchemy.types as types
from typing import Dict, List, Any, Optional

# Create SQLAlchemy Base class
Base = declarative_base()

# Define database models
class User(Base):
    """User model for storing user related data"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True)
    created_at = Column(DateTime, default=datetime.now)
    uploads = relationship("DataUpload", back_populates="user")
    analyses = relationship("Analysis", back_populates="user")

class DataUpload(Base):
    """Model for storing uploaded data files information"""
    __tablename__ = 'data_uploads'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    filename = Column(String(100), nullable=False)
    original_filename = Column(String(100))
    file_size = Column(Integer)  # Size in bytes
    row_count = Column(Integer)
    column_count = Column(Integer)
    upload_date = Column(DateTime, default=datetime.now)
    file_hash = Column(String(64))  # For detecting duplicate uploads
    file_metadata = Column(Text)  # JSON string for storing additional metadata
    
    user = relationship("User", back_populates="uploads")
    analyses = relationship("Analysis", back_populates="data_upload")

class Analysis(Base):
    """Model for storing analysis results"""
    __tablename__ = 'analyses'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    data_upload_id = Column(Integer, ForeignKey('data_uploads.id'))
    analysis_type = Column(String(50), nullable=False)  # e.g., 'exploratory', 'regression'
    analysis_method = Column(String(50))  # e.g., 'linear_regression', 'k_means'
    parameters = Column(Text)  # JSON string for storing analysis parameters
    results = Column(Text)  # JSON string for storing analysis results
    created_at = Column(DateTime, default=datetime.now)
    execution_time = Column(Float)  # Time taken to run analysis in seconds
    is_successful = Column(Boolean, default=True)
    error_message = Column(Text)
    
    user = relationship("User", back_populates="analyses")
    data_upload = relationship("DataUpload", back_populates="analyses")
    visualizations = relationship("Visualization", back_populates="analysis")

class Visualization(Base):
    """Model for storing visualization metadata"""
    __tablename__ = 'visualizations'
    
    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analyses.id'))
    visualization_type = Column(String(50), nullable=False)  # e.g., 'scatter', 'histogram'
    title = Column(String(200))
    description = Column(Text)
    configuration = Column(Text)  # JSON string for storing visualization configuration
    created_at = Column(DateTime, default=datetime.now)
    image_data = Column(Text)  # Base64 encoded image or URL to stored image
    
    analysis = relationship("Analysis", back_populates="visualizations")

# Database Connection Helper
class DatabaseConnection:
    """Helper class to manage database connections"""
    
    def __init__(self):
        """Initialize database connection using SQLite"""
        # Use SQLite database file in the project directory
        self.database_url = "sqlite:///./data_analytics.db"
        self.engine = None
        self.Session = None
        
        # Initialize the database connection
        self._connect()
    
    def _connect(self):
        """Establish connection to the database"""
        try:
            # Create engine with connection pool settings
            self.engine = create_engine(
                self.database_url,
                # Keep connections alive and handle reconnections
                pool_pre_ping=True,
                # Configure pool recycle time to avoid stale connections
                pool_recycle=3600
            )
            self.Session = sessionmaker(bind=self.engine)
        except Exception as e:
            st.error(f"Failed to connect to database: {str(e)}")
            raise
    
    def create_tables(self):
        """Create all defined tables in the database"""
        Base.metadata.create_all(self.engine)
    
    def get_session(self):
        """Get a new database session"""
        try:
            if not self.Session:
                self._connect()
            return self.Session()
        except Exception as e:
            st.error(f"Error creating database session: {str(e)}")
            # Attempt to reconnect
            try:
                self._connect()
                return self.Session()
            except Exception as reconnect_error:
                st.error(f"Failed to reconnect to database: {str(reconnect_error)}")
                return None

# Initialize database connection in Streamlit session state
def init_database():
    """Initialize database connection and create tables if they don't exist"""
    # Check if we need to force a new connection
    need_new_connection = False
    
    if 'db_connection' in st.session_state:
        # Test the existing connection
        try:
            existing_conn = st.session_state['db_connection']
            test_session = existing_conn.get_session()
            if test_session:
                # Connection is working
                test_session.close()
            else:
                # Session creation failed, need new connection
                need_new_connection = True
        except Exception:
            # Any error means we need a new connection
            need_new_connection = True
    else:
        # No connection exists yet
        need_new_connection = True
    
    # Create a new connection if needed
    if need_new_connection:
        try:
            # Create database connection
            db_connection = DatabaseConnection()
            
            # Create tables if they don't exist
            db_connection.create_tables()
            
            # Store in session state
            st.session_state['db_connection'] = db_connection
            
            return db_connection
        except Exception as e:
            st.error(f"Failed to initialize database: {str(e)}")
            return None
    
    # Return existing connection
    return st.session_state['db_connection']

# Helper functions for common database operations
def get_or_create_user(username: str, email: Optional[str] = None) -> User:
    """Get existing user or create a new one"""
    db_connection = init_database()
    if not db_connection:
        st.error("Database connection failed")
        return None
    
    session = db_connection.get_session()
    if not session:
        st.error("Failed to create database session")
        return None
    
    try:
        # Check if user exists
        user = session.query(User).filter(User.username == username).first()
        
        if not user:
            # Create new user
            user = User(username=username, email=email)
            session.add(user)
            session.commit()
        
        return user
    except Exception as e:
        if session:
            try:
                session.rollback()
            except Exception:
                pass  # Already failed, just continue
        st.error(f"Error creating user: {str(e)}")
        return None
    finally:
        if session:
            try:
                session.close()
            except Exception:
                pass  # Ignore close errors

def save_data_upload(user_id: int, filename: str, original_filename: str, 
                    file_size: int, row_count: int, column_count: int,
                    file_hash: str, metadata: Dict[str, Any]) -> DataUpload:
    """
    Save data upload information to database
    
    Parameters:
    user_id - The ID of the user who uploaded the data
    filename - The filename to save the data as
    original_filename - The original filename of the uploaded file
    file_size - The size of the file in bytes
    row_count - The number of rows in the dataset
    column_count - The number of columns in the dataset
    file_hash - A hash of the file contents for detecting duplicates
    metadata - Dictionary containing metadata about the upload (will be stored as file_metadata)
    
    Returns:
    DataUpload object if successful, None otherwise
    """
    # Establish a fresh database connection
    db_connection = init_database()
    if not db_connection:
        st.error("Database connection failed")
        return None
    
    session = db_connection.get_session()
    if not session:
        st.error("Failed to create database session")
        return None
    
    try:
        # Create new data upload record
        data_upload = DataUpload(
            user_id=user_id,
            filename=filename,
            original_filename=original_filename,
            file_size=file_size,
            row_count=row_count,
            column_count=column_count,
            file_hash=file_hash,
            file_metadata=json.dumps(metadata)
        )
        
        session.add(data_upload)
        session.commit()
        
        # Detach object from session to avoid issues later
        session.expunge(data_upload)
        
        return data_upload
    except Exception as e:
        if session:
            try:
                session.rollback()
            except Exception:
                pass  # Already failed, just continue
        st.error(f"Error saving data upload: {str(e)}")
        return None
    finally:
        if session:
            try:
                session.close()
            except Exception:
                pass  # Ignore close errors

def save_analysis(user_id: int, data_upload_id: int, analysis_type: str,
                 analysis_method: str, parameters: Dict[str, Any],
                 results: Dict[str, Any], execution_time: float,
                 is_successful: bool = True, error_message: str = None) -> Analysis:
    """Save analysis results to database"""
    db_connection = init_database()
    if not db_connection:
        return None
    
    session = db_connection.get_session()
    try:
        # Create new analysis record
        analysis = Analysis(
            user_id=user_id,
            data_upload_id=data_upload_id,
            analysis_type=analysis_type,
            analysis_method=analysis_method,
            parameters=json.dumps(parameters),
            results=json.dumps(results),
            execution_time=execution_time,
            is_successful=is_successful,
            error_message=error_message
        )
        
        session.add(analysis)
        session.commit()
        
        return analysis
    except Exception as e:
        session.rollback()
        st.error(f"Error saving analysis: {str(e)}")
        return None
    finally:
        session.close()

def save_visualization(analysis_id: int, visualization_type: str,
                      title: str, description: str, configuration: Dict[str, Any],
                      image_data: str = None) -> Visualization:
    """Save visualization metadata to database"""
    db_connection = init_database()
    if not db_connection:
        return None
    
    session = db_connection.get_session()
    try:
        # Create new visualization record
        visualization = Visualization(
            analysis_id=analysis_id,
            visualization_type=visualization_type,
            title=title,
            description=description,
            configuration=json.dumps(configuration),
            image_data=image_data
        )
        
        session.add(visualization)
        session.commit()
        
        return visualization
    except Exception as e:
        session.rollback()
        st.error(f"Error saving visualization: {str(e)}")
        return None
    finally:
        session.close()

def get_user_uploads(user_id: int) -> List[DataUpload]:
    """Get all data uploads for a user"""
    db_connection = init_database()
    if not db_connection:
        st.error("Database connection failed")
        return []
    
    session = db_connection.get_session()
    if not session:
        st.error("Failed to create database session")
        return []
    
    try:
        # Load data with eager loading of relationships to avoid detached session issues
        uploads = session.query(DataUpload).options(
            # Eagerly load related data to avoid "not bound to session" errors
            joinedload(DataUpload.user),
            joinedload(DataUpload.analyses)
        ).filter(DataUpload.user_id == user_id).all()
        
        # Create a detached copy of the results to avoid session issues
        result = []
        for upload in uploads:
            # Make sure we have all the data before closing session
            if upload.file_metadata:
                # Trigger access to ensure data is loaded
                json.loads(upload.file_metadata) 
            
            # Expunge to detach from session
            session.expunge(upload)
            result.append(upload)
            
        return result
    except Exception as e:
        st.error(f"Error retrieving user uploads: {str(e)}")
        return []
    finally:
        if session:
            try:
                session.close()
            except Exception:
                pass  # Ignore close errors

def get_user_analyses(user_id: int) -> List[Analysis]:
    """Get all analyses for a user"""
    db_connection = init_database()
    if not db_connection:
        st.error("Database connection failed")
        return []
    
    session = db_connection.get_session()
    if not session:
        st.error("Failed to create database session")
        return []
    
    try:
        # Load data with eager loading of relationships to avoid detached session issues
        analyses = session.query(Analysis).options(
            # Eagerly load related data to avoid "not bound to session" errors
            joinedload(Analysis.user),
            joinedload(Analysis.data_upload),
            joinedload(Analysis.visualizations)
        ).filter(Analysis.user_id == user_id).all()
        
        # Create a detached copy of the results to avoid session issues
        result = []
        for analysis in analyses:
            # Make sure we have all the data before closing session
            if analysis.parameters:
                # Trigger access to ensure data is loaded
                json.loads(analysis.parameters)
            if analysis.results:
                json.loads(analysis.results)
            
            # Expunge to detach from session
            session.expunge(analysis)
            result.append(analysis)
            
        return result
    except Exception as e:
        st.error(f"Error retrieving user analyses: {str(e)}")
        return []
    finally:
        if session:
            try:
                session.close()
            except Exception:
                pass  # Ignore close errors

def get_analysis_visualizations(analysis_id: int) -> List[Visualization]:
    """Get all visualizations for an analysis"""
    db_connection = init_database()
    if not db_connection:
        st.error("Database connection failed")
        return []
    
    session = db_connection.get_session()
    if not session:
        st.error("Failed to create database session")
        return []
    
    try:
        # Load data with eager loading of relationships
        visualizations = session.query(Visualization).options(
            joinedload(Visualization.analysis)
        ).filter(Visualization.analysis_id == analysis_id).all()
        
        # Create a detached copy of the results to avoid session issues
        result = []
        for viz in visualizations:
            # Make sure we have all the data before closing session
            if viz.configuration:
                # Trigger access to ensure data is loaded
                json.loads(viz.configuration)
            
            # Expunge to detach from session
            session.expunge(viz)
            result.append(viz)
            
        return result
    except Exception as e:
        st.error(f"Error retrieving analysis visualizations: {str(e)}")
        return []
    finally:
        if session:
            try:
                session.close()
            except Exception:
                pass  # Ignore close errors