from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
load_dotenv()
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import os
import pandas as pd
import numpy as np
from datetime import datetime
import json
import re
import psycopg2
from psycopg2.extras import RealDictCursor
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import logging
from functools import wraps
import hashlib
import uuid
from dateutil.parser import parse as parse_date
import io
import math
import traceback
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
# Fix Windows console encoding for emoji/unicode in log messages
import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'clinical-data-pipeline-secret-key')

# Use absolute path for SQLite so it always persists in the instance folder
# regardless of working directory
_BASE_DIR = os.path.abspath(os.path.dirname(__file__))
_INSTANCE_DIR = os.path.join(_BASE_DIR, 'instance')
os.makedirs(_INSTANCE_DIR, exist_ok=True)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(_INSTANCE_DIR, "clinical_pipeline.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure required directories exist on startup
os.makedirs('uploads', exist_ok=True)
os.makedirs('uploads/tmp_chunks', exist_ok=True)

db = SQLAlchemy(app)

def normalize_date_bulk(series):
    """Ultra-fast bulk date normalization using vectorized operations"""
    if series is None or len(series) == 0:
        return series
    
    # Convert to string and strip
    series = series.astype(str).str.strip()
    
    # Handle empty/nan values
    mask = (series == '') | (series == 'nan') | (series == 'None')
    series = series.mask(mask, None)
    
    # Fast regex patterns for common formats
    # YYYY-MM-DD
    ymd_mask = series.str.match(r'^\d{4}-\d{2}-\d{2}$', na=False)
    series = series.mask(ymd_mask, series)
    
    # DD-MM-YYYY
    dmy_mask = series.str.match(r'^\d{2}-\d{2}-\d{4}$', na=False)
    if dmy_mask.any():
        dmy_series = series[dmy_mask]
        try:
            parsed = pd.to_datetime(dmy_series, format='%d-%m-%Y', errors='coerce')
            series = series.mask(dmy_mask, parsed.dt.strftime('%Y-%m-%d'))
        except:
            pass
    
    # MM/DD/YYYY
    mdy_mask = series.str.match(r'^\d{2}/\d{2}/\d{4}$', na=False)
    if mdy_mask.any():
        mdy_series = series[mdy_mask]
        try:
            parsed = pd.to_datetime(mdy_series, format='%m/%d/%Y', errors='coerce')
            series = series.mask(mdy_mask, parsed.dt.strftime('%Y-%m-%d'))
        except:
            pass
    
    # DD/MM/YYYY
    ddm_mask = series.str.match(r'^\d{2}/\d{2}/\d{4}$', na=False)
    if ddm_mask.any():
        ddm_series = series[ddm_mask]
        try:
            parsed = pd.to_datetime(ddm_series, format='%d/%m/%Y', errors='coerce')
            series = series.mask(ddm_mask, parsed.dt.strftime('%Y-%m-%d'))
        except:
            pass
    
    return series

def normalize_date_fast(date_value):
    """Fast date normalization with optimized parsing"""
    if pd.isna(date_value) or date_value == '' or date_value is None:
        return None
    
    try:
        # Handle different date formats automatically
        if isinstance(date_value, str):
            original_value = date_value.strip()
            
            # Fast path: Check for common patterns first
            # YYYY-MM-DD format
            if re.match(r'^\d{4}-\d{2}-\d{2}$', original_value):
                return original_value
            # DD-MM-YYYY format
            elif re.match(r'^\d{2}-\d{2}-\d{4}$', original_value):
                try:
                    parsed_date = datetime.strptime(original_value, '%d-%m-%Y')
                    return parsed_date.strftime('%Y-%m-%d')
                except ValueError:
                    pass
            # MM/DD/YYYY format
            elif re.match(r'^\d{2}/\d{2}/\d{4}$', original_value):
                try:
                    parsed_date = datetime.strptime(original_value, '%m/%d/%Y')
                    return parsed_date.strftime('%Y-%m-%d')
                except ValueError:
                    pass
            # DD/MM/YYYY format
            elif re.match(r'^\d{2}/\d{2}/\d{4}$', original_value):
                try:
                    parsed_date = datetime.strptime(original_value, '%d/%m/%Y')
                    return parsed_date.strftime('%Y-%m-%d')
                except ValueError:
                    pass
            
            # Fallback to trying common formats
            date_formats = [
                '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%d/%m/%Y',
                '%Y/%m/%d', '%m-%d-%Y', '%Y%m%d', '%d-%b-%Y',
                '%d-%B-%Y', '%b %d %Y', '%d %b %Y', '%B %d %Y',
                '%d %B %Y', '%Y-%d-%m', '%d.%m.%Y', '%Y.%m.%d'
            ]
            
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(original_value, fmt)
                    return parsed_date.strftime('%Y-%m-%d')
                except ValueError:
                    continue
            
            # If no format worked, return original
            return original_value
            
        elif hasattr(date_value, 'strftime'):
            # Already a datetime object
            return date_value.strftime('%Y-%m-%d')
        else:
            # Try to convert to string and parse
            str_value = str(date_value)
            if str_value and str_value != 'nan':
                return normalize_date_fast(str_value)
            return None
            
    except Exception:
        return str(date_value) if date_value else None

def normalize_date(date_value):
    """Legacy normalize_date function for backward compatibility"""
    return normalize_date_fast(date_value)

AE_COLS = [
    "subject_id", "subjectid", "subject_no", "subject_number", "subject_code", "subj_id", "subj_no",
    "participant_id", "participant_no", "patient_id", "patient_no", "pat_id", "usubjid",
    "ae_id", "aeid", "ae_no", "ae_number", "adverse_event_id", "adverseevent_id",
    "adverse_event_no", "ae_record_id", "event_id",
    "pt_code", "ptcode", "meddra_pt_code", "meddra_code", "pt_meddra_code",
    "preferred_term_code", "ae_pt_code",
    "pt_name", "pt_term", "pt_description", "meddra_pt", "meddra_term",
    "preferred_term", "ae_term", "adverse_event_term",
    "severity", "ae_severity", "severity_grade", "ae_sev",
    "intensity", "ae_intensity", "toxicity_grade", "ctcae_grade",
    "related", "related_to_drug", "causality", "relationship",
    "ae_relationship", "relationship_to_treatment", "treatment_related",
    "drug_related", "causality_assessment",
    "serious", "is_serious", "sae_flag", "sae",
    "serious_flag", "serious_event", "seriousness", "is_sae",
    "ae_start_dt", "ae_start_date", "ae_onset_date", "onset_date",
    "event_start_date", "start_date", "ae_onset_dt", "aestdt"
]

SUBJECT_COLS = [
    "subject_id", "subjectid", "subject_no", "subject_number", "subject_code",
    "subj_id", "usubjid", "participant_id", "patient_id",
    "site_id", "siteid", "site_no", "site_number", "center_id",
    "centre_id", "investigator_site", "study_site",
    "sex", "gender", "biological_sex", "subject_sex",
    "dob", "date_of_birth", "birth_date", "birth_dt",
    "arm", "treatment_arm", "study_arm", "arm_name",
    "randomization_arm", "rand_arm", "cohort",
    "start_date", "study_start_date", "enrollment_date",
    "enrolled_date", "first_dose_date", "treatment_start_date",
    "rfstdtc"
]

LAB_COLS = [
    "subject_id", "subjectid", "subj_id", "usubjid", "participant_id", "patient_id",
    "visit_no", "visit_number", "visitnum", "visit_id", "visit",
    "test_code", "testcode", "lab_test_code", "lbtestcd", "analyte_code",
    "result_value", "result", "test_result", "lab_result",
    "lbresult", "lbres", "result_numeric",
    "units", "unit", "result_unit", "lab_unit", "lborresu",
    "ref_low", "reference_low", "normal_low", "lower_limit",
    "lbnrlo", "ref_range_low",
    "ref_high", "reference_high", "normal_high", "upper_limit",
    "lbnrhi", "ref_range_high",
    "result_date", "test_date", "lab_date", "collection_date",
    "specimen_date", "lbdtc"
]

class DatabaseConnection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(500), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_connected = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)

class UploadedFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(50), nullable=False)  # subjects, labs, aes
    file_size = db.Column(db.Integer)
    record_count = db.Column(db.Integer)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    processed = db.Column(db.Boolean, default=False)
    processing_status = db.Column(db.String(100))
    error_message = db.Column(db.Text)
    database_id = db.Column(db.Integer, db.ForeignKey('database_connection.id'))

class UnprocessedData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    file_id = db.Column(db.Integer, db.ForeignKey('uploaded_file.id'))
    table_name = db.Column(db.String(100))
    row_data = db.Column(db.Text)  # JSON string
    error_reason = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class PipelineStatus(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    database_id = db.Column(db.Integer, db.ForeignKey('database_connection.id'))
    pipeline_stage = db.Column(db.String(50))  # bronze, silver, gold, prediction
    status = db.Column(db.String(20))  # running, completed, failed
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    error_message = db.Column(db.Text)
    records_processed = db.Column(db.Integer, default=0)

# ── RBAC Models ────────────────────────────────────────────────────────────
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='data_analyst')
    # roles: admin, data_engineer, data_analyst
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# ── Pipeline Schedule Models ───────────────────────────────────────────────
class PipelineSchedule(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    database_id = db.Column(db.Integer, db.ForeignKey('database_connection.id'), nullable=False)
    schedule_type = db.Column(db.String(20), nullable=False)  # daily, weekly, monthly, custom
    cron_expression = db.Column(db.String(100), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_run = db.Column(db.DateTime, nullable=True)
    next_run = db.Column(db.DateTime, nullable=True)
    run_count = db.Column(db.Integer, default=0)

class ScheduleRun(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    schedule_id = db.Column(db.Integer, db.ForeignKey('pipeline_schedule.id'), nullable=False)
    database_id = db.Column(db.Integer, db.ForeignKey('database_connection.id'), nullable=False)
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String(20), default='running')  # running, completed, failed
    triggered_by = db.Column(db.String(20), default='schedule')  # schedule, manual
    error_message = db.Column(db.Text, nullable=True)

# ── Auth decorators (must be defined before any route uses them) ───────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated

def role_required(*roles):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if 'user_id' not in session:
                return redirect(url_for('login'))
            if session.get('user_role') not in roles:
                flash('Access denied: insufficient permissions.', 'danger')
                return redirect(url_for('home'))
            return f(*args, **kwargs)
        return decorated
    return decorator

@app.route('/api/database/fix-schema', methods=['POST'])
def fix_database_schema():
    """Force fix database schema for existing databases"""
    try:
        database_id = request.json.get('database_id') if request.is_json else request.form.get('database_id')
        
        if not database_id:
            return jsonify({'success': False, 'message': 'Database ID required'})
        
        db_conn = DatabaseConnection.query.get(database_id)
        if not db_conn:
            return jsonify({'success': False, 'message': 'Database not found'})
        
        logger.info(f"Force fixing schema for database: {db_conn.name}")
        
        # Connect to database and fix schema
        conn = psycopg2.connect(db_conn.url)
        conn.autocommit = True  # Enable autocommit to avoid transaction issues
        cursor = conn.cursor()
        
        # Create uploaded_files table if it doesn't exist
        try:
            cursor.execute("SELECT * FROM public.uploaded_files LIMIT 1")
            logger.info("uploaded_files table already exists")
        except psycopg2.errors.UndefinedTable:
            logger.info("Creating uploaded_files table...")
            cursor.execute("""
                CREATE TABLE public.uploaded_files (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    original_filename VARCHAR(255) NOT NULL,
                    file_type VARCHAR(50) NOT NULL,
                    database_id INTEGER NOT NULL,
                    file_size BIGINT NOT NULL,
                    record_count INTEGER DEFAULT 0,
                    processed BOOLEAN DEFAULT TRUE,
                    bronze_loaded BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            logger.info("Created uploaded_files table")
        
        # Ensure all required columns exist (for existing tables)
        cursor.execute("""
            ALTER TABLE public.uploaded_files 
            ADD COLUMN IF NOT EXISTS record_count INTEGER DEFAULT 0
        """)
        cursor.execute("""
            ALTER TABLE public.uploaded_files 
            ADD COLUMN IF NOT EXISTS processed BOOLEAN DEFAULT TRUE
        """)
        cursor.execute("""
            ALTER TABLE public.uploaded_files 
            ADD COLUMN IF NOT EXISTS bronze_loaded BOOLEAN DEFAULT TRUE
        """)
        cursor.execute("""
            ALTER TABLE public.uploaded_files 
            ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        """)
        cursor.execute("""
            ALTER TABLE public.uploaded_files 
            ADD COLUMN IF NOT EXISTS file_size BIGINT NOT NULL DEFAULT 0
        """)
        
        # Fix phi_site_summary table
        try:
            cursor.execute("SELECT database_id FROM gold.phi_site_summary LIMIT 1")
            logger.info("phi_site_summary already has database_id")
        except psycopg2.errors.UndefinedColumn:
            logger.info("Adding database_id to phi_site_summary...")
            cursor.execute("ALTER TABLE gold.phi_site_summary ADD COLUMN database_id INTEGER")
            # Update existing rows with database_id
            cursor.execute("UPDATE gold.phi_site_summary SET database_id = %s WHERE database_id IS NULL", (database_id,))
            logger.info("Fixed phi_site_summary table")
        
        # Fix site_predictions table
        try:
            cursor.execute("SELECT database_id FROM prediction.site_predictions LIMIT 1")
            logger.info("site_predictions already has database_id")
        except psycopg2.errors.UndefinedColumn:
            logger.info("Adding database_id to site_predictions...")
            cursor.execute("ALTER TABLE prediction.site_predictions ADD COLUMN database_id INTEGER")
            # Update existing rows with database_id
            cursor.execute("UPDATE prediction.site_predictions SET database_id = %s WHERE database_id IS NULL", (database_id,))
            logger.info("Fixed site_predictions table")
        
        cursor.close()
        conn.close()
        
        logger.info("Database schema fixed successfully!")
        return jsonify({'success': True, 'message': 'Database schema fixed successfully'})
        
    except Exception as e:
        logger.error(f"Error fixing database schema: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

def update_database_schema(connection_string):
    """Update existing database schema with missing database_id columns"""
    try:
        logger.info(f"Updating database schema: {connection_string}")
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        
        # Check if phi_site_summary table needs database_id column
        try:
            cursor.execute("SELECT database_id FROM gold.phi_site_summary LIMIT 1")
            logger.info("phi_site_summary already has database_id column")
        except psycopg2.errors.UndefinedColumn:
            logger.info("Adding database_id column to phi_site_summary...")
            cursor.execute("ALTER TABLE gold.phi_site_summary ADD COLUMN database_id INTEGER")
            cursor.execute("ALTER TABLE gold.phi_site_summary ADD PRIMARY KEY (database_id, site_id)")
            logger.info("Added database_id column to phi_site_summary")
        
        # Check if site_predictions table needs database_id column
        try:
            cursor.execute("SELECT database_id FROM prediction.site_predictions LIMIT 1")
            logger.info("site_predictions already has database_id column")
        except psycopg2.errors.UndefinedColumn:
            logger.info("Adding database_id column to site_predictions...")
            cursor.execute("ALTER TABLE prediction.site_predictions ADD COLUMN database_id INTEGER")
            cursor.execute("ALTER TABLE prediction.site_predictions ADD PRIMARY KEY (database_id, site_id, prediction_date)")
            logger.info("Added database_id column to site_predictions")
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Database schema updated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error updating database schema: {str(e)}")
        return False

def create_database_schemas(connection_string):
    """Create all required schemas and tables in the database"""
    try:
        logger.info(f"Attempting to connect to database: {connection_string}")
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        logger.info("Database connection established successfully")
        
        schemas = ['bronze', 'silver', 'gold', 'prediction']
        
        for schema in schemas:
            cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            logger.info(f"Created schema: {schema}")
        
        # Bronze tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bronze.bronze_subjects (
                database_id INTEGER,
                subject_id TEXT,
                site_id TEXT,
                sex TEXT,
                dob DATE,
                arm TEXT,
                start_date DATE,
                raw_data JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bronze.bronze_labs (
                database_id INTEGER,
                subject_id TEXT,
                visitno TEXT,
                testcode TEXT,
                resultvalue TEXT,
                units TEXT,
                reflow TEXT,
                refhigh TEXT,
                resultdate TEXT,
                raw_data JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bronze.bronze_aes (
                database_id INTEGER,
                subject_id TEXT,
                ae_id TEXT,
                pt_code TEXT,
                pt_name TEXT,
                severity TEXT,
                related TEXT,
                serious TEXT,
                ae_start_date TEXT,
                raw_data JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Silver tables
        # Silver tables — canonical schema matching process_silver_layer inserts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS silver.silver_subjects (
                id SERIAL PRIMARY KEY,
                database_id INTEGER NOT NULL,
                subject_hash_id TEXT NOT NULL,
                subject_id_raw TEXT,
                site_id TEXT,
                sex TEXT,
                dob DATE,
                arm TEXT,
                start_date DATE,
                age_at_start INTEGER,
                ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS silver.silver_labs (
                lab_id SERIAL PRIMARY KEY,
                database_id INTEGER NOT NULL,
                subject_hash_id TEXT NOT NULL,
                subject_id TEXT,
                visit_no TEXT,
                test_code TEXT,
                result_value FLOAT,
                ref_low FLOAT,
                ref_high FLOAT,
                units TEXT,
                result_date DATE,
                result_status TEXT,
                data_quality_flag TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS silver.silver_ae (
                id SERIAL PRIMARY KEY,
                database_id INTEGER NOT NULL,
                subject_hash_id TEXT NOT NULL,
                subject_id TEXT,
                ae_id TEXT,
                pt_code TEXT,
                pt_name TEXT,
                severity TEXT,
                related TEXT,
                serious_flag TEXT,
                ae_start_date DATE,
                data_quality_flag TEXT,
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Gold tables — canonical schema (no fact_* tables)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS gold.gold_subjects (
                database_id INTEGER NOT NULL,
                subject_hash_id TEXT NOT NULL,
                site_id TEXT,
                sex TEXT,
                age_at_start INTEGER,
                age_group TEXT,
                arm TEXT,
                start_date DATE,
                study_year INTEGER,
                study_month INTEGER,
                ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS gold.gold_labs (
                lab_fact_id SERIAL PRIMARY KEY,
                database_id INTEGER NOT NULL,
                subject_hash_id TEXT NOT NULL,
                visit_no TEXT,
                test_code TEXT,
                result_value FLOAT,
                ref_low FLOAT,
                ref_high FLOAT,
                result_status TEXT,
                result_date DATE,
                units TEXT,
                uln_ratio FLOAT,
                alt_3x_uln_flag BOOLEAN DEFAULT FALSE,
                safety_signal TEXT,
                study_day INTEGER,
                ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS gold.gold_ae (
                id SERIAL PRIMARY KEY,
                database_id INTEGER NOT NULL,
                subject_hash_id TEXT NOT NULL,
                ae_id TEXT,
                pt_code TEXT,
                pt_name TEXT,
                severity TEXT,
                related TEXT,
                serious_flag TEXT,
                ae_start_date DATE,
                severity_score INTEGER,
                safety_signal TEXT,
                ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Prediction tables — ae_predictions is intentionally excluded
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction.site_predictions (
                database_id INTEGER,
                site_id TEXT,
                prediction_date DATE,
                predicted_total_ae INTEGER,
                predicted_total_serious_events INTEGER,
                predicted_total_ae_signal_count INTEGER,
                predicted_avg_lab_ratio FLOAT,
                predicted_new_subjects INTEGER,
                predicted_risk_group TEXT,
                total_subjects INTEGER,
                total_aes INTEGER,
                risk_level TEXT,
                predicted_enrollments INTEGER,
                confidence_score FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS gold.phi_site_summary (
                database_id INTEGER,
                site_id TEXT,
                total_subjects INTEGER,
                total_lab_tests INTEGER,
                total_adverse_events INTEGER,
                total_serious_events INTEGER,
                total_ae_signal_count INTEGER,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (database_id, site_id)
            )
        """)

        conn.commit()
        cursor.close()
        conn.close()
        logger.info("All schemas and tables created successfully")

        # Drop ae_predictions table if it exists (no longer needed)
        try:
            _drop_conn = psycopg2.connect(connection_string)
            _drop_cur = _drop_conn.cursor()
            _drop_cur.execute("DROP TABLE IF EXISTS prediction.ae_predictions")
            _drop_conn.commit()
            _drop_cur.close()
            _drop_conn.close()
        except Exception:
            pass
        
        # Create SQLAlchemy tables (for UploadedFile and DatabaseConnection)
        with app.app_context():
            # Temporarily update SQLAlchemy URI to use PostgreSQL
            original_uri = app.config['SQLALCHEMY_DATABASE_URI']
            app.config['SQLALCHEMY_DATABASE_URI'] = connection_string
            db.create_all()
            # Restore original URI (for local SQLite operations)
            app.config['SQLALCHEMY_DATABASE_URI'] = original_uri
            logger.info("SQLAlchemy tables created successfully")
        
        # Explicitly create uploaded_files table in public schema if needed
        try:
            uf_conn = psycopg2.connect(connection_string)
            uf_cursor = uf_conn.cursor()
            uf_cursor.execute("""
                CREATE TABLE IF NOT EXISTS public.uploaded_files (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    original_filename VARCHAR(255) NOT NULL,
                    file_type VARCHAR(50) NOT NULL,
                    database_id INTEGER NOT NULL,
                    file_size BIGINT NOT NULL,
                    record_count INTEGER DEFAULT 0,
                    processed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            uf_conn.commit()
            uf_cursor.close()
            uf_conn.close()
            logger.info("Explicit uploaded_files table created in public schema")
        except Exception as e:
            logger.error(f"Error creating uploaded_files table: {str(e)}")
        
        return True
        
    except psycopg2.OperationalError as e:
        error_msg = str(e)
        if "connection to server" in error_msg.lower():
            logger.error(f"Database connection failed: Unable to connect to PostgreSQL server. Please check:")
            logger.error("  1. PostgreSQL is running on the specified host and port")
            logger.error("  2. The connection string format is correct")
            logger.error("  3. Firewall settings allow the connection")
            logger.error(f"  Connection string: {connection_string}")
        else:
            logger.error(f"Database operational error: {error_msg}")
        return False
    except psycopg2.Error as e:
        logger.error(f"PostgreSQL error: {str(e)}")
        logger.error(f"Connection string used: {connection_string}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error creating database schemas: {str(e)}")
        logger.error(f"Connection string used: {connection_string}")
        return False

def validate_columns(file_path, file_type):
    """Validate file columns against expected column lists"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xml'):
            df = pd.read_xml(file_path)
        elif file_path.endswith('.ndjson'):
            df = pd.read_json(file_path, lines=True)
        else:
            return False, "Unsupported file format"
        
        file_columns = df.columns.str.lower().str.strip()
        
        # Debug logging
        logger.info(f"Validating {file_type} file: {file_path}")
        logger.info(f"File columns: {file_columns.tolist()}")
        
        if file_type == 'subjects':
            # Check for at least one subject ID column (more flexible)
            subject_id_cols = ['subject_id', 'subjectid', 'subject_no', 'subject_number', 'subject_code', 'subj_id', 'usubjid', 'participant_id', 'patient_id', 'subject', 'subj', 'subjid']
            has_subject_id = any(col in file_columns.values for col in subject_id_cols)
            logger.info(f"Subject ID check: {has_subject_id} (looking for: {subject_id_cols})")
            
            # Check for at least one site ID column (more flexible)
            site_id_cols = ['site_id', 'siteid', 'site_no', 'site_number', 'center_id', 'centre_id', 'investigator_site', 'study_site', 'site', 'center']
            has_site_id = any(col in file_columns.values for col in site_id_cols)
            logger.info(f"Site ID check: {has_site_id} (looking for: {site_id_cols})")
            
            # Check for at least one sex column (more flexible)
            sex_cols = ['sex', 'gender', 'biological_sex', 'subject_sex', 'male_female', 'm_f']
            has_sex = any(col in file_columns.values for col in sex_cols)
            logger.info(f"Sex check: {has_sex} (looking for: {sex_cols})")
            
            # For subjects, only require subject_id (make site_id and sex optional)
            if has_subject_id:
                logger.info("Subjects file validation PASSED (has subject_id)")
                return True, "Subjects file is valid"
            else:
                logger.error("Subjects file validation FAILED - missing subject_id")
                return False, "Subjects file must contain a subject ID column"
            
            # Check for at least one DOB column
            dob_cols = ['dob', 'date_of_birth', 'birth_date', 'birth_dt']
            has_dob = any(col in file_columns.values for col in dob_cols)
            logger.info(f"DOB check: {has_dob} (looking for: {dob_cols})")
            
            # Check for at least one arm column
            arm_cols = ['arm', 'treatment_arm', 'study_arm', 'arm_name', 'randomization_arm', 'rand_arm', 'cohort']
            has_arm = any(col in file_columns.values for col in arm_cols)
            logger.info(f"Arm check: {has_arm} (looking for: {arm_cols})")
            
            # Check for at least one start date column
            start_date_cols = ['start_date', 'study_start_date', 'enrollment_date', 'enrolled_date', 'first_dose_date', 'treatment_start_date', 'rfstdtc']
            has_start_date = any(col in file_columns.values for col in start_date_cols)
            logger.info(f"Start date check: {has_start_date} (looking for: {start_date_cols})")
            
            missing_requirements = []
            if not has_subject_id: missing_requirements.append("subject identifier (subject_id, subjectid, etc.)")
            if not has_site_id: missing_requirements.append("site identifier (site_id, siteid, etc.)")
            if not has_sex: missing_requirements.append("sex/gender column")
            if not has_dob: missing_requirements.append("date of birth column")
            if not has_arm: missing_requirements.append("treatment arm column")
            if not has_start_date: missing_requirements.append("start date column")
            
        elif file_type == 'labs':
            # Check for at least one subject ID column (more flexible)
            subject_id_cols = ['subject_id', 'subjectid', 'subj_id', 'usubjid', 'participant_id', 'patient_id', 'subject', 'patient', 'subjid', 'SubjectID']
            has_subject_id = any(col in file_columns.values for col in subject_id_cols)
            logger.info(f"Subject ID check: {has_subject_id} (looking for: {subject_id_cols})")
            
            # Check for at least one visit column (more flexible)
            visit_cols = ['visitno', 'visit_no', 'visit_number', 'visit', 'visitname', 'visit_name', 'visit_id', 'VisitNo']
            has_visit = any(col in file_columns.values for col in visit_cols)
            logger.info(f"Visit check: {has_visit} (looking for: {visit_cols})")
            
            # Check for at least one test code column (more flexible)
            testcode_cols = ['testcode', 'test_code', 'test_name', 'lab_test', 'test', 'parameter', 'parameter_cd', 'paramcd', 'TestCode']
            has_testcode = any(col in file_columns.values for col in testcode_cols)
            logger.info(f"Test code check: {has_testcode} (looking for: {testcode_cols})")
            
            # For labs, only require subject_id (make others optional)
            if has_subject_id:
                logger.info("Labs file validation PASSED (has subject_id)")
                return True, "Labs file is valid"
            else:
                logger.error("Labs file validation FAILED - missing subject_id")
                return False, "Labs file must contain a subject ID column"
            visit_cols = ['visit_no', 'visit_number', 'visitnum', 'visit_id', 'visit', 'visitno', 'visitnumber', 'visit_num']
            has_visit = any(col in file_columns.values for col in visit_cols)
            logger.info(f"Visit check: {has_visit} (looking for: {visit_cols})")
            
            # Check for at least one test code column
            testcode_cols = ['test_code', 'testcode', 'lab_test_code', 'lbtestcd', 'analyte_code', 'test', 'test_name', 'analyte']
            has_testcode = any(col in file_columns.values for col in testcode_cols)
            logger.info(f"Test code check: {has_testcode} (looking for: {testcode_cols})")
            
            # Check for at least one result value column - EXTENDED LIST
            result_cols = ['result_value', 'result', 'test_result', 'lab_result', 'lbresult', 'lbres', 'result_numeric', 'value', 'result_val', 'resultvalue', 'test_value', 'lab_value', 'measurement', 'reading']
            has_result = any(col in file_columns.values for col in result_cols)
            logger.info(f"Result value check: {has_result} (looking for: {result_cols})")
            
            # Check for at least one units column
            units_cols = ['units', 'unit', 'result_unit', 'lab_unit', 'lborresu', 'unit_of_measure']
            has_units = any(col in file_columns.values for col in units_cols)
            logger.info(f"Units check: {has_units} (looking for: {units_cols})")
            
            # Check for result date column - EXTENDED LIST
            date_cols = ['result_date', 'test_date', 'lab_date', 'collection_date', 'specimen_date', 'lbdtc', 'date', 'test_datetime', 'resultdate', 'testdate', 'labdate', 'collectiondate', 'specimendate']
            has_date = any(col in file_columns.values for col in date_cols)
            logger.info(f"Date check: {has_date} (looking for: {date_cols})")
            
            # Fallback: try fuzzy matching if exact matching fails
            if not has_result:
                logger.warning("Exact result value matching failed, trying fuzzy matching")
                for col in file_columns.values:
                    if any(keyword in col for keyword in ['result', 'value', 'test', 'lab', 'measurement', 'reading']):
                        logger.info(f"Fuzzy matched result column: {col}")
                        has_result = True
                        break
            
            if not has_date:
                logger.warning("Exact date matching failed, trying fuzzy matching")
                for col in file_columns.values:
                    if any(keyword in col for keyword in ['date', 'time', 'dt', 'when', 'collected', 'specimen']):
                        logger.info(f"Fuzzy matched date column: {col}")
                        has_date = True
                        break
            
            missing_requirements = []
            if not has_subject_id: missing_requirements.append("subject identifier")
            if not has_visit: missing_requirements.append("visit number")
            if not has_testcode: missing_requirements.append("test code")
            if not has_result: missing_requirements.append("result value")
            if not has_units: missing_requirements.append("units")
            if not has_date: missing_requirements.append("result date")
            
        elif file_type == 'aes':
            # Check for at least one subject ID column (more flexible)
            subject_id_cols = ['subject_id', 'subjectid', 'subj_id', 'usubjid', 'participant_id', 'patient_id', 'subject', 'patient', 'subjid', 'SubjectID']
            has_subject_id = any(col in file_columns.values for col in subject_id_cols)
            logger.info(f"Subject ID check: {has_subject_id} (looking for: {subject_id_cols})")
            
            # For AES, only require subject_id (make others optional)
            if has_subject_id:
                logger.info("AES file validation PASSED (has subject_id)")
                return True, "AES file is valid"
            else:
                logger.error("AES file validation FAILED - missing subject_id")
                return False, "AES file must contain a subject ID column"
            
            # Check for at least one AE ID column
            ae_id_cols = ['ae_id', 'aeid', 'ae_no', 'ae_number', 'adverse_event_id', 'adverseevent_id', 'adverse_event_no', 'ae_record_id', 'event_id']
            has_ae_id = any(col in file_columns.values for col in ae_id_cols)
            
            # Check for preferred term columns
            pt_code_cols = ['pt_code', 'ptcode', 'meddra_pt_code', 'meddra_code', 'pt_meddra_code', 'preferred_term_code', 'ae_pt_code']
            pt_name_cols = ['pt_name', 'pt_term', 'pt_description', 'meddra_pt', 'meddra_term', 'preferred_term', 'ae_term', 'adverse_event_term']
            has_pt = any(col in file_columns.values for col in pt_code_cols + pt_name_cols)
            
            # Check for severity column
            severity_cols = ['severity', 'ae_severity', 'severity_grade', 'ae_sev', 'intensity', 'ae_intensity', 'toxicity_grade', 'ctcae_grade']
            has_severity = any(col in file_columns.values for col in severity_cols)
            
            # Check for relatedness column
            related_cols = ['related', 'related_to_drug', 'causality', 'relationship', 'ae_relationship', 'relationship_to_treatment', 'treatment_related', 'drug_related', 'causality_assessment']
            has_related = any(col in file_columns.values for col in related_cols)
            
            # Check for seriousness column
            serious_cols = ['serious', 'is_serious', 'sae_flag', 'sae', 'serious_flag', 'serious_event', 'seriousness', 'is_sae']
            has_serious = any(col in file_columns.values for col in serious_cols)
            
            # Check for start date column
            start_date_cols = ['ae_start_dt', 'ae_start_date', 'ae_onset_date', 'onset_date', 'event_start_date', 'start_date', 'ae_onset_dt', 'aestdt']
            has_start_date = any(col in file_columns.values for col in start_date_cols)
            
            missing_requirements = []
            if not has_subject_id: missing_requirements.append("subject identifier")
            if not has_ae_id: missing_requirements.append("adverse event identifier")
            if not has_pt: missing_requirements.append("preferred term (pt_code or pt_name)")
            if not has_severity: missing_requirements.append("severity")
            if not has_related: missing_requirements.append("relatedness")
            if not has_serious: missing_requirements.append("seriousness")
            if not has_start_date: missing_requirements.append("start date")
            
        else:
            return False, "Invalid file type"
        
        logger.info(f"Missing requirements: {missing_requirements}")
        
        if missing_requirements:
            return False, f"Missing required columns: {', '.join(missing_requirements)}"
        
        return True, "Column validation passed"
        
    except Exception as e:
        logger.error(f"Error validating columns: {str(e)}")
        return False, f"Error validating columns: {str(e)}"

def normalize_visit_no(value):
    """Extract numeric part of visit number"""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.search(r'\d+', value)
        if match:
            return int(match.group())
    return None

def normalize_lab_units_row(row):
    """Normalize laboratory units to U/L"""
    unit = str(row.get("units", "")).strip().upper()
    
    conversion_factors = {
        "U/L": 1, "IU/L": 1, "UNITS/L": 1,
        "U/ML": 1000, "IU/ML": 1000,
        "U/DL": 10, "IU/DL": 10
    }
    
    if unit in conversion_factors:
        factor = conversion_factors[unit]
        try:
            row["resultvalue"] = float(row["resultvalue"]) * factor
        except:
            pass
        row["units"] = "U/L"
    
    return row

def generate_subject_pseudonym(subject_id):
    """Generate pseudonym for subject ID"""
    return hashlib.sha256(subject_id.encode()).hexdigest()[:16]

@app.route('/')
@login_required
def home():
    return render_template('index.html')

@app.route('/upload')
@login_required
@role_required('admin', 'data_engineer')
def upload():
    databases = DatabaseConnection.query.all()
    return render_template('upload.html', databases=databases)

@app.route('/file-history')
@login_required
@role_required('admin', 'data_engineer')
def file_history():
    return render_template('file_history.html')

@app.route('/unprocessed-files')
@login_required
@role_required('admin', 'data_engineer')
def unprocessed_files():
    return render_template('unprocessed_files.html')

@app.route('/results')
@login_required
@role_required('admin', 'data_engineer')
def results():
    return render_template('results.html')

@app.route('/api/results/data')
def get_results_data():
    """Get results data from database"""
    try:
        database_id = request.args.get('database_id') or current_database_session.get('database_id')
        if not database_id:
            return jsonify({'success': False, 'message': 'No database selected'})

        database_id = int(database_id)
        db_conn = DatabaseConnection.query.get(database_id)

        if not db_conn:
            return jsonify({'success': False, 'message': 'Database connection not found'})

        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        cursor = conn.cursor()

        # Bronze layer
        cursor.execute("SELECT COUNT(*) FROM bronze.bronze_subjects WHERE database_id = %s", (database_id,))
        bronze_subjects = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM bronze.bronze_labs WHERE database_id = %s", (database_id,))
        bronze_labs = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM bronze.bronze_aes WHERE database_id = %s", (database_id,))
        bronze_aes = cursor.fetchone()[0]

        # Silver layer (silver_ae not silver_aes)
        cursor.execute("SELECT COUNT(*) FROM silver.silver_subjects WHERE database_id = %s", (database_id,))
        silver_subjects = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM silver.silver_labs WHERE database_id = %s", (database_id,))
        silver_labs = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM silver.silver_ae WHERE database_id = %s", (database_id,))
        silver_aes = cursor.fetchone()[0]

        # Gold layer (gold_subjects/gold_labs/gold_ae not fact_*)
        cursor.execute("SELECT COUNT(*) FROM gold.gold_subjects WHERE database_id = %s", (database_id,))
        gold_subjects = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM gold.gold_labs WHERE database_id = %s", (database_id,))
        gold_labs = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM gold.gold_ae WHERE database_id = %s", (database_id,))
        gold_aes = cursor.fetchone()[0]

        # Prediction layer
        cursor.execute("SELECT COUNT(*) FROM prediction.site_predictions WHERE database_id = %s", (database_id,))
        predictions = cursor.fetchone()[0]

        cursor.close()
        conn.close()

        return jsonify({
            'success': True,
            'data': {
                'bronze': {'subjects': bronze_subjects, 'labs': bronze_labs, 'aes': bronze_aes},
                'silver': {'subjects': silver_subjects, 'labs': silver_labs, 'aes': silver_aes},
                'gold':   {'subjects': gold_subjects,   'labs': gold_labs,   'aes': gold_aes},
                'predictions': predictions
            }
        })

    except Exception as e:
        logger.error(f"Error getting results data: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/analytics')
@login_required
@role_required('admin', 'data_analyst', 'data_engineer')
def analytics():
    return render_template('analytics.html')

@app.route('/predictions')
@login_required
@role_required('admin', 'data_analyst', 'data_engineer')
def predictions():
    return render_template('predictions.html')

@app.route('/report')
@login_required
@role_required('admin', 'data_engineer')
def report():
    return render_template('report.html')

@app.route('/api/report/data')
def api_report_data():
    """Generate all data needed for the Clinical Safety Summary Report"""
    try:
        database_id = request.args.get('database_id')
        if not database_id:
            database_id = current_database_session.get('database_id')
        if not database_id:
            return jsonify({'success': False, 'message': 'No database selected'})
        database_id = int(database_id)

        db_conn = DatabaseConnection.query.get(database_id)
        if not db_conn:
            return jsonify({'success': False, 'message': 'Database not found'})

        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        cursor = conn.cursor()

        def q(sql, params=()):
            try:
                cursor.execute(sql, params)
                return cursor.fetchall()
            except Exception:
                conn.rollback()
                return []

        def q1(sql, params=()):
            rows = q(sql, params)
            return rows[0][0] if rows and rows[0][0] is not None else 0

        # ── Study Overview ────────────────────────────────────────────────────
        total_subjects  = q1("SELECT COUNT(*) FROM gold.gold_subjects WHERE database_id=%s", (database_id,))
        total_sites     = q1("SELECT COUNT(DISTINCT site_id) FROM gold.gold_subjects WHERE database_id=%s", (database_id,))
        arms_rows       = q("SELECT DISTINCT arm FROM gold.gold_subjects WHERE database_id=%s AND arm IS NOT NULL", (database_id,))
        arms            = [r[0] for r in arms_rows]
        study_start     = q("SELECT MIN(start_date) FROM gold.gold_subjects WHERE database_id=%s", (database_id,))
        study_end       = q("SELECT MAX(start_date) FROM gold.gold_subjects WHERE database_id=%s", (database_id,))
        study_start_dt  = study_start[0][0].strftime('%Y-%m-%d') if study_start and study_start[0][0] else 'N/A'
        study_end_dt    = study_end[0][0].strftime('%Y-%m-%d')   if study_end   and study_end[0][0]   else 'N/A'

        # ── Demographics ──────────────────────────────────────────────────────
        male_count   = q1("SELECT COUNT(*) FROM gold.gold_subjects WHERE database_id=%s AND sex='MALE'",   (database_id,))
        female_count = q1("SELECT COUNT(*) FROM gold.gold_subjects WHERE database_id=%s AND sex='FEMALE'", (database_id,))
        age_rows     = q("SELECT AVG(age_at_start), MIN(age_at_start), MAX(age_at_start) FROM gold.gold_subjects WHERE database_id=%s AND age_at_start IS NOT NULL", (database_id,))
        avg_age      = round(float(age_rows[0][0]), 1) if age_rows and age_rows[0][0] else 0
        min_age      = int(age_rows[0][1]) if age_rows and age_rows[0][1] else 0
        max_age      = int(age_rows[0][2]) if age_rows and age_rows[0][2] else 0
        subjects_per_site = q("SELECT site_id, COUNT(*) FROM gold.gold_subjects WHERE database_id=%s GROUP BY site_id ORDER BY site_id", (database_id,))

        # ── Lab Summary ───────────────────────────────────────────────────────
        total_labs    = q1("SELECT COUNT(*) FROM gold.gold_labs WHERE database_id=%s", (database_id,))
        normal_labs   = q1("SELECT COUNT(*) FROM gold.gold_labs WHERE database_id=%s AND result_status='NORMAL'", (database_id,))
        abnormal_labs = q1("SELECT COUNT(*) FROM gold.gold_labs WHERE database_id=%s AND result_status IN ('HIGH','LOW')", (database_id,))
        safety_labs   = q1("SELECT COUNT(*) FROM gold.gold_labs WHERE database_id=%s AND safety_signal IS NOT NULL AND safety_signal!=''", (database_id,))
        alt_count     = q1("SELECT COUNT(*) FROM gold.gold_labs WHERE database_id=%s AND test_code='ALT'", (database_id,))
        ast_count     = q1("SELECT COUNT(*) FROM gold.gold_labs WHERE database_id=%s AND test_code='AST'", (database_id,))
        lab_by_test   = q("SELECT test_code, COUNT(*) FROM gold.gold_labs WHERE database_id=%s GROUP BY test_code ORDER BY COUNT(*) DESC LIMIT 10", (database_id,))

        # ── Safety Flags ──────────────────────────────────────────────────────
        alt_3x   = q1("SELECT COUNT(DISTINCT subject_hash_id) FROM gold.gold_labs WHERE database_id=%s AND test_code='ALT' AND uln_ratio>=3", (database_id,))
        ast_3x   = q1("SELECT COUNT(DISTINCT subject_hash_id) FROM gold.gold_labs WHERE database_id=%s AND test_code='AST' AND uln_ratio>=3", (database_id,))
        bili_2x  = q1("SELECT COUNT(DISTINCT subject_hash_id) FROM gold.gold_labs WHERE database_id=%s AND test_code='TBILI' AND uln_ratio>=2", (database_id,))
        hys_law  = q1("""SELECT COUNT(DISTINCT a.subject_hash_id)
            FROM gold.gold_labs a JOIN gold.gold_labs b ON a.subject_hash_id=b.subject_hash_id AND a.database_id=b.database_id
            WHERE a.database_id=%s AND a.test_code='ALT' AND a.uln_ratio>=3 AND b.test_code='TBILI' AND b.uln_ratio>=2""", (database_id,))
        total_safety_alerts = alt_3x + ast_3x + bili_2x + hys_law

        # ── AE Summary ────────────────────────────────────────────────────────
        total_aes    = q1("SELECT COUNT(*) FROM gold.gold_ae WHERE database_id=%s", (database_id,))
        serious_aes  = q1("SELECT COUNT(*) FROM gold.gold_ae WHERE database_id=%s AND serious_flag='Y'", (database_id,))
        related_aes  = q1("SELECT COUNT(*) FROM gold.gold_ae WHERE database_id=%s AND related='Y'", (database_id,))
        moderate_aes = q1("SELECT COUNT(*) FROM gold.gold_ae WHERE database_id=%s AND severity='MODERATE'", (database_id,))
        severe_aes   = q1("SELECT COUNT(*) FROM gold.gold_ae WHERE database_id=%s AND severity='SEVERE'", (database_id,))
        top_ae_rows  = q("SELECT COALESCE(pt_name,pt_code,'Unknown'), COUNT(*) FROM gold.gold_ae WHERE database_id=%s GROUP BY 1 ORDER BY 2 DESC LIMIT 1", (database_id,))
        most_common_ae = top_ae_rows[0][0] if top_ae_rows else 'N/A'

        # ── Site Level Summary ────────────────────────────────────────────────
        site_summary = q("""
            SELECT gs.site_id,
                   COUNT(DISTINCT gs.subject_hash_id) AS subjects,
                   COUNT(DISTINCT ga.id) AS total_aes,
                   COUNT(DISTINCT CASE WHEN ga.serious_flag='Y' THEN ga.id END) AS serious_aes,
                   COUNT(DISTINCT CASE WHEN gl.safety_signal IS NOT NULL AND gl.safety_signal!='' THEN gl.lab_fact_id END) AS safety_flags
            FROM gold.gold_subjects gs
            LEFT JOIN gold.gold_ae ga ON gs.subject_hash_id=ga.subject_hash_id AND gs.database_id=ga.database_id
            LEFT JOIN gold.gold_labs gl ON gs.subject_hash_id=gl.subject_hash_id AND gs.database_id=gl.database_id
            WHERE gs.database_id=%s
            GROUP BY gs.site_id ORDER BY gs.site_id
        """, (database_id,))

        # ── Drug Safety Assessment ────────────────────────────────────────────
        ae_rate = (total_aes / total_subjects * 100) if total_subjects else 0
        serious_rate = (serious_aes / total_subjects * 100) if total_subjects else 0
        if alt_3x > 5 or hys_law > 0 or serious_rate > 10:
            overall_safety = 'High Risk'
            risk_level = 'HIGH'
        elif alt_3x > 0 or ast_3x > 0 or serious_rate > 5:
            overall_safety = 'Moderate Risk'
            risk_level = 'MODERATE'
        else:
            overall_safety = 'Safe'
            risk_level = 'LOW'

        liver_tox = 'Observed' if (alt_3x > 0 or ast_3x > 0) else 'Not Observed'

        if risk_level == 'HIGH':
            safety_interp = 'Significant safety signals detected. Elevated liver enzymes and/or Hy\'s Law cases require immediate clinical review.'
            safety_rec    = 'Immediate review of subjects with ALT/AST ≥3×ULN. Consider dose reduction or study suspension pending safety review.'
        elif risk_level == 'MODERATE':
            safety_interp = 'Mild to moderate safety signals detected. Liver enzyme elevations observed in a subset of subjects.'
            safety_rec    = 'Continue monitoring liver enzymes at every visit. Evaluate subjects with ALT >3×ULN for dose adjustment.'
        else:
            safety_interp = 'No critical safety signals identified. Lab results and adverse events are within acceptable ranges.'
            safety_rec    = 'Maintain standard monitoring schedule. Continue routine safety reviews per protocol.'

        # ── Data Processing Summary ───────────────────────────────────────────
        file_rows = q("SELECT file_type, COUNT(*), SUM(record_count) FROM uploaded_files WHERE database_id=%s GROUP BY file_type", (database_id,))
        file_map  = {r[0]: {'files': r[1], 'records': r[2] or 0} for r in file_rows}
        total_records = sum(v['records'] for v in file_map.values())

        cursor.close()
        conn.close()

        return jsonify({'success': True, 'data': {
            'study_overview': {
                'study_id': f'STUDY-{database_id:04d}',
                'study_start': study_start_dt,
                'study_end': study_end_dt,
                'total_sites': int(total_sites),
                'total_subjects': int(total_subjects),
                'treatment_arms': arms,
                'processing_date': datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'),
                'pipeline_status': 'Completed'
            },
            'demographics': {
                'total_subjects': int(total_subjects),
                'male': int(male_count),
                'female': int(female_count),
                'avg_age': avg_age,
                'age_range': f'{min_age}–{max_age}',
                'subjects_per_site': [{'site_id': r[0], 'count': r[1]} for r in subjects_per_site]
            },
            'lab_summary': {
                'total': int(total_labs),
                'normal': int(normal_labs),
                'abnormal': int(abnormal_labs),
                'critical': int(safety_labs),
                'safety_flags': int(safety_labs),
                'alt_count': int(alt_count),
                'ast_count': int(ast_count),
                'by_test': [{'test': r[0], 'count': r[1]} for r in lab_by_test]
            },
            'safety_flags': {
                'alt_3x_uln': int(alt_3x),
                'ast_3x_uln': int(ast_3x),
                'bili_2x_uln': int(bili_2x),
                'hys_law': int(hys_law),
                'total_alerts': int(total_safety_alerts)
            },
            'ae_summary': {
                'total': int(total_aes),
                'serious': int(serious_aes),
                'non_serious': int(total_aes - serious_aes),
                'drug_related': int(related_aes),
                'most_common': most_common_ae,
                'moderate': int(moderate_aes),
                'severe': int(severe_aes)
            },
            'site_summary': [{'site_id': r[0], 'subjects': r[1], 'total_aes': r[2], 'serious_aes': r[3], 'safety_flags': r[4]} for r in site_summary],
            'drug_safety': {
                'overall_status': overall_safety,
                'risk_level': risk_level,
                'liver_toxicity': liver_tox,
                'interpretation': safety_interp,
                'recommendation': safety_rec
            },
            'processing': {
                'lab_files': int(file_map.get('labs', {}).get('files', 0)),
                'ae_files': int(file_map.get('aes', {}).get('files', 0)),
                'subject_records': int(file_map.get('subjects', {}).get('records', 0)),
                'total_records': int(total_records),
                'validation_errors': 0,
                'rejected_records': 0,
                'status': 'Completed'
            }
        }})
    except Exception as e:
        logger.error(f"Report data error: {e}")
        import traceback; logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/database/connect', methods=['POST'])
def connect_database():
    data = request.get_json()
    db_url = data.get('url')
    db_name = data.get('name', 'Unnamed Database')
    
    if not db_url:
        return jsonify({'success': False, 'message': 'Database URL is required'})
    
    logger.info(f"Attempting to connect to database: {db_name} at {db_url}")
    
    # Check for duplicate name or URL before testing connection
    dup_url  = DatabaseConnection.query.filter_by(url=db_url).first()
    dup_name = DatabaseConnection.query.filter(
        DatabaseConnection.name == db_name,
        DatabaseConnection.url  != db_url
    ).first()
    if dup_name:
        return jsonify({'success': False, 'message': f'A database named "{db_name}" already exists. Please use a different name.'})
    
    # ALWAYS test connection first, whether new or existing database
    logger.info("Testing database connection before proceeding")
    try:
        # Quick connection test
        test_conn = psycopg2.connect(db_url)
        test_cursor = test_conn.cursor()
        test_cursor.execute("SELECT 1")
        test_cursor.close()
        test_conn.close()
        logger.info("Connection test successful")
    except Exception as e:
        logger.error(f"Connection test failed: {str(e)}")
        # If database doesn't exist, remove it from local records
        existing_db = DatabaseConnection.query.filter_by(url=db_url).first()
        if existing_db:
            logger.info(f"Removing non-existent database {existing_db.name} from local records")
            db.session.delete(existing_db)
            db.session.commit()
        return jsonify({'success': False, 'message': f'Database connection failed: {str(e)}'})
    
    # Check if database already exists in local records
    existing_db = DatabaseConnection.query.filter_by(url=db_url).first()
    
    if existing_db:
        logger.info(f"Database already exists, testing connection and updating schemas")
        # Test connection to existing database and update/create schemas if needed
        if update_database_schema(db_url) and create_database_schemas(db_url):
            existing_db.last_connected = datetime.utcnow()
            existing_db.is_active = True
            db.session.commit()
            
            # Auto-switch to this database
            current_database_session['database_id'] = existing_db.id
            current_database_session['name'] = existing_db.name
            current_database_session['connection'] = existing_db.url
            
            logger.info(f"Successfully reconnected and auto-switched to existing database: {existing_db.name}")
            return jsonify({'success': True, 'message': 'Connected to existing database', 'database_id': existing_db.id, 'redirect_to': '/upload'})
        else:
            logger.error(f"Failed to connect to existing database or create schemas: {existing_db.name}")
            return jsonify({'success': False, 'message': 'Failed to connect to existing database or create schemas. Check application logs for details.'})
    
    # Create schemas for new database
    logger.info("Creating database schemas")
    if create_database_schemas(db_url):
        # Save new database connection
        new_db = DatabaseConnection(url=db_url, name=db_name, last_connected=datetime.utcnow())
        db.session.add(new_db)
        db.session.commit()
        
        # Auto-switch to this database
        current_database_session['database_id'] = new_db.id
        current_database_session['name'] = new_db.name
        current_database_session['connection'] = new_db.url
        
        logger.info(f"Successfully connected, created schemas, and auto-switched to new database: {db_name}")
        return jsonify({'success': True, 'message': 'Database connected and schemas created', 'database_id': new_db.id, 'redirect_to': '/upload'})
    else:
        logger.error(f"Failed to create schemas for new database: {db_name}")
        return jsonify({'success': False, 'message': 'Failed to create database schemas. Check application logs for details.'})

@app.route('/api/database/list')
def list_databases():
    """Get list of databases"""
    validate_connections = request.args.get('validate', 'false').lower() == 'true'
    
    databases = DatabaseConnection.query.all()
    result = []

    for db_conn in databases:
        is_available = db_conn.is_active  # default to cached

        if validate_connections:
            try:
                test_conn = psycopg2.connect(db_conn.url, connect_timeout=10)
                test_cursor = test_conn.cursor()
                test_cursor.execute("SELECT 1")
                test_cursor.close()
                test_conn.close()
                is_available = True
                db_conn.is_active = True
                db_conn.last_connected = datetime.utcnow()
            except Exception as e:
                logger.warning(f"Database {db_conn.name} unavailable: {str(e)}")
                is_available = False
                db_conn.is_active = False

        result.append({
            'id': db_conn.id,
            'name': db_conn.name,
            'url': db_conn.url,
            'created_at': db_conn.created_at.isoformat(),
            'last_connected': db_conn.last_connected.isoformat() if db_conn.last_connected else None,
            'is_active': db_conn.is_active,
            'is_available': is_available
        })

    if validate_connections:
        try:
            db.session.commit()
        except Exception as e:
            logger.error(f"Failed to commit database status updates: {str(e)}")
            db.session.rollback()

    logger.info(f"Returning {len(result)} databases (validate={validate_connections})")
    return jsonify(result)

@app.route('/api/database/validate', methods=['POST'])
def validate_database():
    """Validate if a database connection still works"""
    data = request.get_json()
    db_url = data.get('url')
    
    if not db_url:
        return jsonify({'success': False, 'message': 'Database URL is required'})
    
    try:
        test_conn = psycopg2.connect(db_url, connect_timeout=10)
        test_cursor = test_conn.cursor()
        test_cursor.execute("SELECT 1")
        test_cursor.close()
        test_conn.close()
        return jsonify({'success': True, 'message': 'Database connection is valid'})
    except Exception as e:
        logger.warning(f"Database validation failed: {str(e)}")
        return jsonify({'success': False, 'message': f'Database not accessible: {str(e)}'})

@app.route('/api/database/<int:db_id>/delete', methods=['DELETE'])
def delete_database(db_id):
    """Delete a database connection and its associated data"""
    db_conn = DatabaseConnection.query.get(db_id)
    if not db_conn:
        return jsonify({'success': False, 'message': 'Database not found'})
    
    # Try to delete from uploaded_files table (may fail if database is deleted)
    try:
        conn = psycopg2.connect(db_conn.url, connect_timeout=3)
        cursor = conn.cursor()
        
        cursor.execute("SELECT filename FROM uploaded_files WHERE database_id = %s", (db_id,))
        file_records = cursor.fetchall()
        
        for file_record in file_records:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_record[0])
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted file: {file_record[0]}")
            
            cursor.execute("DELETE FROM uploaded_files WHERE database_id = %s AND filename = %s", (db_id, file_record[0]))
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"Successfully cleaned up files for database {db_conn.name}")
        
    except Exception as e:
        # Database connection failed (likely deleted), but we can still delete the local record
        logger.warning(f"Could not connect to database {db_conn.name} for cleanup: {str(e)}")
        logger.info(f"Proceeding with local record deletion for {db_conn.name}")
    
    # Delete database connection from local database
    try:
        db.session.delete(db_conn)
        db.session.commit()
        logger.info(f"Successfully deleted database connection: {db_conn.name}")
        return jsonify({'success': True, 'message': 'Database connection deleted successfully'})
    except Exception as e:
        logger.error(f"Error deleting database connection: {str(e)}")
        db.session.rollback()
        return jsonify({'success': False, 'message': f'Error deleting database connection: {str(e)}'})
    
    return jsonify({'success': True, 'message': 'Database connection deleted successfully'})
    
    # Clear current session if this was the selected database
    if current_database_session['database_id'] == db_id:
        current_database_session['database_id'] = None
        current_database_session['connection'] = None
        current_database_session['name'] = None
    
    return jsonify({'success': True, 'message': 'Database deleted successfully'})

@app.route('/api/bronze/data', methods=['GET'])
def get_bronze_data():
    """Get bronze layer data for current database session"""
    try:
        database_id = request.args.get('database_id') or current_database_session['database_id']
        if not database_id:
            return jsonify({'success': False, 'message': 'No database selected'})
        database_id = int(database_id)

        db_conn = DatabaseConnection.query.get(database_id)
        if not db_conn:
            return jsonify({'success': False, 'message': 'Database not found'})
        
        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT COUNT(*) FROM bronze.bronze_subjects WHERE database_id = %s", (database_id,))
            total_subjects = cursor.fetchone()[0]
        except Exception:
            conn.rollback(); total_subjects = 0
        
        try:
            cursor.execute("SELECT COUNT(*) FROM bronze.bronze_labs WHERE database_id = %s", (database_id,))
            total_labs = cursor.fetchone()[0]
        except Exception:
            conn.rollback(); total_labs = 0
        
        try:
            cursor.execute("SELECT COUNT(*) FROM bronze.bronze_aes WHERE database_id = %s", (database_id,))
            total_aes = cursor.fetchone()[0]
        except Exception:
            conn.rollback(); total_aes = 0
        
        try:
            cursor.execute("SELECT COALESCE(SUM(file_size), 0) FROM public.uploaded_files WHERE database_id = %s", (database_id,))
            total_size = cursor.fetchone()[0] or 0
        except Exception:
            conn.rollback(); total_size = 0
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'data': {
                'subjects': total_subjects,
                'labs': total_labs,
                'aes': total_aes,
                'size': round(total_size / (1024 * 1024), 1)
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting bronze data: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/silver/data', methods=['GET'])
def get_silver_data():
    """Get silver layer data for current database session"""
    try:
        database_id = request.args.get('database_id') or current_database_session['database_id']
        if not database_id:
            return jsonify({'success': False, 'message': 'No database selected'})
        database_id = int(database_id)

        db_conn = DatabaseConnection.query.get(database_id)
        if not db_conn:
            return jsonify({'success': False, 'message': 'Database not found'})
        
        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM silver.silver_subjects WHERE database_id = %s", (database_id,))
        total_subjects = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM silver.silver_labs WHERE database_id = %s", (database_id,))
        total_labs = cursor.fetchone()[0]
        
        try:
            cursor.execute("SELECT COUNT(*) FROM silver.silver_ae WHERE database_id = %s", (database_id,))
            total_aes = cursor.fetchone()[0]
        except Exception:
            conn.rollback(); total_aes = 0
        
        try:
            cursor.execute("SELECT COALESCE(SUM(file_size), 0) FROM public.uploaded_files WHERE database_id = %s", (database_id,))
            total_size = cursor.fetchone()[0] or 0
        except Exception:
            conn.rollback(); total_size = 0
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'data': {
                'subjects': total_subjects,
                'labs': total_labs,
                'aes': total_aes,
                'size': round(total_size / (1024 * 1024), 1),
                'tables': [
                    {'name': 'silver_subjects', 'records': total_subjects, 'quality': 'Validated'},
                    {'name': 'silver_labs',     'records': total_labs,     'quality': 'Validated'},
                    {'name': 'silver_ae',        'records': total_aes,      'quality': 'Validated'},
                ]
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting silver data: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/gold/data', methods=['GET'])
def get_gold_data():
    """Get gold layer data for current database session"""
    try:
        database_id = request.args.get('database_id') or current_database_session['database_id']
        if not database_id:
            return jsonify({'success': False, 'message': 'No database selected'})
        database_id = int(database_id)

        db_conn = DatabaseConnection.query.get(database_id)
        if not db_conn:
            return jsonify({'success': False, 'message': 'Database not found'})
        
        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        cursor = conn.cursor()
        
        # Use correct gold table names
        cursor.execute("SELECT COUNT(*) FROM gold.gold_subjects WHERE database_id = %s", (database_id,))
        total_subjects = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM gold.gold_labs WHERE database_id = %s", (database_id,))
        total_labs = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM gold.gold_ae WHERE database_id = %s", (database_id,))
        total_aes = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT site_id) FROM gold.gold_subjects WHERE database_id = %s", (database_id,))
        total_sites = cursor.fetchone()[0]
        
        cursor.execute("SELECT COALESCE(SUM(file_size), 0) FROM public.uploaded_files WHERE database_id = %s", (database_id,))
        total_size = cursor.fetchone()[0] or 0
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'data': {
                'subjects': total_subjects,
                'labs': total_labs,
                'aes': total_aes,
                'sites': total_sites,
                'size': round(total_size / (1024 * 1024), 1)
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting gold data: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/prediction/data', methods=['GET'])
def get_prediction_data():
    """Get prediction layer data for current database session"""
    try:
        # Accept database_id from query param or fall back to session
        database_id = request.args.get('database_id') or current_database_session['database_id']
        if not database_id:
            return jsonify({'success': False, 'message': 'No database selected'})
        database_id = int(database_id)

        db_conn = DatabaseConnection.query.get(database_id)
        if not db_conn:
            return jsonify({'success': False, 'message': 'Database not found'})
        
        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        cursor = conn.cursor()
        
        # Get real metrics from gold tables (correct names)
        cursor.execute("SELECT COUNT(DISTINCT site_id) FROM gold.gold_subjects WHERE database_id = %s", (database_id,))
        total_sites = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM gold.gold_ae WHERE database_id = %s AND serious_flag = 'Y'", (database_id,))
        high_risk_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM gold.gold_ae WHERE database_id = %s", (database_id,))
        total_aes = cursor.fetchone()[0]
        
        # Calculate risk percentage
        risk_percentage = (high_risk_count / total_aes * 100) if total_aes > 0 else 0
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'data': {
                'sites': total_sites,
                'risks': high_risk_count,
                'confidence': round(100 - risk_percentage, 1)
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting prediction data: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/files/upload_chunked', methods=['POST'])
def upload_file_chunked():
    """Handle chunked file upload with server-side temp file storage"""
    chunk_data = request.files.get('chunk')
    chunk_index = int(request.form.get('chunk_index', 0))
    total_chunks = int(request.form.get('total_chunks', 1))
    file_type = request.form.get('file_type')
    database_id = request.form.get('database_id')
    filename = request.form.get('filename')
    file_size = int(request.form.get('file_size', 0))
    upload_id = request.form.get('upload_id')

    if not chunk_data:
        return jsonify({'success': False, 'message': 'No chunk data provided'})

    if not database_id and current_database_session['database_id']:
        database_id = current_database_session['database_id']
    elif not database_id:
        return jsonify({'success': False, 'message': 'No database selected'})

    # Use a temp directory on disk (avoids Flask session 4KB cookie limit)
    tmp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'tmp_chunks', upload_id)
    os.makedirs(tmp_dir, exist_ok=True)

    chunk_path = os.path.join(tmp_dir, f'chunk_{chunk_index:06d}')
    with open(chunk_path, 'wb') as f:
        f.write(chunk_data.read())

    logger.info(f"Stored chunk {chunk_index + 1}/{total_chunks} for {filename} ({upload_id})")

    if chunk_index == total_chunks - 1:
        # Assemble all chunks in order
        file_bytes = b''
        for i in range(total_chunks):
            part_path = os.path.join(tmp_dir, f'chunk_{i:06d}')
            with open(part_path, 'rb') as f:
                file_bytes += f.read()

        # Clean up temp chunks
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

        logger.info(f"Assembled {len(file_bytes)} bytes from {total_chunks} chunks for {filename}")

        try:
            result = process_uploaded_file_from_memory(file_bytes, filename, file_type, database_id, file_size)
            return result
        except Exception as e:
            logger.error(f"Error processing assembled file: {str(e)}")
            return jsonify({'success': False, 'message': f'Error processing file: {str(e)}'})
    else:
        return jsonify({
            'success': True,
            'message': f'Chunk {chunk_index + 1}/{total_chunks} received',
            'chunk_index': chunk_index,
            'complete': False
        })

def process_uploaded_file_from_memory(file_bytes, filename, file_type, database_id, file_size):
    """Process uploaded file directly from memory"""
    logger.info(f"Processing file {filename} from memory ({len(file_bytes)} bytes)")

    if file_type not in ['subjects', 'labs', 'aes']:
        return jsonify({'success': False, 'message': 'Invalid file type'})

    db_connection = DatabaseConnection.query.get(database_id)
    if not db_connection:
        return jsonify({'success': False, 'message': 'Database connection not found'})

    file_stream = io.BytesIO(file_bytes)
    df = None

    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(file_stream, dtype=str, low_memory=False)
        elif filename.endswith('.xml'):
            import xml.etree.ElementTree as ET
            root = ET.fromstring(file_bytes.decode('utf-8'))
            data = [
                {subchild.tag: subchild.text for subchild in child}
                for child in root
            ]
            df = pd.DataFrame(data) if data else pd.DataFrame()
        elif filename.endswith('.ndjson') or filename.endswith('.json'):
            df = pd.read_json(file_stream, lines=True, dtype=str)
        else:
            return jsonify({'success': False, 'message': 'Unsupported file format'})

        record_count = len(df)
        logger.info(f"Read {record_count} records from {filename}")
    except Exception as e:
        logger.error(f"Error reading file {filename}: {str(e)}")
        return jsonify({'success': False, 'message': f'Error reading file: {str(e)}'})

    is_valid, message = validate_columns_memory(df, file_type)
    if not is_valid:
        return jsonify({'success': False, 'message': f'Validation failed: {message}', 'validation_error': True})

    try:
        conn = psycopg2.connect(db_connection.url)
        cursor = conn.cursor()
        cursor.execute("CREATE SCHEMA IF NOT EXISTS bronze")

        chunk_size = 5000  # large chunks - execute_values handles bulk efficiently
        total_inserted = 0
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size].copy()
            if file_type == 'subjects':
                total_inserted += insert_subjects_to_bronze_chunked(cursor, chunk, database_id)
            elif file_type == 'labs':
                total_inserted += insert_labs_to_bronze_chunked(cursor, chunk, database_id)
            elif file_type == 'aes':
                total_inserted += insert_aes_to_bronze_chunked(cursor, chunk, database_id)

        # Record metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS public.uploaded_files (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                original_filename VARCHAR(255) NOT NULL,
                file_type VARCHAR(50) NOT NULL,
                database_id INTEGER NOT NULL,
                file_size BIGINT NOT NULL DEFAULT 0,
                file_hash VARCHAR(64),
                record_count INTEGER DEFAULT 0,
                processed BOOLEAN DEFAULT TRUE,
                bronze_loaded BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("ALTER TABLE public.uploaded_files ADD COLUMN IF NOT EXISTS file_hash VARCHAR(64)")
        file_hash = calculate_file_hash(file_bytes)
        cursor.execute("""
            INSERT INTO public.uploaded_files
            (database_id, original_filename, filename, file_type, file_size, file_hash, record_count, processed)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (database_id, filename, secure_filename(filename), file_type,
              file_size or len(file_bytes), file_hash, record_count, True))
        conn.commit()
        cursor.close()
        conn.close()

        logger.info(f"Upload complete: {filename} -> {total_inserted} new records inserted into bronze")
        return jsonify({
            'success': True,
            'message': f'File uploaded successfully. {total_inserted} new records inserted into Bronze layer ({record_count} total in file).',
            'file': {
                'filename': secure_filename(filename),
                'original_filename': filename,
                'file_type': file_type,
                'record_count': record_count,
                'inserted_count': total_inserted,
                'database_id': database_id,
                'bronze_loaded': True
            }
        })

    except Exception as e:
        logger.error(f"Error inserting into bronze layer: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        try:
            conn.rollback()
            cursor.close()
            conn.close()
        except:
            pass
        return jsonify({'success': False, 'message': f'Error loading into Bronze layer: {str(e)}'})


def process_uploaded_file(file_path, filename, file_type, database_id, file_size):
    """Process the uploaded file and insert into bronze layer"""
    logger.info(f"Processing file {filename} for bronze layer insertion")
    
    if file_type not in ['subjects', 'labs', 'aes']:
        return jsonify({'success': False, 'message': 'Invalid file type'})
    
    # Get database connection
    db_connection = DatabaseConnection.query.get(database_id)
    if not db_connection:
        return jsonify({'success': False, 'message': 'Database connection not found'})
    
    # Read file based on type
    df = None
    record_count = 0
    
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path, dtype=str, low_memory=False)
        elif filename.endswith('.xml'):
            import xml.etree.ElementTree as ET
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            data = []
            for child in root:
                row = {}
                for subchild in child:
                    row[subchild.tag] = subchild.text
                data.append(row)
            
            if data:
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame()
                logger.warning(f"No data found in XML file: {filename}")
                
        elif filename.endswith('.ndjson'):
            df = pd.read_json(file_path, lines=True)
        else:
            return jsonify({'success': False, 'message': 'Unsupported file format'})
        
        record_count = len(df)
        logger.info(f"Successfully read {record_count} records from {filename}")
        
    except Exception as e:
        logger.error(f"Error reading file {filename}: {str(e)}")
        return jsonify({'success': False, 'message': f'Error reading file: {str(e)}'})
    
    # Validate columns
    is_valid, message = validate_columns_memory(df, file_type)
    
    if not is_valid:
        logger.error(f"File validation failed: {filename} - {message}")
        return jsonify({'success': False, 'message': f'Validation failed: {message}', 'validation_error': True})
    
    # Insert directly into bronze layer
    try:
        logger.info(f"Starting direct bronze layer insertion for {filename}")
        
        conn = psycopg2.connect(db_connection.url)
        cursor = conn.cursor()
        
        # Ensure bronze schema exists
        cursor.execute("CREATE SCHEMA IF NOT EXISTS bronze")
        
        # Use optimized batch insertion instead of slow processing
        if file_type == 'subjects':
            inserted_count = insert_subjects_to_bronze_optimized(cursor, df, database_id, conn)
        elif file_type == 'labs':
            inserted_count = insert_labs_to_bronze_optimized(cursor, df, database_id, conn)
        elif file_type == 'aes':
            inserted_count = insert_aes_to_bronze_optimized(cursor, df, database_id, conn)
        else:
            inserted_count = 0
        
        # Create uploaded_files record (metadata only)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS public.uploaded_files (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                original_filename VARCHAR(255) NOT NULL,
                file_type VARCHAR(50) NOT NULL,
                database_id INTEGER NOT NULL,
                file_size BIGINT NOT NULL,
                record_count INTEGER DEFAULT 0,
                processed BOOLEAN DEFAULT TRUE,
                bronze_loaded BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Ensure all required columns exist
        cursor.execute("""
            ALTER TABLE public.uploaded_files 
            ADD COLUMN IF NOT EXISTS record_count INTEGER DEFAULT 0
        """)
        cursor.execute("""
            ALTER TABLE public.uploaded_files 
            ADD COLUMN IF NOT EXISTS processed BOOLEAN DEFAULT TRUE
        """)
        cursor.execute("""
            ALTER TABLE public.uploaded_files 
            ADD COLUMN IF NOT EXISTS bronze_loaded BOOLEAN DEFAULT TRUE
        """)
        
        cursor.execute("""
            INSERT INTO public.uploaded_files 
            (database_id, original_filename, filename, file_type, file_size, record_count, processed)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (database_id, filename, secure_filename(filename) if filename else f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
              file_type, file_size, record_count, True))
        
        conn.commit()
        
        # Check for duplicates and get actual inserted count
        if file_type == 'subjects':
            cursor.execute("SELECT COUNT(*) FROM bronze.bronze_subjects WHERE database_id = %s", (database_id,))
            total_subjects = cursor.fetchone()[0]
            inserted_count = total_subjects
            message = f"File uploaded successfully. {inserted_count} total subjects in bronze layer."
        elif file_type == 'labs':
            cursor.execute("SELECT COUNT(*) FROM bronze.bronze_labs WHERE database_id = %s", (database_id,))
            total_labs = cursor.fetchone()[0]
            inserted_count = total_labs
            message = f"File uploaded successfully. {inserted_count} total lab records in bronze layer."
        elif file_type == 'aes':
            cursor.execute("SELECT COUNT(*) FROM bronze.bronze_aes WHERE database_id = %s", (database_id,))
            total_aes = cursor.fetchone()[0]
            # Use actual count of new records inserted
            inserted_count = new_aes_count if 'new_aes_count' in locals() else 0
            if inserted_count == 0 and total_aes > 0:
                message = f"File uploaded successfully. {record_count} records processed, all duplicates skipped. {total_aes} total AE records in bronze layer."
            elif inserted_count < record_count:
                message = f"File uploaded successfully. {inserted_count} new records inserted, {record_count - inserted_count} duplicates skipped. {total_aes} total AE records in bronze layer."
            else:
                message = f"File uploaded successfully. {inserted_count} total AE records in bronze layer."
        else:
            inserted_count = record_count
            message = 'File uploaded and loaded into Bronze layer successfully'
        
        cursor.close()
        conn.close()
        
        logger.info(f"File uploaded directly to bronze layer: {filename} ({record_count} records)")
        
        return jsonify({
            'success': True,
            'message': message,
            'file': {
                'filename': secure_filename(filename) if filename else f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'original_filename': filename,
                'file_type': file_type,
                'record_count': record_count,
                'inserted_count': inserted_count,
                'database_id': database_id,
                'bronze_loaded': True
            }
        })
        
    except Exception as e:
        logger.error(f"Error inserting into bronze layer: {str(e)}")
        try:
            conn.rollback()
            cursor.close()
            conn.close()
        except:
            pass
        return jsonify({'success': False, 'message': f'Error loading into Bronze layer: {str(e)}'})

def calculate_file_hash(file_bytes):
    """Calculate SHA256 hash of file content for deduplication"""
    return hashlib.sha256(file_bytes).hexdigest()

def check_file_deduplication(database_id, filename, file_hash, file_size):
    """Check if file has already been uploaded to prevent duplicates"""
    try:
        conn = get_database_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # Create uploaded_files table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS public.uploaded_files (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                original_filename VARCHAR(255) NOT NULL,
                file_type VARCHAR(50) NOT NULL,
                database_id INTEGER NOT NULL,
                file_size BIGINT NOT NULL DEFAULT 0,
                file_hash VARCHAR(64),
                record_count INTEGER DEFAULT 0,
                processed BOOLEAN DEFAULT TRUE,
                bronze_loaded BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("ALTER TABLE public.uploaded_files ADD COLUMN IF NOT EXISTS file_hash VARCHAR(64)")
        conn.commit()
        
        # Check for duplicate by hash only (filename can be reused with different content)
        cursor.execute("""
            SELECT id FROM public.uploaded_files 
            WHERE database_id = %s AND file_hash = %s
        """, (database_id, file_hash))
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if result:
            logger.info(f"Duplicate file detected: {filename} (hash match)")
            return True
        return False
        
    except Exception as e:
        logger.error(f"Error checking file deduplication: {str(e)}")
        return False

def get_database_connection():
    """Get current database connection"""
    database_id = current_database_session.get('database_id')
    if not database_id:
        return None
    
    db_connection = DatabaseConnection.query.get(database_id)
    if not db_connection:
        return None
    
    try:
        return psycopg2.connect(db_connection.url)
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        return None

@app.route('/api/files/upload_optimized', methods=['POST'])
def upload_file_optimized():
    """Optimized file upload with chunked processing, fast date normalization, and deduplication"""
    logger.info("=== OPTIMIZED FILE UPLOAD STARTED ===")
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file provided'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})
    
    file_type = request.form.get('file_type')
    database_id = request.form.get('database_id')
    
    # Use current database session if database_id not provided
    if not database_id and current_database_session['database_id']:
        database_id = current_database_session['database_id']
    elif not database_id:
        return jsonify({'success': False, 'message': 'No database selected'})
    
    if file_type not in ['subjects', 'labs', 'aes']:
        return jsonify({'success': False, 'message': 'Invalid file type'})
    
    try:
        # Read file into memory for processing
        file_bytes = file.read()
        file_size = len(file_bytes)
        
        # Calculate file hash for deduplication
        file_hash = calculate_file_hash(file_bytes)
        
        # Check for duplicate file
        if check_file_deduplication(database_id, file.filename, file_hash, file_size):
            return jsonify({
                'success': False, 
                'message': f'File "{file.filename}" has already been uploaded to this database',
                'duplicate': True
            })
        
        # Process file based on type
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes), dtype=str, low_memory=False)
        elif file.filename.endswith('.xml'):
            import xml.etree.ElementTree as ET
            root = ET.fromstring(file_bytes.decode('utf-8'))
            data = []
            for child in root:
                row = {}
                for subchild in child:
                    row[subchild.tag] = subchild.text
                data.append(row)
            df = pd.DataFrame(data) if data else pd.DataFrame()
        elif file.filename.endswith('.ndjson'):
            df = pd.read_json(io.BytesIO(file_bytes), lines=True)
        else:
            return jsonify({'success': False, 'message': 'Unsupported file format'})
        
        record_count = len(df)
        logger.info(f"Successfully read {record_count} records from {file.filename}")
        
        # Validate columns
        is_valid, message = validate_columns_memory(df, file_type)
        if not is_valid:
            return jsonify({'success': False, 'message': f'Validation failed: {message}', 'validation_error': True})
        
        # Process with chunked insertion
        conn = get_database_connection()
        if not conn:
            return jsonify({'success': False, 'message': 'Database connection failed'})
        
        cursor = conn.cursor()
        
        # Ensure bronze schema exists
        cursor.execute("CREATE SCHEMA IF NOT EXISTS bronze")
        
        # Process in one large batch - execute_values handles bulk efficiently
        chunk_size = 5000
        total_inserted = 0
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            logger.info(f"Processing chunk {i//chunk_size + 1}: {len(chunk)} records")
            
            if file_type == 'subjects':
                inserted = insert_subjects_to_bronze_chunked(cursor, chunk, database_id)
            elif file_type == 'labs':
                inserted = insert_labs_to_bronze_chunked(cursor, chunk, database_id)
            elif file_type == 'aes':
                inserted = insert_aes_to_bronze_chunked(cursor, chunk, database_id)
            
            total_inserted += inserted
        
        # Record file upload metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS public.uploaded_files (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                original_filename VARCHAR(255) NOT NULL,
                file_type VARCHAR(50) NOT NULL,
                database_id INTEGER NOT NULL,
                file_size BIGINT NOT NULL,
                file_hash VARCHAR(64),
                record_count INTEGER DEFAULT 0,
                processed BOOLEAN DEFAULT TRUE,
                bronze_loaded BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(database_id, original_filename, file_hash)
            )
        """)
        
        cursor.execute("""
            INSERT INTO public.uploaded_files 
            (database_id, original_filename, filename, file_type, file_size, file_hash, record_count, processed)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (database_id, file.filename, secure_filename(file.filename), file_type, file_size, file_hash, record_count, True))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Optimized upload completed: {total_inserted} records inserted from {file.filename}")
        
        return jsonify({
            'success': True,
            'message': f'File uploaded successfully. {total_inserted} records inserted.',
            'file': {
                'filename': secure_filename(file.filename),
                'original_filename': file.filename,
                'file_type': file_type,
                'record_count': record_count,
                'inserted_count': total_inserted,
                'database_id': database_id,
                'bronze_loaded': True
            }
        })
        
    except Exception as e:
        logger.error(f"Error in optimized file upload: {str(e)}")
        try:
            conn.rollback()
            cursor.close()
            conn.close()
        except:
            pass
        return jsonify({'success': False, 'message': f'Upload failed: {str(e)}'})

@app.route('/api/files/upload', methods=['POST'])
def upload_file():
    """Handle file upload with direct bronze layer insertion"""
    logger.info("=== REGULAR FILE UPLOAD STARTED ===")
    logger.info(f"=== REGULAR ENDPOINT CALLED AT {datetime.now()} ===")
    logger.info(f"Request method: {request.method}")
    logger.info(f"Request files: {list(request.files.keys())}")
    logger.info(f"Request form: {list(request.form.keys())}")
    
    if 'file' not in request.files:
        logger.error("❌ REGULAR UPLOAD: No file in request.files")
        return jsonify({'success': False, 'message': 'No file provided'})
    
    file = request.files['file']
    logger.info(f"❌ REGULAR UPLOAD: File received: {file.filename if file.filename else 'No filename'}")
    
    if file.filename == '':
        logger.error("❌ REGULAR UPLOAD: Empty filename")
        return jsonify({'success': False, 'message': 'No file selected'})
    
    file_type = request.form.get('file_type')
    logger.info(f"❌ REGULAR UPLOAD: File type: {file_type}")
    
    # Use current database session if database_id not provided
    database_id = request.form.get('database_id')
    if not database_id and current_database_session['database_id']:
        database_id = current_database_session['database_id']
    elif not database_id:
        logger.error("❌ REGULAR UPLOAD: No database selected")
        return jsonify({'success': False, 'message': 'No database selected'})
    
    logger.info(f"❌ REGULAR UPLOAD: Using database ID: {database_id}")
    
    if file_type not in ['subjects', 'labs', 'aes']:
        return jsonify({'success': False, 'message': 'Invalid file type'})
    
    # Get database connection
    db_connection = DatabaseConnection.query.get(database_id)
    if not db_connection:
        return jsonify({'success': False, 'message': 'Database connection not found'})
    
    # Read file into memory (no local storage)
    logger.info(f"❌ REGULAR UPLOAD: Reading file {file.filename} into memory...")
    df = None
    record_count = 0
    
    try:
        # Read file based on type
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file.stream, dtype=str, low_memory=False)
        elif file.filename.endswith('.xml'):
            # Parse XML from stream
            import xml.etree.ElementTree as ET
            file_content = file.stream.read().decode('utf-8')
            root = ET.fromstring(file_content)
            
            data = []
            for child in root:
                row = {}
                for subchild in child:
                    row[subchild.tag] = subchild.text
                data.append(row)
            
            if data:
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame()
                logger.warning(f"No data found in XML file: {file.filename}")
                
        elif file.filename.endswith('.ndjson'):
            df = pd.read_json(file.stream, lines=True)
        else:
            return jsonify({'success': False, 'message': 'Unsupported file format'})
        
        record_count = len(df)
        logger.info(f"❌ REGULAR UPLOAD: Successfully read {record_count} records from {file.filename}")
        
    except Exception as e:
        logger.error(f"❌ REGULAR UPLOAD: Error reading file {file.filename}: {str(e)}")
        return jsonify({'success': False, 'message': f'Error reading file: {str(e)}'})
    
    # Validate columns
    is_valid, message = validate_columns_memory(df, file_type)
    
    if not is_valid:
        logger.error(f"❌ REGULAR UPLOAD: File validation failed: {file.filename} - {message}")
        return jsonify({'success': False, 'message': f'Validation failed: {message}', 'validation_error': True})
    
    # Insert directly into bronze layer
    try:
        logger.info(f"❌ REGULAR UPLOAD: Starting direct bronze layer insertion for {file.filename}")
        
        conn = psycopg2.connect(db_connection.url)
        cursor = conn.cursor()
        
        # Ensure bronze schema exists
        cursor.execute("CREATE SCHEMA IF NOT EXISTS bronze")
        
        # Use optimized batch insertion instead of slow processing
        if file_type == 'subjects':
            inserted_count = insert_subjects_to_bronze_optimized(cursor, df, database_id, conn)
        elif file_type == 'labs':
            inserted_count = insert_labs_to_bronze_optimized(cursor, df, database_id, conn)
        elif file_type == 'aes':
            inserted_count = insert_aes_to_bronze_optimized(cursor, df, database_id, conn)
        else:
            inserted_count = 0
        
        # Create uploaded_files record (metadata only)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS public.uploaded_files (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                original_filename VARCHAR(255) NOT NULL,
                file_type VARCHAR(50) NOT NULL,
                database_id INTEGER NOT NULL,
                file_size BIGINT NOT NULL,
                record_count INTEGER DEFAULT 0,
                processed BOOLEAN DEFAULT TRUE,
                bronze_loaded BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Ensure all required columns exist
        cursor.execute("""
            ALTER TABLE public.uploaded_files 
            ADD COLUMN IF NOT EXISTS record_count INTEGER DEFAULT 0
        """)
        cursor.execute("""
            ALTER TABLE public.uploaded_files 
            ADD COLUMN IF NOT EXISTS processed BOOLEAN DEFAULT TRUE
        """)
        cursor.execute("""
            ALTER TABLE public.uploaded_files 
            ADD COLUMN IF NOT EXISTS bronze_loaded BOOLEAN DEFAULT TRUE
        """)
        
        file_bytes_for_size = file.read() if hasattr(file, 'read') else b''
        actual_file_size = len(file_bytes_for_size)
        cursor.execute("""
            INSERT INTO public.uploaded_files 
            (database_id, original_filename, filename, file_type, file_size, record_count, processed)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (database_id, file.filename, secure_filename(file.filename) if file.filename else f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
              file_type, actual_file_size, record_count, True))
        
        conn.commit()
        
        # Check for duplicates and get actual inserted count
        if file_type == 'subjects':
            cursor.execute("SELECT COUNT(*) FROM bronze.bronze_subjects WHERE database_id = %s", (database_id,))
            total_subjects = cursor.fetchone()[0]
            inserted_count = total_subjects
            message = f"File uploaded successfully. {inserted_count} total subjects in bronze layer."
        elif file_type == 'labs':
            cursor.execute("SELECT COUNT(*) FROM bronze.bronze_labs WHERE database_id = %s", (database_id,))
            total_labs = cursor.fetchone()[0]
            inserted_count = total_labs
            message = f"File uploaded successfully. {inserted_count} total lab records in bronze layer."
        elif file_type == 'aes':
            cursor.execute("SELECT COUNT(*) FROM bronze.bronze_aes WHERE database_id = %s", (database_id,))
            total_aes = cursor.fetchone()[0]
            # Use actual count of new records inserted
            inserted_count = new_aes_count if 'new_aes_count' in locals() else 0
            if inserted_count == 0 and total_aes > 0:
                message = f"File uploaded successfully. {record_count} records processed, all duplicates skipped. {total_aes} total AE records in bronze layer."
            elif inserted_count < record_count:
                message = f"File uploaded successfully. {inserted_count} new records inserted, {record_count - inserted_count} duplicates skipped. {total_aes} total AE records in bronze layer."
            else:
                message = f"File uploaded successfully. {inserted_count} total AE records in bronze layer."
        else:
            inserted_count = record_count
            message = 'File uploaded and loaded into Bronze layer successfully'
        
        cursor.close()
        conn.close()
        
        logger.info(f"❌ REGULAR UPLOAD: File uploaded directly to bronze layer: {file.filename} ({record_count} records)")
        
        return jsonify({
            'success': True,
            'message': message,
            'file': {
                'filename': secure_filename(file.filename) if file.filename else f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'original_filename': file.filename,
                'file_type': file_type,
                'record_count': record_count,
                'inserted_count': inserted_count,
                'database_id': database_id,
                'bronze_loaded': True
            }
        })
        
    except Exception as e:
        logger.error(f"❌ REGULAR UPLOAD: Error inserting into bronze layer: {str(e)}")
        try:
            conn.rollback()
            cursor.close()
            conn.close()
        except:
            pass
        return jsonify({'success': False, 'message': f'Error loading into Bronze layer: {str(e)}'})

def validate_columns_memory(df, file_type):
    """Validate DataFrame columns against expected column lists (production-ready with comprehensive mapping)"""
    try:
        if df.empty:
            return False, "File contains no data"
        
        # Convert all columns to lowercase and strip whitespace for robust matching
        file_columns = df.columns.str.lower().str.strip()
        logger.info(f"Validating {file_type} DataFrame")
        logger.info(f"Original columns: {df.columns.tolist()}")
        logger.info(f"Normalized columns: {file_columns.tolist()}")
        
        if file_type == 'subjects':
            # Comprehensive subject ID column variations
            subject_id_variations = [
                'subject_id', 'subjectid', 'subject_no', 'subjectnumber', 'subject_code', 'subjectcode',
                'subj_id', 'subjid', 'usubjid', 'participant_id', 'participantid', 'patient_id', 'patientid',
                'subject', 'subj', 'sub', 'id', 'subject_identifier', 'subjectidentifier'
            ]
            
            has_subject_id = any(col in file_columns.values for col in subject_id_variations)
            logger.info(f"Subject ID variations checked: {subject_id_variations}")
            logger.info(f"Subject ID check: {has_subject_id}")
            
            if has_subject_id:
                logger.info("Subjects validation PASSED")
                return True, "Subjects file is valid"
            else:
                logger.error("Subjects validation FAILED - missing subject identifier")
                return False, "Subjects file must contain a subject identifier column (e.g., subject_id, subjectid, usubjid, participant_id, etc.)"
                
        elif file_type == 'labs':
            # Comprehensive lab column variations
            subject_id_variations = ['subject_id', 'subjectid', 'subj_id', 'subjid', 'usubjid', 'participant_id', 'patient_id', 'subject', 'subj', 'id']
            visit_variations = ['visit_no', 'visitno', 'visit', 'visitnum', 'visit_number', 'visitnumber', 'visitid', 'visit_id']
            test_code_variations = ['test_code', 'testcode', 'test', 'testname', 'test_name', 'lab_test', 'labtest', 'analyte', 'test_id']
            result_variations = ['result_value', 'resultvalue', 'result', 'value', 'test_result', 'testresult', 'measurement', 'reading']
            
            has_subject_id = any(col in file_columns.values for col in subject_id_variations)
            has_visit = any(col in file_columns.values for col in visit_variations)
            has_test_code = any(col in file_columns.values for col in test_code_variations)
            has_result = any(col in file_columns.values for col in result_variations)
            
            logger.info(f"Labs column variations checked:")
            logger.info(f"  Subject ID: {subject_id_variations} -> Found: {has_subject_id}")
            logger.info(f"  Visit: {visit_variations} -> Found: {has_visit}")
            logger.info(f"  Test Code: {test_code_variations} -> Found: {has_test_code}")
            logger.info(f"  Result: {result_variations} -> Found: {has_result}")
            
            if has_subject_id and has_visit and has_test_code and has_result:
                logger.info("Labs validation PASSED")
                return True, "Labs file is valid"
            else:
                missing = []
                if not has_subject_id:
                    missing.append("subject identifier")
                if not has_visit:
                    missing.append("visit information")
                if not has_test_code:
                    missing.append("test information")
                if not has_result:
                    missing.append("result value")
                
                error_msg = f"Labs file must contain: {', '.join(missing)}. Acceptable column names include: subject_id/subjectid/usubjid, visit_no/visitno/visit, test_code/testcode/test, result_value/result/result"
                logger.error(f"Labs validation FAILED - {error_msg}")
                return False, error_msg
                
        elif file_type == 'aes':
            # Comprehensive AE column variations
            subject_id_variations = ['subject_id', 'subjectid', 'subj_id', 'subjid', 'usubjid', 'participant_id', 'patient_id', 'subject', 'subj', 'id']
            ae_id_variations = ['ae_id', 'aeid', 'adverse_event_id', 'adverseeventid', 'ae_seq', 'aeseq', 'event_id', 'eventid']
            pt_name_variations = ['pt_name', 'ptname', 'preferred_term', 'preferredterm', 'term', 'adverse_event_term', 'adverseeventterm', 'ae_term', 'aeterm']
            severity_variations = ['severity', 'sev', 'grade', 'severity_grade', 'severitygrade', 'intensity', 'seriousness']
            
            has_subject_id = any(col in file_columns.values for col in subject_id_variations)
            has_ae_id = any(col in file_columns.values for col in ae_id_variations)
            has_pt_name = any(col in file_columns.values for col in pt_name_variations)
            has_severity = any(col in file_columns.values for col in severity_variations)
            
            logger.info(f"AES column variations checked:")
            logger.info(f"  Subject ID: {subject_id_variations} -> Found: {has_subject_id}")
            logger.info(f"  AE ID: {ae_id_variations} -> Found: {has_ae_id}")
            logger.info(f"  PT Name: {pt_name_variations} -> Found: {has_pt_name}")
            logger.info(f"  Severity: {severity_variations} -> Found: {has_severity}")
            
            if has_subject_id and has_ae_id and has_pt_name and has_severity:
                logger.info("AES validation PASSED")
                return True, "AES file is valid"
            else:
                missing = []
                if not has_subject_id:
                    missing.append("subject identifier")
                if not has_ae_id:
                    missing.append("adverse event identifier")
                if not has_pt_name:
                    missing.append("event term")
                if not has_severity:
                    missing.append("severity information")
                
                error_msg = f"AES file must contain: {', '.join(missing)}. Acceptable column names include: subject_id/subjectid/usubjid, ae_id/aeid/event_id, pt_name/term/adverse_event_term, severity/grade/intensity"
                logger.error(f"AES validation FAILED - {error_msg}")
                return False, error_msg
        
        return False, "Unknown file type"
        
    except Exception as e:
        logger.error(f"Error in column validation: {str(e)}")
        return False, f"Validation error: {str(e)}"

def insert_subjects_to_bronze_chunked(cursor, df, database_id):
    """Fast bulk insertion for subjects using execute_values (single round-trip)"""
    from psycopg2.extras import execute_values

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bronze.bronze_subjects (
            id SERIAL PRIMARY KEY,
            database_id INTEGER NOT NULL,
            subject_id VARCHAR(100),
            site_id VARCHAR(50),
            sex VARCHAR(20),
            dob TEXT,
            arm VARCHAR(100),
            start_date VARCHAR(50),
            raw_data JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        DO $$ BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'bronze_subjects_database_id_subject_id_key'
            ) THEN
                ALTER TABLE bronze.bronze_subjects
                ADD CONSTRAINT bronze_subjects_database_id_subject_id_key
                UNIQUE (database_id, subject_id);
            END IF;
        END $$;
    """)

    cols = {c.lower(): c for c in df.columns}
    def pick(candidates):
        for c in candidates:
            if c in cols: return cols[c]
        return None

    subject_col    = pick(['subject_id','subjectid','subj_id','subjid','usubjid','participant_id','patient_id','subject','subj','id'])
    site_col       = pick(['site_id','siteid','site_no','site_number','center_id','centre_id','site','center'])
    sex_col        = pick(['sex','gender','biological_sex','subject_sex'])
    dob_col        = pick(['dob','date_of_birth','birth_date','birth_dt'])
    arm_col        = pick(['arm','treatment_arm','study_arm','arm_name','randomization_arm','rand_arm','cohort','group','treatment'])
    start_date_col = pick(['start_date','study_start_date','enrollment_date','enrolled_date','first_dose_date','treatment_start_date','rfstdtc','startdate'])

    if not subject_col:
        logger.error("No subject_id column found")
        return 0

    rows = []
    for _, row in df.iterrows():
        sid = str(row[subject_col]).strip() if pd.notna(row[subject_col]) else None
        if not sid or sid == 'nan':
            continue
        def val(col):
            if col and pd.notna(row[col]):
                v = str(row[col]).strip()
                return None if v == 'nan' else v
            return None
        rows.append((
            database_id, sid,
            val(site_col), val(sex_col),
            normalize_date_fast(row[dob_col]) if dob_col and pd.notna(row[dob_col]) else None,
            val(arm_col),
            normalize_date_fast(row[start_date_col]) if start_date_col and pd.notna(row[start_date_col]) else None,
            json.dumps({k: str(v) for k, v in row.items()})
        ))

    if rows:
        execute_values(cursor, """
            INSERT INTO bronze.bronze_subjects
            (database_id, subject_id, site_id, sex, dob, arm, start_date, raw_data)
            VALUES %s
            ON CONFLICT (database_id, subject_id) DO NOTHING
        """, rows, page_size=500)
        logger.info(f"Subjects bulk insert: {len(rows)} rows")
    return len(rows)

def insert_labs_to_bronze_chunked(cursor, df, database_id):
    """Fast bulk insertion for labs using execute_values (single round-trip)"""
    from psycopg2.extras import execute_values

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bronze.bronze_labs (
            id SERIAL PRIMARY KEY,
            database_id INTEGER NOT NULL,
            subject_id VARCHAR(100),
            visitno TEXT,
            testcode VARCHAR(200),
            resultvalue VARCHAR(100),
            units VARCHAR(50),
            reflow VARCHAR(50),
            refhigh VARCHAR(50),
            resultdate TEXT,
            raw_data JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cols = {c.lower(): c for c in df.columns}
    def pick(candidates):
        for c in candidates:
            if c in cols: return cols[c]
        return None

    subject_col = pick(['subject_id','subjectid','subj_id','subjid','usubjid','participant_id','patient_id','subject','subj','id'])
    visit_col   = pick(['visit_no','visitno','visitnum','visit_number','visitnumber','visit_id','visit'])
    test_col    = pick(['test_code','testcode','lbtestcd','lab_test_code','analyte_code','test_name','testname','test','analyte'])
    result_col  = pick(['result_value','resultvalue','lbresult','lbres','result_numeric','result','test_result','lab_result','value','measurement'])
    unit_col    = pick(['units','unit','result_unit','lab_unit','lborresu'])
    reflow_col  = pick(['ref_low','reflow','reference_low','normal_low','lower_limit','lbnrlo','ref_range_low'])
    refhigh_col = pick(['ref_high','refhigh','reference_high','normal_high','upper_limit','lbnrhi','ref_range_high'])
    date_col    = pick(['result_date','resultdate','lbdtc','test_date','lab_date','collection_date','specimen_date'])

    if not subject_col:
        logger.error("No subject_id column found for labs")
        return 0

    def clean(col, row):
        if col and pd.notna(row[col]):
            v = str(row[col]).strip()
            return None if v == 'nan' else v
        return None

    rows = []
    for _, row in df.iterrows():
        sid = str(row[subject_col]).strip() if pd.notna(row[subject_col]) else None
        if not sid or sid == 'nan':
            continue
        rows.append((
            database_id, sid,
            clean(visit_col, row), clean(test_col, row), clean(result_col, row),
            clean(unit_col, row), clean(reflow_col, row), clean(refhigh_col, row),
            normalize_date_fast(row[date_col]) if date_col and pd.notna(row[date_col]) else None,
            json.dumps({k: str(v) for k, v in row.items()})
        ))

    if rows:
        execute_values(cursor, """
            INSERT INTO bronze.bronze_labs
            (database_id, subject_id, visitno, testcode, resultvalue, units, reflow, refhigh, resultdate, raw_data)
            VALUES %s
        """, rows, page_size=500)
        logger.info(f"Labs bulk insert: {len(rows)} rows")
    return len(rows)

def insert_aes_to_bronze_chunked(cursor, df, database_id):
    """Fast bulk insertion for adverse events using execute_values (single round-trip)"""
    from psycopg2.extras import execute_values

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bronze.bronze_aes (
            id SERIAL PRIMARY KEY,
            database_id INTEGER NOT NULL,
            subject_id VARCHAR(100),
            ae_id VARCHAR(100),
            pt_name VARCHAR(500),
            pt_code VARCHAR(100),
            severity VARCHAR(50),
            related VARCHAR(20),
            serious VARCHAR(20),
            ae_start_date TEXT,
            raw_data JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cols = {c.lower(): c for c in df.columns}
    def pick(candidates):
        for c in candidates:
            if c in cols: return cols[c]
        return None

    subject_col  = pick(['subject_id','subjectid','subj_id','subjid','usubjid','participant_id','patient_id','subject','subj','id'])
    ae_id_col    = pick(['ae_id','aeid','ae_no','ae_number','aenumber','adverse_event_id','adverseeventid','ae_seq','aeseq','event_id','eventid','ae_record_id'])
    pt_name_col  = pick(['pt_name','ptname','pt_term','preferred_term','preferredterm','meddra_pt','meddra_term','adverse_event_term','adverseeventterm','ae_term','aeterm','term','event_name'])
    pt_code_col  = pick(['pt_code','ptcode','meddra_pt_code','meddra_code','pt_meddra_code','preferred_term_code','ae_pt_code'])
    severity_col = pick(['severity','ae_severity','severity_grade','ae_sev','intensity','ae_intensity','toxicity_grade','ctcae_grade','grade','sev'])
    related_col  = pick(['related','related_to_drug','causality','relationship','ae_relationship','relationship_to_treatment','treatment_related','drug_related','causality_assessment'])
    serious_col  = pick(['serious','is_serious','sae_flag','sae','serious_flag','serious_event','seriousness','is_sae'])
    date_col     = pick(['ae_start_dt','ae_start_date','ae_onset_date','onset_date','event_start_date','start_date','ae_onset_dt','aestdt'])

    if not subject_col:
        logger.error("No subject_id column found for AEs")
        return 0

    def clean(col, row):
        if col and pd.notna(row[col]):
            v = str(row[col]).strip()
            return None if v == 'nan' else v
        return None

    rows = []
    for _, row in df.iterrows():
        sid = str(row[subject_col]).strip() if pd.notna(row[subject_col]) else None
        if not sid or sid == 'nan':
            continue
        rows.append((
            database_id, sid,
            clean(ae_id_col, row), clean(pt_name_col, row), clean(pt_code_col, row),
            clean(severity_col, row), clean(related_col, row), clean(serious_col, row),
            normalize_date_fast(row[date_col]) if date_col and pd.notna(row[date_col]) else None,
            json.dumps({k: str(v) for k, v in row.items()})
        ))

    if rows:
        execute_values(cursor, """
            INSERT INTO bronze.bronze_aes
            (database_id, subject_id, ae_id, pt_name, pt_code, severity, related, serious, ae_start_date, raw_data)
            VALUES %s
        """, rows, page_size=500)
        logger.info(f"AEs bulk insert: {len(rows)} rows")
    return len(rows)

def insert_subjects_to_bronze_optimized(cursor, df, database_id, conn):
    """SIMPLE and FAST batch insertion for subjects data"""
    logger.info(f"Starting FAST insertion of {len(df)} subjects")
    
    # Create table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bronze.bronze_subjects (
            id SERIAL PRIMARY KEY,
            database_id INTEGER NOT NULL,
            subject_id VARCHAR(100),
            site_id VARCHAR(50),
            sex VARCHAR(20),
            dob TEXT,
            arm VARCHAR(100),
            start_date VARCHAR(50),
            raw_data JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(database_id, subject_id)
        )
    """)
    
    # Simple and fast column mapping
    logger.info("🚀 Using simple fast column mapping")
    
    # Find columns quickly
    subject_id_col = None
    for col in df.columns:
        if col.lower() in ['subject_id', 'subjectid', 'subj_id', 'usubjid', 'patient_id']:
            subject_id_col = col
            break
    
    if not subject_id_col:
        logger.error("❌ No subject_id column found")
        return 0
    
    # Fast data preparation
    insert_data = []
    for _, row in df.iterrows():
        subject_id = str(row[subject_id_col]).strip() if pd.notna(row[subject_id_col]) else None
        if not subject_id:
            continue
        
        # Get other fields quickly
        site_id = None
        sex = None
        dob = None
        arm = None
        start_date = None
        
        for col in df.columns:
            col_lower = col.lower()
            if site_id is None and col_lower in ['site_id', 'siteid', 'site']:
                site_id = str(row[col]).strip() if pd.notna(row[col]) else None
            elif sex is None and col_lower in ['sex', 'gender']:
                sex = str(row[col]).strip() if pd.notna(row[col]) else None
            elif dob is None and col_lower in ['dob', 'birth_date', 'date_of_birth']:
                dob = normalize_date(row[col]) if pd.notna(row[col]) else None
            elif arm is None and col_lower in ['arm', 'group', 'treatment']:
                arm = str(row[col]).strip() if pd.notna(row[col]) else None
            elif start_date is None and col_lower in ['start_date', 'startdate']:
                start_date = normalize_date(row[col]) if pd.notna(row[col]) else None
        
        insert_data.append((
            database_id, subject_id, site_id, sex, dob, arm, start_date,
            json.dumps(row.to_dict())
        ))
    
    # Fast batch insert with proper deduplication
    if insert_data:
        logger.info(f"🚀 Executing fast batch insert of {len(insert_data)} records with deduplication")
        
        # Get existing subject_ids for deduplication
        cursor.execute("SELECT subject_id FROM bronze.bronze_subjects WHERE database_id = %s", (database_id,))
        existing_subjects = set(row[0] for row in cursor.fetchall())
        logger.info(f"🚀 Found {len(existing_subjects)} existing subjects for deduplication")
        
        # Filter out duplicates
        new_insert_data = []
        for record in insert_data:
            subject_id = record[1]  # subject_id is at index 1
            if subject_id not in existing_subjects:
                new_insert_data.append(record)
        
        logger.info(f"🚀 After deduplication: {len(new_insert_data)} new records to insert")
        
        if new_insert_data:
            cursor.executemany("""
                INSERT INTO bronze.bronze_subjects 
                (database_id, subject_id, site_id, sex, dob, arm, start_date, raw_data)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, new_insert_data)
        else:
            logger.info("🚀 No new records to insert (all duplicates)")
        
        return len(new_insert_data)
    
    logger.info(f"🚀 FAST subjects insertion completed: {len(insert_data)} records processed")
    return len(insert_data)

def insert_labs_to_bronze_optimized(cursor, df, database_id, conn):
    """Optimized batch insertion for labs data"""
    logger.info(f"Starting optimized insertion of {len(df)} labs")
    
    # Create table if not exists with CORRECT column names matching existing code
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bronze.bronze_labs (
            id SERIAL PRIMARY KEY,
            database_id INTEGER NOT NULL,
            subject_id VARCHAR(100),
            visitno TEXT,
            testcode VARCHAR(100),
            resultvalue VARCHAR(100),
            units VARCHAR(50),
            reflow VARCHAR(50),
            refhigh VARCHAR(50),
            resultdate TEXT,
            raw_data JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Batch insert preparation with CORRECT column names
    insert_data = []
    for _, row in df.iterrows():
        # Find subject_id
        subject_id = None
        for col in df.columns:
            if col.lower() in ['subject_id', 'subjectid', 'subj_id', 'usubjid', 'patient_id']:
                subject_id = str(row[col]).strip() if pd.notna(row[col]) else None
                break
        
        if not subject_id:
            continue
            
        # Get other fields with CORRECT column names
        visitno = None
        testcode = None
        resultvalue = None
        units = None
        reflow = None
        refhigh = None
        resultdate = None
        
        for col in df.columns:
            col_lower = col.lower()
            if visitno is None and col_lower in ['visit_no', 'visitnumber', 'visit', 'visitnum', 'visitno']:
                visitno = str(row[col]).strip() if pd.notna(row[col]) else None
            elif testcode is None and col_lower in ['test_code', 'testcode', 'lab_test_code', 'lbtestcd']:
                testcode = str(row[col]).strip() if pd.notna(row[col]) else None
            elif resultvalue is None and col_lower in ['result_value', 'result', 'test_result', 'lab_result', 'resultvalue']:
                resultvalue = str(row[col]).strip() if pd.notna(row[col]) else None
            elif units is None and col_lower in ['units', 'unit', 'result_unit', 'lab_unit']:
                units = str(row[col]).strip() if pd.notna(row[col]) else None
            elif reflow is None and col_lower in ['ref_low', 'reference_low', 'lower_limit', 'reflow']:
                reflow = str(row[col]).strip() if pd.notna(row[col]) else None
            elif refhigh is None and col_lower in ['ref_high', 'reference_high', 'upper_limit', 'refhigh']:
                refhigh = str(row[col]).strip() if pd.notna(row[col]) else None
            elif resultdate is None and col_lower in ['result_date', 'test_date', 'lab_date', 'collection_date', 'resultdate']:
                resultdate = normalize_date(row[col]) if pd.notna(row[col]) else None
        
        insert_data.append((
            database_id, subject_id, visitno, testcode, resultvalue, units, reflow, refhigh, resultdate,
            json.dumps(row.to_dict())
        ))
    
    # Batch insert with CORRECT column names and deduplication
    if insert_data:
        logger.info(f"🚀 Executing labs batch insert of {len(insert_data)} records with deduplication")
        
        # Get existing lab records for deduplication (using subject_id + visitno + testcode)
        cursor.execute("""
            SELECT subject_id, visitno, testcode 
            FROM bronze.bronze_labs 
            WHERE database_id = %s
        """, (database_id,))
        existing_labs = set((row[0], row[1], row[2]) for row in cursor.fetchall())
        logger.info(f"🚀 Found {len(existing_labs)} existing lab records for deduplication")
        
        # Filter out duplicates
        new_insert_data = []
        for record in insert_data:
            lab_key = (record[1], record[2], record[3])  # subject_id, visitno, testcode
            if lab_key not in existing_labs:
                new_insert_data.append(record)
        
        logger.info(f"🚀 After deduplication: {len(new_insert_data)} new lab records to insert")
        
        if new_insert_data:
            cursor.executemany("""
                INSERT INTO bronze.bronze_labs 
                (database_id, subject_id, visitno, testcode, resultvalue, units, reflow, refhigh, resultdate, raw_data)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, new_insert_data)
        else:
            logger.info("🚀 No new lab records to insert (all duplicates)")
        
        return len(new_insert_data)
    
    logger.info(f"Optimized labs insertion completed: {len(insert_data)} records processed")
    return len(insert_data)

def insert_aes_to_bronze_optimized(cursor, df, database_id, conn):
    """Optimized batch insertion for adverse events data"""
    logger.info(f"Starting optimized insertion of {len(df)} AES")
    
    # Create table if not exists with CORRECT column names matching existing code
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bronze.bronze_aes (
            id SERIAL PRIMARY KEY,
            database_id INTEGER NOT NULL,
            subject_id VARCHAR(100),
            ae_id VARCHAR(100),
            pt_name VARCHAR(200),
            pt_code VARCHAR(100),
            severity VARCHAR(50),
            related VARCHAR(20),
            serious VARCHAR(20),
            ae_start_date TEXT,
            raw_data JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(database_id, subject_id, ae_id)
        )
    """)
    
    # Batch insert preparation with CORRECT column names
    insert_data = []
    for _, row in df.iterrows():
        # Find subject_id
        subject_id = None
        for col in df.columns:
            if col.lower() in ['subject_id', 'subjectid', 'subj_id', 'usubjid', 'patient_id']:
                subject_id = str(row[col]).strip() if pd.notna(row[col]) else None
                break
        
        if not subject_id:
            continue
            
        # Get other fields with CORRECT column names
        ae_id = None
        pt_code = None
        pt_name = None
        severity = None
        related = None
        serious = None
        ae_start_date = None
        
        for col in df.columns:
            col_lower = col.lower()
            if ae_id is None and col_lower in ['ae_id', 'aeid', 'ae_no', 'aenumber']:
                ae_id = str(row[col]).strip() if pd.notna(row[col]) else None
            elif pt_code is None and col_lower in ['pt_code', 'ptcode', 'meddra_pt_code']:
                pt_code = str(row[col]).strip() if pd.notna(row[col]) else None
            elif pt_name is None and col_lower in ['pt_name', 'pt_term', 'preferred_term', 'adverse_event_term']:
                pt_name = str(row[col]).strip() if pd.notna(row[col]) else None
            elif severity is None and col_lower in ['severity', 'ae_severity', 'severity_grade']:
                severity = str(row[col]).strip() if pd.notna(row[col]) else None
            elif related is None and col_lower in ['related', 'relationship', 'causality']:
                related = str(row[col]).strip() if pd.notna(row[col]) else None
            elif serious is None and col_lower in ['serious', 'is_serious', 'sae_flag']:
                serious = str(row[col]).strip() if pd.notna(row[col]) else None
            elif ae_start_date is None and col_lower in ['ae_start_dt', 'ae_start_date', 'onset_date', 'ae_start_date']:
                ae_start_date = normalize_date(row[col]) if pd.notna(row[col]) else None
        
        insert_data.append((
            database_id, subject_id, ae_id, pt_name, pt_code, severity, related, serious, ae_start_date,
            json.dumps(row.to_dict())
        ))
    
    # Batch insert with CORRECT column names and deduplication
    if insert_data:
        logger.info(f"🚀 Executing AES batch insert of {len(insert_data)} records with deduplication")
        
        # Get existing AE records for deduplication (using subject_id + ae_id)
        cursor.execute("""
            SELECT subject_id, ae_id 
            FROM bronze.bronze_aes 
            WHERE database_id = %s
        """, (database_id,))
        existing_aes = set((row[0], row[1]) for row in cursor.fetchall())
        logger.info(f"🚀 Found {len(existing_aes)} existing AE records for deduplication")
        
        # Filter out duplicates
        new_insert_data = []
        for record in insert_data:
            ae_key = (record[1], record[2])  # subject_id, ae_id
            if ae_key not in existing_aes:
                new_insert_data.append(record)
        
        logger.info(f"🚀 After deduplication: {len(new_insert_data)} new AE records to insert")
        
        if new_insert_data:
            cursor.executemany("""
                INSERT INTO bronze.bronze_aes 
                (database_id, subject_id, ae_id, pt_name, pt_code, severity, related, serious, ae_start_date, raw_data)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, new_insert_data)
        else:
            logger.info("🚀 No new AE records to insert (all duplicates)")
        
        return len(new_insert_data)
    
    logger.info(f"Optimized AES insertion completed: {len(insert_data)} records processed")
    return len(insert_data)

def insert_subjects_to_bronze(cursor, df, database_id, conn):
    """Insert subjects data directly into bronze layer with optimized performance"""
    logger.info(f"Inserting {len(df)} subjects into bronze layer")
    
    # Create bronze_subjects table if not exists with all required columns
    # Force drop and recreate to fix column type issues
    cursor.execute("DROP TABLE IF EXISTS bronze.bronze_subjects CASCADE")
    logger.info("Dropped existing bronze_subjects table to fix column types")
    
    cursor.execute("""
        CREATE TABLE bronze.bronze_subjects (
            id SERIAL PRIMARY KEY,
            database_id INTEGER NOT NULL,
            subject_id VARCHAR(100),
            site_id VARCHAR(50),
            sex VARCHAR(20),
            dob TEXT,
            arm VARCHAR(100),
            start_date VARCHAR(50),
            raw_data JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(database_id, subject_id)
        )
    """)
    logger.info("Recreated bronze_subjects table with correct column types")
    
    # OPTIMIZATION: Bulk fetch existing subjects instead of individual queries
    cursor.execute("SELECT subject_id FROM bronze.bronze_subjects WHERE database_id = %s", (database_id,))
    existing_subjects = set(row[0] for row in cursor.fetchall())
    logger.info(f"Found {len(existing_subjects)} existing subjects")
    
    # OPTIMIZATION: Vectorized column mapping instead of row-by-row
    column_mapping = {}
    file_columns_lower = df.columns.str.lower().str.strip()
    
    # Map subject_id variations
    subject_id_variations = ['subject_id', 'subjectid', 'subject_no', 'subjectnumber', 'subject_code', 'subjectcode',
                           'subj_id', 'subjid', 'usubjid', 'participant_id', 'participantid', 'patient_id', 'patientid',
                           'subject', 'subj', 'sub', 'id', 'subject_identifier', 'subjectidentifier']
    
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in subject_id_variations and 'subject_id' not in column_mapping:
            column_mapping['subject_id'] = col
            break
    
    # Map other common columns including start_date
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ['site_id', 'siteid', 'site', 'center', 'location'] and 'site_id' not in column_mapping:
            column_mapping['site_id'] = col
        elif col_lower in ['sex', 'gender'] and 'sex' not in column_mapping:
            column_mapping['sex'] = col
        elif col_lower in ['dob', 'birth_date', 'birth_date', 'date_of_birth'] and 'dob' not in column_mapping:
            column_mapping['dob'] = col
        elif col_lower in ['arm', 'group', 'treatment', 'cohort'] and 'arm' not in column_mapping:
            column_mapping['arm'] = col
        elif col_lower in ['start_date', 'startdate', 'start', 'enrollment_date', 'enrollmentdate', 'randomization_date', 'randomizationdate'] and 'start_date' not in column_mapping:
            column_mapping['start_date'] = col
    
    logger.info(f"Subjects column mapping: {column_mapping}")
    
    # OPTIMIZATION: Vectorized filtering instead of row-by-row
    if 'subject_id' not in column_mapping:
        logger.error("No subject_id column found in file")
        return
    
    # Get subject_id column
    subject_id_col = column_mapping['subject_id']
    df['processed_subject_id'] = df[subject_id_col].astype(str).str.strip()
    
    # OPTIMIZATION: Vectorized duplicate filtering
    mask = ~df['processed_subject_id'].isin(existing_subjects) & df['processed_subject_id'].notna()
    new_subjects_df = df[mask].copy()
    
    logger.info(f"Found {len(new_subjects_df)} new subjects to insert, skipped {len(df) - len(new_subjects_df)} duplicates")
    
    if new_subjects_df.empty:
        logger.warning("No new subjects to insert - all records were duplicates or missing required fields")
        return
    
    # OPTIMIZATION: Vectorized data preparation
    def prepare_subject_data(row):
        # Normalize date fields properly
        start_date_col = column_mapping.get('start_date')
        raw_start_date = row.get(start_date_col) if start_date_col else None
        start_date = normalize_date(raw_start_date) if raw_start_date else None
        
        # Debug logging for first few rows
        if hasattr(prepare_subject_data, 'call_count'):
            prepare_subject_data.call_count += 1
        else:
            prepare_subject_data.call_count = 1
            
        if prepare_subject_data.call_count <= 3:
            logger.info(f"DEBUG: Raw start_date: {raw_start_date}, Normalized: {start_date}, Column: {start_date_col}")
        
        return (
            database_id,
            row['processed_subject_id'],
            str(row.get(column_mapping.get('site_id', ''), '')).strip() or None,
            str(row.get(column_mapping.get('sex', ''), '')).strip() or None,
            str(row.get(column_mapping.get('dob', ''), '')).strip() or None,
            str(row.get(column_mapping.get('arm', ''), '')).strip() or None,
            start_date,  # Use normalized date
            json.dumps(row.to_dict())
        )
    
    # OPTIMIZATION: Larger batch sizes (50,000 for maximum performance)
    batch_size = 50000
    new_subjects = [prepare_subject_data(row) for _, row in new_subjects_df.iterrows()]
    
    # OPTIMIZATION: Bulk insert with larger batches
    for i in range(0, len(new_subjects), batch_size):
        batch = new_subjects[i:i+batch_size]
        cursor.executemany("""
            INSERT INTO bronze.bronze_subjects 
            (database_id, subject_id, site_id, sex, dob, arm, start_date, raw_data)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, batch)
        
        # Only commit every 5 batches to improve performance
        if (i // batch_size + 1) % 5 == 0:
            conn.commit()
            logger.info(f"Committed batch {i//batch_size + 1} of {len(batch)} subjects")
        else:
            logger.info(f"Inserted batch {i//batch_size + 1} of {len(batch)} subjects (pending commit)")
    
    # Final commit for any remaining batches
    conn.commit()
    
    logger.info(f"Successfully inserted {len(new_subjects)} new subjects into bronze layer")

def insert_labs_to_bronze(cursor, df, database_id, conn):
    """Insert labs data directly into bronze layer with optimized performance"""
    logger.info(f"Inserting {len(df)} labs into bronze layer")
    logger.info(f"Labs DataFrame columns: {df.columns.tolist()}")
    logger.info(f"Labs DataFrame shape: {df.shape}")
    
    # Create bronze_labs table if not exists with all required columns
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bronze.bronze_labs (
            id SERIAL PRIMARY KEY,
            database_id INTEGER NOT NULL,
            subject_id VARCHAR(100),
            visitno VARCHAR(50),
            testcode VARCHAR(100),
            testname VARCHAR(200),
            resultvalue VARCHAR(100),
            units VARCHAR(50),
            reflow VARCHAR(100),
            refhigh VARCHAR(100),
            resultdate TEXT,
            raw_data JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(database_id, subject_id, visitno, testcode)
        )
    """)
    
    # Add missing columns if they don't exist
    missing_columns = {
        'resultvalue': 'VARCHAR(100)',
        'reflow': 'VARCHAR(100)',
        'refhigh': 'VARCHAR(100)',
        'resultdate': 'TEXT'
    }
    
    for col_name, col_type in missing_columns.items():
        cursor.execute(f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'bronze_labs' 
            AND table_schema = 'bronze' 
            AND column_name = '{col_name}'
        """)
        if not cursor.fetchone():
            cursor.execute(f"ALTER TABLE bronze.bronze_labs ADD COLUMN {col_name} {col_type}")
            logger.info(f"Added {col_name} column to bronze_labs table")
    
    # Get actual column names from existing table
    cursor.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'bronze_labs' 
        AND table_schema = 'bronze'
        ORDER BY ordinal_position
    """)
    existing_columns = [row[0] for row in cursor.fetchall()]
    logger.info(f"Existing bronze_labs columns: {existing_columns}")
    
    # Determine column mapping based on what actually exists
    col_visitno = 'visitno' if 'visitno' in existing_columns else ('visit_no' if 'visit_no' in existing_columns else None)
    col_testcode = 'testcode' if 'testcode' in existing_columns else ('test_code' if 'test_code' in existing_columns else None)
    col_testname = 'testname' if 'testname' in existing_columns else ('test_name' if 'test_name' in existing_columns else None)
    col_resultvalue = 'resultvalue' if 'resultvalue' in existing_columns else ('result_value' if 'result_value' in existing_columns else None)
    col_units = 'units' if 'units' in existing_columns else ('unit' if 'unit' in existing_columns else None)
    
    # Build INSERT statement dynamically based on existing columns
    insert_columns = ['database_id', 'subject_id']
    insert_values = ['%s', '%s']
    
    if col_visitno:
        insert_columns.append(col_visitno)
        insert_values.append('%s')
    if col_testcode:
        insert_columns.append(col_testcode)
        insert_values.append('%s')
    if col_testname:
        insert_columns.append(col_testname)
        insert_values.append('%s')
    if col_resultvalue:
        insert_columns.append(col_resultvalue)
        insert_values.append('%s')
    if col_units:
        insert_columns.append(col_units)
        insert_values.append('%s')
    if 'reflow' in existing_columns:
        insert_columns.append('reflow')
        insert_values.append('%s')
    if 'refhigh' in existing_columns:
        insert_columns.append('refhigh')
        insert_values.append('%s')
    if 'resultdate' in existing_columns:
        insert_columns.append('resultdate')
        insert_values.append('%s')
    
    insert_columns.append('raw_data')
    insert_values.append('%s')
    
    insert_sql = f"""
        INSERT INTO bronze.bronze_labs 
        ({', '.join(insert_columns)})
        VALUES ({', '.join(insert_values)})
    """
    
    logger.info(f"Dynamic INSERT SQL: {insert_sql}")
    
    # OPTIMIZATION: Bulk fetch existing labs instead of individual queries
    existing_labs = set()
    if col_visitno and col_testcode:
        cursor.execute(f"SELECT subject_id, {col_visitno}, {col_testcode} FROM bronze.bronze_labs WHERE database_id = %s", (database_id,))
        for row in cursor.fetchall():
            existing_labs.add((row[0], row[1], row[2]))
    
    logger.info(f"Found {len(existing_labs)} existing lab records")
    
    # OPTIMIZATION: Vectorized column mapping
    column_mapping = {}
    
    # Map subject_id variations
    subject_id_variations = ['subject_id', 'subjectid', 'subj_id', 'subjid', 'usubjid', 'participant_id', 'patient_id', 'subject', 'subj', 'id']
    
    # Map visit variations
    visit_variations = ['visit_no', 'visitno', 'visit', 'visitnum', 'visit_number', 'visitnumber', 'visitid', 'visit_id']
    
    # Map test_code variations
    test_code_variations = ['test_code', 'testcode', 'test', 'testname', 'test_name', 'lab_test', 'labtest', 'analyte', 'test_id']
    
    # Map result_value variations
    result_variations = ['result_value', 'resultvalue', 'result', 'value', 'test_result', 'testresult', 'measurement', 'reading']
    
    # Map units variations
    units_variations = ['units', 'unit', 'uom', 'unit_of_measure', 'unitofmeasure']
    
    # Map test_name variations
    test_name_variations = ['test_name', 'testname', 'test_description', 'testdescription', 'analyte_name', 'analytename']
    
    # Map reference range variations
    reflow_variations = ['reflow', 'ref_low', 'reflow', 'lower_ref', 'lowerref', 'low_ref', 'lowref', 'lower_limit', 'lowerlimit']
    refhigh_variations = ['refhigh', 'ref_high', 'refhigh', 'upper_ref', 'upperref', 'high_ref', 'highref', 'upper_limit', 'upperlimit']
    resultdate_variations = ['resultdate', 'result_date', 'resultdate', 'test_date', 'testdate', 'collection_date', 'collectiondate', 'specimen_date', 'specimendate']
    
    # Build column mapping
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in subject_id_variations and 'subject_id' not in column_mapping:
            column_mapping['subject_id'] = col
        elif col_lower in visit_variations and 'visit_no' not in column_mapping:
            column_mapping['visit_no'] = col
        elif col_lower in test_code_variations and 'test_code' not in column_mapping:
            column_mapping['test_code'] = col
        elif col_lower in result_variations and 'result_value' not in column_mapping:
            column_mapping['result_value'] = col
        elif col_lower in units_variations and 'units' not in column_mapping:
            column_mapping['units'] = col
        elif col_lower in test_name_variations and 'test_name' not in column_mapping:
            column_mapping['test_name'] = col
        elif col_lower in reflow_variations and 'reflow' not in column_mapping:
            column_mapping['reflow'] = col
        elif col_lower in refhigh_variations and 'refhigh' not in column_mapping:
            column_mapping['refhigh'] = col
        elif col_lower in resultdate_variations and 'resultdate' not in column_mapping:
            column_mapping['resultdate'] = col
    
    logger.info(f"Labs column mapping: {column_mapping}")
    
    # OPTIMIZATION: Check required columns
    required_columns = ['subject_id', 'visit_no', 'test_code']
    missing_columns = [col for col in required_columns if col not in column_mapping]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return
    
    # OPTIMIZATION: Vectorized data preparation
    # Create processed columns
    df['processed_subject_id'] = df[column_mapping['subject_id']].astype(str).str.strip()
    df['processed_visit_no'] = df[column_mapping['visit_no']].astype(str).str.strip()
    df['processed_test_code'] = df[column_mapping['test_code']].astype(str).str.strip()
    
    # OPTIMIZATION: Vectorized duplicate filtering
    df['lab_key'] = list(zip(df['processed_subject_id'], df['processed_visit_no'], df['processed_test_code']))
    mask = ~df['lab_key'].isin(existing_labs) & df['processed_subject_id'].notna() & df['processed_visit_no'].notna() & df['processed_test_code'].notna()
    new_labs_df = df[mask].copy()
    
    logger.info(f"Found {len(new_labs_df)} new lab records to insert, skipped {len(df) - len(new_labs_df)} duplicates")
    
    if new_labs_df.empty:
        logger.warning("No new lab records to insert - all records were duplicates or missing required fields")
        return
    
    # OPTIMIZATION: Vectorized data preparation
    def prepare_lab_data(row):
        data = [database_id, row['processed_subject_id']]
        
        if col_visitno:
            data.append(row['processed_visit_no'])
        if col_testcode:
            data.append(row['processed_test_code'])
        if col_testname:
            test_name_col = column_mapping.get('test_name')
            data.append(str(row.get(test_name_col, '')).strip() or None)
        if col_resultvalue:
            result_value_col = column_mapping.get('result_value')
            data.append(str(row.get(result_value_col, '')).strip() or None)
        if col_units:
            units_col = column_mapping.get('units')
            data.append(str(row.get(units_col, '')).strip() or None)
        if 'reflow' in existing_columns:
            reflow_col = column_mapping.get('reflow')
            data.append(str(row.get(reflow_col, '')).strip() or None)
        if 'refhigh' in existing_columns:
            refhigh_col = column_mapping.get('refhigh')
            data.append(str(row.get(refhigh_col, '')).strip() or None)
        if 'resultdate' in existing_columns:
            resultdate_col = column_mapping.get('resultdate')
            resultdate = normalize_date(row.get(resultdate_col)) if resultdate_col else None
            data.append(resultdate)
        
        data.append(json.dumps(row.to_dict()))
        return tuple(data)
    
    # OPTIMIZATION: Larger batch sizes (50,000 for maximum performance)
    batch_size = 50000
    new_labs = [prepare_lab_data(row) for _, row in new_labs_df.iterrows()]
    
    # OPTIMIZATION: Bulk insert with larger batches
    for i in range(0, len(new_labs), batch_size):
        batch = new_labs[i:i+batch_size]
        cursor.executemany(insert_sql, batch)
        
        # Only commit every 5 batches to improve performance
        if (i // batch_size + 1) % 5 == 0:
            conn.commit()
            logger.info(f"Committed batch of {len(batch)} labs")
        else:
            logger.info(f"Inserted batch of {len(batch)} labs (pending commit)")
    
    # Final commit for any remaining batches
    conn.commit()
    
    logger.info(f"Successfully inserted {len(new_labs)} new lab records into bronze layer")

def insert_aes_to_bronze(cursor, df, database_id, conn):
    """Insert adverse events data directly into bronze layer with deduplication"""
    logger.info(f"Inserting {len(df)} adverse events into bronze layer")
    logger.info(f"AES DataFrame columns: {df.columns.tolist()}")
    logger.info(f"AES DataFrame shape: {df.shape}")
    logger.info(f"Sample AES data: {df.head().to_dict() if not df.empty else 'Empty DataFrame'}")
    
    # Create bronze_aes table if not exists with all required columns
    # Force drop and recreate to fix column type issues
    cursor.execute("DROP TABLE IF EXISTS bronze.bronze_aes CASCADE")
    logger.info("Dropped existing bronze_aes table to fix column types")
    
    cursor.execute("""
        CREATE TABLE bronze.bronze_aes (
            id SERIAL PRIMARY KEY,
            database_id INTEGER NOT NULL,
            subject_id VARCHAR(100),
            ae_id VARCHAR(100),
            pt_name VARCHAR(200),
            pt_code VARCHAR(100),
            severity VARCHAR(50),
            related VARCHAR(20),
            serious VARCHAR(20),
            ae_start_date TEXT,
            raw_data JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(database_id, subject_id, ae_id)
        )
    """)
    logger.info("Recreated bronze_aes table with correct column types")
    
    # Check for existing AES to avoid duplicates
    existing_aes = set()
    cursor.execute("SELECT subject_id, ae_id FROM bronze.bronze_aes WHERE database_id = %s", (database_id,))
    for row in cursor.fetchall():
        # Trim whitespace from existing records to fix deduplication
        subject_id = str(row[0]).strip() if row[0] else None
        ae_id = str(row[1]).strip() if row[1] else None
        if subject_id and ae_id:
            existing_aes.add((subject_id, ae_id))
    
    logger.info(f"Found {len(existing_aes)} existing AE records")
    if existing_aes:
        logger.info(f"Sample existing AE keys: {list(existing_aes)[:5]}")
    
    # Filter out duplicates with comprehensive column mapping
    new_aes = []
    skipped_count = 0
    
    # Create comprehensive column mapping for AES
    column_mapping = {}
    
    # Map subject_id variations
    subject_id_variations = ['subject_id', 'subjectid', 'subj_id', 'subjid', 'usubjid', 'participant_id', 'patient_id', 'subject', 'subj', 'id']
    
    # Map ae_id variations
    ae_id_variations = ['ae_id', 'aeid', 'adverse_event_id', 'adverseeventid', 'ae_seq', 'aeseq', 'event_id', 'eventid']
    
    # Map pt_name variations
    pt_name_variations = ['pt_name', 'ptname', 'preferred_term', 'preferredterm', 'term', 'adverse_event_term', 'adverseeventterm', 'ae_term', 'aeterm']
    
    # Map severity variations
    severity_variations = ['severity', 'sev', 'grade', 'severity_grade', 'severitygrade', 'intensity', 'seriousness']
    
    # Map pt_code variations
    pt_code_variations = ['pt_code', 'ptcode', 'preferred_term_code', 'preferredtermcode', 'meddra_code', 'meddracode', 'event_code', 'eventcode']
    
    # Map related variations
    related_variations = ['related', 'relation', 'relationship', 'relatedness', 'causality', 'causal', 'study_drug', 'studydrug']
    
    # Map serious variations
    serious_variations = ['serious', 'seriousness', 'serious_adverse_event', 'seriousae', 'sae', 'serious_criteria', 'seriouscriteria']
    
    # Map ae_start_date variations
    ae_start_date_variations = ['onset_date', 'ae_start_date', 'aestart', 'ae_start', 'start_date', 'event_start_date', 'eventstartdate', 'ae_start_dt', 'aestdt']
    
    # Build column mapping with case-insensitive debugging
    logger.info(f"🔍 Available columns in AE file: {df.columns.tolist()}")
    
    for col in df.columns:
        col_original = col  # Keep original case
        col_lower = col.lower().strip()
        logger.info(f"🔍 Processing column: '{col_original}' (lower: '{col_lower}')")
        
        if col_lower in subject_id_variations and 'subject_id' not in column_mapping:
            column_mapping['subject_id'] = col_original
            logger.info(f"✅ Mapped subject_id to: '{col_original}'")
        elif col_lower in ae_id_variations and 'ae_id' not in column_mapping:
            column_mapping['ae_id'] = col_original
            logger.info(f"✅ Mapped ae_id to: '{col_original}'")
        elif col_lower in pt_name_variations and 'pt_name' not in column_mapping:
            column_mapping['pt_name'] = col_original
            logger.info(f"✅ Mapped pt_name to: '{col_original}'")
        elif col_lower in severity_variations and 'severity' not in column_mapping:
            column_mapping['severity'] = col_original
            logger.info(f"✅ Mapped severity to: '{col_original}'")
        elif col_lower in pt_code_variations and 'pt_code' not in column_mapping:
            column_mapping['pt_code'] = col_original
            logger.info(f"✅ Mapped pt_code to: '{col_original}'")
        elif col_lower in related_variations and 'related' not in column_mapping:
            column_mapping['related'] = col_original
            logger.info(f"✅ Mapped related to: '{col_original}'")
        elif col_lower in serious_variations and 'serious' not in column_mapping:
            column_mapping['serious'] = col_original
            logger.info(f"✅ Mapped serious to: '{col_original}'")
        elif col_lower in ae_start_date_variations and 'ae_start_date' not in column_mapping:
            column_mapping['ae_start_date'] = col_original
            logger.info(f"✅ Mapped ae_start_date to: '{col_original}'")
        else:
            logger.info(f"❓ Column '{col_original}' not matched to any variation")
    
    logger.info(f"AES column mapping: {column_mapping}")
    
    # Debug: Check if ae_start_date mapping exists
    if 'ae_start_date' not in column_mapping:
        logger.warning("ae_start_date not found in column mapping - available columns: " + str(df.columns.tolist()))
    else:
        logger.info(f"✅ ae_start_date mapped to column: {column_mapping['ae_start_date']}")
    
    # Debug: Show sample data from the mapped column
    if 'ae_start_date' in column_mapping and len(df) > 0:
        mapped_col = column_mapping['ae_start_date']
        sample_values = df[mapped_col].head(5).tolist()
        logger.info(f"📊 Sample ae_start_date values from column '{mapped_col}': {sample_values}")
    
    for idx, row in df.iterrows():
        # Get values using mapped columns
        subject_id = row.get(column_mapping.get('subject_id')) if column_mapping.get('subject_id') else None
        ae_id = row.get(column_mapping.get('ae_id')) if column_mapping.get('ae_id') else None
        
        # Clean up values
        if subject_id:
            subject_id = str(subject_id).strip()
        if ae_id:
            ae_id = str(ae_id).strip()
        
        # Debug logging for first few rows
        if idx < 5:
            logger.info(f"Row {idx}: subject_id={subject_id}, ae_id={ae_id}")
            logger.info(f"Row {idx} available columns: {list(row.keys())}")
            logger.info(f"Row {idx} mapping: {column_mapping}")
        
        ae_key = (subject_id, ae_id)
        if subject_id and ae_id:
            if ae_key not in existing_aes:
                # FIXED: Use ae_start_date mapping directly
                ae_start_date = normalize_date(row.get(column_mapping.get('ae_start_date'))) if column_mapping.get('ae_start_date') else None
                
                # Debug logging for first few rows
                if idx < 3:
                    logger.info(f"🔍 Row {idx} ae_start_date processing:")
                    logger.info(f"   - ae_start_date_col: {column_mapping.get('ae_start_date')}")
                    logger.info(f"   - raw_ae_start_date: {row.get(column_mapping.get('ae_start_date'))}")
                    logger.info(f"   - normalized_ae_start_date: {ae_start_date}")
                
                new_aes.append({
                    'database_id': database_id,
                    'subject_id': subject_id,
                    'ae_id': ae_id,
                    'pt_name': row.get(column_mapping.get('pt_name')) if column_mapping.get('pt_name') else None,
                    'pt_code': row.get(column_mapping.get('pt_code')) if column_mapping.get('pt_code') else None,
                    'severity': row.get(column_mapping.get('severity')) if column_mapping.get('severity') else None,
                    'related': row.get(column_mapping.get('related')) if column_mapping.get('related') else None,
                    'serious': row.get(column_mapping.get('serious')) if column_mapping.get('serious') else None,
                    'ae_start_date': ae_start_date,
                    'raw_data': row.to_dict()
                })
                existing_aes.add(ae_key)
            else:
                skipped_count += 1
                logger.info(f"Skipping duplicate AE: {ae_key}")
        else:
            logger.warning(f"Missing required fields - subject_id: {subject_id}, ae_id: {ae_id}")
    
    logger.info(f"Found {len(new_aes)} new AE records to insert, skipped {skipped_count} duplicates")
    
    if not new_aes:
        logger.info("No new AE records to insert")
        return 0  # Return 0 for no new records
    
    # Batch insert with optimized performance
    batch_size = 50000
    for i in range(0, len(new_aes), batch_size):
        batch = new_aes[i:i+batch_size]
        values = [(item['database_id'], item['subject_id'], item['ae_id'],
                  item['pt_name'], item['pt_code'], item['severity'], 
                  item['related'], item['serious'], 
                  item['ae_start_date'], json.dumps(item['raw_data'])) 
                 for item in batch]
        
        cursor.executemany("""
            INSERT INTO bronze.bronze_aes 
            (database_id, subject_id, ae_id, pt_name, pt_code, severity, related, serious, ae_start_date, raw_data)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, values)
        
        # Only commit every 5 batches to improve performance
        if (i // batch_size + 1) % 5 == 0:
            conn.commit()
            logger.info(f"Committed batch of {len(values)} AES")
        else:
            logger.info(f"Inserted batch of {len(values)} AES (pending commit)")
    
    # Final commit for any remaining batches
    conn.commit()
    
    logger.info(f"Successfully inserted {len(new_aes)} new AE records into bronze layer")
    return len(new_aes)  # Return actual count of new records

@app.route('/api/files/list', methods=['GET'])
def list_files():
    """List uploaded files for the current database session"""
    try:
        database_id = request.args.get('database_id')
        
        # Use current database session if database_id not provided
        if not database_id and current_database_session['database_id']:
            database_id = current_database_session['database_id']
        elif not database_id:
            return jsonify({'success': False, 'message': 'No database selected'})
        
        logger.info(f"Loading files for database_id: {database_id}")
        
        # Get database connection
        db_connection = DatabaseConnection.query.get(database_id)
        if not db_connection:
            return jsonify({'success': False, 'message': 'Database connection not found'})
        
        # Query uploaded_files table directly
        conn = psycopg2.connect(db_connection.url)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, original_filename, file_type, file_size, record_count, processed, created_at
            FROM uploaded_files 
            WHERE database_id = %s 
            ORDER BY created_at DESC
        """, (database_id,))
        
        files_data = cursor.fetchall()
        cursor.close()
        conn.close()
        
        logger.info(f"Found {len(files_data)} files for database {database_id}")
        
        return jsonify({
            'success': True,
            'data': [{
                'id': file[0],
                'original_filename': file[1],
                'file_type': file[2],
                'file_size': file[3],
                'record_count': file[4],
                'processed': file[5],
                'uploaded_at': file[6].isoformat() if file[6] else None,
                'database_id': database_id
            } for file in files_data]
        })
        
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/files/history')
def api_file_history():
    """Return full file upload history scoped to the selected database"""
    try:
        database_id = request.args.get('database_id') or current_database_session.get('database_id')
        if not database_id:
            return jsonify({'success': False, 'message': 'No database selected'})
        database_id = int(database_id)
        db_conn = DatabaseConnection.query.get(database_id)
        if not db_conn:
            return jsonify({'success': False, 'message': 'Database not found'})
        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, original_filename, file_type, file_size, record_count,
                   processed, created_at
            FROM public.uploaded_files
            WHERE database_id = %s
            ORDER BY created_at DESC
        """, (database_id,))
        rows = cursor.fetchall()
        cursor.close(); conn.close()
        return jsonify({'success': True, 'database_id': database_id, 'files': [
            {'id': r[0], 'filename': r[1], 'file_type': r[2], 'file_size': r[3],
             'record_count': r[4] or 0, 'processed': r[5], 'bronze_loaded': r[5],
             'uploaded_at': r[6].isoformat() if r[6] else None}
            for r in rows
        ]})
    except Exception as e:
        logger.error(f"api_file_history error: {e}")
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/unprocessed/data')
def api_unprocessed_data():
    """Return all unprocessed / rejected records captured during Silver layer processing"""
    try:
        database_id = request.args.get('database_id') or current_database_session.get('database_id')
        if not database_id:
            return jsonify({'success': False, 'message': 'No database selected'})
        database_id = int(database_id)
        db_conn = DatabaseConnection.query.get(database_id)
        if not db_conn:
            return jsonify({'success': False, 'message': 'Database not found'})
        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        cursor = conn.cursor()

        # Check table exists first
        cursor.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema='silver' AND table_name='unprocessed_records'
            )
        """)
        if not cursor.fetchone()[0]:
            cursor.close(); conn.close()
            return jsonify({'success': True, 'records': [], 'summary': {}, 'total': 0})

        cursor.execute("""
            SELECT id, source_table, subject_id, column_name, raw_value, reason, row_data, created_at
            FROM silver.unprocessed_records
            WHERE database_id = %s
            ORDER BY created_at DESC, source_table, subject_id
        """, (database_id,))
        rows = cursor.fetchall()

        # Summary counts
        cursor.execute("""
            SELECT source_table, reason, COUNT(*)
            FROM silver.unprocessed_records
            WHERE database_id = %s
            GROUP BY source_table, reason
            ORDER BY COUNT(*) DESC
        """, (database_id,))
        summary_rows = cursor.fetchall()
        cursor.close(); conn.close()

        # Map source_table → file type label
        src_label = {'bronze_subjects': 'Subjects', 'bronze_labs': 'Labs', 'bronze_aes': 'Adverse Events'}

        return jsonify({'success': True,
            'total': len(rows),
            'records': [
                {'id': r[0],
                 'source': src_label.get(r[1], r[1]),
                 'source_table': r[1],
                 'subject_id': r[2] or '—',
                 'column_name': r[3] or '—',
                 'raw_value': r[4] or '—',
                 'reason': r[5],
                 'row_data': r[6],
                 'detected_at': r[7].isoformat() if r[7] else None}
                for r in rows
            ],
            'summary': [
                {'source': src_label.get(r[0], r[0]), 'reason': r[1], 'count': r[2]}
                for r in summary_rows
            ]
        })
    except Exception as e:
        logger.error(f"api_unprocessed_data error: {e}")
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/debug/bronze-counts', methods=['GET'])
def debug_bronze_counts():
    """Debug endpoint to check bronze table contents"""
    try:
        database_id = request.args.get('database_id')
        if not database_id and current_database_session['database_id']:
            database_id = current_database_session['database_id']
        elif not database_id:
            return jsonify({'success': False, 'message': 'No database selected'})
        
        db_conn = DatabaseConnection.query.get(database_id)
        if not db_conn:
            return jsonify({'success': False, 'message': 'Database connection not found'})
        
        conn = psycopg2.connect(db_conn.url)
        cursor = conn.cursor()
        
        # Get counts from all bronze tables
        cursor.execute("SELECT COUNT(*) FROM bronze.bronze_subjects WHERE database_id = %s", (database_id,))
        subjects_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM bronze.bronze_labs WHERE database_id = %s", (database_id,))
        labs_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM bronze.bronze_aes WHERE database_id = %s", (database_id,))
        aes_count = cursor.fetchone()[0]
        
        # Get sample data from labs
        cursor.execute("SELECT subject_id, visit_no, test_code FROM bronze.bronze_labs WHERE database_id = %s LIMIT 5", (database_id,))
        sample_labs = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'counts': {
                'subjects': subjects_count,
                'labs': labs_count,
                'aes': aes_count
            },
            'sample_labs': sample_labs
        })
        
    except Exception as e:
        logger.error(f"Error debugging bronze counts: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/analytics/metrics', methods=['GET'])
def get_analytics_metrics():
    """Get analytics metrics from current database session"""
    try:
        # Accept database_id from query param or fall back to session
        database_id = request.args.get('database_id') or current_database_session['database_id']
        if not database_id:
            return jsonify({'success': False, 'message': 'No database selected'})
        database_id = int(database_id)

        db_conn = DatabaseConnection.query.get(database_id)
        if not db_conn:
            return jsonify({'success': False, 'message': 'Database not found'})
        
        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        cursor = conn.cursor()
        
        # Use correct gold table names
        try:
            cursor.execute("SELECT COUNT(*) FROM gold.gold_subjects WHERE database_id = %s", (database_id,))
            total_subjects = cursor.fetchone()[0]
        except Exception:
            conn.rollback()
            total_subjects = 0
        
        try:
            cursor.execute("SELECT COUNT(*) FROM gold.gold_ae WHERE database_id = %s", (database_id,))
            total_aes = cursor.fetchone()[0]
        except Exception:
            conn.rollback()
            total_aes = 0
        
        try:
            cursor.execute("SELECT COUNT(*) FROM gold.gold_ae WHERE database_id = %s AND serious_flag = 'Y'", (database_id,))
            total_serious = cursor.fetchone()[0]
        except Exception:
            conn.rollback()
            total_serious = 0
        
        try:
            cursor.execute("SELECT COUNT(*) FROM gold.gold_labs WHERE database_id = %s AND alt_3x_uln_flag = TRUE", (database_id,))
            total_abnormal_labs = cursor.fetchone()[0]
        except Exception:
            conn.rollback()
            total_abnormal_labs = 0
        
        try:
            cursor.execute("SELECT COUNT(*) FROM gold.gold_labs WHERE database_id = %s", (database_id,))
            total_labs = cursor.fetchone()[0]
        except Exception:
            conn.rollback()
            total_labs = 0
        
        avg_lab_ratio = (total_abnormal_labs / total_labs * 100) if total_labs > 0 else 0
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'metrics': {
                'totalSubjects': total_subjects,
                'totalAEs': total_aes,
                'seriousAEs': total_serious,
                'avgLabRatio': round(avg_lab_ratio, 1)
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting analytics metrics: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/analytics/ae-trend')
def get_ae_trend():
    """Get AE trend data from database"""
    try:
        database_id = request.args.get('database_id') or current_database_session['database_id']
        if not database_id:
            return jsonify({'success': False, 'message': 'No database selected'})
        database_id = int(database_id)
        db_conn = DatabaseConnection.query.get(database_id)
        if not db_conn:
            return jsonify({'success': False, 'message': 'Database not found'})
        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DATE_TRUNC('month', ae_start_date) as month, 
                   COUNT(*) as ae_count
            FROM gold.gold_ae
            WHERE database_id = %s AND ae_start_date IS NOT NULL
            GROUP BY DATE_TRUNC('month', ae_start_date)
            ORDER BY month DESC
            LIMIT 12
        """, (database_id,))
        
        data = cursor.fetchall()
        cursor.close()
        conn.close()
        
        dates = [row[0].strftime('%Y-%m') for row in reversed(data)]
        counts = [row[1] for row in reversed(data)]
        
        return jsonify({
            'success': True,
            'data': {'dates': dates, 'counts': counts}
        })
        
    except Exception as e:
        logger.error(f"Error getting AE trend: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/analytics/site-performance')
def get_site_performance():
    """Get site performance data from database"""
    try:
        database_id = request.args.get('database_id') or current_database_session['database_id']
        if not database_id:
            return jsonify({'success': False, 'message': 'No database selected'})
        database_id = int(database_id)
        db_conn = DatabaseConnection.query.get(database_id)
        if not db_conn:
            return jsonify({'success': False, 'message': 'Database not found'})
        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT site_id, COUNT(*) as count
            FROM gold.gold_subjects
            WHERE database_id = %s
            GROUP BY site_id
            ORDER BY count DESC
            LIMIT 10
        """, (database_id,))
        
        sites = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'data': {
                'sites': [row[0] for row in sites],
                'counts': [row[1] for row in sites]
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting site performance: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/predictions/data')
def get_predictions_data():
    """Get predictions data from database"""
    try:
        # Accept database_id from query param or fall back to session
        database_id = request.args.get('database_id') or current_database_session['database_id']
        if not database_id:
            return jsonify({'success': False, 'message': 'No database selected'})
        database_id = int(database_id)
        db_conn = DatabaseConnection.query.get(database_id)
        
        if not db_conn:
            return jsonify({'success': False, 'message': 'Database connection not found'})
        
        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT site_id, predicted_total_ae, predicted_total_serious_events, 
                   predicted_total_ae_signal_count, predicted_avg_lab_ratio, 
                   predicted_new_subjects, predicted_risk_group, prediction_date
            FROM prediction.site_predictions 
            WHERE database_id = %s
            ORDER BY site_id ASC, prediction_date DESC
            LIMIT 20
        """, (database_id,))
        
        predictions = cursor.fetchall()
        
        # Get site summary data
        cursor.execute("""
            SELECT site_id, total_subjects, total_lab_tests, total_adverse_events, 
                   total_serious_events, total_ae_signal_count
            FROM gold.phi_site_summary 
            WHERE database_id = %s
            ORDER BY site_id
        """, (database_id,))
        
        site_summary = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'data': {
                'predictions': [
                    {
                        'site_id': p[0],
                        'predicted_total_ae': p[1],
                        'predicted_total_serious_events': p[2],
                        'predicted_total_ae_signal_count': p[3],
                        'predicted_avg_lab_ratio': p[4],
                        'predicted_new_subjects': p[5],
                        'predicted_risk_group': p[6],
                        'prediction_date': p[7].strftime('%Y-%m-%d') if p[7] else None
                    }
                    for p in predictions
                ],
                'site_summary': [
                    {
                        'site_id': s[0],
                        'total_subjects': s[1],
                        'total_lab_tests': s[2],
                        'total_adverse_events': s[3],
                        'total_serious_events': s[4],
                        'total_ae_signal_count': s[5]
                    }
                    for s in site_summary
                ]
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting predictions data: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/analytics/data')
def get_analytics_data():
    """Get analytics data from database"""
    try:
        database_id = request.args.get('database_id') or current_database_session['database_id']
        if not database_id:
            return jsonify({'success': False, 'message': 'No database selected'})
        database_id = int(database_id)

        db_conn = DatabaseConnection.query.get(database_id)
        if not db_conn:
            return jsonify({'success': False, 'message': 'Database connection not found'})

        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        cursor = conn.cursor()

        swh, sparams = _subj_filters(database_id)
        awh, aparams = _ae_filters(database_id)
        lwh, lparams = _lab_filters(database_id)

        cursor.execute(f"SELECT COUNT(*) FROM gold.gold_subjects WHERE {swh}", sparams)
        total_subjects = cursor.fetchone()[0]

        cursor.execute(f"SELECT COUNT(*) FROM gold.gold_labs WHERE {lwh}", lparams)
        total_labs = cursor.fetchone()[0]

        cursor.execute(f"SELECT COUNT(*) FROM gold.gold_ae WHERE {awh}", aparams)
        total_aes = cursor.fetchone()[0]

        cursor.execute(f"SELECT COUNT(*) FROM gold.gold_ae WHERE {awh} AND serious_flag='Y'", aparams)
        serious_aes = cursor.fetchone()[0]

        cursor.execute(f"""
            SELECT COALESCE(pt_name, pt_code, 'Unknown'), COUNT(*) FROM gold.gold_ae
            WHERE {awh} GROUP BY COALESCE(pt_name, pt_code, 'Unknown') ORDER BY 2 DESC LIMIT 10
        """, aparams)
        top_aes_rows = cursor.fetchall()

        cursor.execute(f"""
            SELECT site_id, COUNT(*) FROM gold.gold_subjects WHERE {swh}
            GROUP BY site_id ORDER BY 2 DESC
        """, sparams)
        site_distribution = cursor.fetchall()

        cursor.close(); conn.close()

        return jsonify({
            'success': True,
            'data': {
                'total_subjects': total_subjects,
                'total_labs': total_labs,
                'total_aes': total_aes,
                'serious_aes': serious_aes,
                'top_aes': [{'pt_name': r[0], 'count': r[1]} for r in top_aes_rows],
                'site_distribution': [{'site_id': s[0], 'subject_count': s[1]} for s in site_distribution]
            }
        })

    except Exception as e:
        logger.error(f"Error getting analytics data: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

def _subj_filters(database_id):
    """Build WHERE clause + params for gold_subjects based on request.args."""
    where = ["database_id=%s"]
    params = [database_id]
    site_id = request.args.get('site_id')
    arm = request.args.get('arm')
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')
    if site_id: where.append("site_id=%s"); params.append(site_id)
    if arm: where.append("arm=%s"); params.append(arm)
    if date_from: where.append("start_date>=%s"); params.append(date_from)
    if date_to: where.append("start_date<=%s"); params.append(date_to)
    return ' AND '.join(where), params

def _ae_filters(database_id):
    """Build WHERE clause + params for gold_ae based on request.args."""
    where = ["database_id=%s"]
    params = [database_id]
    severity = request.args.get('severity')
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')
    site_id = request.args.get('site_id')
    arm = request.args.get('arm')
    if severity: where.append("severity=%s"); params.append(severity)
    if date_from: where.append("ae_start_date>=%s"); params.append(date_from)
    if date_to: where.append("ae_start_date<=%s"); params.append(date_to)
    if site_id or arm:
        # join to subjects for site/arm filter
        where.append("""subject_hash_id IN (
            SELECT subject_hash_id FROM gold.gold_subjects
            WHERE database_id=%s""" + (" AND site_id=%s" if site_id else "") + (" AND arm=%s" if arm else "") + ")")
        params.append(database_id)
        if site_id: params.append(site_id)
        if arm: params.append(arm)
    return ' AND '.join(where), params

def _lab_filters(database_id):
    """Build WHERE clause + params for gold_labs based on request.args."""
    where = ["database_id=%s"]
    params = [database_id]
    test_code = request.args.get('test_code')
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')
    site_id = request.args.get('site_id')
    arm = request.args.get('arm')
    if test_code: where.append("test_code=%s"); params.append(test_code)
    if date_from: where.append("result_date>=%s"); params.append(date_from)
    if date_to: where.append("result_date<=%s"); params.append(date_to)
    if site_id or arm:
        where.append("""subject_hash_id IN (
            SELECT subject_hash_id FROM gold.gold_subjects
            WHERE database_id=%s""" + (" AND site_id=%s" if site_id else "") + (" AND arm=%s" if arm else "") + ")")
        params.append(database_id)
        if site_id: params.append(site_id)
        if arm: params.append(arm)
    return ' AND '.join(where), params

@app.route('/api/analytics/enrollment-by-site')
def api_enrollment_by_site():
    try:
        database_id = int(request.args.get('database_id') or current_database_session['database_id'])
        db_conn = DatabaseConnection.query.get(database_id)
        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        cursor = conn.cursor()
        wh, params = _subj_filters(database_id)
        cursor.execute(f"SELECT COALESCE(site_id,'Unknown'), COUNT(*) FROM gold.gold_subjects WHERE {wh} GROUP BY site_id ORDER BY site_id", params)
        rows = cursor.fetchall()
        cursor.close(); conn.close()
        return jsonify({'success': True, 'sites': [r[0] for r in rows], 'counts': [r[1] for r in rows]})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/analytics/gender-distribution')
def api_gender_distribution():
    try:
        database_id = int(request.args.get('database_id') or current_database_session['database_id'])
        db_conn = DatabaseConnection.query.get(database_id)
        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        cursor = conn.cursor()
        wh, params = _subj_filters(database_id)
        cursor.execute(f"SELECT COALESCE(sex,'Unknown'), COUNT(*) FROM gold.gold_subjects WHERE {wh} GROUP BY sex", params)
        rows = cursor.fetchall()
        cursor.close(); conn.close()
        return jsonify({'success': True, 'labels': [r[0] for r in rows], 'values': [r[1] for r in rows]})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/analytics/age-distribution')
def api_age_distribution():
    try:
        database_id = int(request.args.get('database_id') or current_database_session['database_id'])
        db_conn = DatabaseConnection.query.get(database_id)
        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        cursor = conn.cursor()
        wh, params = _subj_filters(database_id)
        cursor.execute(f"SELECT age_at_start FROM gold.gold_subjects WHERE {wh} AND age_at_start IS NOT NULL", params)
        rows = cursor.fetchall()
        cursor.close(); conn.close()
        return jsonify({'success': True, 'ages': [r[0] for r in rows]})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/analytics/arm-distribution')
def api_arm_distribution():
    try:
        database_id = int(request.args.get('database_id') or current_database_session['database_id'])
        db_conn = DatabaseConnection.query.get(database_id)
        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        cursor = conn.cursor()
        wh, params = _subj_filters(database_id)
        cursor.execute(f"SELECT COALESCE(arm,'Unknown'), COUNT(*) FROM gold.gold_subjects WHERE {wh} GROUP BY arm ORDER BY arm", params)
        rows = cursor.fetchall()
        cursor.close(); conn.close()
        return jsonify({'success': True, 'arms': [r[0] for r in rows], 'counts': [r[1] for r in rows]})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/analytics/ae-by-severity')
def api_ae_by_severity():
    try:
        database_id = int(request.args.get('database_id') or current_database_session['database_id'])
        db_conn = DatabaseConnection.query.get(database_id)
        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        cursor = conn.cursor()
        wh, params = _ae_filters(database_id)
        cursor.execute(f"SELECT COALESCE(severity,'Unknown'), COUNT(*) FROM gold.gold_ae WHERE {wh} GROUP BY severity ORDER BY severity", params)
        rows = cursor.fetchall()
        cursor.close(); conn.close()
        return jsonify({'success': True, 'severities': [r[0] for r in rows], 'counts': [r[1] for r in rows]})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/analytics/top-aes')
def api_top_aes():
    try:
        database_id = int(request.args.get('database_id') or current_database_session['database_id'])
        db_conn = DatabaseConnection.query.get(database_id)
        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        cursor = conn.cursor()
        wh, params = _ae_filters(database_id)
        cursor.execute(f"""
            SELECT COALESCE(pt_name, pt_code, 'Unknown'), COUNT(*) FROM gold.gold_ae
            WHERE {wh} GROUP BY COALESCE(pt_name, pt_code, 'Unknown')
            ORDER BY COUNT(*) DESC LIMIT 15
        """, params)
        rows = cursor.fetchall()
        cursor.close(); conn.close()
        return jsonify({'success': True, 'names': [r[0] for r in rows], 'counts': [r[1] for r in rows]})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/analytics/ae-by-site')
def api_ae_by_site():
    try:
        database_id = int(request.args.get('database_id') or current_database_session['database_id'])
        db_conn = DatabaseConnection.query.get(database_id)
        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        cursor = conn.cursor()
        wh, params = _ae_filters(database_id)
        cursor.execute(f"""
            SELECT gs.site_id, COUNT(ga.id) FROM gold.gold_ae ga
            JOIN gold.gold_subjects gs ON ga.subject_hash_id=gs.subject_hash_id AND ga.database_id=gs.database_id
            WHERE {wh.replace('database_id=%s', 'ga.database_id=%s', 1)} GROUP BY gs.site_id ORDER BY gs.site_id
        """, params)
        rows = cursor.fetchall()
        cursor.close(); conn.close()
        return jsonify({'success': True, 'sites': [r[0] for r in rows], 'counts': [r[1] for r in rows]})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/analytics/ae-timeline')
def api_ae_timeline():
    try:
        database_id = int(request.args.get('database_id') or current_database_session['database_id'])
        db_conn = DatabaseConnection.query.get(database_id)
        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        cursor = conn.cursor()
        wh, params = _ae_filters(database_id)
        cursor.execute(f"""
            SELECT DATE_TRUNC('month', ae_start_date), COUNT(*) FROM gold.gold_ae
            WHERE {wh} AND ae_start_date IS NOT NULL
            GROUP BY DATE_TRUNC('month', ae_start_date) ORDER BY 1
        """, params)
        rows = cursor.fetchall()
        cursor.close(); conn.close()
        return jsonify({'success': True,
                        'dates': [r[0].strftime('%Y-%m') for r in rows],
                        'counts': [r[1] for r in rows]})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/analytics/alt-trend')
def api_alt_trend():
    try:
        database_id = int(request.args.get('database_id') or current_database_session['database_id'])
        db_conn = DatabaseConnection.query.get(database_id)
        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        cursor = conn.cursor()
        wh, params = _lab_filters(database_id)
        # Force ALT for this endpoint regardless of test_code filter
        if 'test_code=%s' not in wh:
            wh += " AND test_code=%s"
            params.append('ALT')
        cursor.execute(f"""
            SELECT visit_no, ROUND(AVG(result_value)::numeric,2) FROM gold.gold_labs
            WHERE {wh} AND result_value IS NOT NULL
            GROUP BY visit_no ORDER BY visit_no
        """, params)
        rows = cursor.fetchall()
        cursor.close(); conn.close()
        return jsonify({'success': True, 'visits': [r[0] for r in rows], 'avg_values': [float(r[1]) for r in rows]})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/analytics/abnormal-labs')
def api_abnormal_labs():
    try:
        database_id = int(request.args.get('database_id') or current_database_session['database_id'])
        db_conn = DatabaseConnection.query.get(database_id)
        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        cursor = conn.cursor()
        wh, params = _lab_filters(database_id)
        cursor.execute(f"""
            SELECT test_code, COUNT(*) FROM gold.gold_labs
            WHERE {wh} AND safety_signal IS NOT NULL AND safety_signal != ''
            GROUP BY test_code ORDER BY COUNT(*) DESC
        """, params)
        rows = cursor.fetchall()
        cursor.close(); conn.close()
        return jsonify({'success': True, 'tests': [r[0] for r in rows], 'counts': [r[1] for r in rows]})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/analytics/safety-flags')
def api_safety_flags():
    try:
        database_id = int(request.args.get('database_id') or current_database_session['database_id'])
        db_conn = DatabaseConnection.query.get(database_id)
        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        cursor = conn.cursor()
        wh, params = _lab_filters(database_id)
        cursor.execute(f"""
            SELECT
                SUM(CASE WHEN test_code='ALT' AND uln_ratio>=3 THEN 1 ELSE 0 END) AS alt_3x,
                SUM(CASE WHEN test_code='AST' AND uln_ratio>=3 THEN 1 ELSE 0 END) AS ast_3x,
                SUM(CASE WHEN test_code='TBILI' AND uln_ratio>=2 THEN 1 ELSE 0 END) AS bili_2x,
                SUM(CASE WHEN test_code='ALT' AND uln_ratio>=3 AND
                    subject_hash_id IN (
                        SELECT subject_hash_id FROM gold.gold_labs
                        WHERE database_id=%s AND test_code='TBILI' AND uln_ratio>=2
                    ) THEN 1 ELSE 0 END) AS hys_law
            FROM gold.gold_labs WHERE {wh}
        """, [database_id] + params)
        row = cursor.fetchone()
        cursor.close(); conn.close()
        flags = ['ALT ≥3×ULN','AST ≥3×ULN','BILIRUBIN ≥2×ULN',"HYS LAW"]
        counts = [int(row[0] or 0), int(row[1] or 0), int(row[2] or 0), int(row[3] or 0)]
        return jsonify({'success': True, 'flags': flags, 'counts': counts})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/analytics/site-safety')
def api_site_safety():
    try:
        database_id = int(request.args.get('database_id') or current_database_session['database_id'])
        db_conn = DatabaseConnection.query.get(database_id)
        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        cursor = conn.cursor()
        wh, params = _lab_filters(database_id)
        cursor.execute(f"""
            SELECT gs.site_id, COUNT(gl.lab_fact_id) FROM gold.gold_labs gl
            JOIN gold.gold_subjects gs ON gl.subject_hash_id=gs.subject_hash_id AND gl.database_id=gs.database_id
            WHERE {wh.replace('database_id=%s', 'gl.database_id=%s', 1)} AND gl.safety_signal IS NOT NULL AND gl.safety_signal!=''
            GROUP BY gs.site_id ORDER BY COUNT(*) DESC
        """, params)
        rows = cursor.fetchall()
        cursor.close(); conn.close()
        return jsonify({'success': True, 'sites': [r[0] for r in rows], 'counts': [r[1] for r in rows]})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/analytics/serious-vs-nonserious')
def api_serious_vs_nonserious():
    try:
        database_id = int(request.args.get('database_id') or current_database_session['database_id'])
        db_conn = DatabaseConnection.query.get(database_id)
        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        cursor = conn.cursor()
        wh, params = _ae_filters(database_id)
        cursor.execute(f"""
            SELECT CASE WHEN serious_flag='Y' THEN 'Serious' ELSE 'Non-Serious' END, COUNT(*)
            FROM gold.gold_ae WHERE {wh} GROUP BY 1
        """, params)
        rows = cursor.fetchall()
        cursor.close(); conn.close()
        return jsonify({'success': True, 'labels': [r[0] for r in rows], 'values': [r[1] for r in rows]})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/analytics/lab-trend-over-time')
def api_lab_trend_over_time():
    try:
        database_id = int(request.args.get('database_id') or current_database_session['database_id'])
        test_code = request.args.get('test_code', 'ALT')
        db_conn = DatabaseConnection.query.get(database_id)
        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        cursor = conn.cursor()
        wh, params = _lab_filters(database_id)
        # Override test_code with explicit param
        if 'test_code=%s' not in wh:
            wh += " AND test_code=%s"
            params.append(test_code)
        cursor.execute(f"""
            SELECT DATE_TRUNC('month', result_date), ROUND(AVG(result_value)::numeric,2)
            FROM gold.gold_labs
            WHERE {wh} AND result_date IS NOT NULL AND result_value IS NOT NULL
            GROUP BY 1 ORDER BY 1
        """, params)
        rows = cursor.fetchall()
        cursor.close(); conn.close()
        return jsonify({'success': True,
                        'dates': [r[0].strftime('%Y-%m') for r in rows],
                        'values': [float(r[1]) for r in rows],
                        'test_code': test_code})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/analytics/filters')
def api_analytics_filters():
    """Return distinct filter values for the analytics page"""
    try:
        database_id = int(request.args.get('database_id') or current_database_session['database_id'])
        db_conn = DatabaseConnection.query.get(database_id)
        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT site_id FROM gold.gold_subjects WHERE database_id=%s AND site_id IS NOT NULL ORDER BY site_id", (database_id,))
        sites = [r[0] for r in cursor.fetchall()]
        cursor.execute("SELECT DISTINCT arm FROM gold.gold_subjects WHERE database_id=%s AND arm IS NOT NULL ORDER BY arm", (database_id,))
        arms = [r[0] for r in cursor.fetchall()]
        cursor.execute("SELECT DISTINCT test_code FROM gold.gold_labs WHERE database_id=%s AND test_code IS NOT NULL ORDER BY test_code", (database_id,))
        tests = [r[0] for r in cursor.fetchall()]
        cursor.execute("SELECT DISTINCT severity FROM gold.gold_ae WHERE database_id=%s AND severity IS NOT NULL ORDER BY severity", (database_id,))
        severities = [r[0] for r in cursor.fetchall()]
        cursor.close(); conn.close()
        return jsonify({'success': True, 'sites': sites, 'arms': arms, 'tests': tests, 'severities': severities})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/files/upload_status', methods=['GET'])
def get_upload_status():
    """Check upload status to enable/disable Run Pipeline button"""
    try:
        database_id = request.args.get('database_id')
        
        # Use current database session if database_id not provided
        if not database_id and current_database_session['database_id']:
            database_id = current_database_session['database_id']
        elif not database_id:
            return jsonify({'success': False, 'message': 'No database selected'})
        
        # Ensure database_id is an integer
        try:
            database_id = int(database_id)
        except (ValueError, TypeError):
            return jsonify({'success': False, 'message': 'Invalid database_id'})
        
        # Get database connection
        db_connection = DatabaseConnection.query.get(database_id)
        if not db_connection:
            return jsonify({'success': False, 'message': 'Database connection not found'})
        
        # Check uploaded files
        conn = psycopg2.connect(db_connection.url)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT file_type, COUNT(*) as count, 
                   SUM(CASE WHEN processed = true THEN 1 ELSE 0 END) as processed_count
            FROM public.uploaded_files 
            WHERE database_id = %s 
            GROUP BY file_type
        """, (database_id,))
        
        files_by_type = cursor.fetchall()
        
        # Check required file types
        required_types = ['subjects', 'labs', 'aes']
        upload_status = {}
        all_required_uploaded = True
        all_validated = True
        
        for file_type in required_types:
            type_files = [f for f in files_by_type if f[0] == file_type]
            if type_files:
                upload_status[file_type] = {
                    'uploaded': True,
                    'count': type_files[0][1],
                    'processed': type_files[0][2],
                    'validated': type_files[0][2] > 0  # Consider processed as validated
                }
                if type_files[0][2] == 0:
                    all_validated = False
            else:
                upload_status[file_type] = {
                    'uploaded': False,
                    'count': 0,
                    'processed': 0,
                    'validated': False
                }
                all_required_uploaded = False
        
        # Check unprocessed files for validation failures
        cursor.execute("""
            SELECT original_filename, file_type, 
                   CASE WHEN processed = false THEN 'Validation Failed' ELSE 'Validated' END as status
            FROM public.uploaded_files 
            WHERE database_id = %s AND processed = false
            ORDER BY created_at DESC
        """, (database_id,))
        
        unprocessed_files = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'data': {
                'upload_status': upload_status,
                'all_required_uploaded': all_required_uploaded,
                'all_validated': all_validated,
                'can_run_pipeline': all_required_uploaded and all_validated,
                'unprocessed_files': [{
                    'filename': file[0],
                    'file_type': file[1],
                    'status': file[2]
                } for file in unprocessed_files]
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting upload status: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})
@app.route('/api/files/<int:file_id>/delete', methods=['DELETE'])
def delete_file(file_id):
    file_record = UploadedFile.query.get_or_404(file_id)
    
    # Delete physical file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_record.filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # Delete database record
    db.session.delete(file_record)
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'File deleted'})

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    logger.info("=== TEST ENDPOINT CALLED ===")
    return jsonify({'success': True, 'message': 'Test endpoint working'})

@app.route('/api/pipeline/reset', methods=['POST'])
def reset_pipeline():
    """Reset a stuck pipeline"""
    try:
        data = request.get_json()
        database_id = data.get('database_id')
        
        if not database_id:
            return jsonify({'success': False, 'message': 'Database ID required'})
        
        # Reset pipeline status to idle
        pipeline_status = PipelineStatus.query.filter_by(database_id=database_id).first()
        if pipeline_status:
            pipeline_status.bronze = 'idle'
            pipeline_status.silver = 'idle'
            pipeline_status.gold = 'idle'
            pipeline_status.prediction = 'idle'
            pipeline_status.overall_status = 'idle'
            pipeline_status.current_stage = 'idle'
            pipeline_status.stage_status = 'idle'
            db.session.commit()
            logger.info(f"Pipeline status reset for database {database_id}")
        
        # Mark all files as unprocessed
        db_conn = DatabaseConnection.query.get(database_id)
        if db_conn:
            conn = psycopg2.connect(db_conn.url)
            cursor = conn.cursor()
            cursor.execute("UPDATE uploaded_files SET processed = false WHERE database_id = %s", (database_id,))
            conn.commit()
            cursor.close()
            conn.close()
            logger.info(f"Reset {cursor.rowcount} files to unprocessed for database {database_id}")
        
        return jsonify({'success': True, 'message': 'Pipeline reset successfully'})
        
    except Exception as e:
        logger.error(f"Error resetting pipeline: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/pipeline/reprocess', methods=['POST'])
def reprocess_pipeline():
    """Manually mark all files for reprocessing and run pipeline"""
    try:
        data = request.get_json() if request.is_json else request.form.to_dict()
        database_id = data.get('database_id')
        
        if not database_id:
            return jsonify({'success': False, 'message': 'Database ID required'})
        
        # Get database connection
        db_connection = DatabaseConnection.query.get(database_id)
        if not db_connection:
            return jsonify({'success': False, 'message': 'Database not found'})
        
        # Mark all files as unprocessed
        conn = psycopg2.connect(db_connection.url)
        cursor = conn.cursor()
        
        cursor.execute("UPDATE uploaded_files SET processed = false WHERE database_id = %s", (database_id,))
        affected_rows = cursor.rowcount
        conn.commit()
        
        cursor.close()
        conn.close()
        
        logger.info(f"Marked {affected_rows} files for reprocessing for database {database_id}")
        
        # Start pipeline
        import requests
        pipeline_response = requests.post(f"http://127.0.0.1:5000/api/pipeline/run", 
                                         json={"database_id": database_id})
        
        return jsonify({
            'success': True, 
            'message': f'Marked {affected_rows} files for reprocessing and started pipeline',
            'affected_files': affected_rows
        })
        
    except Exception as e:
        logger.error(f"Error reprocessing pipeline: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/pipeline/force_complete', methods=['POST'])
def force_complete_pipeline():
    """Force complete a stuck pipeline for testing"""
    try:
        data = request.get_json()
        database_id = data.get('database_id')
        
        if not database_id:
            return jsonify({'success': False, 'message': 'Database ID required'})
        
        # Force set pipeline status to completed
        pipeline_status = PipelineStatus.query.filter_by(database_id=database_id).first()
        if pipeline_status:
            pipeline_status.bronze = 'completed'
            pipeline_status.silver = 'completed'
            pipeline_status.gold = 'completed'
            pipeline_status.prediction = 'completed'
            pipeline_status.overall_status = 'idle'
            pipeline_status.current_stage = 'completed'
            pipeline_status.stage_status = 'completed'
            db.session.commit()
            logger.info(f"Pipeline force completed for database {database_id}")
        
        # Mark all files as processed
        db_conn = DatabaseConnection.query.get(database_id)
        if db_conn:
            conn = psycopg2.connect(db_conn.url)
            cursor = conn.cursor()
            cursor.execute("UPDATE uploaded_files SET processed = true WHERE database_id = %s", (database_id,))
            conn.commit()
            cursor.close()
            conn.close()
            logger.info(f"Marked {cursor.rowcount} files as processed for database {database_id}")
        
        return jsonify({'success': True, 'message': 'Pipeline force completed successfully'})
        
    except Exception as e:
        logger.error(f"Error force completing pipeline: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/pipeline/run', methods=['POST'])
def run_pipeline():
    logger.info("=== PIPELINE RUN ENDPOINT CALLED ===")
    logger.info(f"Request method: {request.method}")
    logger.info(f"Request URL: {request.url}")
    logger.info(f"Request form data: {dict(request.form)}")
    
    # Get database_id from form data
    database_id = request.form.get('database_id')
    if not database_id:
        # Try JSON as fallback
        try:
            data = request.get_json()
            database_id = data.get('database_id')
        except:
            pass
    
    if not database_id:
        logger.error("Database ID is required")
        return jsonify({'success': False, 'message': 'Database ID is required'})
    
    # Convert to integer
    try:
        database_id = int(database_id)
    except ValueError:
        logger.error(f"Invalid database_id format: {database_id}")
        return jsonify({'success': False, 'message': 'Invalid database ID format'})
    
    logger.info(f"Database ID: {database_id} (type: {type(database_id)})")
    
    db_conn = DatabaseConnection.query.get(database_id)
    if not db_conn:
        logger.error("Database connection not found")
        return jsonify({'success': False, 'message': 'Database connection not found'})
    
    try:
        logger.info(f"Starting pipeline for database {database_id}")
        
        # Set global status immediately
        global pipeline_status
        pipeline_status['database_id'] = int(database_id)
        pipeline_status['current_stage'] = 'bronze'
        pipeline_status['started_at'] = datetime.utcnow()
        
        # Start pipeline in background thread
        import threading
        logger.info("Creating background thread...")
        pipeline_thread = threading.Thread(target=run_pipeline_background, args=(int(database_id),))
        pipeline_thread.daemon = True
        pipeline_thread.start()
        logger.info("Background thread started")
        
        # Return JSON response
        return jsonify({'success': True, 'message': 'Pipeline started successfully'})
    
    except Exception as e:
        logger.error(f"Pipeline start error: {str(e)}")
        import traceback
        logger.error(f"Pipeline start traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'message': f'Pipeline failed to start: {str(e)}'})

def run_pipeline_background(database_id):
    """Run pipeline in background thread with real-time status updates"""
    global pipeline_status
    
    # Create Flask application context for background thread
    with app.app_context():
        try:
            import socket
            socket.setdefaulttimeout(60)   # DNS + connect timeout for all sockets

            # Initialize pipeline status
            pipeline_status['database_id'] = database_id
            pipeline_status['started_at'] = datetime.utcnow()
            
            # Step 1: Bronze Layer
            logger.info("Processing Bronze layer...")
            pipeline_status['current_stage'] = 'bronze'
            update_pipeline_status(database_id, 'bronze', 'running')
            process_bronze_layer(database_id)
            update_pipeline_status(database_id, 'bronze', 'completed')
            logger.info("Bronze layer completed")
            
            # Step 2: Silver Layer
            logger.info("Processing Silver layer...")
            pipeline_status['current_stage'] = 'silver'
            update_pipeline_status(database_id, 'silver', 'running')
            process_silver_layer(database_id)
            update_pipeline_status(database_id, 'silver', 'completed')
            logger.info("Silver layer completed")
            
            # Step 3: Gold Layer
            logger.info("Processing Gold layer...")
            pipeline_status['current_stage'] = 'gold'
            update_pipeline_status(database_id, 'gold', 'running')
            process_gold_layer(database_id)
            update_pipeline_status(database_id, 'gold', 'completed')
            logger.info("Gold layer completed")
            
            # Step 4: Prediction Layer
            logger.info("Processing Prediction layer...")
            pipeline_status['current_stage'] = 'prediction'
            update_pipeline_status(database_id, 'prediction', 'running')
            try:
                import threading as _threading
                pred_result = [None]
                pred_error = [None]

                def _run_pred():
                    try:
                        with app.app_context():
                            process_prediction_layer(database_id)
                        pred_result[0] = 'ok'
                    except Exception as e:
                        pred_error[0] = e

                pred_thread = _threading.Thread(target=_run_pred, daemon=True)
                pred_thread.start()
                pred_thread.join(timeout=60)   # hard 60-second wall-clock limit

                if pred_thread.is_alive():
                    logger.error("Prediction layer timed out after 60s — marking as failed and continuing")
                    update_pipeline_status(database_id, 'prediction', 'failed')
                elif pred_error[0]:
                    logger.error(f"Prediction layer failed: {pred_error[0]}")
                    update_pipeline_status(database_id, 'prediction', 'failed')
                else:
                    update_pipeline_status(database_id, 'prediction', 'completed')
                    logger.info("Prediction layer completed")
            except Exception as pred_err:
                logger.error(f"Prediction layer failed (non-fatal): {pred_err}")
                update_pipeline_status(database_id, 'prediction', 'failed')
            
            # Reset pipeline status
            pipeline_status['current_stage'] = None
            pipeline_status['database_id'] = None
            pipeline_status['started_at'] = None
            
            logger.info("Pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            import traceback
            logger.error(f"Pipeline traceback: {traceback.format_exc()}")
            
            # Reset pipeline status on error
            pipeline_status['current_stage'] = None
            pipeline_status['database_id'] = None
            pipeline_status['started_at'] = None

def update_pipeline_status(database_id, layer, status):
    """Update pipeline status in database for real-time tracking"""
    try:
        # This could be enhanced to store progress in a status table
        logger.info(f"Pipeline status update: {layer} -> {status}")
    except Exception as e:
        logger.error(f"Error updating pipeline status: {str(e)}")

# Global pipeline status tracking
pipeline_status = {
    'current_stage': None,
    'database_id': None,
    'started_at': None
}

# Current active database session for isolation
current_database_session = {
    'database_id': None,
    'connection': None,
    'name': None
}

@app.route('/api/database/select', methods=['POST'])
def select_database():
    """Select and set the active database for the session"""
    data = request.get_json()
    database_id = data.get('database_id')
    
    if not database_id:
        return jsonify({'success': False, 'message': 'Database ID required'})
    
    db_conn = DatabaseConnection.query.get(database_id)
    if not db_conn:
        return jsonify({'success': False, 'message': 'Database not found'})
    
    # Set current database session
    current_database_session['database_id'] = database_id
    current_database_session['connection'] = db_conn.url
    current_database_session['name'] = db_conn.name
    
    logger.info(f"Selected database: {db_conn.name} (ID: {database_id})")
    
    return jsonify({
        'success': True, 
        'message': f'Database {db_conn.name} selected',
        'database': {
            'id': database_id,
            'name': db_conn.name,
            'url': db_conn.url
        }
    })

@app.route('/api/database/set_current', methods=['POST'])
def set_current_database():
    """Set the current database session"""
    try:
        data = request.get_json()
        database_id = data.get('database_id')
        
        if not database_id:
            return jsonify({'success': False, 'message': 'Database ID required'})
        
        # Get database connection
        db_connection = DatabaseConnection.query.get(database_id)
        if not db_connection:
            return jsonify({'success': False, 'message': 'Database not found'})
        
        # Set current database session
        current_database_session['database_id'] = database_id
        current_database_session['name'] = db_connection.name
        current_database_session['connection'] = db_connection.url
        
        logger.info(f"Set current database to {db_connection.name} (ID: {database_id})")
        
        return jsonify({'success': True, 'message': 'Current database set successfully'})
        
    except Exception as e:
        logger.error(f"Error setting current database: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/database/current', methods=['GET'])
def get_current_database():
    """Get the currently selected database"""
    if current_database_session['database_id']:
        return jsonify({
            'success': True,
            'database': {
                'id': current_database_session['database_id'],
                'name': current_database_session['name'],
                'url': current_database_session['connection']
            }
        })
    else:
        return jsonify({'success': False, 'message': 'No database selected'})

@app.route('/api/pipeline/status', methods=['GET'])
def get_pipeline_status():
    """Get real-time sequential pipeline status"""
    database_id = request.args.get('database_id')
    
    if not database_id:
        return jsonify({'success': False, 'message': 'Database ID required'})
    
    try:
        # Check if pipeline is currently running
        status = {
            'bronze': 'idle',
            'silver': 'idle', 
            'gold': 'idle',
            'prediction': 'idle'
        }
        
        # If pipeline is running, set current stage to 'running'
        if pipeline_status['database_id'] == int(database_id) and pipeline_status['current_stage']:
            current_stage = pipeline_status['current_stage']
            if current_stage in status:
                status[current_stage] = 'running'
        
        # Check pipeline status by querying each layer sequentially
        conn_str = DatabaseConnection.query.get(database_id).url
        conn = psycopg2.connect(conn_str, connect_timeout=10)
        cursor = conn.cursor()
        
        # Check bronze layer completion
        try:
            cursor.execute("SELECT COUNT(*) FROM bronze.bronze_subjects WHERE database_id = %s", (database_id,))
            bronze_count = cursor.fetchone()[0]
            if bronze_count > 0:
                status['bronze'] = 'completed'
                
                # Only check silver if bronze is completed
                try:
                    cursor.execute("SELECT COUNT(*) FROM silver.silver_subjects WHERE database_id = %s", (database_id,))
                    silver_count = cursor.fetchone()[0]
                    if silver_count > 0:
                        status['silver'] = 'completed'
                        
                        # Only check gold if silver is completed
                        try:
                            cursor.execute("""
                                SELECT COUNT(*) FROM gold.gold_subjects WHERE database_id = %s
                            """, (database_id,))
                            gold_count = cursor.fetchone()[0]
                            if gold_count > 0:
                                status['gold'] = 'completed'
                                
                                # Only check prediction if gold is completed
                                try:
                                    cursor.execute("SELECT COUNT(*) FROM prediction.site_predictions WHERE database_id = %s", (database_id,))
                                    prediction_count = cursor.fetchone()[0]
                                    if prediction_count > 0:
                                        status['prediction'] = 'completed'
                                    elif pipeline_status['database_id'] == int(database_id) and pipeline_status['current_stage'] == 'prediction':
                                        status['prediction'] = 'running'
                                except Exception as e:
                                    logger.info(f"Prediction layer not yet created: {str(e)}")
                                    if pipeline_status['database_id'] == int(database_id) and pipeline_status['current_stage'] == 'prediction':
                                        status['prediction'] = 'running'
                                    else:
                                        status['prediction'] = 'idle'
                        except Exception as e:
                            logger.info(f"Gold layer not yet created: {str(e)}")
                            status['gold'] = 'idle'
                except Exception as e:
                    logger.info(f"Silver layer not yet created: {str(e)}")
                    status['silver'] = 'idle'
        except Exception as e:
            logger.info(f"Bronze layer not yet created: {str(e)}")
            status['bronze'] = 'idle'
        
        conn.close()
        
        # Return the correct structure with current_stage and overall_status
        # If pipeline thread is active for this database, force overall_status to 'running'
        is_pipeline_running = (
            pipeline_status['database_id'] == int(database_id) and
            pipeline_status['current_stage'] is not None
        )

        if is_pipeline_running:
            overall_status = 'running'
        elif all(stage == 'completed' for stage in status.values()):
            overall_status = 'completed'
        else:
            overall_status = 'idle'
        
        # Fix current_stage logic - if all stages are completed, current_stage should be null
        current_stage = None
        if overall_status != 'completed':
            current_stage = pipeline_status['current_stage'] if pipeline_status['database_id'] == int(database_id) else None
        
        return jsonify({
            'success': True,
            'data': {
                'current_stage': current_stage,
                'overall_status': overall_status,
                'stage_status': status.get(current_stage, 'idle') if current_stage else None,
                'bronze': status['bronze'],
                'silver': status['silver'],
                'gold': status['gold'],
                'prediction': status['prediction']
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting pipeline status: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

def process_bronze_layer(database_id):
    """Process bronze layer - data is already loaded via direct upload"""
    logger.info(f"=== STARTING BRONZE LAYER PROCESSING FOR DATABASE {database_id} ===")
    logger.info("Direct-to-bronze upload enabled - checking if data is already loaded")
    
    # Check ALL files for this database
    db_conn = DatabaseConnection.query.get(database_id)
    if not db_conn:
        logger.error("Database connection not found")
        return
    
    conn = psycopg2.connect(db_conn.url, connect_timeout=30)
    cursor = conn.cursor()
    
    # Check if bronze data exists
    cursor.execute("SELECT COUNT(*) FROM bronze.bronze_subjects WHERE database_id = %s", (database_id,))
    bronze_subjects = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM bronze.bronze_labs WHERE database_id = %s", (database_id,))
    bronze_labs = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM bronze.bronze_aes WHERE database_id = %s", (database_id,))
    bronze_aes = cursor.fetchone()[0]
    
    total_bronze_records = bronze_subjects + bronze_labs + bronze_aes
    logger.info(f"Bronze layer records found: {bronze_subjects} subjects, {bronze_labs} labs, {bronze_aes} AES")
    
    # Check uploaded_files for bronze_loaded status (column may not exist on older DBs)
    try:
        cursor.execute("SELECT COUNT(*) FROM public.uploaded_files WHERE database_id = %s AND bronze_loaded = true", (database_id,))
        bronze_loaded_files = cursor.fetchone()[0]
    except Exception:
        conn.rollback()
        # Column doesn't exist — fall back to counting all processed files
        cursor.execute("SELECT COUNT(*) FROM public.uploaded_files WHERE database_id = %s AND processed = true", (database_id,))
        bronze_loaded_files = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM public.uploaded_files WHERE database_id = %s", (database_id,))
    total_files = cursor.fetchone()[0]
    
    cursor.close()
    conn.close()
    
    if total_bronze_records > 0 or bronze_loaded_files > 0:
        logger.info(f"Bronze layer already contains data ({total_bronze_records} records from {bronze_loaded_files} files)")
        logger.info("Skipping bronze layer processing - data already loaded directly from uploads")
        update_pipeline_status(database_id, 'bronze', 'completed')
        return
    else:
        logger.info("No bronze data found - this should not happen with direct-to-bronze uploads")
        logger.warning("Checking for legacy file-based uploads...")
        
        # Fallback to legacy processing if no bronze data found
        legacy_process_bronze_layer(database_id)
        return

def legacy_process_bronze_layer(database_id):
    """Legacy bronze layer processing for file-based uploads (fallback)"""
    logger.info(f"=== LEGACY BRONZE LAYER PROCESSING FOR DATABASE {database_id} ===")
    
    # Get database connection
    db_conn = DatabaseConnection.query.get(database_id)
    if not db_conn:
        logger.error(f"Database connection not found for database_id {database_id}")
        return
    
    conn = psycopg2.connect(db_conn.url)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM uploaded_files WHERE database_id = %s AND processed = false", (database_id,))
    unprocessed_files_count = cursor.fetchone()[0]
    
    if not unprocessed_files_count:
        logger.info("No unprocessed files found")
        cursor.close()
        conn.close()
        update_pipeline_status(database_id, 'bronze', 'completed')
        return
    
    # Get unprocessed files
    cursor.execute("SELECT original_filename, file_type FROM uploaded_files WHERE database_id = %s AND processed = false", (database_id,))
    unprocessed_files = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    # Process each file (legacy method)
    for file_record in unprocessed_files:
        original_filename = file_record[0]
        file_type = file_record[1]
        
        logger.info(f"Processing file: {original_filename} (type: {file_type})")
        
        # Find the actual filename in uploads folder
        filename = secure_filename(original_filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            continue
        
        # Process file using legacy method
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, dtype=str, low_memory=False, engine='c')
            elif file_path.endswith('.xml'):
                df = pd.read_xml(file_path)
            elif file_path.endswith('.ndjson'):
                df = pd.read_json(file_path, lines=True)
            else:
                logger.warning(f"Unsupported file format: {file_path}")
                continue
            
            logger.info(f"Successfully read {len(df)} rows from {original_filename}")
            
            # Insert into bronze using existing functions
            conn = psycopg2.connect(db_conn.url)
            cursor = conn.cursor()
            
            if 'subject' in file_type.lower():
                batch_insert_subjects(cursor, df, database_id, conn)
            elif 'lab' in file_type.lower():
                batch_insert_labs(cursor, df, database_id)
                conn.commit()
            elif 'aes' in file_type.lower():
                batch_insert_aes(cursor, df, database_id)
                conn.commit()
            
            cursor.close()
            conn.close()
            
            # Mark file as processed
            conn = psycopg2.connect(db_conn.url)
            cursor = conn.cursor()
            cursor.execute("UPDATE uploaded_files SET processed = true WHERE database_id = %s AND original_filename = %s", 
                          (database_id, original_filename))
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error processing file {original_filename}: {str(e)}")
            continue
    
    logger.info("Legacy bronze layer processing completed")
    update_pipeline_status(database_id, 'bronze', 'completed')

_DATE_CACHE: dict = {}
_DATE_FORMATS = [
    '%Y-%m-%d',    # 2024-01-15  (most common — try first)
    '%d/%m/%Y',    # 15/01/2024
    '%m/%d/%Y',    # 01/15/2024
    '%d-%m-%Y',    # 15-01-2024
    '%m-%d-%Y',    # 01-15-2024
    '%Y/%m/%d',    # 2024/01/15
    '%Y.%m.%d',    # 2024.01.15
    '%d.%m.%Y',    # 15.01.2024
    '%d %b %Y',    # 15 Jan 2024
    '%d %B %Y',    # 15 January 2024
]

def parse_date_flexible(date_str):
    """Parse various date formats flexibly — cached for performance"""
    if not date_str or date_str == '' or date_str is None:
        return None
    try:
        s = str(date_str).strip()
        if not s or s.lower() in ('none', 'nan', 'null', 'nat'):
            return None
        cached = _DATE_CACHE.get(s)
        if cached is not None:
            return cached
        result = None
        for fmt in _DATE_FORMATS:
            try:
                result = datetime.strptime(s, fmt).date()
                break
            except ValueError:
                continue
        if result is None:
            try:
                from dateutil.parser import parse as _du_parse
                result = _du_parse(s).date()
            except Exception:
                result = None
        # Cache up to 50k unique values to avoid unbounded growth
        if len(_DATE_CACHE) < 50000:
            _DATE_CACHE[s] = result
        return result
    except Exception:
        return None

def batch_insert_subjects(cursor, df, database_id, conn):
    """Batch insert subjects for better performance - optimized for large datasets"""
    # Use larger batch size for better performance
    batch_size = 1000  # Increased from 100 for better performance
    total_rows = len(df)
    
    logger.info(f"Starting batch insert for {total_rows} subjects...")
    
    for i in range(0, total_rows, batch_size):
        batch = df.iloc[i:i+batch_size]
        
        # Prepare data for batch insert
        values = []
        for _, row in batch.iterrows():
            try:
                # Handle dates as text (no parsing needed)
                dob_text = str(row.get('dob') or row.get('date_of_birth') or '').strip()
                start_date_text = str(row.get('start_date') or '').strip()
                
                # Create row dict
                row_dict = row.to_dict()
                
                values.append((
                    database_id,
                    str(row.get('subject_id', '')).strip().upper(),
                    str(row.get('site_id', '')).strip().upper(),
                    str(row.get('sex', '')).strip().upper(),
                    dob_text,  # Store as text
                    str(row.get('arm', '')).strip().upper(),
                    start_date_text,  # Store as text
                    json.dumps(row_dict, default=str)
                ))
            except Exception as e:
                logger.error(f"Error preparing subject data: {str(e)}")
                continue
        
        # Batch insert using executemany for better performance
        if values:
            try:
                cursor.executemany("""
                    INSERT INTO bronze.bronze_subjects 
                    (database_id, subject_id, site_id, sex, dob, arm, raw_data, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                """, values)
                
                # Commit after each batch to prevent hanging
                conn.commit()
                logger.info(f"Inserted {len(values)} subjects (batch {i//batch_size + 1})")
                
            except Exception as e:
                logger.error(f"Error in batch insert subjects: {str(e)}")
                conn.rollback()
                raise

def batch_insert_labs(cursor, df, database_id):
    """Simple batch insert for labs"""
    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT INTO bronze.bronze_labs 
                (database_id, subject_id, visitno, testcode, resultvalue, units, reflow, refhigh, resultdate, raw_data, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            """, (
                database_id,
                str(row.get("subject_id", row.get("SubjectID", "")).strip().upper()),
                str(row.get("visitno", row.get("VisitNo", "")).strip()),
                str(row.get("testcode", row.get("TestCode", "")).strip().upper()),
                str(row.get("resultvalue", row.get("ResultValue", "")).strip()),
                str(row.get("units", row.get("Units", "")).strip().upper()),
                str(row.get("reflow", row.get("RefLow", "")).strip()),
                str(row.get("refhigh", row.get("RefHigh", "")).strip()),
                str(row.get("resultdate", row.get("ResultDate", "")).strip()),
                json.dumps(row.to_dict())
            ))
        except Exception as e:
            logger.error(f"Error inserting lab data: {str(e)}")
    logger.info(f"Inserted {len(df)} lab records")

def _hmac_hash(subject_id: str) -> str:
    """Deterministic HMAC-SHA256 pseudonymization of subject_id"""
    import hmac as _hmac
    secret = os.environ.get('SUBJECT_HASH_SECRET', 'clinical-pipeline-default-secret')
    return _hmac.new(secret.encode(), subject_id.encode(), 'sha256').hexdigest()


def _safe_float(val):
    """Cast to float, return None on failure"""
    try:
        s = str(val).strip()
        if s in ('', 'None', 'nan', 'NaN', 'NULL'):
            return None
        # Strip leading operators like '<', '>'
        s = s.lstrip('<>~')
        return float(s)
    except Exception:
        return None


def _norm_serious(val):
    """Normalize serious flag to Y/N"""
    v = str(val or '').strip().upper()
    if v in ('Y', 'YES', 'TRUE', '1', 'T'):
        return 'Y'
    if v in ('N', 'NO', 'FALSE', '0', 'F'):
        return 'N'
    return None


def _norm_related(val):
    """Normalize causality/related field - preserves PROBABLE/POSSIBLE/UNRELATED"""
    v = str(val or '').strip().upper()
    if v in ('Y', 'YES', 'TRUE', '1', 'T', 'RELATED'):
        return 'Y'
    if v in ('N', 'NO', 'FALSE', '0', 'F', 'UNRELATED', 'NOT RELATED', 'NOT_RELATED'):
        return 'N'
    if v in ('PROBABLE', 'PROBABLY'):
        return 'PROBABLE'
    if v in ('POSSIBLE', 'POSSIBLY', 'MAYBE'):
        return 'POSSIBLE'
    if v in ('UNLIKELY',):
        return 'UNLIKELY'
    return v or None


def _norm_severity(val):
    """Normalize severity to MILD/MODERATE/SEVERE"""
    v = str(val or '').strip().upper()
    if v in ('MILD', 'GRADE 1', 'GRADE I', 'GR1', '1', 'LOW'):
        return 'MILD'
    if v in ('MODERATE', 'GRADE 2', 'GRADE II', 'GR2', '2', 'MEDIUM', 'MED', 'MOD'):
        return 'MODERATE'
    if v in ('SEVERE', 'GRADE 3', 'GRADE III', 'GR3', '3', 'HIGH', 'SEV',
             'LIFE-THREATENING', 'GRADE 4', '4', 'FATAL', 'GRADE 5', '5'):
        return 'SEVERE'
    return v or None


def _norm_sex(val):
    """Normalize sex to MALE / FEMALE"""
    v = str(val or '').strip().upper()
    if v in ('M', 'MALE', 'MAN', 'BOY', '1'):
        return 'MALE'
    if v in ('F', 'FEMALE', 'WOMAN', 'WOMEN', 'GIRL', '2', 'FEM'):
        return 'FEMALE'
    return None


def _norm_arm(val):
    """Normalize treatment arm to canonical form"""
    v = str(val or '').strip().upper()
    if any(x in v for x in ('200', 'HIGH', 'ARM A', 'BLU-200', 'BLUM 200')):
        return 'BLUMETINIB 200MG QD'
    if any(x in v for x in ('100', 'LOW', 'ARM B', 'BLU-100', 'BLUM 100')):
        return 'BLUMETINIB 100MG QD'
    if any(x in v for x in ('PLACEBO', 'PBO', 'ARM C', 'CONTROL')):
        return 'PLACEBO'
    return v or None


def _norm_id(val):
    """Strip spaces and uppercase an ID field"""
    return re.sub(r'\s+', '', str(val or '')).upper() or None


# ═════════════════════════════════════════════════════════════════════════════
# UNIT NORMALISATION MODULE
# Bronze → Silver transformation: standardise all lab values to clinical units
# ═════════════════════════════════════════════════════════════════════════════

# Standard target units per test code
_STANDARD_UNITS = {
    'WBC':   'x10^9/L',
    'RBC':   'x10^12/L',
    'HGB':   'g/dL',
    'HCT':   '%',
    'PLT':   'x10^9/L',
    'NEUT':  'x10^9/L',
    'ALT':   'U/L',
    'AST':   'U/L',
    'ALP':   'U/L',
    'TBILI': 'mg/dL',
    'CREAT': 'mg/dL',
    'GLUC':  'mg/dL',
}

# Per-test conversion table:
# { TEST_CODE: { INCOMING_UNIT_UPPER: (factor, operation) } }
# result_std = result_raw * factor   (operation='multiply')
# result_std = result_raw / factor   (operation='divide')
_TEST_UNIT_CONVERSIONS = {

    # ── GLUC → mg/dL ─────────────────────────────────────────────────────────
    'GLUC': {
        'MG/DL':   (1.0,    'multiply'),   # already standard
        'MMOL/L':  (18.0,   'multiply'),   # mmol/L × 18 = mg/dL
    },

    # ── CREAT → mg/dL ────────────────────────────────────────────────────────
    'CREAT': {
        'MG/DL':            (1.0,   'multiply'),
        'UMOL/L':           (88.4,  'divide'),
        '\u00b5MOL/L':      (88.4,  'divide'),   # µmol/L
        '\u039cMOL/L':      (88.4,  'divide'),   # uppercase µ variant
        'MICROMOL/L':       (88.4,  'divide'),
    },

    # ── TBILI → mg/dL ────────────────────────────────────────────────────────
    'TBILI': {
        'MG/DL':            (1.0,   'multiply'),
        'UMOL/L':           (17.1,  'divide'),
        '\u00b5MOL/L':      (17.1,  'divide'),
        '\u039cMOL/L':      (17.1,  'divide'),   # uppercase µ variant
        'MICROMOL/L':       (17.1,  'divide'),
    },

    # ── HGB → g/dL ───────────────────────────────────────────────────────────
    'HGB': {
        'G/DL': (1.0,  'multiply'),
        'G/L':  (10.0, 'divide'),             # g/L ÷ 10 = g/dL
    },

    # ── HCT → % ──────────────────────────────────────────────────────────────
    'HCT': {
        '%':    (1.0,   'multiply'),
        'L/L':  (0.01,  'divide'),            # L/L ÷ 0.01 = %  (i.e. ×100)
    },

    # ── WBC → x10^9/L ────────────────────────────────────────────────────────
    'WBC': {
        'X10^9/L':          (1.0,    'multiply'),
        '10^9/L':           (1.0,    'multiply'),
        'X10^9/L':          (1.0,    'multiply'),
        'GI/L':             (1.0,    'multiply'),
        'CELLS/\u00b5L':    (1000.0, 'divide'),   # cells/µL ÷ 1000 = x10^9/L
        'CELLS/\u039cL':    (1000.0, 'divide'),   # uppercase µ variant
        'CELLS/UL':         (1000.0, 'divide'),
        '/\u00b5L':         (1000.0, 'divide'),
        '/UL':              (1000.0, 'divide'),
        'K/UL':             (1.0,    'multiply'),
        'K/\u00b5L':        (1.0,    'multiply'),
        '10^3/UL':          (1.0,    'multiply'),
        '10^3/\u00b5L':     (1.0,    'multiply'),
    },

    # ── PLT → x10^9/L (same conversions as WBC) ──────────────────────────────
    'PLT': {
        'X10^9/L':          (1.0,    'multiply'),
        '10^9/L':           (1.0,    'multiply'),
        'GI/L':             (1.0,    'multiply'),
        'CELLS/\u00b5L':    (1000.0, 'divide'),
        'CELLS/\u039cL':    (1000.0, 'divide'),
        'CELLS/UL':         (1000.0, 'divide'),
        '/\u00b5L':         (1000.0, 'divide'),
        '/UL':              (1000.0, 'divide'),
        'K/UL':             (1.0,    'multiply'),
        'K/\u00b5L':        (1.0,    'multiply'),
        '10^3/UL':          (1.0,    'multiply'),
        '10^3/\u00b5L':     (1.0,    'multiply'),
    },

    # ── NEUT → x10^9/L ───────────────────────────────────────────────────────
    'NEUT': {
        'X10^9/L':          (1.0,    'multiply'),
        '10^9/L':           (1.0,    'multiply'),
        'CELLS/\u00b5L':    (1000.0, 'divide'),
        'CELLS/\u039cL':    (1000.0, 'divide'),
        'CELLS/UL':         (1000.0, 'divide'),
        'K/UL':             (1.0,    'multiply'),
        'K/\u00b5L':        (1.0,    'multiply'),
    },

    # ── RBC → x10^12/L ───────────────────────────────────────────────────────
    'RBC': {
        'X10^12/L':         (1.0,       'multiply'),
        '10^12/L':          (1.0,       'multiply'),
        'TI/L':             (1.0,       'multiply'),
        'M/\u00b5L':        (1.0,       'multiply'),  # M/µL = 10^6/µL = 10^12/L
        'M/\u039cL':        (1.0,       'multiply'),  # uppercase µ variant
        'M/UL':             (1.0,       'multiply'),
        '10^6/\u00b5L':     (1.0,       'multiply'),
        '10^6/UL':          (1.0,       'multiply'),
        'CELLS/\u00b5L':    (1_000_000, 'divide'),
        'CELLS/\u039cL':    (1_000_000, 'divide'),
        'CELLS/UL':         (1_000_000, 'divide'),
    },

    # ── ALT → U/L ────────────────────────────────────────────────────────────
    'ALT': {
        'U/L':              (1.0,    'multiply'),
        'IU/L':             (1.0,    'multiply'),
        'U/ML':             (1000.0, 'multiply'),
        'MU/ML':            (1.0,    'multiply'),
        'MU/L':             (0.001,  'multiply'),
        '\u00b5KAT/L':      (60.0,   'multiply'),  # µkat/L × 60 = U/L
        '\u039cKAT/L':      (60.0,   'multiply'),  # uppercase µ variant
        'UKAT/L':           (60.0,   'multiply'),
        'NKAT/L':           (0.06,   'multiply'),
    },

    # ── AST → U/L ────────────────────────────────────────────────────────────
    'AST': {
        'U/L':              (1.0,    'multiply'),
        'IU/L':             (1.0,    'multiply'),
        'U/ML':             (1000.0, 'multiply'),
        'MU/ML':            (1.0,    'multiply'),
        'MU/L':             (0.001,  'multiply'),
        '\u00b5KAT/L':      (60.0,   'multiply'),
        '\u039cKAT/L':      (60.0,   'multiply'),
        'UKAT/L':           (60.0,   'multiply'),
        'NKAT/L':           (0.06,   'multiply'),
    },

    # ── ALP → U/L ────────────────────────────────────────────────────────────
    'ALP': {
        'U/L':              (1.0,    'multiply'),
        'IU/L':             (1.0,    'multiply'),
        'U/ML':             (1000.0, 'multiply'),
        'MU/ML':            (1.0,    'multiply'),
        'MU/L':             (0.001,  'multiply'),
        '\u00b5KAT/L':      (60.0,   'multiply'),
        '\u039cKAT/L':      (60.0,   'multiply'),
        'UKAT/L':           (60.0,   'multiply'),
        'NKAT/L':           (0.06,   'multiply'),
    },
}


def _normalise_lab_unit(value, raw_unit, test_code):
    """
    Normalise a single lab result to its standard clinical unit.

    Returns:
        (normalised_value, standard_unit, unit_validation_flag)
        unit_validation_flag = True  → unit was NULL / unrecognised (keep original)
        unit_validation_flag = False → conversion applied or unit already standard
    """
    tc  = str(test_code or '').strip().upper()
    raw = str(raw_unit  or '').strip()

    # Normalise µ: U+00B5 (micro sign) and U+03BC (Greek mu) both → \u00b5
    # .upper() converts µ (U+00B5) → Μ (U+039C), so we fix that before lookup
    unit_key = raw.upper().replace('\u039c', '\u00b5').replace('\u03bc', '\u00b5')

    std_unit = _STANDARD_UNITS.get(tc)

    # ── NULL / empty unit ─────────────────────────────────────────────────────
    if not unit_key:
        return (round(value, 4) if value is not None else None), (raw or ''), True

    # ── Test code not in our standard list → pass through unchanged ───────────
    if std_unit is None:
        return (round(value, 4) if value is not None else None), raw, False

    # ── Look up conversion rule for this test ─────────────────────────────────
    test_rules = _TEST_UNIT_CONVERSIONS.get(tc, {})
    rule       = test_rules.get(unit_key)

    if rule is None:
        # Unrecognised unit — flag it but only log at DEBUG to avoid flooding logs
        logger.debug(f"Unit normalisation: unrecognised unit '{raw}' for test_code={tc}")
        return (round(value, 4) if value is not None else None), raw, True

    if value is None:
        return None, std_unit, False

    factor, operation = rule
    converted = value * factor if operation == 'multiply' else value / factor
    return round(converted, 4), std_unit, False


def _ensure_silver_schema(cursor, conn):
    """Create silver tables if missing, then migrate any missing columns on existing tables"""
    cursor.execute("CREATE SCHEMA IF NOT EXISTS silver")

    # ── silver_subjects ───────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS silver.silver_subjects (
            id SERIAL PRIMARY KEY,
            database_id INTEGER NOT NULL,
            subject_hash_id TEXT NOT NULL,
            subject_id_raw TEXT,
            site_id TEXT,
            sex TEXT,
            dob DATE,
            arm TEXT,
            start_date DATE,
            age_at_start INTEGER,
            ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    # Add UNIQUE constraint idempotently (may not exist on older tables)
    try:
        cursor.execute("""
            ALTER TABLE silver.silver_subjects
            ADD CONSTRAINT silver_subjects_db_hash_uq UNIQUE (database_id, subject_hash_id)
        """)
        conn.commit()
    except Exception:
        conn.rollback()
    _add_cols(cursor, conn, 'silver.silver_subjects', [
        ('subject_hash_id',     'TEXT'),
        ('subject_id_raw',      'TEXT'),
        ('site_id',             'TEXT'),
        ('sex',                 'TEXT'),
        ('dob',                 'DATE'),
        ('arm',                 'TEXT'),
        ('start_date',          'DATE'),
        ('age_at_start',        'INTEGER'),
        ('ingestion_timestamp', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
    ])
    # Drop legacy subject_id column if it exists (was replaced by subject_id_raw)
    _drop_cols(cursor, conn, 'silver.silver_subjects', ['subject_id'])

    # ── silver_labs ───────────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS silver.silver_labs (
            lab_id SERIAL PRIMARY KEY,
            database_id INTEGER NOT NULL,
            subject_hash_id TEXT NOT NULL,
            subject_id TEXT,
            visit_no TEXT,
            test_code TEXT,
            result_value FLOAT,
            ref_low FLOAT,
            ref_high FLOAT,
            units TEXT,
            result_date DATE,
            result_status TEXT,
            abnormal_flag BOOLEAN DEFAULT FALSE,
            unit_validation_flag BOOLEAN DEFAULT FALSE,
            data_quality_flag TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    _add_cols(cursor, conn, 'silver.silver_labs', [
        ('subject_hash_id',      'TEXT'),
        ('subject_id',           'TEXT'),
        ('visit_no',             'TEXT'),
        ('test_code',            'TEXT'),
        ('result_value',         'FLOAT'),
        ('ref_low',              'FLOAT'),
        ('ref_high',             'FLOAT'),
        ('units',                'TEXT'),
        ('result_date',          'DATE'),
        ('result_status',        'TEXT'),
        ('abnormal_flag',        'BOOLEAN DEFAULT FALSE'),
        ('unit_validation_flag', 'BOOLEAN DEFAULT FALSE'),
        ('data_quality_flag',    'TEXT'),
        ('created_at',           'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
        ('updated_at',           'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
    ])
    # Drop legacy columns that were replaced by properly-named equivalents
    _drop_cols(cursor, conn, 'silver.silver_labs',
               ['visitno', 'testcode', 'resultvalue', 'reflow', 'refhigh', 'resultdate'])

    # ── silver_ae ─────────────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS silver.silver_ae (
            id SERIAL PRIMARY KEY,
            database_id INTEGER NOT NULL,
            subject_hash_id TEXT NOT NULL,
            subject_id TEXT,
            ae_id TEXT,
            pt_code TEXT,
            pt_name TEXT,
            severity TEXT,
            related TEXT,
            serious_flag TEXT,
            ae_start_date DATE,
            data_quality_flag TEXT,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    _add_cols(cursor, conn, 'silver.silver_ae', [
        ('subject_hash_id',   'TEXT'),
        ('subject_id',        'TEXT'),
        ('ae_id',             'TEXT'),
        ('pt_code',           'TEXT'),
        ('pt_name',           'TEXT'),
        ('severity',          'TEXT'),
        ('related',           'TEXT'),
        ('serious_flag',      'TEXT'),
        ('ae_start_date',     'DATE'),
        ('data_quality_flag', 'TEXT'),
        ('ingested_at',       'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
    ])

    # Drop the unwanted silver_aes table if it was created by an older version
    try:
        cursor.execute("DROP TABLE IF EXISTS silver.silver_aes")
        conn.commit()
    except Exception:
        conn.rollback()

    # ── unprocessed_records ───────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS silver.unprocessed_records (
            id SERIAL PRIMARY KEY,
            database_id INTEGER NOT NULL,
            source_table TEXT NOT NULL,
            file_name TEXT,
            subject_id TEXT,
            column_name TEXT,
            raw_value TEXT,
            reason TEXT NOT NULL,
            row_data JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def _ensure_gold_schema(cursor, conn):
    """Create gold tables if missing, then migrate any missing columns on existing tables"""
    cursor.execute("CREATE SCHEMA IF NOT EXISTS gold")

    # ── gold_subjects ─────────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS gold.gold_subjects (
            database_id INTEGER NOT NULL,
            subject_hash_id TEXT NOT NULL,
            site_id TEXT,
            sex TEXT,
            age_at_start INTEGER,
            age_group TEXT,
            arm TEXT,
            start_date DATE,
            study_year INTEGER,
            study_month INTEGER,
            ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    # Add PK/UNIQUE constraint idempotently (may not exist on older tables)
    try:
        cursor.execute("""
            ALTER TABLE gold.gold_subjects
            ADD CONSTRAINT gold_subjects_pk PRIMARY KEY (database_id, subject_hash_id)
        """)
        conn.commit()
    except Exception:
        conn.rollback()
    _add_cols(cursor, conn, 'gold.gold_subjects', [
        ('subject_hash_id',     'TEXT'),
        ('site_id',             'TEXT'),
        ('sex',                 'TEXT'),
        ('age_at_start',        'INTEGER'),
        ('age_group',           'TEXT'),
        ('arm',                 'TEXT'),
        ('start_date',          'DATE'),
        ('study_year',          'INTEGER'),
        ('study_month',         'INTEGER'),
        ('ingestion_timestamp', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
    ])

    # ── gold_labs ─────────────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS gold.gold_labs (
            lab_fact_id SERIAL PRIMARY KEY,
            database_id INTEGER NOT NULL,
            subject_hash_id TEXT NOT NULL,
            visit_no TEXT,
            test_code TEXT,
            result_value FLOAT,
            ref_low FLOAT,
            ref_high FLOAT,
            result_status TEXT,
            result_date DATE,
            units TEXT,
            uln_ratio FLOAT,
            alt_3x_uln_flag BOOLEAN DEFAULT FALSE,
            safety_signal TEXT,
            study_day INTEGER,
            ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    _add_cols(cursor, conn, 'gold.gold_labs', [
        ('subject_hash_id',     'TEXT'),
        ('visit_no',            'TEXT'),
        ('test_code',           'TEXT'),
        ('result_value',        'FLOAT'),
        ('ref_low',             'FLOAT'),
        ('ref_high',            'FLOAT'),
        ('result_status',       'TEXT'),
        ('result_date',         'DATE'),
        ('units',               'TEXT'),
        ('uln_ratio',           'FLOAT'),
        ('alt_3x_uln_flag',     'BOOLEAN DEFAULT FALSE'),
        ('safety_signal',       'TEXT'),
        ('study_day',           'INTEGER'),
        ('ingestion_timestamp', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
    ])

    # ── gold_ae ───────────────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS gold.gold_ae (
            id SERIAL PRIMARY KEY,
            database_id INTEGER NOT NULL,
            subject_hash_id TEXT NOT NULL,
            ae_id TEXT,
            pt_code TEXT,
            pt_name TEXT,
            severity TEXT,
            related TEXT,
            serious_flag TEXT,
            ae_start_date DATE,
            severity_score INTEGER,
            safety_signal TEXT,
            ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    _add_cols(cursor, conn, 'gold.gold_ae', [
        ('subject_hash_id',     'TEXT'),
        ('ae_id',               'TEXT'),
        ('pt_code',             'TEXT'),
        ('pt_name',             'TEXT'),
        ('severity',            'TEXT'),
        ('related',             'TEXT'),
        ('serious_flag',        'TEXT'),
        ('ae_start_date',       'DATE'),
        ('severity_score',      'INTEGER'),
        ('safety_signal',       'TEXT'),
        ('ingestion_timestamp', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
    ])

    # ── phi_site_summary (legacy - kept for prediction layer) ─────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS gold.phi_site_summary (
            database_id INTEGER,
            site_id TEXT,
            total_subjects INTEGER,
            total_lab_tests INTEGER,
            total_adverse_events INTEGER,
            total_serious_events INTEGER,
            total_ae_signal_count INTEGER,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (database_id, site_id)
        )
    """)
    _add_cols(cursor, conn, 'gold.phi_site_summary', [
        ('total_subjects',        'INTEGER'),
        ('total_lab_tests',       'INTEGER'),
        ('total_adverse_events',  'INTEGER'),
        ('total_serious_events',  'INTEGER'),
        ('total_ae_signal_count', 'INTEGER'),
        ('updated_at',            'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
    ])

    # Drop legacy fact_* tables created by older versions
    for tbl in ('gold.fact_subjects', 'gold.fact_labs', 'gold.fact_aes'):
        try:
            cursor.execute(f"DROP TABLE IF EXISTS {tbl}")
            conn.commit()
        except Exception:
            conn.rollback()

    conn.commit()


def _drop_cols(cursor, conn, table, columns):
    """Drop columns from an existing table (safe, idempotent)"""
    for col_name in columns:
        try:
            cursor.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS {col_name}")
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.debug(f"_drop_cols {table}.{col_name}: {e}")


def _add_cols(cursor, conn, table, columns):
    """Add missing columns to an existing table (safe, idempotent)"""
    for col_name, col_def in columns:
        try:
            cursor.execute(
                f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col_name} {col_def}"
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.debug(f"_add_cols {table}.{col_name}: {e}")


def process_silver_layer(database_id):
    """Process bronze layer data into silver layer with full transformations"""
    logger.info(f"=== STARTING SILVER LAYER PROCESSING FOR DATABASE {database_id} ===")

    try:
        from psycopg2.extras import execute_values

        db_conn = DatabaseConnection.query.get(database_id)
        if not db_conn:
            raise Exception(f"Database connection not found for id {database_id}")

        conn = psycopg2.connect(db_conn.url, connect_timeout=60)
        conn.autocommit = False
        cursor = conn.cursor()

        _ensure_silver_schema(cursor, conn)

        # --- Clean slate for this database_id ---
        cursor.execute("DELETE FROM silver.silver_subjects    WHERE database_id = %s", (database_id,))
        cursor.execute("DELETE FROM silver.silver_labs        WHERE database_id = %s", (database_id,))
        cursor.execute("DELETE FROM silver.silver_ae          WHERE database_id = %s", (database_id,))
        cursor.execute("DELETE FROM silver.unprocessed_records WHERE database_id = %s", (database_id,))
        conn.commit()

        now = datetime.utcnow()
        import json as _json

        # Batch reject buffer — flushed once at the end (avoids per-row DB round-trips)
        _reject_buffer = []

        def _reject(source, subject_id, column_name, raw_value, reason, row_data=None):
            _reject_buffer.append((
                database_id, source, str(subject_id or ''), column_name,
                str(raw_value or ''), reason,
                _json.dumps(row_data) if row_data else None, now
            ))

        def _flush_rejects():
            if _reject_buffer:
                execute_values(cursor, """
                    INSERT INTO silver.unprocessed_records
                        (database_id, source_table, subject_id, column_name,
                         raw_value, reason, row_data, created_at)
                    VALUES %s
                """, _reject_buffer, page_size=500)
                conn.commit()
                logger.info(f"Unprocessed records: flushed {len(_reject_buffer)} entries")
                _reject_buffer.clear()

        # ── SUBJECTS ──────────────────────────────────────────────────────────
        cursor.execute("""
            SELECT subject_id, site_id, sex, dob, arm, start_date
            FROM bronze.bronze_subjects
            WHERE database_id = %s
        """, (database_id,))
        bronze_subjects = cursor.fetchall()
        logger.info(f"Silver subjects: {len(bronze_subjects)} raw rows")

        seen_subjects = set()
        subject_rows = []
        rejected = 0
        for row in bronze_subjects:
            # Normalize subject_id: strip spaces, uppercase
            subject_id_raw = _norm_id(row[0])
            if not subject_id_raw:
                rejected += 1
                _reject('bronze_subjects', row[0], 'subject_id', row[0], 'NULL or empty subject_id',
                        {'subject_id': str(row[0]), 'site_id': str(row[1]), 'sex': str(row[2])})
                continue

            dedup_key = (database_id, subject_id_raw)
            if dedup_key in seen_subjects:
                continue
            seen_subjects.add(dedup_key)

            subject_hash_id = _hmac_hash(subject_id_raw)
            site_id = _norm_id(row[1])
            sex = _norm_sex(row[2])
            dob = parse_date_flexible(str(row[3])) if row[3] else None
            arm = _norm_arm(row[4])
            start_date = parse_date_flexible(str(row[5])) if row[5] else None

            age_at_start = None
            if start_date and dob:
                try:
                    age_at_start = int((start_date - dob).days // 365)
                except Exception:
                    pass

            # Reject rows with null critical fields — do NOT insert into silver
            null_issues = []
            if not site_id:
                null_issues.append(('site_id', row[1], 'NULL or empty site_id'))
            if not sex:
                null_issues.append(('sex', row[2], 'Unrecognised or NULL sex value'))
            if not dob:
                null_issues.append(('dob', row[3], 'NULL or unparseable date of birth'))
            if not start_date:
                null_issues.append(('start_date', row[5], 'NULL or unparseable start_date'))
            if not arm:
                null_issues.append(('arm', row[4], 'Unrecognised or NULL arm value'))

            if null_issues:
                rejected += 1
                for col, raw_val, reason in null_issues:
                    _reject('bronze_subjects', subject_id_raw, col, raw_val, reason,
                            {'subject_id': subject_id_raw, 'site_id': str(row[1]),
                             'sex': str(row[2]), 'arm': str(row[4])})
                continue

            subject_rows.append((
                database_id, subject_hash_id, subject_id_raw,
                site_id, sex, dob, arm, start_date, age_at_start, now
            ))

        def _subj_sort_key(sid):
            """Extract trailing number from subject_id for numeric sort"""
            m = re.search(r'(\d+)$', str(sid or ''))
            return int(m.group(1)) if m else 0

        if subject_rows:
            subject_rows.sort(key=lambda r: _subj_sort_key(r[2]))  # r[2] = subject_id_raw
            execute_values(cursor, """
                INSERT INTO silver.silver_subjects
                    (database_id, subject_hash_id, subject_id_raw,
                     site_id, sex, dob, arm, start_date, age_at_start, ingestion_timestamp)
                VALUES %s
            """, subject_rows, page_size=500)
            conn.commit()
        logger.info(f"Silver subjects: inserted {len(subject_rows)}, rejected {rejected}")

        # ── LABS ──────────────────────────────────────────────────────────────
        cursor.execute("""
            SELECT subject_id, visitno, testcode, resultvalue, units, reflow, refhigh, resultdate
            FROM bronze.bronze_labs
            WHERE database_id = %s
        """, (database_id,))
        bronze_labs = cursor.fetchall()
        logger.info(f"Silver labs: {len(bronze_labs)} raw rows")

        # Build per-subject ordered list to assign visit_no via ROW_NUMBER when blank
        from collections import defaultdict
        subj_lab_counter = defaultdict(int)

        seen_labs = set()
        lab_rows = []
        rejected = 0
        for row in bronze_labs:
            subject_id = _norm_id(row[0])
            if not subject_id:
                rejected += 1
                _reject('bronze_labs', row[0], 'subject_id', row[0], 'NULL or empty subject_id',
                        {'subject_id': str(row[0]), 'testcode': str(row[2])})
                continue

            # visit_no: use value if present, else ROW_NUMBER per subject ordered by result_date
            raw_visit = str(row[1] or '').strip()
            if raw_visit and raw_visit.lower() not in ('none', 'nan', 'null', ''):
                visit_no = raw_visit
            else:
                subj_lab_counter[subject_id] += 1
                visit_no = str(subj_lab_counter[subject_id])

            # test_code: strip spaces, uppercase
            test_code = re.sub(r'\s+', '', str(row[2] or '')).upper()
            if not test_code:
                rejected += 1
                _reject('bronze_labs', subject_id, 'testcode', row[2], 'NULL or empty test_code',
                        {'subject_id': subject_id, 'visitno': str(row[1])})
                continue

            # ── UNIT NORMALISATION ────────────────────────────────────────────
            result_value = _safe_float(row[3])
            ref_low      = _safe_float(row[5])
            ref_high     = _safe_float(row[6])
            raw_units = str(row[4] or '').strip()

            result_value_norm, units_out, unit_val_flag = _normalise_lab_unit(result_value, raw_units, test_code)
            ref_low_norm,  _, _ = _normalise_lab_unit(ref_low,  raw_units, test_code)
            ref_high_norm, _, _ = _normalise_lab_unit(ref_high, raw_units, test_code)

            # ── DATA QUALITY FLAG — reject nulls/invalid, don't insert into silver ──
            if result_value is None:
                rejected += 1
                _reject('bronze_labs', subject_id, 'resultvalue', row[3],
                        'NULL result_value — row sent to unprocessed_records only',
                        {'subject_id': subject_id, 'testcode': test_code, 'visitno': visit_no})
                continue
            elif result_value < 0:
                rejected += 1
                _reject('bronze_labs', subject_id, 'resultvalue', row[3],
                        f'Negative result_value ({result_value}) is invalid',
                        {'subject_id': subject_id, 'testcode': test_code, 'resultvalue': str(result_value)})
                continue

            if unit_val_flag:
                data_quality_flag = 'UNIT_UNKNOWN'
                _reject('bronze_labs', subject_id, 'units', raw_units,
                        f'Unrecognised unit "{raw_units}" for test_code {test_code}',
                        {'subject_id': subject_id, 'testcode': test_code, 'units': raw_units})
            else:
                data_quality_flag = 'VALID'

            result_date = parse_date_flexible(str(row[7])) if row[7] else None
            if not result_date:
                rejected += 1
                _reject('bronze_labs', subject_id, 'resultdate', row[7],
                        'NULL or unparseable result_date — row sent to unprocessed_records only',
                        {'subject_id': subject_id, 'testcode': test_code, 'resultdate': str(row[7])})
                continue
            if result_value_norm is None:
                result_status = None
                abnormal_flag = False
            elif ref_low_norm is not None and result_value_norm < ref_low_norm:
                result_status = 'LOW'
                abnormal_flag = True
            elif ref_high_norm is not None and result_value_norm > ref_high_norm:
                result_status = 'HIGH'
                abnormal_flag = True
            else:
                result_status = 'NORMAL'
                abnormal_flag = False

            dedup_key = (database_id, subject_id, visit_no, test_code, str(result_date))
            if dedup_key in seen_labs:
                continue
            seen_labs.add(dedup_key)

            subject_hash_id = _hmac_hash(subject_id)
            lab_rows.append((
                database_id, subject_hash_id, subject_id,
                visit_no, test_code,
                result_value_norm, ref_low_norm, ref_high_norm,
                units_out, result_date,
                result_status, abnormal_flag, unit_val_flag, data_quality_flag,
                now, now
            ))

        if lab_rows:
            lab_rows.sort(key=lambda r: (_subj_sort_key(r[2]), str(r[9] or '')))  # r[2]=subject_id, r[9]=result_date
            execute_values(cursor, """
                INSERT INTO silver.silver_labs
                    (database_id, subject_hash_id, subject_id,
                     visit_no, test_code,
                     result_value, ref_low, ref_high,
                     units, result_date,
                     result_status, abnormal_flag, unit_validation_flag, data_quality_flag,
                     created_at, updated_at)
                VALUES %s
            """, lab_rows, page_size=500)
            conn.commit()
        logger.info(f"Silver labs: inserted {len(lab_rows)}, rejected {rejected}")

        # ── ADVERSE EVENTS ────────────────────────────────────────────────────
        cursor.execute("""
            SELECT subject_id, ae_id, pt_code, pt_name, severity, related, serious, ae_start_date
            FROM bronze.bronze_aes
            WHERE database_id = %s
        """, (database_id,))
        bronze_aes = cursor.fetchall()
        logger.info(f"Silver AEs: {len(bronze_aes)} raw rows")

        seen_aes = set()
        ae_rows = []
        rejected = 0
        for row in bronze_aes:
            subject_id = _norm_id(row[0])
            if not subject_id:
                rejected += 1
                _reject('bronze_aes', row[0], 'subject_id', row[0], 'NULL or empty subject_id',
                        {'subject_id': str(row[0]), 'ae_id': str(row[1])})
                continue

            ae_id        = str(row[1] or '').strip() or None
            pt_code      = re.sub(r'\s+', '', str(row[2] or '')).upper() or None
            pt_name      = str(row[3] or '').strip().upper() or None
            severity     = _norm_severity(row[4])
            related      = _norm_related(row[5])
            serious_flag = _norm_serious(row[6])
            ae_start_date = parse_date_flexible(str(row[7])) if row[7] else None

            # data_quality_flag + reject rows with critical nulls
            if not pt_code and not pt_name:
                rejected += 1
                _reject('bronze_aes', subject_id, 'pt_name/pt_code', f'{row[2]}/{row[3]}',
                        'Both pt_code and pt_name are NULL — row sent to unprocessed_records only',
                        {'subject_id': subject_id, 'ae_id': str(ae_id)})
                continue
            elif not ae_start_date:
                rejected += 1
                _reject('bronze_aes', subject_id, 'ae_start_date', row[7],
                        'NULL or unparseable ae_start_date — row sent to unprocessed_records only',
                        {'subject_id': subject_id, 'ae_id': str(ae_id), 'ae_start_date': str(row[7])})
                continue
            else:
                dq_flag = 'VALID'

            if not severity:
                _reject('bronze_aes', subject_id, 'severity', row[4],
                        f'Unrecognised severity value "{row[4]}" — row inserted with null',
                        {'subject_id': subject_id, 'ae_id': str(ae_id), 'severity': str(row[4])})
            if not serious_flag:
                _reject('bronze_aes', subject_id, 'serious', row[6],
                        f'Unrecognised serious flag "{row[6]}" — row inserted with null',
                        {'subject_id': subject_id, 'ae_id': str(ae_id), 'serious': str(row[6])})

            dedup_key = (database_id, subject_id, ae_id)
            if dedup_key in seen_aes:
                continue
            seen_aes.add(dedup_key)

            subject_hash_id = _hmac_hash(subject_id)
            ae_rows.append((
                database_id, subject_hash_id, subject_id,
                ae_id, pt_code, pt_name,
                severity, related, serious_flag,
                ae_start_date, dq_flag, now
            ))

        if ae_rows:
            ae_rows.sort(key=lambda r: (_subj_sort_key(r[2]), str(r[3] or '')))
            execute_values(cursor, """
                INSERT INTO silver.silver_ae
                    (database_id, subject_hash_id, subject_id,
                     ae_id, pt_code, pt_name,
                     severity, related, serious_flag,
                     ae_start_date, data_quality_flag, ingested_at)
                VALUES %s
            """, ae_rows, page_size=500)
            conn.commit()
        logger.info(f"Silver AEs: inserted {len(ae_rows)}, rejected {rejected}")

        _flush_rejects()   # batch insert all unprocessed records at once
        cursor.close()
        conn.close()
        logger.info("=== SILVER LAYER PROCESSING COMPLETED ===")

    except Exception as e:
        logger.error(f"Error in silver layer processing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def process_gold_layer(database_id):
    """Process silver layer data into gold layer — fully SQL-driven, no Python row loops"""
    logger.info(f"=== STARTING GOLD LAYER PROCESSING FOR DATABASE {database_id} ===")

    try:
        db_conn = DatabaseConnection.query.get(database_id)
        if not db_conn:
            raise Exception(f"Database connection not found for id {database_id}")

        conn = psycopg2.connect(db_conn.url, connect_timeout=120)
        conn.autocommit = False
        cursor = conn.cursor()

        _ensure_gold_schema(cursor, conn)

        # --- Clean slate ---
        for tbl in ('gold.gold_subjects', 'gold.gold_labs', 'gold.gold_ae',
                    'gold.phi_site_summary'):
            try:
                cursor.execute(f"DELETE FROM {tbl} WHERE database_id = %s", (database_id,))
            except Exception:
                conn.rollback()
        conn.commit()

        # ── GOLD SUBJECTS — single SQL INSERT...SELECT ────────────────────────
        cursor.execute("""
            INSERT INTO gold.gold_subjects
                (database_id, subject_hash_id,
                 site_id, sex, age_at_start, age_group,
                 arm, start_date, study_year, study_month, ingestion_timestamp)
            SELECT
                database_id,
                subject_hash_id,
                site_id,
                sex,
                age_at_start,
                CASE
                    WHEN age_at_start < 30  THEN 'YOUNG'
                    WHEN age_at_start <= 50 THEN 'MIDDLE'
                    ELSE 'SENIOR'
                END,
                arm,
                start_date,
                EXTRACT(YEAR  FROM start_date)::INTEGER,
                EXTRACT(MONTH FROM start_date)::INTEGER,
                NOW()
            FROM silver.silver_subjects
            WHERE database_id = %s
        """, (database_id,))
        conn.commit()
        cursor.execute("SELECT COUNT(*) FROM gold.gold_subjects WHERE database_id = %s", (database_id,))
        logger.info(f"Gold subjects: inserted {cursor.fetchone()[0]}")

        # ── GOLD LABS — single SQL INSERT...SELECT ────────────────────────────
        cursor.execute("""
            INSERT INTO gold.gold_labs
                (database_id, subject_hash_id,
                 visit_no, test_code,
                 result_value, ref_low, ref_high,
                 result_status, result_date, units,
                 uln_ratio, alt_3x_uln_flag, safety_signal,
                 study_day, ingestion_timestamp)
            SELECT
                sl.database_id,
                sl.subject_hash_id,
                sl.visit_no,
                sl.test_code,
                sl.result_value,
                sl.ref_low,
                sl.ref_high,
                sl.result_status,
                sl.result_date,
                sl.units,
                -- uln_ratio
                CASE
                    WHEN sl.result_value IS NOT NULL AND sl.ref_high > 0
                    THEN ROUND((sl.result_value / sl.ref_high)::NUMERIC, 4)
                    ELSE NULL
                END,
                -- alt_3x_uln_flag
                CASE
                    WHEN sl.test_code = 'ALT'
                     AND sl.result_value IS NOT NULL
                     AND sl.ref_high > 0
                     AND (sl.result_value / sl.ref_high) > 3.0
                    THEN TRUE ELSE FALSE
                END,
                -- safety_signal
                CASE
                    WHEN sl.test_code = 'ALT'
                     AND sl.result_value IS NOT NULL
                     AND sl.ref_high > 0
                     AND (sl.result_value / sl.ref_high) > 3.0
                    THEN 'ALT_CRITICAL'
                    WHEN sl.result_status = 'HIGH' THEN 'LAB_HIGH'
                    ELSE 'NORMAL'
                END,
                -- study_day
                CASE
                    WHEN sl.result_date IS NOT NULL AND ss.start_date IS NOT NULL
                    THEN (sl.result_date - ss.start_date)
                    ELSE NULL
                END,
                NOW()
            FROM silver.silver_labs sl
            LEFT JOIN silver.silver_subjects ss
                ON ss.database_id = sl.database_id
               AND ss.subject_hash_id = sl.subject_hash_id
            WHERE sl.database_id = %s
        """, (database_id,))
        conn.commit()
        cursor.execute("SELECT COUNT(*) FROM gold.gold_labs WHERE database_id = %s", (database_id,))
        logger.info(f"Gold labs: inserted {cursor.fetchone()[0]}")

        # ── GOLD AE — single SQL INSERT...SELECT ─────────────────────────────
        cursor.execute("""
            INSERT INTO gold.gold_ae
                (database_id, subject_hash_id,
                 ae_id, pt_code, pt_name,
                 severity, related, serious_flag,
                 ae_start_date, severity_score, safety_signal, ingestion_timestamp)
            SELECT
                database_id,
                subject_hash_id,
                ae_id,
                pt_code,
                pt_name,
                severity,
                related,
                serious_flag,
                ae_start_date,
                -- severity_score
                CASE UPPER(severity)
                    WHEN 'MILD'     THEN 1
                    WHEN 'MODERATE' THEN 2
                    WHEN 'SEVERE'   THEN 3
                    ELSE NULL
                END,
                -- safety_signal
                CASE
                    WHEN UPPER(serious_flag) = 'Y' AND UPPER(severity) = 'SEVERE'
                        THEN 'SERIOUS_SEVERE'
                    WHEN UPPER(serious_flag) = 'Y'
                        THEN 'SERIOUS'
                    WHEN UPPER(severity) = 'SEVERE'
                        THEN 'SEVERE_AE'
                    ELSE 'NORMAL'
                END,
                NOW()
            FROM silver.silver_ae
            WHERE database_id = %s
        """, (database_id,))
        conn.commit()
        cursor.execute("SELECT COUNT(*) FROM gold.gold_ae WHERE database_id = %s", (database_id,))
        logger.info(f"Gold AEs: inserted {cursor.fetchone()[0]}")

        # ── phi_site_summary ──────────────────────────────────────────────────
        cursor.execute("""
            INSERT INTO gold.phi_site_summary
                (database_id, site_id, total_subjects, total_lab_tests,
                 total_adverse_events, total_serious_events, total_ae_signal_count)
            SELECT
                gs.database_id,
                gs.site_id,
                COUNT(DISTINCT gs.subject_hash_id),
                COUNT(DISTINCT gl.lab_fact_id),
                COUNT(DISTINCT ga.id),
                COUNT(DISTINCT CASE WHEN ga.serious_flag = 'Y' THEN ga.id END),
                COUNT(DISTINCT CASE WHEN ga.safety_signal != 'NORMAL' THEN ga.id END)
            FROM gold.gold_subjects gs
            LEFT JOIN gold.gold_labs gl
                ON gl.database_id = gs.database_id
               AND gl.subject_hash_id = gs.subject_hash_id
            LEFT JOIN gold.gold_ae ga
                ON ga.database_id = gs.database_id
               AND ga.subject_hash_id = gs.subject_hash_id
            WHERE gs.database_id = %s
            GROUP BY gs.database_id, gs.site_id
            ON CONFLICT (database_id, site_id) DO UPDATE SET
                total_subjects        = EXCLUDED.total_subjects,
                total_lab_tests       = EXCLUDED.total_lab_tests,
                total_adverse_events  = EXCLUDED.total_adverse_events,
                total_serious_events  = EXCLUDED.total_serious_events,
                total_ae_signal_count = EXCLUDED.total_ae_signal_count,
                updated_at            = CURRENT_TIMESTAMP
        """, (database_id,))
        conn.commit()
        cursor.execute("SELECT COUNT(*) FROM gold.phi_site_summary WHERE database_id = %s", (database_id,))
        logger.info(f"Gold phi_site_summary: {cursor.fetchone()[0]} sites")

        cursor.close()
        conn.close()
        logger.info("=== GOLD LAYER PROCESSING COMPLETED ===")

    except Exception as e:
        logger.error(f"Error in gold layer processing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def process_prediction_layer(database_id):
    """Generate predictions from gold layer data"""
    logger.info(f"=== STARTING PREDICTION LAYER PROCESSING FOR DATABASE {database_id} ===")

    try:
        db_conn = DatabaseConnection.query.get(database_id)
        if not db_conn:
            raise Exception(f"Database connection not found for id {database_id}")

        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        conn.autocommit = False
        cursor = conn.cursor()

        # Ensure schema exists and table has correct schema — drop and recreate if needed
        cursor.execute("CREATE SCHEMA IF NOT EXISTS prediction")
        conn.commit()

        # Drop and recreate to guarantee correct columns (handles stale schema from old runs)
        cursor.execute("DROP TABLE IF EXISTS prediction.site_predictions")
        cursor.execute("""
            CREATE TABLE prediction.site_predictions (
                id SERIAL PRIMARY KEY,
                database_id INTEGER NOT NULL,
                site_id TEXT NOT NULL,
                prediction_date DATE NOT NULL,
                predicted_total_ae INTEGER,
                predicted_total_serious_events INTEGER,
                predicted_total_ae_signal_count INTEGER,
                predicted_avg_lab_ratio FLOAT,
                predicted_new_subjects INTEGER,
                predicted_risk_group TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT uq_site_pred UNIQUE (database_id, site_id, prediction_date)
            )
        """)
        conn.commit()
        # Clear existing predictions for this database
        cursor.execute("DELETE FROM prediction.site_predictions WHERE database_id = %s", (database_id,))
        conn.commit()

        cursor.execute("""
            SELECT database_id, site_id, total_subjects, total_lab_tests,
                   total_adverse_events, total_serious_events, total_ae_signal_count
            FROM gold.phi_site_summary
            WHERE database_id = %s
        """, (database_id,))
        sites = cursor.fetchall()
        logger.info(f"Found {len(sites)} sites for prediction processing")

        today = datetime.now().date()
        pred_rows = []
        for site_data in sites:
            current_site_id       = site_data[1]
            total_subjects        = site_data[2] or 0
            total_lab_tests       = site_data[3] or 0
            total_adverse_events  = site_data[4] or 0
            total_serious_events  = site_data[5] or 0
            total_ae_signal_count = site_data[6] or 0

            if total_serious_events > 5 or total_ae_signal_count > 3:
                predicted_risk = 'High'
            elif total_serious_events > 2 or total_ae_signal_count > 1:
                predicted_risk = 'Medium'
            else:
                predicted_risk = 'Low'

            predicted_ae = max(0, int(total_adverse_events * 1.1))

            pred_rows.append((
                database_id, current_site_id, today,
                predicted_ae, total_serious_events, total_ae_signal_count,
                round(total_lab_tests / max(total_subjects, 1), 2),
                max(1, total_subjects // 10),
                predicted_risk
            ))
            logger.info(f"Prediction for site {current_site_id}: {predicted_risk} risk")

        if pred_rows:
            from psycopg2.extras import execute_values
            execute_values(cursor, """
                INSERT INTO prediction.site_predictions
                    (database_id, site_id, prediction_date,
                     predicted_total_ae, predicted_total_serious_events,
                     predicted_total_ae_signal_count, predicted_avg_lab_ratio,
                     predicted_new_subjects, predicted_risk_group)
                VALUES %s
                ON CONFLICT (database_id, site_id, prediction_date) DO UPDATE SET
                    predicted_total_ae              = EXCLUDED.predicted_total_ae,
                    predicted_total_serious_events  = EXCLUDED.predicted_total_serious_events,
                    predicted_total_ae_signal_count = EXCLUDED.predicted_total_ae_signal_count,
                    predicted_risk_group            = EXCLUDED.predicted_risk_group
            """, pred_rows, page_size=100)

        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("=== PREDICTION LAYER PROCESSING COMPLETED SUCCESSFULLY ===")
        
    except Exception as e:
        logger.error(f"Error in prediction layer processing: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

@app.route('/api/home/metrics', methods=['GET'])
def get_home_metrics():
    """Return summary metrics for the home page cards"""
    try:
        databases = DatabaseConnection.query.all()
        total_databases = len(databases)
        total_files = 0
        total_records = 0
        total_pipeline_runs = 0
        for db_conn in databases:
            try:
                conn = psycopg2.connect(db_conn.url, connect_timeout=10)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*), COALESCE(SUM(record_count),0) FROM public.uploaded_files WHERE database_id=%s", (db_conn.id,))
                row = cursor.fetchone()
                if row:
                    total_files += row[0] or 0
                    total_records += row[1] or 0
                # Count pipeline runs as number of rows in prediction.site_predictions
                try:
                    cursor.execute("SELECT COUNT(DISTINCT prediction_date) FROM prediction.site_predictions WHERE database_id=%s", (db_conn.id,))
                    pr = cursor.fetchone()
                    total_pipeline_runs += pr[0] if pr else 0
                except Exception:
                    conn.rollback()
                cursor.close()
                conn.close()
            except Exception:
                pass
        return jsonify({'success': True, 'databases': total_databases, 'files': total_files,
                        'records': total_records, 'pipeline_runs': total_pipeline_runs})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/layer/download', methods=['GET'])
def download_layer_zip():
    """Download all tables of a given layer as a ZIP of CSVs"""
    import zipfile, io as _io
    database_id = request.args.get('database_id') or current_database_session.get('database_id')
    layer = request.args.get('layer', 'bronze').lower()
    if not database_id:
        return jsonify({'success': False, 'message': 'No database selected'}), 400
    database_id = int(database_id)
    db_conn = DatabaseConnection.query.get(database_id)
    if not db_conn:
        return jsonify({'success': False, 'message': 'Database not found'}), 404

    layer_tables = {
        'bronze': ['bronze.bronze_subjects', 'bronze.bronze_labs', 'bronze.bronze_aes'],
        'silver': ['silver.silver_subjects', 'silver.silver_labs', 'silver.silver_ae'],
        'gold':   ['gold.gold_subjects', 'gold.gold_labs', 'gold.gold_ae', 'gold.phi_site_summary'],
        'prediction': ['prediction.site_predictions'],
    }
    tables = layer_tables.get(layer, [])
    if not tables:
        return jsonify({'success': False, 'message': 'Invalid layer'}), 400

    try:
        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        cursor = conn.cursor()
        zip_buffer = _io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for full_table in tables:
                schema, tname = full_table.split('.')
                try:
                    cursor.execute(f"SELECT * FROM {full_table} WHERE database_id = %s LIMIT 50000", (database_id,))
                    rows = cursor.fetchall()
                    cols = [desc[0] for desc in cursor.description]
                    csv_buf = _io.StringIO()
                    csv_buf.write(','.join(cols) + '\n')
                    for row in rows:
                        csv_buf.write(','.join(f'"{str(v)}"' if v is not None else '' for v in row) + '\n')
                    zf.writestr(f'{tname}.csv', csv_buf.getvalue())
                except Exception as te:
                    conn.rollback()
                    zf.writestr(f'{tname}_error.txt', str(te))
        cursor.close()
        conn.close()
        zip_buffer.seek(0)
        from flask import send_file
        return send_file(zip_buffer, mimetype='application/zip',
                         as_attachment=True,
                         download_name=f'{layer}_layer_{database_id}.zip')
    except Exception as e:
        logger.error(f"Layer download error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/unprocessed/download', methods=['GET'])
def download_unprocessed():
    """Download unprocessed records as CSV"""
    import io as _io
    database_id = request.args.get('database_id') or current_database_session.get('database_id')
    if not database_id:
        return jsonify({'success': False, 'message': 'No database selected'}), 400
    database_id = int(database_id)
    db_conn = DatabaseConnection.query.get(database_id)
    if not db_conn:
        return jsonify({'success': False, 'message': 'Database not found'}), 404
    try:
        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT EXISTS (SELECT 1 FROM information_schema.tables
                WHERE table_schema='silver' AND table_name='unprocessed_records')
        """)
        if not cursor.fetchone()[0]:
            cursor.close(); conn.close()
            return jsonify({'success': False, 'message': 'No unprocessed records table found'}), 404
        cursor.execute("""
            SELECT id, source_table, subject_id, column_name, raw_value, reason, created_at
            FROM silver.unprocessed_records WHERE database_id=%s ORDER BY created_at DESC
        """, (database_id,))
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
        cursor.close(); conn.close()
        buf = _io.StringIO()
        buf.write(','.join(cols) + '\n')
        for row in rows:
            buf.write(','.join(f'"{str(v)}"' if v is not None else '' for v in row) + '\n')
        buf.seek(0)
        from flask import Response
        return Response(buf.getvalue(), mimetype='text/csv',
                        headers={'Content-Disposition': f'attachment; filename=unprocessed_records_{database_id}.csv'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/layer/table-data', methods=['GET'])
def get_layer_table_data():
    """Return paginated rows from a layer table for preview"""
    database_id = request.args.get('database_id') or current_database_session.get('database_id')
    table = request.args.get('table')
    limit = min(int(request.args.get('limit', 100)), 500)
    if not database_id or not table:
        return jsonify({'success': False, 'message': 'database_id and table required'}), 400
    # Whitelist allowed tables
    allowed = {'bronze.bronze_subjects','bronze.bronze_labs','bronze.bronze_aes',
                'silver.silver_subjects','silver.silver_labs','silver.silver_ae',
                'gold.gold_subjects','gold.gold_labs','gold.gold_ae','gold.phi_site_summary',
                'prediction.site_predictions'}
    if table not in allowed:
        return jsonify({'success': False, 'message': 'Table not allowed'}), 403
    database_id = int(database_id)
    db_conn = DatabaseConnection.query.get(database_id)
    if not db_conn:
        return jsonify({'success': False, 'message': 'Database not found'}), 404
    try:
        conn = psycopg2.connect(db_conn.url, connect_timeout=30)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(f"SELECT * FROM {table} WHERE database_id=%s LIMIT %s", (database_id, limit))
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description] if cursor.description else []
        cursor.close(); conn.close()
        return jsonify({'success': True, 'columns': cols, 'rows': [dict(r) for r in rows]})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════
# AUTH & RBAC
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = User.query.filter_by(username=username, is_active=True).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['user_role'] = user.role
            session['user_email'] = user.email
            flash(f'Welcome back, {user.username}!', 'success')
            return redirect(request.args.get('next') or url_for('home'))
        flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/admin/users')
@login_required
@role_required('admin')
def admin_users():
    users = User.query.order_by(User.created_at.desc()).all()
    return render_template('admin_users.html', users=users)

@app.route('/api/admin/users', methods=['GET'])
@login_required
@role_required('admin')
def api_list_users():
    users = User.query.order_by(User.created_at.desc()).all()
    return jsonify([{
        'id': u.id, 'username': u.username, 'email': u.email,
        'role': u.role, 'is_active': u.is_active,
        'created_at': u.created_at.strftime('%Y-%m-%d %H:%M') if u.created_at else None
    } for u in users])

@app.route('/api/admin/users', methods=['POST'])
@login_required
@role_required('admin')
def api_create_user():
    data = request.get_json()
    username = data.get('username', '').strip()
    email = data.get('email', '').strip()
    role = data.get('role', 'data_analyst')
    password = data.get('password', '').strip()
    if not all([username, email, role, password]):
        return jsonify({'success': False, 'message': 'All fields are required'})
    if role not in ('admin', 'data_engineer', 'data_analyst'):
        return jsonify({'success': False, 'message': 'Invalid role'})
    if User.query.filter_by(username=username).first():
        return jsonify({'success': False, 'message': 'Username already exists'})
    if User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'message': 'Email already exists'})
    user = User(username=username, email=email, role=role, created_by=session['user_id'])
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    # Try to send welcome email (non-blocking)
    try:
        _send_welcome_email(email, username, password, role)
    except Exception as e:
        logger.warning(f'Could not send welcome email: {e}')
    return jsonify({'success': True, 'message': f'User {username} created successfully'})

@app.route('/api/admin/users/<int:user_id>', methods=['PUT'])
@login_required
@role_required('admin')
def api_update_user(user_id):
    user = User.query.get_or_404(user_id)
    data = request.get_json()
    if 'role' in data and data['role'] in ('admin', 'data_engineer', 'data_analyst'):
        user.role = data['role']
    if 'is_active' in data:
        user.is_active = bool(data['is_active'])
    if 'password' in data and data['password']:
        user.set_password(data['password'])
    db.session.commit()
    return jsonify({'success': True, 'message': 'User updated'})

@app.route('/api/admin/users/<int:user_id>', methods=['DELETE'])
@login_required
@role_required('admin')
def api_delete_user(user_id):
    if user_id == session['user_id']:
        return jsonify({'success': False, 'message': 'Cannot delete your own account'})
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    return jsonify({'success': True, 'message': 'User deleted'})

def _build_email_html(username, password, role):
    role_labels = {'admin': 'Administrator', 'data_engineer': 'Data Engineer', 'data_analyst': 'Data Analyst'}
    return f"""
    <div style="font-family:Arial,sans-serif;max-width:520px;margin:auto;border:1px solid #e0e0e0;border-radius:8px;padding:32px;">
      <h2 style="color:#2c3e50;">Welcome to Clinical Data Pipeline</h2>
      <p>Your account has been created. Here are your login credentials:</p>
      <table style="border-collapse:collapse;width:100%;">
        <tr><td style="padding:8px;font-weight:bold;width:120px;">Username</td><td style="padding:8px;">{username}</td></tr>
        <tr style="background:#f9f9f9;"><td style="padding:8px;font-weight:bold;">Password</td><td style="padding:8px;">{password}</td></tr>
        <tr><td style="padding:8px;font-weight:bold;">Role</td><td style="padding:8px;">{role_labels.get(role, role)}</td></tr>
      </table>
      <p style="margin-top:24px;color:#e74c3c;font-size:13px;">Please change your password after your first login.</p>
    </div>
    """


def _send_via_sendgrid(to_email, subject, html_body):
    """Send email using SendGrid HTTP API (no port restrictions, works on all hosts)."""
    import requests as req
    api_key = os.environ.get('SENDGRID_API_KEY', '')
    sender = os.environ.get('SENDGRID_FROM_EMAIL', os.environ.get('SMTP_USER', ''))
    if not api_key or not sender:
        raise ValueError('SENDGRID_API_KEY and SENDGRID_FROM_EMAIL are not configured')
    payload = {
        'personalizations': [{'to': [{'email': to_email}]}],
        'from': {'email': sender},
        'subject': subject,
        'content': [{'type': 'text/html', 'value': html_body}]
    }
    resp = req.post(
        'https://api.sendgrid.com/v3/mail/send',
        headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
        json=payload,
        timeout=15
    )
    if resp.status_code not in (200, 202):
        raise RuntimeError(f'SendGrid returned {resp.status_code}: {resp.text}')


def _send_via_smtp(to_email, subject, html_body):
    """Send email using SMTP with STARTTLS."""
    smtp_host = os.environ.get('SMTP_HOST', '')
    smtp_port = int(os.environ.get('SMTP_PORT', 587))
    smtp_user = os.environ.get('SMTP_USER', '')
    smtp_pass = os.environ.get('SMTP_PASS', '')
    if not smtp_host or not smtp_user:
        raise ValueError('SMTP_HOST and SMTP_USER are not configured')
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = smtp_user
    msg['To'] = to_email
    msg.attach(MIMEText(html_body, 'html'))
    with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, to_email, msg.as_string())


def _send_welcome_email(to_email, username, password, role):
    """Send welcome email with credentials.
    Tries SendGrid first (recommended for production), falls back to SMTP.
    Configure via environment variables — see render.yaml for required keys.
    """
    subject = 'Clinical Data Pipeline — Your Account Credentials'
    html_body = _build_email_html(username, password, role)

    sendgrid_key = os.environ.get('SENDGRID_API_KEY', '')
    if sendgrid_key:
        logger.info(f'Sending welcome email to {to_email} via SendGrid')
        _send_via_sendgrid(to_email, subject, html_body)
        logger.info(f'Welcome email sent successfully to {to_email}')
        return

    smtp_host = os.environ.get('SMTP_HOST', '')
    if smtp_host:
        logger.info(f'Sending welcome email to {to_email} via SMTP ({smtp_host})')
        _send_via_smtp(to_email, subject, html_body)
        logger.info(f'Welcome email sent successfully to {to_email}')
        return

    raise ValueError(
        'No email provider configured. Set SENDGRID_API_KEY (recommended) '
        'or SMTP_HOST + SMTP_USER + SMTP_PASS environment variables.'
    )

# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE SCHEDULING (APScheduler-based DAG simulation)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_next_run(cron_expr):
    """Compute next run datetime from a simple cron expression using croniter if available."""
    try:
        from croniter import croniter
        base = datetime.utcnow()
        return croniter(cron_expr, base).get_next(datetime)
    except Exception:
        # Fallback: next run in 24h
        from datetime import timedelta
        return datetime.utcnow() + timedelta(hours=24)

def _run_scheduled_pipeline(schedule_id, database_id):
    """Execute pipeline for a scheduled run (runs in background thread)."""
    import threading
    def _execute():
        with app.app_context():
            run = ScheduleRun(schedule_id=schedule_id, database_id=database_id,
                              triggered_by='schedule', status='running')
            db.session.add(run)
            db.session.commit()
            run_id = run.id
            try:
                # Reuse existing pipeline trigger logic
                import requests as _req
                _req.post(f'http://127.0.0.1:5000/api/pipeline/run',
                          json={'database_id': database_id}, timeout=5)
                run.status = 'completed'
                run.completed_at = datetime.utcnow()
            except Exception as e:
                run.status = 'failed'
                run.error_message = str(e)
                run.completed_at = datetime.utcnow()
            # Update schedule
            sched = PipelineSchedule.query.get(schedule_id)
            if sched:
                sched.last_run = datetime.utcnow()
                sched.run_count = (sched.run_count or 0) + 1
                sched.next_run = _compute_next_run(sched.cron_expression)
            db.session.commit()
    threading.Thread(target=_execute, daemon=True).start()

def _start_scheduler():
    """Start the background scheduler that checks for due schedules every minute."""
    import threading, time
    def _loop():
        while True:
            time.sleep(60)
            try:
                with app.app_context():
                    now = datetime.utcnow()
                    due = PipelineSchedule.query.filter(
                        PipelineSchedule.is_active == True,
                        PipelineSchedule.next_run <= now
                    ).all()
                    for sched in due:
                        logger.info(f'Scheduler: triggering pipeline for db {sched.database_id}')
                        _run_scheduled_pipeline(sched.id, sched.database_id)
                        sched.next_run = _compute_next_run(sched.cron_expression)
                    if due:
                        db.session.commit()
            except Exception as e:
                logger.error(f'Scheduler loop error: {e}')
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    logger.info('Background scheduler started')

@app.route('/automation')
@login_required
@role_required('admin', 'data_engineer')
def automation():
    return render_template('automation.html')

@app.route('/api/schedules', methods=['GET'])
@login_required
def api_list_schedules():
    database_id = request.args.get('database_id')
    q = PipelineSchedule.query
    if database_id:
        q = q.filter_by(database_id=int(database_id))
    schedules = q.order_by(PipelineSchedule.created_at.desc()).all()
    result = []
    for s in schedules:
        db_conn = DatabaseConnection.query.get(s.database_id)
        creator = User.query.get(s.created_by) if s.created_by else None
        result.append({
            'id': s.id,
            'database_id': s.database_id,
            'database_name': db_conn.name if db_conn else 'Unknown',
            'schedule_type': s.schedule_type,
            'cron_expression': s.cron_expression,
            'is_active': s.is_active,
            'created_by': creator.username if creator else 'system',
            'created_at': s.created_at.strftime('%Y-%m-%d %H:%M') if s.created_at else None,
            'last_run': s.last_run.strftime('%Y-%m-%d %H:%M') if s.last_run else None,
            'next_run': s.next_run.strftime('%Y-%m-%d %H:%M') if s.next_run else None,
            'run_count': s.run_count or 0,
        })
    return jsonify({'success': True, 'schedules': result})

@app.route('/api/schedules', methods=['POST'])
@login_required
@role_required('admin', 'data_engineer')
def api_create_schedule():
    data = request.get_json()
    database_id = data.get('database_id')
    schedule_type = data.get('schedule_type', 'daily')
    cron_expression = data.get('cron_expression', '0 2 * * *')
    if not database_id:
        return jsonify({'success': False, 'message': 'database_id required'})
    # Validate cron
    try:
        from croniter import croniter
        if not croniter.is_valid(cron_expression):
            return jsonify({'success': False, 'message': 'Invalid cron expression'})
    except ImportError:
        pass
    sched = PipelineSchedule(
        database_id=int(database_id),
        schedule_type=schedule_type,
        cron_expression=cron_expression,
        created_by=session.get('user_id'),
        next_run=_compute_next_run(cron_expression)
    )
    db.session.add(sched)
    db.session.commit()
    return jsonify({'success': True, 'message': 'Schedule created', 'id': sched.id})

@app.route('/api/schedules/<int:sched_id>', methods=['PUT'])
@login_required
@role_required('admin', 'data_engineer')
def api_update_schedule(sched_id):
    sched = PipelineSchedule.query.get_or_404(sched_id)
    data = request.get_json()
    if 'is_active' in data:
        sched.is_active = bool(data['is_active'])
    if 'cron_expression' in data:
        sched.cron_expression = data['cron_expression']
        sched.next_run = _compute_next_run(data['cron_expression'])
    if 'schedule_type' in data:
        sched.schedule_type = data['schedule_type']
    db.session.commit()
    return jsonify({'success': True, 'message': 'Schedule updated'})

@app.route('/api/schedules/<int:sched_id>', methods=['DELETE'])
@login_required
@role_required('admin', 'data_engineer')
def api_delete_schedule(sched_id):
    sched = PipelineSchedule.query.get_or_404(sched_id)
    db.session.delete(sched)
    db.session.commit()
    return jsonify({'success': True, 'message': 'Schedule deleted'})

@app.route('/api/schedules/<int:sched_id>/run-now', methods=['POST'])
@login_required
@role_required('admin', 'data_engineer')
def api_run_schedule_now(sched_id):
    sched = PipelineSchedule.query.get_or_404(sched_id)
    _run_scheduled_pipeline(sched.id, sched.database_id)
    return jsonify({'success': True, 'message': 'Pipeline triggered manually'})

@app.route('/api/schedules/runs', methods=['GET'])
@login_required
def api_schedule_runs():
    database_id = request.args.get('database_id')
    limit = int(request.args.get('limit', 20))
    q = ScheduleRun.query
    if database_id:
        q = q.filter_by(database_id=int(database_id))
    runs = q.order_by(ScheduleRun.started_at.desc()).limit(limit).all()
    result = []
    for r in runs:
        sched = PipelineSchedule.query.get(r.schedule_id)
        db_conn = DatabaseConnection.query.get(r.database_id)
        duration = None
        if r.completed_at and r.started_at:
            duration = int((r.completed_at - r.started_at).total_seconds())
        result.append({
            'id': r.id,
            'schedule_id': r.schedule_id,
            'database_name': db_conn.name if db_conn else 'Unknown',
            'cron': sched.cron_expression if sched else '—',
            'started_at': r.started_at.strftime('%Y-%m-%d %H:%M:%S') if r.started_at else None,
            'completed_at': r.completed_at.strftime('%Y-%m-%d %H:%M:%S') if r.completed_at else None,
            'duration_sec': duration,
            'status': r.status,
            'triggered_by': r.triggered_by,
            'error_message': r.error_message,
        })
    return jsonify({'success': True, 'runs': result})

# Run on startup regardless of how the app is launched (gunicorn or direct)
with app.app_context():
    db.create_all()
    if not User.query.first():
        admin = User(username='admin', email='admin@clinicalpipeline.com', role='admin')
        admin.set_password('Admin@123')
        db.session.add(admin)
        db.session.commit()
        logger.info('Default admin user created: admin / Admin@123')
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('uploads/tmp_chunks', exist_ok=True)

_start_scheduler()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
