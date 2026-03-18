from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.email import EmailOperator
from airflow.models import Variable
import pandas as pd
import numpy as np
import logging
import json
import hashlib
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# Configure logging
logger = logging.getLogger(__name__)

# Default arguments for the DAG
default_args = {
    'owner': 'clinical_pipeline',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'clinical_data_pipeline',
    default_args=default_args,
    description='Clinical Data Processing Pipeline - Bronze, Silver, Gold, Prediction Layers',
    schedule_interval=None,  # Manual trigger or can be set to cron schedule
    catchup=False,
    tags=['clinical', 'etl', 'pipeline'],
    max_active_runs=1,
)

# Column validation lists
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

def get_database_connection():
    """Get database connection from Airflow connections"""
    postgres_hook = PostgresHook(postgres_conn_id='clinical_db')
    return postgres_hook.get_conn()

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

def bronze_layer_processing(**context):
    """Process and load data into bronze layer"""
    logger.info("Starting Bronze Layer Processing")
    
    conn = get_database_connection()
    cursor = conn.cursor()
    
    try:
        # Process subjects data
        cursor.execute("SELECT * FROM bronze.bronze_subjects WHERE processed = FALSE")
        subjects = cursor.fetchall()
        
        for subject in subjects:
            try:
                # Raw data processing logic here
                logger.info(f"Processing subject: {subject[0]}")
                # Mark as processed
                cursor.execute(
                    "UPDATE bronze.bronze_subjects SET processed = TRUE WHERE subject_id = %s",
                    (subject[0],)
                )
            except Exception as e:
                logger.error(f"Error processing subject {subject[0]}: {str(e)}")
                # Log to unprocessed table
                cursor.execute("""
                    INSERT INTO processing_errors 
                    (table_name, record_id, error_message, error_time)
                    VALUES (%s, %s, %s, %s)
                """, ('bronze_subjects', subject[0], str(e), datetime.now()))
        
        # Process labs data
        cursor.execute("SELECT * FROM bronze.bronze_labs WHERE processed = FALSE")
        labs = cursor.fetchall()
        
        for lab in labs:
            try:
                logger.info(f"Processing lab record: {lab[0]}")
                cursor.execute(
                    "UPDATE bronze.bronze_labs SET processed = TRUE WHERE subject_id = %s AND testcode = %s",
                    (lab[0], lab[2])
                )
            except Exception as e:
                logger.error(f"Error processing lab record {lab[0]}: {str(e)}")
                cursor.execute("""
                    INSERT INTO processing_errors 
                    (table_name, record_id, error_message, error_time)
                    VALUES (%s, %s, %s, %s)
                """, ('bronze_labs', f"{lab[0]}_{lab[2]}", str(e), datetime.now()))
        
        # Process adverse events data
        cursor.execute("SELECT * FROM bronze.bronze_aes WHERE processed = FALSE")
        aes = cursor.fetchall()
        
        for ae in aes:
            try:
                logger.info(f"Processing AE record: {ae[0]}")
                cursor.execute(
                    "UPDATE bronze.bronze_aes SET processed = TRUE WHERE subject_id = %s AND ae_id = %s",
                    (ae[0], ae[1])
                )
            except Exception as e:
                logger.error(f"Error processing AE record {ae[0]}: {str(e)}")
                cursor.execute("""
                    INSERT INTO processing_errors 
                    (table_name, record_id, error_message, error_time)
                    VALUES (%s, %s, %s, %s)
                """, ('bronze_aes', f"{ae[0]}_{ae[1]}", str(e), datetime.now()))
        
        conn.commit()
        logger.info("Bronze Layer Processing Completed Successfully")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Bronze Layer Processing Failed: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

def silver_layer_processing(**context):
    """Process bronze layer data and load into silver layer"""
    logger.info("Starting Silver Layer Processing")
    
    conn = get_database_connection()
    cursor = conn.cursor()
    
    try:
        # Process subjects from bronze to silver
        cursor.execute("SELECT * FROM bronze.bronze_subjects")
        subjects = cursor.fetchall()
        
        for subject in subjects:
            try:
                raw_data = json.loads(subject[8]) if subject[8] else {}
                
                # Apply transformations
                subject_id = str(raw_data.get('subject_id', '')).strip().upper()
                site_id = str(raw_data.get('site_id', '')).strip().upper()
                sex = str(raw_data.get('sex', '')).strip().upper()
                dob = raw_data.get('dob')
                arm = str(raw_data.get('arm', '')).strip().upper()
                start_date = raw_data.get('start_date')
                
                if not subject_id:
                    continue
                
                cursor.execute("""
                    INSERT INTO silver.silver_subjects 
                    (subject_id, site_id, sex, dob, arm, start_date)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (subject_id) DO UPDATE SET
                    site_id = EXCLUDED.site_id,
                    sex = EXCLUDED.sex,
                    dob = EXCLUDED.dob,
                    arm = EXCLUDED.arm,
                    start_date = EXCLUDED.start_date
                """, (subject_id, site_id, sex, dob, arm, start_date))
                
            except Exception as e:
                logger.error(f"Error processing subject in silver layer: {str(e)}")
                continue
        
        # Process labs from bronze to silver
        cursor.execute("SELECT * FROM bronze.bronze_labs")
        labs = cursor.fetchall()
        
        for lab in labs:
            try:
                raw_data = json.loads(lab[8]) if lab[8] else {}
                
                subject_id = str(raw_data.get('subject_id', '')).strip().upper()
                visitno = normalize_visit_no(raw_data.get('visitno'))
                testcode = str(raw_data.get('testcode', '')).strip().upper()
                resultvalue = raw_data.get('resultvalue')
                units = str(raw_data.get('units', '')).strip().upper()
                reflow = raw_data.get('reflow')
                refhigh = raw_data.get('refhigh')
                resultdate = raw_data.get('resultdate')
                
                if not subject_id or not testcode:
                    continue
                
                # Normalize units and result value
                lab_row = {'units': units, 'resultvalue': resultvalue}
                lab_row = normalize_lab_units_row(lab_row)
                
                try:
                    resultvalue = int(np.ceil(float(lab_row['resultvalue'])))
                except:
                    resultvalue = None
                
                try:
                    reflow = int(np.ceil(float(reflow)))
                except:
                    reflow = None
                
                try:
                    refhigh = int(np.ceil(float(refhigh)))
                except:
                    refhigh = None
                
                cursor.execute("""
                    INSERT INTO silver.silver_labs 
                    (subject_id, visitno, testcode, resultvalue, units, reflow, refhigh, resultdate)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (subject_id, visitno, testcode, resultvalue, lab_row['units'], reflow, refhigh, resultdate))
                
            except Exception as e:
                logger.error(f"Error processing lab in silver layer: {str(e)}")
                continue
        
        # Process adverse events from bronze to silver
        cursor.execute("SELECT * FROM bronze.bronze_aes")
        aes = cursor.fetchall()
        
        for ae in aes:
            try:
                raw_data = json.loads(ae[8]) if ae[8] else {}
                
                subject_id = str(raw_data.get('subject_id', '')).strip().upper()
                ae_id = str(raw_data.get('ae_id', '')).strip()
                pt_code = str(raw_data.get('pt_code', '')).strip().upper()
                pt_name = str(raw_data.get('pt_name', '')).strip()
                severity = str(raw_data.get('severity', '')).strip()
                related = str(raw_data.get('related', '')).strip()
                serious = str(raw_data.get('serious', '')).strip()
                ae_start_dt = raw_data.get('ae_start_dt')
                
                if not subject_id:
                    continue
                
                cursor.execute("""
                    INSERT INTO silver.silver_aes 
                    (subject_id, ae_id, pt_code, pt_name, severity, related, serious, ae_start_dt)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (subject_id, ae_id, pt_code, pt_name, severity, related, serious, ae_start_dt))
                
            except Exception as e:
                logger.error(f"Error processing AE in silver layer: {str(e)}")
                continue
        
        conn.commit()
        logger.info("Silver Layer Processing Completed Successfully")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Silver Layer Processing Failed: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

def gold_layer_processing(**context):
    """Process silver layer data and load into gold layer"""
    logger.info("Starting Gold Layer Processing")
    
    conn = get_database_connection()
    cursor = conn.cursor()
    
    try:
        # Process subjects with pseudonymization
        cursor.execute("SELECT DISTINCT subject_id FROM silver.silver_subjects")
        subjects = cursor.fetchall()
        
        subject_mapping = {}
        for subject in subjects:
            subject_id = subject[0]
            pseudonym = generate_subject_pseudonym(subject_id)
            subject_mapping[subject_id] = pseudonym
        
        # Load fact_subjects
        cursor.execute("SELECT * FROM silver.silver_subjects")
        silver_subjects = cursor.fetchall()
        
        for subject in silver_subjects:
            subject_id = subject[0]
            pseudonym = subject_mapping.get(subject_id)
            
            cursor.execute("""
                INSERT INTO gold.fact_subjects 
                (subject_pseudonym, site_id, sex, dob, arm, start_date)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (subject_pseudonym) DO UPDATE SET
                site_id = EXCLUDED.site_id,
                sex = EXCLUDED.sex,
                dob = EXCLUDED.dob,
                arm = EXCLUDED.arm,
                start_date = EXCLUDED.start_date
            """, (pseudonym, subject[1], subject[2], subject[3], subject[4], subject[5]))
        
        # Load fact_labs with safety flags
        cursor.execute("SELECT * FROM silver.silver_labs")
        silver_labs = cursor.fetchall()
        
        for lab in silver_labs:
            subject_id = lab[0]
            pseudonym = subject_mapping.get(subject_id)
            
            # Apply safety flag logic (example: ALT > 3x ULN)
            safety_flag = False
            if lab[3] and lab[6]:  # resultvalue and refhigh
                try:
                    if float(lab[3]) > (3 * float(lab[6])):
                        safety_flag = True
                except:
                    pass
            
            cursor.execute("""
                INSERT INTO gold.fact_labs 
                (subject_pseudonym, visitno, testcode, resultvalue, units, reflow, refhigh, resultdate, safety_flag)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (pseudonym, lab[1], lab[2], lab[3], lab[4], lab[5], lab[6], lab[7], safety_flag))
        
        # Load fact_aes
        cursor.execute("SELECT * FROM silver.silver_aes")
        silver_aes = cursor.fetchall()
        
        for ae in silver_aes:
            subject_id = ae[0]
            pseudonym = subject_mapping.get(subject_id)
            
            cursor.execute("""
                INSERT INTO gold.fact_aes 
                (subject_pseudonym, ae_id, pt_code, pt_name, severity, related, serious, ae_start_dt)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (pseudonym, ae[1], ae[2], ae[3], ae[4], ae[5], ae[6], ae[7]))
        
        # Create site summary
        cursor.execute("SELECT site_id, COUNT(*) FROM gold.fact_subjects GROUP BY site_id")
        site_subjects = dict(cursor.fetchall())
        
        cursor.execute("SELECT COUNT(*) FROM gold.fact_labs")
        total_labs = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM gold.fact_aes")
        total_aes = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM gold.fact_aes WHERE serious IN ('Y', 'YES', 'TRUE')")
        total_serious = cursor.fetchone()[0]
        
        for site_id, subject_count in site_subjects.items():
            cursor.execute("""
                INSERT INTO gold.phi_site_summary 
                (site_id, total_subjects, total_lab_tests, total_adverse_events, total_serious_events, total_ae_signal_count)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (site_id) DO UPDATE SET
                total_subjects = EXCLUDED.total_subjects,
                total_lab_tests = EXCLUDED.total_lab_tests,
                total_adverse_events = EXCLUDED.total_adverse_events,
                total_serious_events = EXCLUDED.total_serious_events,
                total_ae_signal_count = EXCLUDED.total_ae_signal_count,
                updated_at = CURRENT_TIMESTAMP
            """, (site_id, subject_count, total_labs, total_aes, total_serious, 0))
        
        conn.commit()
        logger.info("Gold Layer Processing Completed Successfully")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Gold Layer Processing Failed: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

def prediction_layer_processing(**context):
    """Generate predictions from gold layer data"""
    logger.info("Starting Prediction Layer Processing")
    
    conn = get_database_connection()
    cursor = conn.cursor()
    
    try:
        # Get site-level historical data
        cursor.execute("SELECT * FROM gold.phi_site_summary")
        sites = cursor.fetchall()
        
        for site in sites:
            site_id = site[0]
            
            # Get historical trends (simplified example)
            cursor.execute("""
                SELECT DATE_TRUNC('month', created_at) as month, 
                       COUNT(*) as ae_count
                FROM gold.fact_aes 
                WHERE subject_pseudonym IN (
                    SELECT subject_pseudonym FROM gold.fact_subjects WHERE site_id = %s
                )
                GROUP BY DATE_TRUNC('month', created_at)
                ORDER BY month
                LIMIT 12
            """, (site_id,))
            
            monthly_data = cursor.fetchall()
            
            if len(monthly_data) >= 3:
                # Simple exponential smoothing for prediction
                ae_counts = [row[1] for row in monthly_data]
                model = SimpleExpSmoothing(ae_counts).fit()
                predicted_ae = int(model.forecast(1)[0])
            else:
                predicted_ae = site[3] if site[3] else 0  # Use current total
            
            # Risk group assignment (simplified)
            features = [
                predicted_ae,
                site[4] if site[4] else 0,  # serious events
                site[5] if site[5] else 0,  # signal count
                site[2] if site[2] else 0,  # lab tests
                site[1] if site[1] else 0   # subjects
            ]
            
            if sum(features) > 0:
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform([features])
                
                kmeans = KMeans(n_clusters=3, random_state=42)
                risk_group = kmeans.fit_predict(features_scaled)[0]
                
                risk_labels = ['Low', 'Medium', 'High']
                predicted_risk = risk_labels[risk_group]
            else:
                predicted_risk = 'Low'
            
            cursor.execute("""
                INSERT INTO prediction.site_predictions 
                (site_id, prediction_date, predicted_total_ae, predicted_total_serious_events, 
                 predicted_total_ae_signal_count, predicted_avg_lab_ratio, predicted_new_subjects, predicted_risk_group)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (site_id, prediction_date) DO UPDATE SET
                predicted_total_ae = EXCLUDED.predicted_total_ae,
                predicted_total_serious_events = EXCLUDED.predicted_total_serious_events,
                predicted_total_ae_signal_count = EXCLUDED.predicted_total_ae_signal_count,
                predicted_avg_lab_ratio = EXCLUDED.predicted_avg_lab_ratio,
                predicted_new_subjects = EXCLUDED.predicted_new_subjects,
                predicted_risk_group = EXCLUDED.predicted_risk_group
            """, (
                site_id, datetime.now().date(), predicted_ae, 
                site[4] if site[4] else 0,
                site[5] if site[5] else 0,
                0.1,  # placeholder ratio
                5,    # placeholder new subjects
                predicted_risk
            ))
        
        conn.commit()
        logger.info("Prediction Layer Processing Completed Successfully")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Prediction Layer Processing Failed: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

def send_pipeline_notification(**context):
    """Send pipeline completion notification"""
    logger.info("Sending pipeline completion notification")
    
    # Get pipeline execution details
    execution_date = context['execution_date']
    dag_id = context['dag'].dag_id
    
    # Send email notification (configure email settings in Airflow)
    try:
        EmailOperator(
            task_id='send_email_notification',
            to=Variable.get('pipeline_notification_email', default_var='admin@example.com'),
            subject=f'Clinical Pipeline Completed - {execution_date}',
            html_content=f'''
            <h2>Clinical Data Pipeline Execution Report</h2>
            <p><strong>DAG:</strong> {dag_id}</p>
            <p><strong>Execution Date:</strong> {execution_date}</p>
            <p><strong>Status:</strong> Completed Successfully</p>
            
            <h3>Pipeline Stages Completed:</h3>
            <ul>
                <li>✅ Bronze Layer - Raw data ingestion</li>
                <li>✅ Silver Layer - Data cleaning and normalization</li>
                <li>✅ Gold Layer - Fact tables and analytics</li>
                <li>✅ Prediction Layer - Forecasting and risk assessment</li>
            </ul>
            
            <p>Visit the <a href="http://localhost:5000">Clinical Dashboard</a> to view results.</p>
            '''
        ).execute(context)
        
    except Exception as e:
        logger.error(f"Failed to send notification: {str(e)}")

# Define the tasks
bronze_task = PythonOperator(
    task_id='bronze_layer_processing',
    python_callable=bronze_layer_processing,
    dag=dag,
)

silver_task = PythonOperator(
    task_id='silver_layer_processing',
    python_callable=silver_layer_processing,
    dag=dag,
)

gold_task = PythonOperator(
    task_id='gold_layer_processing',
    python_callable=gold_layer_processing,
    dag=dag,
)

prediction_task = PythonOperator(
    task_id='prediction_layer_processing',
    python_callable=prediction_layer_processing,
    dag=dag,
)

notification_task = PythonOperator(
    task_id='send_pipeline_notification',
    python_callable=send_pipeline_notification,
    dag=dag,
    trigger_rule='all_success',
)

# Define task dependencies
bronze_task >> silver_task >> gold_task >> prediction_task >> notification_task
