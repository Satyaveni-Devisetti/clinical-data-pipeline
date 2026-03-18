# Clinical Data Pipeline

A comprehensive clinical data processing pipeline with advanced ETL capabilities and predictive analytics.

## Features

- **Database Management**: PostgreSQL/Neon DB connection with schema management
- **File Upload & Validation**: CSV, XML, NDJSON support with column validation
- **ETL Pipeline**: Bronze → Silver → Gold → Prediction layers
- **Data Processing**: Cleaning, normalization, transformations
- **Analytics & Predictions**: Time series forecasting and risk assessment
- **Professional UI**: Clinical environment interface with animations
- **Apache Airflow**: Automated pipeline orchestration

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open browser to `http://localhost:5000`

## Architecture

### Database Layers

- **Bronze**: Raw data storage (as uploaded)
- **Silver**: Cleaned and normalized data
- **Gold**: Analytics-ready fact tables with pseudonymization
- **Prediction**: Site-level forecasting and risk assessment

### File Types Supported

- **Subjects**: CSV files with subject information
- **Laboratory**: XML files with test results
- **Adverse Events**: NDJSON files with event data

### Data Transformations

- Subject ID normalization and pseudonymization
- Laboratory unit conversion to U/L
- Visit number standardization
- Date format normalization
- Safety flag application

## Configuration

### Database Connection

Example Neon DB connection:
```
postgresql://neondb_owner:npg_bxsDXUdeE57B@ep-sweet-water-aikpodpl-pooler.c-4.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require
```

### Apache Airflow

1. Configure Airflow connection: `clinical_db`
2. Set notification email variable: `pipeline_notification_email`
3. Enable DAG: `clinical_data_pipeline`

## API Endpoints

- `POST /api/database/connect` - Connect database
- `GET /api/database/list` - List databases
- `POST /api/upload` - Upload files
- `GET /api/files/list` - List uploaded files
- `POST /api/pipeline/run` - Run pipeline

## Pages

- **Home**: Dashboard with metrics and pipeline status
- **Upload**: File upload with validation
- **Results**: View processed data by layer
- **Analytics**: Interactive dashboards and charts
- **Predictions**: Site-level forecasting and risk assessment

## Error Handling

- Comprehensive error logging
- Unprocessed data tracking
- Pipeline continues on individual record failures
- User notifications for all operations

## Security

- Subject ID pseudonymization
- No PHI in prediction layer
- Secure database connections
- Input validation and sanitization
