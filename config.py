# config.py
import os

# Load configuration from environment variables
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-2')
TAI_KEY = os.environ['TAI_KEY']  # Together AI API key

# Database Lambda function name
DB_SELECT_LAMBDA = os.environ['DB_SELECT_LAMBDA']  # Single Lambda for all DB operations 