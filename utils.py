import json
import boto3
from typing import Dict, Any, Union, Optional
from botocore.exceptions import ClientError
import logging
from config import logger, AWS_REGION

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
lambda_client = boto3.client('lambda', region_name=AWS_REGION)
dynamodb = boto3.resource('dynamodb')
sessions_table = dynamodb.Table('Sessions')

class LambdaError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(f"[{status_code}] {message}")

def create_response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
        "body": json.dumps(body),
    }

def invoke_lambda(function_name, payload, invocation_type="RequestResponse"):
    try:
        logger.info(f"Invoking {function_name}...")
        response = lambda_client.invoke(
            FunctionName=function_name,
            InvocationType=invocation_type,
            Payload=json.dumps(payload),
        )
        response_payload = response["Payload"].read().decode("utf-8")
        if not response_payload:
            return {}
        return json.loads(response_payload)
    except ClientError as e:
        raise LambdaError(500, f"Failed to invoke {function_name}: {e.response['Error']['Message']}")
    except Exception as e:
        raise LambdaError(500, f"An unexpected error occurred invoking {function_name}: {e}")

def parse_event(event):
    response = invoke_lambda('ParseEvent', event)
    if response.get('statusCode') != 200:
        raise LambdaError(response.get('statusCode', 500), "Failed to parse event.")
    return json.loads(response.get('body', '{}'))

def authorize(user_id, session_id):
    response = invoke_lambda('Authorize', {'user_id': user_id, 'session_id': session_id})
    body = json.loads(response.get('body', '{}'))
    if response.get('statusCode') != 200 or not body.get('authorized'):
        raise LambdaError(response.get('statusCode', 401), body.get('message', 'ACS: Unauthorized'))

def db_select(table_name, index_name, key_name, key_value, account_id, session_id):
    payload = {'body': {'table_name': table_name, 'index_name': index_name, 'key_name': key_name, 'key_value': key_value, 'account_id': account_id, 'session': session_id}}
    response = invoke_lambda('DBSelect', payload)
    if response.get('statusCode') != 200:
        raise LambdaError(response.get('statusCode', 500), response.get('body', {}).get('error', 'DB select failed.'))
    return json.loads(response.get('body', '[]'))

def db_update(table_name, index_name, key_name, key_value, update_data, account_id, session_id):
    payload = {'body': {'table_name': table_name, 'index_name': index_name, 'key_name': key_name, 'key_value': key_value, 'update_data': update_data, 'account_id': account_id, 'session': session_id}}
    response = invoke_lambda('DBUpdate', payload)
    if response.get('statusCode') != 200:
        raise LambdaError(response.get('statusCode', 500), response.get('body', {}).get('error', 'DB update failed.'))
    return json.loads(response.get('body', '{}'))

def select(table_name: str, index_name: str, key_name: str, key_value: str, account_id: str, session_id: str) -> Dict[str, Any]:
    """
    Select a record from a DynamoDB table by key
    
    Args:
        table_name (str): The name of the DynamoDB table
        index_name (str): The name of the index to use
        key_name (str): The name of the key to use
        key_value (str): The value of the key to use
        account_id (str): The account ID to validate ownership
        session_id (str): The session ID to validate
        
    Returns:
        Dict[str, Any]: The selected record
        
    Raises:
        AuthorizationError: If authorization fails
    """
    try:
        # Invoke the select Lambda function
        response = db_select(table_name, index_name, key_name, key_value, account_id, session_id)
        
        return response
        
    except Exception as e:
        logger.error(f"Error selecting record: {str(e)}")
        raise

def update(table_name: str, index_name: str, key_name: str, key_value: str, account_id: str, session_id: str) -> Dict[str, Any]:
    """
    Update a record in a DynamoDB table by key
    
    Args:
        table_name (str): The name of the DynamoDB table
        index_name (str): The name of the index to use
        key_name (str): The name of the key to use
        key_value (str): The value of the key to use
        account_id (str): The account ID to validate ownership
        session_id (str): The session ID to validate
        
    Returns:
        Dict[str, Any]: The updated record
        
    Raises:
        AuthorizationError: If authorization fails
    """
    try:
        # Invoke the update Lambda function
        response = db_update(table_name, index_name, key_name, key_value, {}, account_id, session_id)
        
        return response
    
    except Exception as e:
        logger.error(f"Error updating record: {str(e)}")
        raise
    