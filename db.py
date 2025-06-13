# db.py
import json
import boto3
import logging
from typing import Dict, Any, Optional, List, Tuple
from config import AWS_REGION, DB_SELECT_LAMBDA
import time
import uuid

logger = logging.getLogger()
logger.setLevel(logging.INFO)

lambda_client = boto3.client('lambda', region_name=AWS_REGION)
dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)

def invoke_db_select(table_name: str, index_name: Optional[str], key_name: str, key_value: Any) -> Optional[List[Dict[str, Any]]]:
    """
    Generic function to invoke the db-select Lambda for read operations only.
    Returns a list of items or None if the invocation failed.
    """
    try:
        payload = {
            'table_name': table_name,
            'index_name': index_name,
            'key_name': key_name,
            'key_value': key_value
        }
        
        response = lambda_client.invoke(
            FunctionName=DB_SELECT_LAMBDA,
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )
        
        response_payload = json.loads(response['Payload'].read())
        if response_payload['statusCode'] != 200:
            logger.error(f"Database Lambda failed: {response_payload}")
            return None
            
        result = json.loads(response_payload['body'])
        logger.info(f"Database Lambda response: {result}")
        return result if isinstance(result, list) else None
    except Exception as e:
        logger.error(f"Error invoking database Lambda: {str(e)}")
        return None

def get_conversation_id(message_id: str) -> Optional[str]:
    """Get conversation ID by message ID."""
    if not message_id:
        return None
    
    result = invoke_db_select(
        table_name='Conversations',
        index_name='response_id-index',
        key_name='response_id',
        key_value=message_id
    )
    
    # Handle list response
    if isinstance(result, list) and result:
        return result[0].get('conversation_id')
    return None

def get_associated_account(email: str) -> Optional[str]:
    """Get account ID by email."""
    result = invoke_db_select(
        table_name='Users',
        index_name='responseEmail-index',
        key_name='responseEmail',
        key_value=email.lower()
    )
    
    # Handle list response
    if isinstance(result, list) and result:
        return result[0].get('id')
    return None

def get_email_chain(conversation_id: str) -> List[Dict[str, Any]]:
    """Get email chain for a conversation."""
    result = invoke_db_select(
        table_name='Conversations',
        index_name='conversation_id-index',
        key_name='conversation_id',
        key_value=conversation_id
    )
    
    # Handle list response directly
    if not isinstance(result, list):
        return []
        
    # Sort by timestamp and format items
    sorted_items = sorted(result, key=lambda x: x.get('timestamp', ''))
    
    return [{
        'subject': item.get('subject', ''),
        'body': item.get('body', ''),
        'sender': item.get('sender', ''),
        'timestamp': item.get('timestamp', ''),
        'type': item.get('type', '')
    } for item in sorted_items]

def get_account_email(account_id: str) -> Optional[str]:
    """Get account email by account ID."""
    result = invoke_db_select(
        table_name='Users',
        index_name='id-index',
        key_name='id',
        key_value=account_id
    )
    
    # Handle list response
    if isinstance(result, list) and result:
        return result[0].get('responseEmail')
    return None

def update_thread_ev(conversation_id: str, ev_score: int) -> bool:
    """Update thread with new EV score using direct DynamoDB access."""
    try:
        threads_table = dynamodb.Table('Threads')
        threads_table.update_item(
            Key={'conversation_id': conversation_id},
            UpdateExpression='SET #flag = :flag',
            ExpressionAttributeNames={'#flag': 'flag'},
            ExpressionAttributeValues={':flag': ev_score >= 80}
        )
        logger.info(f"Successfully updated thread EV score for conversation {conversation_id}")
        return True
    except Exception as e:
        logger.error(f"Error updating thread EV score: {str(e)}")
        return False

def update_conversation_ev(conversation_id: str, message_id: str, ev_score: int) -> bool:
    """Update conversation with EV score using direct DynamoDB access."""
    try:
        conversations_table = dynamodb.Table('Conversations')
        conversations_table.update_item(
            Key={
                'conversation_id': conversation_id,
                'response_id': message_id
            },
            UpdateExpression='SET ev_score = :ev',
            ExpressionAttributeValues={':ev': str(ev_score)}
        )
        logger.info(f"Successfully updated conversation EV score for {conversation_id} message {message_id}")
        return True
    except Exception as e:
        logger.error(f"Error updating conversation EV score: {str(e)}")
        return False

def store_ai_invocation(
    associated_account: str,
    input_tokens: int,
    output_tokens: int,
    llm_email_type: str,
    model_name: str,
    conversation_id: Optional[str] = None
) -> bool:
    """
    Store an AI invocation record in DynamoDB.
    
    Args:
        associated_account: The user's account ID
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens generated
        llm_email_type: Type of LLM invocation (e.g., 'flag', 'ev_calculation')
        model_name: Name of the model used
        conversation_id: Optional conversation ID if applicable
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        invocations_table = dynamodb.Table('Invocations')
        
        # Create timestamp for sorting
        timestamp = int(time.time() * 1000)  # Current time in milliseconds
        
        item = {
            'id': str(uuid.uuid4()),  # Generate unique ID for the invocation
            'associated_account': associated_account,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'llm_email_type': llm_email_type,
            'model_name': model_name,
            'timestamp': timestamp
        }
        
        # Add conversation_id if provided
        if conversation_id:
            item['conversation_id'] = conversation_id
            
        invocations_table.put_item(Item=item)
        logger.info(f"Successfully stored AI invocation record for account {associated_account}")
        return True
        
    except Exception as e:
        logger.error(f"Error storing AI invocation record: {str(e)}")
        return False

def get_user_rate_limits(account_id: str) -> Dict[str, int]:
    """
    Get rate limits for a user from the Users table.
    Returns a dict with 'rl_aws' and 'rl_ai' limits.
    """
    try:
        users_table = dynamodb.Table('Users')
        response = users_table.get_item(
            Key={'id': account_id}
        )
        
        if 'Item' not in response:
            logger.error(f"User {account_id} not found in Users table")
            return {'rl_aws': 0, 'rl_ai': 0}
            
        user = response['Item']
        return {
            'rl_aws': int(user.get('rl_aws', 0)),
            'rl_ai': int(user.get('rl_ai', 0))
        }
    except Exception as e:
        logger.error(f"Error getting user rate limits: {str(e)}")
        return {'rl_aws': 0, 'rl_ai': 0}

def check_and_update_aws_rate_limit(account_id: str) -> Tuple[bool, str]:
    """
    Check and update AWS rate limit for an account.
    Returns (is_allowed, message).
    """
    try:
        # Get user's rate limit
        rate_limits = get_user_rate_limits(account_id)
        aws_limit = rate_limits['rl_aws']
        
        # Get current invocation count
        rl_table = dynamodb.Table('RL_AWS')
        response = rl_table.get_item(
            Key={'associated_account': account_id}
        )
        
        current_invocations = 0
        if 'Item' in response:
            current_invocations = int(response['Item'].get('invocations', 0))
        else:
            # If no record exists, create one with TTL set to 1 minute from now
            current_time_s = int(time.time())
            ttl_time_s = current_time_s + 60  # 1 minute from now in seconds
            
            rl_table.put_item(
                Item={
                    'associated_account': account_id,
                    'invocations': 1,
                    'ttl': ttl_time_s
                }
            )
            return True, ""
            
        # Check if limit exceeded
        if current_invocations >= aws_limit:
            return False, "AWS rate limit exceeded"
            
        # Update invocation count
        rl_table.update_item(
            Key={'associated_account': account_id},
            UpdateExpression='SET invocations = :val, #ttl = :ttl',
            ExpressionAttributeValues={
                ':val': current_invocations + 1,
                ':ttl': ttl_time_s
            },
            ExpressionAttributeNames={
                '#ttl': 'ttl'
            }
        )
        
        return True, ""
        
    except Exception as e:
        logger.error(f"Error checking AWS rate limit: {str(e)}")
        return False, "Error checking AWS rate limit"

def check_and_update_ai_rate_limit(account_id: str) -> Tuple[bool, str]:
    """
    Check and update AI rate limit for an account.
    Returns (is_allowed, message).
    """
    try:
        # Get user's rate limit
        rate_limits = get_user_rate_limits(account_id)
        ai_limit = rate_limits['rl_ai']
        
        # Get current invocation count
        rl_table = dynamodb.Table('RL_AI')
        response = rl_table.get_item(
            Key={'associated_account': account_id}
        )
        
        current_invocations = 0
        if 'Item' in response:
            current_invocations = int(response['Item'].get('invocations', 0))
        else:
            # If no record exists, create one with TTL set to 1 minute from now
            current_time_s = int(time.time())
            ttl_time_s = current_time_s + 60  # 1 minute from now in seconds
            
            rl_table.put_item(
                Item={
                    'associated_account': account_id,
                    'invocations': 1,
                    'ttl': ttl_time_s
                }
            )
            return True, ""
            
        # Check if limit exceeded
        if current_invocations >= ai_limit:
            return False, "AI rate limit exceeded"
            
        # Update invocation count
        rl_table.update_item(
            Key={'associated_account': account_id},
            UpdateExpression='SET invocations = :val',
            ExpressionAttributeValues={':val': current_invocations + 1}
        )
        
        return True, ""
        
    except Exception as e:
        logger.error(f"Error checking AI rate limit: {str(e)}")
        return False, "Error checking AI rate limit" 