# db.py
import json
import boto3
import logging
from typing import Dict, Any, Optional, List, Tuple
from config import AWS_REGION, DB_SELECT_LAMBDA
import time
from utils import *
import uuid


logger = logging.getLogger()
logger.setLevel(logging.INFO)

lambda_client = boto3.client('lambda', region_name=AWS_REGION)
dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)


def get_conversation_id(message_id: str, account_id: str, session_id: str) -> Optional[str]:
    """Get conversation ID by message ID."""
    if not message_id:
        return None
    
    result = select(
        table_name='Conversations',
        index_name='response_id-index',
        key_name='response_id',
        key_value=message_id,
        account_id=account_id,
        session_id=session_id
    )
    
    # Handle list response
    if isinstance(result, list) and result:
        return result[0].get('conversation_id')
    return None

def get_associated_account(email: str, account_id: str, session_id: str) -> Optional[str]:
    """Get account ID by email."""
    result = select(
        table_name='Users',
        index_name='responseEmail-index',
        key_name='responseEmail',
        key_value=email.lower(),
        account_id=account_id,
        session_id=session_id
    )
    
    # Handle list response
    if isinstance(result, list) and result:
        return result[0].get('id')
    return None

def get_email_chain(conversation_id: str, account_id: str, session_id: str) -> List[Dict[str, Any]]:
    """Get email chain for a conversation."""
    result = select(
        table_name='Conversations',
        index_name='conversation_id-index',
        key_name='conversation_id',
        key_value=conversation_id,
        account_id=account_id,
        session_id=session_id
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

def get_account_email(account_id: str, session_id: str) -> Optional[str]:
    """Get account email by account ID."""
    result = select(
        table_name='Users',
        index_name='id-index',
        key_name='id',
        key_value=account_id,
        account_id=account_id,
        session_id=session_id
    )
    
    # Handle list response
    if isinstance(result, list) and result:
        return result[0].get('responseEmail')
    return None

def update_thread_ev(conversation_id: str, ev_score: int, flag_decision: int, account_id: str, session_id: str) -> bool:
    """
    Update the EV score and flag status for a thread.
    
    Args:
        conversation_id (str): The conversation ID to update
        ev_score (int): The calculated EV score
        flag_decision (int): The flag decision from the LLM
        account_id (str): The account ID for authorization
        session_id (str): The session ID for authorization
    
    Returns:
        bool: True if update was successful, False otherwise
    """
    try:
        update_data = {
            'flag': flag_decision,
            'ev_score': ev_score,
            'updated_at': int(time.time())
        }
        
        success = update(
            table_name='Threads',
            key_name='conversation_id',
            key_value=conversation_id,
            index_name='conversation_id-index',
            update_data=update_data,
            account_id=account_id
        )
        
        if success:
            logger.info(f"Updated thread EV score to {ev_score} and flag to {flag_decision} for conversation {conversation_id}")
        else:
            logger.error(f"Failed to update thread EV score for conversation {conversation_id}")
        
        return success
        
    except Exception as e:
        logger.error(f"Error updating thread EV: {str(e)}")
        return False

def update_conversation_ev(conversation_id: str, ev_score: int, account_id: str, session_id: str) -> bool:
    """
    Update the EV score for a conversation.
    
    Args:
        conversation_id (str): The conversation ID to update
        ev_score (int): The calculated EV score
        account_id (str): The account ID for authorization
        session_id (str): The session ID for authorization
    
    Returns:
        bool: True if update was successful, False otherwise
    """
    try:
        update_data = {
            'ev_score': ev_score,
            'updated_at': int(time.time())
        }
        
        success = update(
            table_name='Conversations',
            key_name='conversation_id',
            key_value=conversation_id,
            index_name='conversation_id-index',
            update_data=update_data,
            account_id=account_id
        )
        
        if success:
            logger.info(f"Updated conversation EV score to {ev_score} for conversation {conversation_id}")
        else:
            logger.error(f"Failed to update conversation EV score for conversation {conversation_id}")
        
        return success
        
    except Exception as e:
        logger.error(f"Error updating conversation EV: {str(e)}")
        return False

def store_ai_invocation(
    associated_account: str,
    input_tokens: int,
    output_tokens: int,
    llm_email_type: str,
    model_name: str,
    conversation_id: str,
    session_id: str
) -> bool:
    """
    Store a record of an AI invocation in the Invocations table.
    
    Args:
        associated_account (str): The account ID associated with the invocation
        input_tokens (int): Number of input tokens used
        output_tokens (int): Number of output tokens used
        llm_email_type (str): Type of LLM invocation (e.g., 'ev_calculation', 'flag')
        model_name (str): Name of the model used
        conversation_id (str): The conversation ID
        session_id (str): The session ID for authorization
    
    Returns:
        bool: True if storage was successful, False otherwise
    """
    try:
        timestamp = int(time.time())
        invocation_id = f"{conversation_id}_{timestamp}_{llm_email_type}"
        
        update_data = {
            'associated_account': associated_account,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'llm_email_type': llm_email_type,
            'model_name': model_name,
            'conversation_id': conversation_id,
            'created_at': timestamp,
            'updated_at': timestamp
        }
        
        success = update(
            table_name='Invocations',
            key_name='invocation_id',
            key_value=invocation_id,
            index_name='invocation_id-index',
            update_data=update_data,
            account_id=associated_account
        )
        
        if success:
            logger.info(f"Stored AI invocation record for conversation {conversation_id}")
        else:
            logger.error(f"Failed to store AI invocation record for conversation {conversation_id}")
        
        return success
        
    except Exception as e:
        logger.error(f"Error storing AI invocation: {str(e)}")
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
        # Calculate TTL at the start of the function
        current_time_s = int(time.time())
        ttl_time_s = current_time_s + 60  # 1 minute from now in seconds

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