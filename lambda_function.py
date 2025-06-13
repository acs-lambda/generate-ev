import json
import boto3
import logging
from typing import Dict, Any, List

from ev_calculator import calc_ev, parse_messages
from db import get_email_chain, get_account_email, check_and_update_aws_rate_limit
from flag_llm import invoke_flag_llm

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodb = boto3.resource('dynamodb')

def update_thread_ev(conversation_id: str, ev_score: int, should_flag: bool) -> None:
    """
    Updates the thread with the new EV score and flag status.
    """
    try:
        threads_table = dynamodb.Table('Threads')
        threads_table.update_item(
            Key={
                'conversation_id': conversation_id
            },
            UpdateExpression='SET #flag = :flag, ev_score = :ev',
            ExpressionAttributeNames={
                '#flag': 'flag'
            },
            ExpressionAttributeValues={
                ':flag': should_flag,
                ':ev': str(ev_score)
            }
        )
        logger.info(f"Updated thread flag for conversation {conversation_id} with EV score {ev_score} and flag {should_flag}")
    except Exception as e:
        logger.error(f"Error updating thread flag: {str(e)}")
        raise

def update_conversation_ev(conversation_id: str, message_id: str, ev_score: int) -> None:
    """
    Updates the conversation with the EV score.
    """
    try:
        conversations_table = dynamodb.Table('Conversations')
        conversations_table.update_item(
            Key={
                'conversation_id': conversation_id,
                'response_id': message_id
            },
            UpdateExpression='SET ev_score = :ev',
            ExpressionAttributeValues={
                ':ev': str(ev_score)
            }
        )
        logger.info(f"Updated conversation EV score for {conversation_id} message {message_id}")
    except Exception as e:
        logger.error(f"Error updating conversation EV score: {str(e)}")
        raise

def calculate_ev_for_conversation(conversation_id: str, message_id: str, account_id: str) -> Dict[str, Any]:
    """
    Calculates EV for a conversation and updates the database.
    Returns the EV score and updated status.
    """
    try:
        # Get the email chain and realtor email
        chain = get_email_chain(conversation_id)
        realtor_email = get_account_email(account_id)
        
        if not chain or not realtor_email:
            raise ValueError("Could not get email chain or realtor email")

        # Calculate EV
        ev, ev_token_usage = calc_ev(parse_messages(realtor_email, chain), account_id, conversation_id)
        logger.info(f"Calculated EV score: {ev} for conversation {conversation_id}")

        # Determine if the email should be flagged using the flag LLM
        should_flag, flag_token_usage = invoke_flag_llm(chain, account_id, conversation_id)
        logger.info(f"Flag LLM decision: {should_flag} for conversation {conversation_id}")

        # Update both thread and conversation
        update_thread_ev(conversation_id, ev, should_flag)
        update_conversation_ev(conversation_id, message_id, ev)

        return {
            'ev_score': ev,
            'conversation_id': conversation_id,
            'message_id': message_id,
            'status': 'success',
            'token_usage': {
                'ev_calculation': ev_token_usage,
                'flag_determination': flag_token_usage
            }
        }
    except Exception as e:
        logger.error(f"Error calculating EV: {str(e)}")
        return {
            'status': 'error',
            'error': str(e)
        }

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for EV calculation.
    Handles both direct invocation and API Gateway events.
    
    Direct invocation format:
    {
        'conversation_id': str,
        'message_id': str,
        'account_id': str
    }
    
    API Gateway format:
    {
        'body': JSON string containing {
            'conversation_id': str,
            'response_id': str,  # Used as message_id
            'account_id': str
        }
    }
    """
    try:
        # Check if this is an API Gateway event
        if 'body' in event and isinstance(event['body'], str):
            try:
                payload = json.loads(event['body'])
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON in request body")
            
            # Map API Gateway fields to expected format
            event_data = {
                'conversation_id': payload.get('conversation_id'),
                'message_id': payload.get('response_id'),  # Map response_id to message_id
                'account_id': payload.get('account_id')
            }
        else:
            # Direct invocation format
            event_data = event

        # Validate input
        required_fields = ['conversation_id', 'message_id', 'account_id']
        missing_fields = [field for field in required_fields if not event_data.get(field)]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

        # Check AWS rate limit
        is_allowed, message = check_and_update_aws_rate_limit(event_data['account_id'])
        if not is_allowed:
            return {
                'statusCode': 429,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'status': 'error',
                    'error': message
                })
            }

        # Calculate EV and update database
        result = calculate_ev_for_conversation(
            event_data['conversation_id'],
            event_data['message_id'],
            event_data['account_id']
        )

        # Format response based on success/failure
        status_code = 200 if result['status'] == 'success' else 500
        response = {
            'statusCode': status_code,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(result)
        }
        
        return response

    except Exception as e:
        logger.error(f"Error in lambda handler: {str(e)}")
        error_response = {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'status': 'error',
                'error': str(e)
            })
        }
        return error_response 