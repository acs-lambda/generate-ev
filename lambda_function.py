import json
import boto3
import logging
from typing import Dict, Any, List, Tuple
import time

from ev_calculator import calc_ev, parse_messages
from db import get_email_chain, get_account_email
from flag_llm import invoke_flag_llm
from utils import parse_event, authorize, AuthorizationError, invoke_lambda, create_response, LambdaError
import os
from ev_logic import calculate_ev_for_conversation

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

AUTH_BP = os.environ.get('AUTH_BP', '')

dynamodb = boto3.resource('dynamodb')

def store_ai_invocation(associated_account: str, input_tokens: int, output_tokens: int, 
                       llm_email_type: str, model_name: str, conversation_id: str, 
                       session_id: str) -> bool:
    """Store AI invocation record in DynamoDB."""
    try:
        ai_invocations_table = dynamodb.Table('Invocations')
        ai_invocations_table.put_item(
            Item={
                'associated_account': associated_account,
                'timestamp': int(time.time()),
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'llm_email_type': llm_email_type,
                'model_name': model_name,
                'conversation_id': conversation_id,
                'session_id': session_id
            }
        )
        return True
    except Exception as e:
        logger.error(f"Error storing AI invocation: {str(e)}")
        return False

def update_thread_ev(conversation_id: str, ev_score: int, should_flag: bool, account_id: str, session_id: str) -> bool:
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
        return True
    except Exception as e:
        logger.error(f"Error updating thread flag: {str(e)}")
        return False

def update_conversation_ev(conversation_id: str, message_id: str, ev_score: int, account_id: str, session_id: str) -> bool:
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
        return True
    except Exception as e:
        logger.error(f"Error updating conversation EV score: {str(e)}")
        return False

def calculate_ev_for_conversation(conversation_id: str, account_id: str, session_id: str) -> Tuple[int, Dict[str, int]]:
    """
    Calculate the EV score for a conversation.
    
    Args:
        conversation_id (str): The conversation ID
        account_id (str): The account ID for authorization
        session_id (str): The session ID for authorization
    
    Returns:
        Tuple[int, Dict[str, int]]: (ev_score, token_usage)
    """
    # Get the email chain
    chain = get_email_chain(conversation_id, account_id, session_id)
    if not chain:
        logger.error(f"Failed to get email chain for conversation {conversation_id}")
        return -1, {'input_tokens': 0, 'output_tokens': 0}
    
    # Calculate EV score
    ev_score, token_usage = calc_ev(chain, account_id, conversation_id, session_id)
    if ev_score < 0:
        logger.error(f"Failed to calculate EV score for conversation {conversation_id}")
        return ev_score, token_usage
    
    # Get flag decision
    flag_decision = invoke_flag_llm(chain, account_id, conversation_id, session_id)
    if flag_decision < 0:
        logger.error(f"Failed to get flag decision for conversation {conversation_id}")
        return flag_decision, token_usage
    
    # Update thread EV with flag decision
    if not update_thread_ev(conversation_id, ev_score, flag_decision, account_id, session_id):
        logger.error(f"Failed to update thread EV for conversation {conversation_id}")
        return -1, token_usage
    
    # Update conversation EV
    if not update_conversation_ev(conversation_id, ev_score, account_id, session_id):
        logger.error(f"Failed to update conversation EV for conversation {conversation_id}")
        return -1, token_usage
    
    # Store AI invocation records
    if not store_ai_invocation(
        associated_account=account_id,
        input_tokens=token_usage['input_tokens'],
        output_tokens=token_usage['output_tokens'],
        llm_email_type='ev_calculation',
        model_name='meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
        conversation_id=conversation_id,
        session_id=session_id
    ):
        logger.error(f"Failed to store EV calculation invocation record for conversation {conversation_id}")
    
    if not store_ai_invocation(
        associated_account=account_id,
        input_tokens=token_usage['input_tokens'],
        output_tokens=token_usage['output_tokens'],
        llm_email_type='flag',
        model_name='meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
        conversation_id=conversation_id,
        session_id=session_id
    ):
        logger.error(f"Failed to store flag invocation record for conversation {conversation_id}")
    
    logger.info(f"Calculated EV score {ev_score} and flag decision {flag_decision} for conversation {conversation_id}")
    
    return ev_score, token_usage

def check_aws_rate_limit(account_id, session_id):
    """
    Checks the AWS rate limit for a given account by invoking the rate-limit-aws lambda.
    """
    payload = {'client_id': account_id, 'session': session_id}
    # This invocation is already designed to raise LambdaError on failure
    invoke_lambda('RateLimitAWS', payload)

def lambda_handler(event, context):
    try:
        parsed_event = parse_event(event)
        
        conversation_id = parsed_event.get('conversation_id')
        account_id = parsed_event.get('account_id') or parsed_event.get('account') or parsed_event.get('client_id')
        session_id = parsed_event.get('session_id') or parsed_event.get('session')
        
        if not all([conversation_id, account_id, session_id]):
            raise LambdaError(400, "Missing required fields: conversation_id, account_id, or session_id.")
        
        if session_id != AUTH_BP:
            authorize(account_id, session_id)
            check_aws_rate_limit(account_id, session_id)
        
        ev_score, token_usage = calculate_ev_for_conversation(
            conversation_id=conversation_id,
            account_id=account_id,
            session_id=session_id
        )
        
        response_body = {
            'ev_score': ev_score,
            'conversation_id': conversation_id,
            'status': 'success',
            'token_usage': token_usage
        }
        return create_response(200, response_body)
        
    except LambdaError as e:
        logger.error(f"Error processing EV calculation: {e.message}")
        return create_response(e.status_code, {"status": "error", "error": e.message})
    except Exception as e:
        logger.error(f"An unexpected error occurred in lambda_handler: {e}")
        return create_response(500, {"status": "error", "error": "An internal server error occurred."}) 