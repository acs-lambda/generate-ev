import time
import boto3
from config import logger
from utils import LambdaError
from ev_calculator import calc_ev
from db import get_email_chain, update_thread_attributes, store_ai_invocation
from flag_llm import invoke_flag_llm

dynamodb = boto3.resource('dynamodb')

def store_ai_invocation(associated_account, input_tokens, output_tokens, llm_email_type, model_name, conversation_id, session_id):
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
    except Exception as e:
        logger.error(f"Error storing AI invocation for {conversation_id}: {e}")
        # Non-critical, so just log and continue

def update_thread_ev(conversation_id, ev_score, should_flag):
    """Updates the thread with the new EV score and flag status."""
    try:
        threads_table = dynamodb.Table('Threads')
        threads_table.update_item(
            Key={'conversation_id': conversation_id},
            UpdateExpression='SET #flag = :flag, ev_score = :ev',
            ExpressionAttributeNames={'#flag': 'flag'},
            ExpressionAttributeValues={':flag': should_flag, ':ev': ev_score}
        )
    except Exception as e:
        raise LambdaError(500, f"Error updating thread EV for {conversation_id}: {e}")

def update_conversation_ev(conversation_id, message_id, ev_score):
    """Updates the conversation with the EV score."""
    try:
        conversations_table = dynamodb.Table('Conversations')
        conversations_table.update_item(
            Key={'conversation_id': conversation_id, 'response_id': message_id},
            UpdateExpression='SET ev_score = :ev',
            ExpressionAttributeValues={':ev': ev_score}
        )
    except Exception as e:
        raise LambdaError(500, f"Error updating conversation EV for {conversation_id}: {e}")

def calculate_ev_for_conversation(conversation_id, account_id, session_id):
    """Calculate the EV score for a conversation."""
    chain, realtor_email = get_email_chain(conversation_id, account_id, session_id)
    if not chain:
        raise LambdaError(404, f"Failed to get email chain for conversation {conversation_id}")
    
    ev_score, token_usage_ev = calc_ev(chain, account_id, conversation_id, session_id)
    if ev_score < 0:
        raise LambdaError(500, f"Failed to calculate EV score for conversation {conversation_id}")
    
    should_flag, token_usage_flag = invoke_flag_llm(chain, account_id, conversation_id, session_id)
    
    thread_attrs = {'ev_score': ev_score, 'flag': should_flag}
    update_thread_attributes(conversation_id, thread_attrs, account_id, session_id)
    
    total_input_tokens = token_usage_ev.get('input_tokens', 0) + token_usage_flag.get('input_tokens', 0)
    total_output_tokens = token_usage_ev.get('output_tokens', 0) + token_usage_flag.get('output_tokens', 0)

    # Store EV calculation invocation
    store_ai_invocation(
        account_id=account_id, conversation_id=conversation_id,
        llm_email_type='ev_calculation', model_name='meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
        input_tokens=token_usage_ev.get('input_tokens', 0),
        output_tokens=token_usage_ev.get('output_tokens', 0),
        session_id=session_id
    )
    
    # Store flag invocation
    store_ai_invocation(
        account_id=account_id, conversation_id=conversation_id,
        llm_email_type='flag', model_name='meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
        input_tokens=token_usage_flag.get('input_tokens', 0),
        output_tokens=token_usage_flag.get('output_tokens', 0),
        session_id=session_id
    )
    
    return ev_score, {"input_tokens": total_input_tokens, "output_tokens": total_output_tokens}
