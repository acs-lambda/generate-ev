import json
import boto3
import logging
from typing import Dict, Any, List

from ev_calculator import calc_ev, parse_messages
from db import get_email_chain, get_account_email

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodb = boto3.resource('dynamodb')

def update_thread_ev(conversation_id: str, ev_score: int) -> None:
    """
    Updates the thread with the new EV score and flag status.
    """
    try:
        threads_table = dynamodb.Table('Threads')
        threads_table.update_item(
            Key={
                'conversation_id': conversation_id
            },
            UpdateExpression='SET #flag = :flag',
            ExpressionAttributeNames={
                '#flag': 'flag'
            },
            ExpressionAttributeValues={
                ':flag': ev_score >= 80
            }
        )
        logger.info(f"Updated thread flag for conversation {conversation_id} with EV score {ev_score}")
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
        ev = calc_ev(parse_messages(realtor_email, chain))
        logger.info(f"Calculated EV score: {ev} for conversation {conversation_id}")

        # Update both thread and conversation
        update_thread_ev(conversation_id, ev)
        update_conversation_ev(conversation_id, message_id, ev)

        return {
            'ev_score': ev,
            'conversation_id': conversation_id,
            'message_id': message_id,
            'status': 'success'
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
    Expected event format:
    {
        'conversation_id': str,
        'message_id': str,
        'account_id': str
    }
    """
    try:
        # Validate input
        required_fields = ['conversation_id', 'message_id', 'account_id']
        for field in required_fields:
            if field not in event:
                raise ValueError(f"Missing required field: {field}")

        # Calculate EV and update database
        result = calculate_ev_for_conversation(
            event['conversation_id'],
            event['message_id'],
            event['account_id']
        )

        return {
            'statusCode': 200 if result['status'] == 'success' else 500,
            'body': json.dumps(result)
        }
    except Exception as e:
        logger.error(f"Error in lambda handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'status': 'error',
                'error': str(e)
            })
        } 