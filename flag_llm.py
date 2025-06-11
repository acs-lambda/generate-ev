import json
import boto3
import logging
from typing import Dict, Any, List, Tuple
from db import store_ai_invocation

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
lambda_client = boto3.client('lambda')

def format_conversation_for_llm(chain: List[Dict[str, Any]]) -> str:
    """
    Format the conversation chain for the LLM prompt.
    """
    formatted_chain = []
    for msg in chain:
        formatted_msg = f"From: {msg['sender']}\n"
        formatted_msg += f"Subject: {msg['subject']}\n"
        formatted_msg += f"Body: {msg['body']}\n"
        formatted_msg += "---\n"
        formatted_chain.append(formatted_msg)
    
    return "\n".join(formatted_chain)

def invoke_flag_llm(conversation_chain: List[Dict[str, Any]], account_id: str, conversation_id: str) -> Tuple[bool, Dict[str, int]]:
    """
    Invokes the Together AI LLM to determine if the email should be flagged.
    Returns a tuple of (flag_decision, token_usage).
    
    Args:
        conversation_chain: List of email messages
        account_id: The user's account ID
        conversation_id: The conversation ID
    
    Returns:
        Tuple[bool, Dict[str, int]]: (flag decision, token usage info)
    """
    try:
        # Format the conversation for the LLM
        formatted_conversation = format_conversation_for_llm(conversation_chain)
        
        # Create the prompt
        prompt = f"""You are a specialized real estate deal progression analyzer. Your task is to identify when a conversation needs human intervention to close a deal or handle tasks that AI cannot perform.

IMPORTANT: You can ONLY respond with either "flag" or "false". No other text or explanation.

Flag the email if ANY of these conditions indicate the conversation is ready for human intervention:
1. The client shows clear buying/selling intent and is ready to move forward
2. There are specific requests for property viewings or showings
3. The client wants to discuss or negotiate specific terms of a deal
4. There are requests for in-person meetings or calls
5. The client is ready to make an offer or discuss pricing
6. There are questions about contracts, legal documents, or signing processes
7. The client needs help with mortgage pre-approval or financing options
8. There are specific scheduling requests for property tours or inspections
9. The client wants to discuss property-specific details that require human expertise
10. There are requests for market analysis or property comparisons

Do NOT flag if:
1. The conversation is still in early stages of general inquiry
2. The client is just gathering initial information
3. The email is purely informational or confirmatory
4. The conversation can be handled by AI responses
5. There are no specific actions or decisions needed from a human

Remember: The goal is to identify when a human realtor needs to step in to close a deal or handle tasks that AI cannot perform.

Here is the email conversation:
{formatted_conversation}

Based on the conversation above, should this email be flagged for human realtor intervention? Respond with ONLY "flag" or "false":"""

        # Invoke Together AI Lambda
        response = lambda_client.invoke(
            FunctionName='together-ai',
            InvocationType='RequestResponse',
            Payload=json.dumps({
                'prompt': prompt,
                'max_tokens': 5,  # We only need a very short response
                'temperature': 0.0,  # Zero temperature for deterministic responses
                'top_p': 0.1,  # Low top_p for focused sampling
                'frequency_penalty': 0.0,  # No frequency penalty needed
                'presence_penalty': 0.0,  # No presence penalty needed
                'stop': ['\n', '.', ' ', ',']  # Stop at any punctuation or space
            })
        )
        
        # Parse the response
        response_payload = json.loads(response['Payload'].read())
        if response_payload['statusCode'] != 200:
            logger.error(f"Failed to get LLM response: {response_payload}")
            return False, {'input_tokens': 0, 'output_tokens': 0}
            
        result = json.loads(response_payload['body'])
        if result['status'] != 'success':
            logger.error(f"LLM response generation failed: {result}")
            return False, {'input_tokens': 0, 'output_tokens': 0}
            
        # Get token usage from response
        token_usage = {
            'input_tokens': result.get('usage', {}).get('prompt_tokens', 0),
            'output_tokens': result.get('usage', {}).get('completion_tokens', 0)
        }
        
        # Store the invocation record
        store_ai_invocation(
            associated_account=account_id,
            input_tokens=token_usage['input_tokens'],
            output_tokens=token_usage['output_tokens'],
            llm_email_type='flag',
            model_name='meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
            conversation_id=conversation_id
        )
            
        # Get the response text and clean it
        response_text = result['response'].strip().lower()
        logger.info(f"Flag LLM response: {response_text}")
        
        # Return True if the response is "flag", False otherwise
        return response_text == "flag", token_usage
        
    except Exception as e:
        logger.error(f"Error invoking flag LLM: {str(e)}")
        return False, {'input_tokens': 0, 'output_tokens': 0} 