import json
import urllib3
import logging
from typing import Dict, Any, List, Tuple
from config import TAI_KEY
from db import check_and_update_ai_rate_limit
from utils import store_ai_invocation

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize urllib3 pool manager
http = urllib3.PoolManager()

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

def invoke_flag_llm(conversation_chain: List[Dict[str, Any]], account_id: str, conversation_id: str, session_id: str) -> Tuple[bool, Dict[str, int]]:
    """
    Invoke the flag LLM to determine if a conversation should be flagged.
    
    Args:
        conversation_chain: List of conversation messages
        account_id: The account ID
        conversation_id: The conversation ID
        session_id: The session ID for authorization
    
    Returns:
        Tuple[bool, Dict[str, int]]: (should_flag, token_usage)
    """
    try:
        # Format conversation for LLM
        formatted_chain = format_conversation_for_llm(conversation_chain)
        
        # Prepare system prompt
        system_prompt = {
            "role": "system",
            "content": (
                "You are an assistant that determines if a conversation between a realtor and a buyer "
                "should be flagged for review. Return exactly one word: 'flag' if the conversation "
                "should be flagged, or 'ok' if it should not be flagged.\n\n"
                "Flag the conversation if:\n"
                "1. The buyer expresses serious interest in buying\n"
                "2. The buyer asks about viewing the property\n"
                "3. The buyer discusses financing or pre-approval\n"
                "4. The buyer asks about making an offer\n"
                "5. The conversation shows strong buying signals\n\n"
                "Do not flag if:\n"
                "1. The buyer is just asking general questions\n"
                "2. The buyer is not showing clear buying interest\n"
                "3. The conversation is just about scheduling or logistics\n"
                "4. The buyer is not ready to move forward\n"
                "5. The conversation is ambiguous or unclear\n\n"
                "Return ONLY 'flag' or 'ok', nothing else."
            )
        }
        
        # Prepare user message
        user_message = {
            "role": "user",
            "content": formatted_chain
        }
        
        # Make API call
        url = "https://api.together.xyz/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {TAI_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            "messages": [system_prompt, user_message],
            "max_tokens": 5,  # We only need a very short response
            "temperature": 0.0,  # Zero temperature for deterministic responses
            "top_p": 0.1,  # Low top_p for focused sampling
            "frequency_penalty": 0.0,  # No frequency penalty needed
            "presence_penalty": 0.0,  # No presence penalty needed
            "stop": ["\n", ".", " ", ","]  # Stop at any punctuation or space
        }

        # Make the API call
        encoded_data = json.dumps(payload).encode('utf-8')
        response = http.request(
            'POST',
            url,
            body=encoded_data,
            headers=headers
        )

        if response.status != 200:
            logger.error(f"API call failed with status {response.status}: {response.data.decode('utf-8')}")
            return False, {'input_tokens': 0, 'output_tokens': 0}

        response_data = json.loads(response.data.decode('utf-8'))
        if "choices" not in response_data:
            logger.error(f"Invalid API response format: {response_data}")
            return False, {'input_tokens': 0, 'output_tokens': 0}

        # Get token usage from response
        token_usage = {
            'input_tokens': response_data.get('usage', {}).get('prompt_tokens', 0),
            'output_tokens': response_data.get('usage', {}).get('completion_tokens', 0)
        }

        # Get the response text and clean it
        response_text = response_data["choices"][0]["message"]["content"].strip().lower()
        logger.info(f"Flag LLM response: {response_text}")

        # Store the invocation record
        store_ai_invocation(
            associated_account=account_id,
            input_tokens=token_usage['input_tokens'],
            output_tokens=token_usage['output_tokens'],
            llm_email_type='flag',
            model_name='meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
            conversation_id=conversation_id,
            session_id=session_id
        )

        # Return True if the response is "flag", False otherwise
        return response_text == "flag", token_usage

    except Exception as e:
        logger.error(f"Error invoking flag LLM: {str(e)}")
        return False, {'input_tokens': 0, 'output_tokens': 0} 