# ev_calculator.py
import urllib3
import json
import logging
from config import TAI_KEY

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize urllib3 pool manager
http = urllib3.PoolManager()

def calc_ev(messages: list) -> int:
    logger.info(f"Calculating EV for {len(messages)} messages")
    url = "https://api.together.xyz/v1/chat/completions"

    headers = {
            "Authorization": f"Bearer {TAI_KEY}",
            "Content-Type": "application/json"
    }
    payload_messages = [{
        "role": "system",
        "content": '''You are an assistant that must ONLY respond with a single integer number from 0 to 100, with NO extra text, explanation, or formatting. Do not say anything else, do not explain, do not apologize, do not repeat the question. Just output a number from 0 to 100. This number will indicate how interested a prospective buyer of a property is, based on a series of emails. Each email will be sent as a separate message by the user, and the first word will indicate if the email was sent by a buyer or a realtor.\nIf the first word is BUYER: the email that follows is sent by a buyer.\nIf the first word is REALTOR: the email that follows is sent by a realtor.\nRegardless of the input, reply ONLY with a number from 0 to 100.'''
    }]
    payload_messages.extend(messages)
    payload = {
        "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "messages": payload_messages,
        "max_tokens": 5,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1,
        "stop": ["<|im_end|>", "<|endoftext|>"],
        "stream": False
    }

    try:
        logger.info("Sending request to Together AI API")
        encoded_data = json.dumps(payload).encode('utf-8')
        response = http.request(
            'POST',
            url,
            body=encoded_data,
            headers=headers
        )
        
        if response.status != 200:
            logger.error(f"API call failed with status {response.status}: {response.data.decode('utf-8')}")
            return -3

        response_data = json.loads(response.data.decode('utf-8'))
        if "choices" not in response_data:
            logger.error(f"Invalid API response format: {response_data}")
            return -3

        logger.info("EV Response: " + response_data["choices"][0]["message"]["content"])
        try:
            ev = int(response_data["choices"][0]["message"]["content"])
            return ev
        except ValueError:
            logger.error('The AI did not return a valid number')
            return -2
    except Exception as e:
        logger.error(f"Error in calc_ev: {str(e)}")
        return -3


def parse_messages(realtor_email: str, emails: list) -> list:
    logger.info(f"Parsing messages for EV calculation. Chain length: {len(emails)}")
    for i, email in enumerate(emails):
        logger.info(f"Message {i+1} - Sender: {email['sender']}, Body length: {len(email['body'])}")
    
    messages = []
    for email in emails:
        if email['sender'].split('@')[1] == 'lgw.automatedconsultancy.com':
            continue
        messages.append({'role': 'user', 'content':  
                                        ('REALTOR: ' if email['sender'] == realtor_email else 'BUYER: ')
                                        + email['body']})
    return messages 