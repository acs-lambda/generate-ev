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
    """
    Sends a chain of messages to the LLM to get a single integer (0–100)
    indicating the percent chance the lead will convert. The system prompt
    has been optimized so the AI evaluates urgency, intent, tone, and next-step
    indicators in the conversation—just like a human would—then returns a single
    integer with no extra text.
    Even with minimal context, the AI must make a rough speculative guess.
    """
    logger.info(f"Calculating EV for {len(messages)} messages")
    url = "https://api.together.xyz/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {TAI_KEY}",
        "Content-Type": "application/json"
    }

    system_prompt = {
        "role": "system",
        "content": (
            "You are an assistant that assesses how likely a prospective buyer is to convert based on "
            "a series of emails exchanged with a realtor. Your job is to produce a single integer "
            "from 0 to 100 (representing 0%–100% chance of conversion) with no additional text or formatting.  \n"
            "When you evaluate, consider:  \n"
            "• Buyer urgency (e.g., asking to schedule a viewing immediately).  \n"
            "• Specific questions about financing, timelines, or next steps.  \n"
            "• Positive signals (e.g., “I’d like to move forward,” “This looks perfect”).  \n"
            "• Hesitations or vague interest (e.g., “Maybe later,” “Just browsing”).  \n"
            "• The number of back-and-forths and overall engagement level.  \n"
            "• Any explicit mention of being pre-approved, ready to tour, or ready to make an offer.  \n"
            "Even if you have very little information, make a reasonable speculative guess based on whatever context is available.  \n"
            "Always return a single integer between 0 and 100 inclusive—no explanations or extra text."
        )
    }

    payload_messages = [system_prompt] + messages

    payload = {
        "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "messages": payload_messages,
        "max_tokens": 5,
        "temperature": 0.2,
        "top_p": 0.9,
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

        raw_content = response_data["choices"][0]["message"]["content"].strip()
        logger.info("EV raw response: " + raw_content)

        try:
            ev = int(raw_content)
            return max(0, min(100, ev))
        except ValueError:
            logger.error("The AI did not return a valid integer")
            return -2

    except Exception as e:
        logger.error(f"Error in calc_ev: {str(e)}")
        return -3


def parse_messages(realtor_email: str, emails: list) -> list:
    """
    Transforms a list of email dicts into the format expected by calc_ev:
      - Tags each email as REALTOR or BUYER.
      - Returns a list of {"role": "user", "content": "..."} items.
    """
    logger.info(f"Parsing messages for EV calculation. Chain length: {len(emails)}")
    for i, email in enumerate(emails):
        logger.info(f"Message {i+1} - Sender: {email['sender']}, Body length: {len(email['body'])}")

    messages = []
    for email in emails:
        tag = "REALTOR: " if email['sender'] == realtor_email else "BUYER: "
        content = tag + email['body']
        messages.append({'role': 'user', 'content': content})

    return messages
