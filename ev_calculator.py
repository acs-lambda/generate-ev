# ev_calculator.py
import urllib3
import json
import logging
import re
from typing import Dict, Any, Tuple
from config import TAI_KEY
from db import store_ai_invocation

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize urllib3 pool manager
http = urllib3.PoolManager()

def calc_ev(messages: list, account_id: str, conversation_id: str) -> Tuple[int, Dict[str, int]]:
    """
    Sends a chain of messages to the LLM to get a single integer (0–100)
    indicating the percent chance the lead will convert. The system prompt
    has been updated so the AI outputs a granular integer (e.g., 14, 27, 63)
    rather than rounding to multiples of 5 or 10. We also retry once if the model
    fails to return a valid integer.
    
    Returns a tuple of (ev_score, token_usage).
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
            "You are an assistant that assesses how likely a prospective buyer is to convert—"
            "expressed as an integer percentage from 0 to 100—based solely on the email thread "
            "between a realtor and a buyer.\n\n"
            "RULES (follow exactly):\n"
            "1. Always return exactly one integer between 0 and 100, with no extra text, no explanations, "
            "and no punctuation.\n"
            "2. Provide highly granular scores that reflect subtle differences in buyer behavior. "
            "For example:\n"
            "   • A buyer asking about viewing next week might be 37% vs 35% if they also mentioned "
            "   their pre-approval status\n"
            "   • A buyer expressing interest but with some hesitation might be 42% vs 40% if they "
            "   asked specific questions about the property\n"
            "3. NEVER default to round numbers (like 20, 25, 30) unless the signals are truly ambiguous. "
            "Use the full range of numbers to capture nuanced differences.\n"
            "4. Evaluate all of the following factors with precise weighting:\n"
            "   • Buyer's urgency (e.g., 'Can I tour tomorrow?' = +15-20%, 'Maybe next month' = +5-10%)\n"
            "   • Specific questions about financing (+8-12%), timelines (+5-10%), or next steps (+7-15%)\n"
            "   • Positive signals (e.g., 'Looks perfect' = +20-25%, 'Interesting' = +5-8%)\n"
            "   • Hesitations or vague interest (-10-15% for each major hesitation)\n"
            "   • Message frequency and engagement (each back-and-forth = +3-7%)\n"
            "   • Pre-approval status (+12-18%), tour readiness (+15-20%), offer readiness (+25-30%)\n"
            "5. If you have very little context, still make your best speculative guess "
            "but use precise numbers (e.g., 'Thanks, I'll pass for now' = 7, not 5 or 10).\n"
            "6. Under no circumstances output anything other than that single integer. "
            "No labels, no notes, no newlines—nothing but a numeric digit string.\n\n"
            "FEW‐SHOT EXAMPLES (for illustration only; do not include anything but the integer after you see the real emails):\n\n"
            "Example 1:\n"
            "Buyer: 'Thanks, I'll think about it. Just browsing for now.'\n"
            "Realtor: 'Sure, let me know if you have any questions.'\n"
            "→ 7\n\n"
            "Example 2:\n"
            "Buyer: 'I'm pre‐approved and would like to tour 123 Main St this Saturday.'\n"
            "Realtor: 'Booked for Saturday at 2 p.m.; let me know if you need anything else.'\n"
            "→ 67\n\n"
            "Example 3:\n"
            "Buyer: 'Can you send comps and more details on HOA fees? I'm hoping to submit an offer next week.'\n"
            "Realtor: 'Absolutely—comps are attached. I'll schedule a call for tomorrow.'\n"
            "→ 83\n\n"
            "Example 4:\n"
            "Buyer: 'The house looks nice but I'm still considering a few other options.'\n"
            "Realtor: 'I understand. Would you like to see some similar properties?'\n"
            "→ 31\n\n"
            "Now evaluate the following messages in chronological order and return exactly one integer "
            "(0–100) that best estimates the percent chance of conversion.\n"
        )
    }

    payload_messages = [system_prompt] + messages

    payload = {
        "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "messages": payload_messages,
        "max_tokens": 3,
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.2,
        "stop": ["<|im_end|>", "<|endoftext|>"],
        "stream": False
    }

    try:
        # We'll allow up to 2 attempts if the output isn't a clean integer.
        for attempt in range(2):
            logger.info(f"Sending request to Together AI API (attempt {attempt+1})")
            encoded_data = json.dumps(payload).encode('utf-8')
            response = http.request(
                'POST',
                url,
                body=encoded_data,
                headers=headers
            )

            if response.status != 200:
                logger.error(f"API call failed with status {response.status}: {response.data.decode('utf-8')}")
                return -3, {'input_tokens': 0, 'output_tokens': 0}

            response_data = json.loads(response.data.decode('utf-8'))
            if "choices" not in response_data:
                logger.error(f"Invalid API response format: {response_data}")
                return -3, {'input_tokens': 0, 'output_tokens': 0}

            # Get token usage from response
            token_usage = {
                'input_tokens': response_data.get('usage', {}).get('prompt_tokens', 0),
                'output_tokens': response_data.get('usage', {}).get('completion_tokens', 0)
            }

            raw_content = response_data["choices"][0]["message"]["content"].strip()
            logger.info("EV raw response: " + raw_content)

            # Check if the content is exactly a 1–3 digit integer (0–100)
            if re.fullmatch(r"\d{1,3}", raw_content):
                ev = int(raw_content)
                ev_score = max(0, min(100, ev))
                
                # Store the invocation record
                store_ai_invocation(
                    associated_account=account_id,
                    input_tokens=token_usage['input_tokens'],
                    output_tokens=token_usage['output_tokens'],
                    llm_email_type='ev_calculation',
                    model_name='meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
                    conversation_id=conversation_id
                )
                
                return ev_score, token_usage
            else:
                logger.warning(f"Attempt {attempt+1}: AI returned non‐integer \"{raw_content}\". Retrying...")

        # If we exit the loop without returning, it never gave a valid integer.
        logger.error("The AI did not return a valid integer after retries")
        return -2, {'input_tokens': 0, 'output_tokens': 0}

    except Exception as e:
        logger.error(f"Error in calc_ev: {str(e)}")
        return -3, {'input_tokens': 0, 'output_tokens': 0}


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
