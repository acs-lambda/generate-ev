# ev_calculator.py
import urllib3
import json
import logging
import re
from config import TAI_KEY

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize urllib3 pool manager
http = urllib3.PoolManager()

def calc_ev(messages: list) -> int:
    """
    Sends a chain of messages to the LLM to get a single integer (0–100)
    indicating the percent chance the lead will convert. The system prompt
    has been updated so the AI outputs a granular integer (e.g., 14, 27, 63)
    rather than rounding to multiples of 5 or 10. We also retry once if the model
    fails to return a valid integer.
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
            "2. Do NOT default to multiples of 10 or 5 unless it genuinely reflects your judgment. "
            "If you feel there is, say, roughly a 14% chance, output 14; if it looks like a 27% chance, output 27.\n"
            "3. Evaluate all of the following factors:\n"
            "   • Buyer’s urgency (e.g., “Can I tour tomorrow?” vs. “Just browsing”).\n"
            "   • Specific questions about financing, timelines, or next steps.\n"
            "   • Positive signals (e.g., “Looks perfect—I want to move forward”).\n"
            "   • Any hesitations or vague interest (“Maybe later,” “Just looking around”).\n"
            "   • The total number of back‐and‐forth messages and overall engagement level.\n"
            "   • Mentions of being pre‐approved, ready to tour, or ready to make an offer.\n"
            "4. If you have very little context, still make your best speculative guess "
            "(e.g., even with just “Thanks, I’ll pass for now,” you might output 5 or 8).\n"
            "5. Under no circumstances output anything other than that single integer. "
            "No labels, no notes, no newlines—nothing but a numeric digit string (e.g., “14”).\n\n"
            "FEW‐SHOT EXAMPLES (for illustration only; do not include anything but the integer after you see the real emails):\n\n"
            "Example 1:\n"
            "Buyer: “Thanks, I’ll think about it. Just browsing for now.”\n"
            "Realtor: “Sure, let me know if you have any questions.”\n"
            "→ 8\n\n"
            "Example 2:\n"
            "Buyer: “I’m pre‐approved and would like to tour 123 Main St this Saturday.”\n"
            "Realtor: “Booked for Saturday at 2 p.m.; let me know if you need anything else.”\n"
            "→ 63\n\n"
            "Example 3:\n"
            "Buyer: “Can you send comps and more details on HOA fees? I’m hoping to submit an offer next week.”\n"
            "Realtor: “Absolutely—comps are attached. I’ll schedule a call for tomorrow.”\n"
            "→ 81\n\n"
            "Now evaluate the following messages in chronological order and return exactly one integer "
            "(0–100) that best estimates the percent chance of conversion.\n"
        )
    }

    payload_messages = [system_prompt] + messages

    payload = {
        "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "messages": payload_messages,
        "max_tokens": 3,
        "temperature": 0.0,
        "top_p": 0.0,
        "top_k": 1,
        "repetition_penalty": 1,
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
                return -3

            response_data = json.loads(response.data.decode('utf-8'))
            if "choices" not in response_data:
                logger.error(f"Invalid API response format: {response_data}")
                return -3

            raw_content = response_data["choices"][0]["message"]["content"].strip()
            logger.info("EV raw response: " + raw_content)

            # Check if the content is exactly a 1–3 digit integer (0–100)
            if re.fullmatch(r"\d{1,3}", raw_content):
                ev = int(raw_content)
                return max(0, min(100, ev))
            else:
                logger.warning(f"Attempt {attempt+1}: AI returned non‐integer \"{raw_content}\". Retrying...")

        # If we exit the loop without returning, it never gave a valid integer.
        logger.error("The AI did not return a valid integer after retries")
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
