"""
prompt_template.py
Defines the conversational template for CohereLabs/tiny-aya-earth and helper utilities
to format African-UltraChat conversational data into the prompt format expected by the model.
"""

# Cohere / TinyAya special turn tokens
START_TURN = "<|START_OF_TURN_TOKEN|>"
END_TURN = "<|END_OF_TURN_TOKEN|>"
USER_TOKEN = "<|USER_TOKEN|>"
CHATBOT_TOKEN = "<|CHATBOT_TOKEN|>"
SYSTEM_TOKEN = "<|SYSTEM_TOKEN|>"

def format_single_turn(instruction: str, response: str, system_prompt: str = None) -> str:
    """
    Format a single instruction-response pair into the TinyAya turn format.
    """
    formatted = ""
    if system_prompt:
        formatted += f"{START_TURN}{SYSTEM_TOKEN}{system_prompt.strip()}{END_TURN}"
    formatted += f"{START_TURN}{USER_TOKEN}{instruction.strip()}{END_TURN}"
    formatted += f"{START_TURN}{CHATBOT_TOKEN}{response.strip()}{END_TURN}"
    return formatted

def format_multiturn(messages: list[dict[str, str]]) -> str:
    """
    Format a list of message dictionaries:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    formatted = ""
    for msg in messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "").strip()
        if role in ["system"]:
            formatted += f"{START_TURN}{SYSTEM_TOKEN}{content}{END_TURN}"
        elif role in ["user", "human"]:
            formatted += f"{START_TURN}{USER_TOKEN}{content}{END_TURN}"
        elif role in ["assistant", "bot", "chatbot"]:
            formatted += f"{START_TURN}{CHATBOT_TOKEN}{content}{END_TURN}"
    return formatted

def format_generation_prompt(instruction: str, system_prompt: str = None) -> str:
    """
    Format an inference prompt up to the start of the assistant's turn.
    """
    prompt = ""
    if system_prompt:
        prompt += f"{START_TURN}{SYSTEM_TOKEN}{system_prompt.strip()}{END_TURN}"
    prompt += f"{START_TURN}{USER_TOKEN}{instruction.strip()}{END_TURN}"
    prompt += f"{START_TURN}{CHATBOT_TOKEN}"
    return prompt
