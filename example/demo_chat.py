import json
from dotenv import load_dotenv
load_dotenv()
from config.encapsulation.llm.chat.openai import OpenAIChatConfig

JSON_CONFIG = """
{
    "type": "openai_chat",
    "model_name": "gpt-4o-mini",
    "temperature": 0.0,
    "max_tokens": 200
}
"""


def main(query: str) -> None:
    config_data = json.loads(JSON_CONFIG)
    llm_config = OpenAIChatConfig.model_validate(config_data)
    llm = llm_config.build()

    messages = [
        {"role": "system", "content": "You are a helpful Chinese assistant."},
        {"role": "user", "content": query}
    ]

    response = llm.chat(messages)

    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main("你是谁？")

