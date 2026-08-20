import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/ft-reward-retraining/outputs/sft_grpo/grpo/checkpoint-800"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

model.eval()

print("Model loaded.")
print("Device:", next(model.parameters()).device)


def generate(prompt, max_new_tokens=1024):

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[-1]

    generated_tokens = outputs[0][input_length:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response


if __name__ == "__main__":

    prompt = """
You are a precise function-calling assistant. Given a user query and a list of available tools, respond with a JSON array containing all the tool calls needed to answer the query.

Each element of the array must be a JSON object with exactly two keys:
  "name"      : the tool name (string)
  "arguments" : a dict of argument name → value

Wrap your entire response in a single ```json ... ``` code block.
Do not include any explanation outside the code block.

User query:
Fetch news articles about cricket from India Today and health-related news in the US

Available tools:
[
  {
    "name": "tag_search",
    "description": "Fetches news articles based on the provided tag from the India Today API.",
    "parameters": {
      "tag": {
        "description": "The tag or keyword to search for in the news articles.",
        "type": "str",
        "default": "India"
      }
    }
  },
  {
    "name": "health",
    "description": "Fetches health-related news from Google News using the provided language region and API key.",
    "parameters": {
      "lr": {
        "description": "Language region for the news, e.g., 'en-US'.",
        "type": "str",
        "default": "en-US"
      }
    }
  }
]
"""

    response = generate(prompt)

    print("\n" + "=" * 80)
    print("MODEL OUTPUT")
    print("=" * 80)
    print(response)