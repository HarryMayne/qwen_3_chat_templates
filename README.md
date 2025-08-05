# Qwen 3 Chat Templates

Modified chat templates for Qwen 3 8B. Useful for multi-turn reinforcement learning.

## Templates

### `all_assistant.jinja`
Returns a mask with **all assistant tokens**. The generation mask is applied to all assistant responses in the conversation.
Credit to [Alexander Kovrigin](https://huggingface.co/Qwen/Qwen3-8B/discussions/14) for writing this.

### `final_assistant.jinja` 
Returns a mask with **only the final assistant tokens**. The generation mask is applied only to the last assistant response in the conversation.

## Usage

See `usage_example.ipynb` for a complete example. Basic usage:

```python
import transformers

# Load tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# Load custom template
with open('all_assistant.jinja', 'r') as f:
    tokenizer.chat_template = f.read()

# Apply template with assistant token masking
conversation = [
    {"role": "user", "content": "Hello assistant"},
    {"role": "assistant", "content": "Hello user"},
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "I'm good"},
]

tokenized_output = tokenizer.apply_chat_template(
    conversation,
    return_assistant_tokens_mask=True,
    return_dict=True,
)
```

These templates are useful for multi-turn RL training where you need different masking strategies for computing losses on assistant responses.
