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

# BEFORE
# {'input_ids': [151644, 872, 198, 9707, 17847, 151645, 198, 151644, 77091, 198, 9707, 1196, 151645, 198, 151644, 872, 198, 4340, 525, 498, 30, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271, 40, 2776, 1661, 151645, 198], 
#  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#  'assistant_masks': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# }

# UNDER ALL ASSISTANT
# {'input_ids': [151644, 872, 198, 9707, 17847, 151645, 198, 151644, 77091, 198, 9707, 1196, 151645, 198, 151644, 872, 198, 4340, 525, 498, 30, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271, 40, 2776, 1661, 151645, 198], 
#  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#  'assistant_masks': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# }

# UNDER FINAL ASSISTANT
# {'input_ids': [151644, 872, 198, 9707, 17847, 151645, 198, 151644, 77091, 198, 9707, 1196, 151645, 198, 151644, 872, 198, 4340, 525, 498, 30, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271, 40, 2776, 1661, 151645, 198], 
#  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#  'assistant_masks': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# }

```
<img width="195" height="180" alt="original" src="https://github.com/user-attachments/assets/d8655a4a-931b-4eb5-aecf-60b47b4da708" />
<img width="191" height="179" alt="all_assistant" src="https://github.com/user-attachments/assets/f63d0c6d-8cb8-48b3-b8d4-bb70f02134a8" />
<img width="195" height="179" alt="final_assistant" src="https://github.com/user-attachments/assets/c1842f4e-55f3-4e1f-8732-fe214089fb0b" />

