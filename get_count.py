import json
import requests

# function to call COUNT query for Infini-gram (takes token ids)
def get_count(query_tokens):
    payload = {
    'index': 'v4_piletrain_llama',
    'query_type': 'count',
    'query_ids': query_tokens
    }
    response = requests.post('https://api.infini-gram.io/', json=payload)
    if response.status_code == 200:
        result = response.json()
        if "count" in result:
            count = result["count"]
            return count
    return -1

if __name__ == "__main__":
    from transformers import AutoTokenizer
    token = os.getenv('HF_READ_ONLY_KEY')
    infinigram_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=token)
    