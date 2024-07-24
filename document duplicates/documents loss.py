import json
import pickle
import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import numpy as np
import torch.nn.functional as F
import os

model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-1b",
  revision="step143000",
)

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-1b",
  revision="step143000",
)

collection_paths = {
    "Pile-CC": "/home/maggiew/JSONL Collection/pile_cc.jsonl",
    "PubMed Central": "/home/maggiew/JSONL Collection/pubmed_central.jsonl",
    "Books3": "/home/maggiew/JSONL Collection/books3.jsonl",
    "OpenWebText2": "/home/maggiew/JSONL Collection/openwebtext2.jsonl",
    "ArXiv": "/home/maggiew/JSONL Collection/arxiv.jsonl",
    "Github": "/home/maggiew/JSONL Collection/github.jsonl",
    "FreeLaw": "/home/maggiew/JSONL Collection/freelaw.jsonl",
    "StackExchange": "/home/maggiew/JSONL Collection/stack_exchange.jsonl",
    "USPTO Backgrounds": "/home/maggiew/JSONL Collection/uspto_backgrounds.jsonl",
    "PubMed Abstracts": "/home/maggiew/JSONL Collection/pubmed_abstracts.jsonl",
    "Gutenberg (PG-19)": "/home/maggiew/JSONL Collection/gutenberg.jsonl",
    "OpenSubtitles": "/home/maggiew/JSONL Collection/opensubtitles.jsonl",
    "Wikipedia (en)": "/home/maggiew/JSONL Collection/wikipedia.jsonl",
    "DM Mathematics": "/home/maggiew/JSONL Collection/dm_mathematics.jsonl",
    "Ubuntu IRC": "/home/maggiew/JSONL Collection/ubuntu_irc.jsonl",
    "BookCorpus2": "/home/maggiew/JSONL Collection/bookcorpus2.jsonl",
    "EuroParl": "/home/maggiew/JSONL Collection/europarl.jsonl",
    "HackerNews": "/home/maggiew/JSONL Collection/hackernews.jsonl",
    "YoutubeSubtitles": "/home/maggiew/JSONL Collection/youtubesubtitles.jsonl",
    "PhilPapers": "/home/maggiew/JSONL Collection/philpapers.jsonl",
    "NIH ExPorter": "/home/maggiew/JSONL Collection/nih_exporter.jsonl",
    "Enron Emails": "/home/maggiew/JSONL Collection/enron_emails.jsonl"
}

def get_mink(input_path, output_path, duplicates_to_mink_paths, duplicates_to_mink):
    with open(input_path, 'r') as input_file, open(output_path, 'a') as output_file:
        for i, line in enumerate(input_file):
            if i >= 2001:
                break
            json_data = json.loads(line)
            document_ID = json_data['document_id']
            query_chars = json_data["query_chars"]
            num_duplicates = json_data['num_duplicates']
            original_token_ids = tokenizer(query_chars, return_tensors="pt")["input_ids"]
            trunc_500_token_ids = original_token_ids[:, :500]
            trunc_500_token_ids_list = trunc_500_token_ids.tolist()
            trunc_chars = tokenizer.decode(trunc_500_token_ids[0], skip_special_tokens=True)
            trunc_500_tokens = {
                'input_ids': trunc_500_token_ids,
                'attention_mask': torch.ones(trunc_500_token_ids.shape, dtype=torch.long)
            }
            
            outputs = model(**trunc_500_tokens, labels=trunc_500_tokens["input_ids"])
            logits = outputs.logits
            input_ids = trunc_500_tokens["input_ids"][0][1:].unsqueeze(-1)  # Exclude the first token (usually the start token)
            log_probs = F.log_softmax(logits[0, :-1], dim=-1)
            token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)

            # Calculate Min-K% scores for different ratios
            scores = {}
            for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                k_length = max(1, int(len(token_log_probs) * ratio))  # Ensure at least one token is selected
                topk = np.sort(token_log_probs.cpu().detach().numpy())[:k_length]
                if len(topk) > 0:
                    scores[f'mink_{ratio}'] = float(np.mean(topk))  # Convert to native Python float
                else:
                    scores[f'mink_{ratio}'] = float('nan')  # Handle empty slice case

            json.dump({
                'document_id': document_ID,
                'num_duplicates': num_duplicates,
                'trunc_500_token_ids': trunc_500_token_ids_list,
                'trunc_chars': trunc_chars,
                **{f'mink_{ratio}': scores[f'mink_{ratio}'] for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
            }, output_file)
            output_file.write('\n')

            for ratio in duplicates_to_mink_paths:
                duplicates_to_mink[ratio].append((num_duplicates, scores[f'mink_{ratio}']))

            # checkpoint every 100 entries
            if i % 100 == 0:
                for ratio, path in duplicates_to_mink_paths.items():
                    with open(path, 'wb') as dups_file:
                        pickle.dump(duplicates_to_mink[ratio], dups_file)
                print(str(i))
                
    # Final write to ensure all data is saved
    for ratio, path in duplicates_to_mink_paths.items():
        with open(path, 'wb') as dups_file:
            pickle.dump(duplicates_to_mink[ratio], dups_file)           


collection_paths = {
    "Pile-CC": "/home/maggiew/JSONL Collection/components/pile_cc.jsonl",
    "PubMed Central": "/home/maggiew/JSONL Collection/components/pubmed_central.jsonl",
    "Books3": "/home/maggiew/JSONL Collection/components/books3.jsonl",
    "OpenWebText2": "/home/maggiew/JSONL Collection/components/openwebtext2.jsonl",
    "ArXiv": "/home/maggiew/JSONL Collection/components/arxiv.jsonl",
    "Github": "/home/maggiew/JSONL Collection/components/github.jsonl",
    "FreeLaw": "/home/maggiew/JSONL Collection/components/freelaw.jsonl",
    "StackExchange": "/home/maggiew/JSONL Collection/components/stack_exchange.jsonl",
    "USPTO Backgrounds": "/home/maggiew/JSONL Collection/components/uspto_backgrounds.jsonl",
    "PubMed Abstracts": "/home/maggiew/JSONL Collection/components/pubmed_abstracts.jsonl",
    "Gutenberg (PG-19)": "/home/maggiew/JSONL Collection/components/gutenberg.jsonl",
    "OpenSubtitles": "/home/maggiew/JSONL Collection/components/opensubtitles.jsonl",
    "Wikipedia (en)": "/home/maggiew/JSONL Collection/components/wikipedia.jsonl",
    "DM Mathematics": "/home/maggiew/JSONL Collection/components/dm_mathematics.jsonl",
    "Ubuntu IRC": "/home/maggiew/JSONL Collection/components/ubuntu_irc.jsonl",
    "BookCorpus2": "/home/maggiew/JSONL Collection/components/bookcorpus2.jsonl",
    "EuroParl": "/home/maggiew/JSONL Collection/components/europarl.jsonl",
    "HackerNews": "/home/maggiew/JSONL Collection/components/hackernews.jsonl",
    "YoutubeSubtitles": "/home/maggiew/JSONL Collection/components/youtubesubtitles.jsonl",
    "PhilPapers": "/home/maggiew/JSONL Collection/components/philpapers.jsonl",
    "NIH ExPorter": "/home/maggiew/JSONL Collection/components/nih_exporter.jsonl",
    "Enron Emails": "/home/maggiew/JSONL Collection/components/enron_emails.jsonl"
}

# call function
for component in collection_paths:
    if component == 'Pile-CC':
        continue
    duplicates_to_mink_paths = {
    0.1: f'/home/maggiew/Collection/track/{component}/duplicates_to_mink10.pkl',
    0.2: f'/home/maggiew/Collection/track/{component}/duplicates_to_mink20.pkl',
    0.3: f'/home/maggiew/Collection/track/{component}/duplicates_to_mink30.pkl',
    0.4: f'/home/maggiew/Collection/track/{component}/duplicates_to_mink40.pkl',
    0.5: f'/home/maggiew/Collection/track/{component}/duplicates_to_mink50.pkl',
    0.6: f'/home/maggiew/Collection/track/{component}/duplicates_to_mink60.pkl',
    0.7: f'/home/maggiew/Collection/track/{component}/duplicates_to_mink70.pkl',
    0.8: f'/home/maggiew/Collection/track/{component}/duplicates_to_mink80.pkl',
    0.9: f'/home/maggiew/Collection/track/{component}/duplicates_to_mink90.pkl',
    1.0: f'/home/maggiew/Collection/track/{component}/duplicates_to_mink100.pkl'
    }
    
    duplicates_to_mink = {ratio: [] for ratio in duplicates_to_mink_paths}
    
    output_path = f'/home/maggiew/JSONL Collection/{component}_mink.jsonl'
    input_path = collection_paths[component]
    get_mink(input_path, output_path, duplicates_to_mink_paths, duplicates_to_mink)
    