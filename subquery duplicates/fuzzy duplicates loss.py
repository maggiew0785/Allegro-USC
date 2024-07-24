import json
import pickle
import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import numpy as np
import torch.nn.functional as F
import os
import heapq

model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-1b",
  revision="step143000",
)

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-1b",
  revision="step143000",
)

def get_rank(lst, value):
    total = len(lst)
    count = 0
    for elem in lst:
        if elem < value:
            count += 1
    return total - (count + 1) + 1

def get_mink(input_file_path, output_file_path, output_logits_path, counts_path, loss_path):
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile, open(output_logits_path, 'w') as logitsfile:
        counts_array = []
        loss_array = []
        for i, line in enumerate(infile):
            data = json.loads(line)
            document_id = data["document_id"] # 1
            document_text = data["document"] # 2
            subsequences = data["subsequence_chars"] # 3
            subsequence_counts = data["subsequence_counts"] # 4
            average_subcount = data["average_subcount"] # 5
            mode_subcount = data["mode_subcount"] # 6
            converged = data["converged"] # 7
            num_queries = data["num_queries"]

            subqueries_results = []
            total_loss = 0
            subquery_count = 0

            for idx, subsequence in enumerate(subsequences):
                subsequence_tokens = tokenizer(subsequence, return_tensors="pt")
                outputs = model(**subsequence_tokens, labels=subsequence_tokens["input_ids"])
                loss = outputs.loss.item() # 8
                loss_array.append(loss)
                subcount = subsequence_counts[idx]
                counts_array.append(subcount)
                total_loss += loss
                subquery_count += 1

                logits = outputs.logits
                token_ids = subsequence_tokens["input_ids"][0][1:].unsqueeze(-1) # 9
                num_tokens = token_ids.size(0)
                log_probs = F.log_softmax(logits[0, :-1], dim=-1)
                token_id_log_probs = log_probs.gather(dim=-1, index=token_ids).squeeze(-1) # 10

                scores = {}
                for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                    k_length = max(1, int(len(token_id_log_probs) * ratio))  # Ensure at least one token is selected
                    topk = np.sort(token_id_log_probs.cpu().detach().numpy())[:k_length]
                    scores[f'mink_{ratio}'] = float(np.mean(topk)) if len(topk) > 0 else float('nan')
                
                subquery_logits = {}
                token_ranks = []
                token_log_probs = []
                for id_index in range(num_tokens):
                    id = token_ids[id_index].item()
                    id_log_probs = log_probs[id_index].tolist()
                    largest_500_log_probs = heapq.nlargest(500, id_log_probs)
                    subquery_logits[id] = largest_500_log_probs
                    
                    id_log_prob = id_log_probs[id]
                    rank = get_rank(id_log_probs, id_log_prob)
                    token_ranks.append(rank)
                    token_log_probs.append(id_log_prob)
    
                average_rank = np.mean(token_ranks)
                unique_subsequence_id = f"{document_id}_{idx}"
                subqueries_results.append({
                    "subsequence_id": unique_subsequence_id,
                    "subsequence_text": subsequence,
                    "subsequence_tokens": token_ids.tolist(),
                    "loss": loss,
                    "subsequence_count": subcount,
                    "mink_scores": scores,
                    "token_log_probs": token_log_probs,
                    "token_ranks": token_ranks,
                    "average_rank": average_rank
                })
            
            average_loss = total_loss / subquery_count if subquery_count > 0 else float('nan')

            document_result = {
                "document_id": document_id,
                "num_queries": num_queries,
                "average_loss": average_loss,
                "average_subcount": average_subcount,
                "mode_subcount": mode_subcount,
                "converged": converged,
                "subsequence_model_info": subqueries_results
            }

            document_logits = {
                "document_id": document_id,
                "subquery_logits": subquery_logits
            }

            outfile.write(json.dumps(document_result) + "\n")
            logitsfile.write(json.dumps(document_logits) + "\n")
            if i % 100 == 0:
                print(f"flushing: {i}")
                outfile.flush()
                logitsfile.flush()
                with open(counts_path, 'wb') as counts_out:
                    pickle.dump(counts_array, counts_out)
                with open(loss_path, 'wb') as loss_out:
                    pickle.dump(loss_array, loss_out)

        with open(counts_path, 'wb') as counts_out:
            pickle.dump(counts_array, counts_out)
        with open(loss_path, 'wb') as loss_out:
            pickle.dump(loss_array, loss_out)

components = [
    "Pile-CC",
    "PubMed Central",
    "Books3",
    "OpenWebText2",
    "ArXiv",
    "Github",
    "FreeLaw",
    "StackExchange",
    "USPTO Backgrounds",
    "PubMed Abstracts",
    "Gutenberg (PG-19)",
    "OpenSubtitles",
    "Wikipedia (en)",
    "DM Mathematics",
    "Ubuntu IRC",
    "BookCorpus2",
    "EuroParl",
    "HackerNews",
    "YoutubeSubtitles",
    "PhilPapers",
    "NIH ExPorter",
    "Enron Emails"
]

# call function
for component in components:
    input_file_path = f"/home/maggiew/JSONL Fuzzy 3/{component}.jsonl"
    output_file_path = f"/home/maggiew/Fuzzy Dupes/track fuzzy loss/{component}.jsonl"
    output_logits_path = f"/home/maggiew/Fuzzy Dupes/fuzzy logits/{component}.jsonl"
    counts_path = f"/home/maggiew/Fuzzy Dupes/track counts/{component}.pkl"
    loss_path = f"/home/maggiew/Fuzzy Dupes/track loss/{component}.pkl"

    get_mink(input_file_path, output_file_path, output_logits_path, counts_path, loss_path)