import json
import requests
import os
from transformers import AutoTokenizer
import pickle
import random
from scipy.stats import mode
import numpy as np

token = os.getenv('HF_READ_ONLY_KEY')
infinigram_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=token)

def get_count(query):
    payload = {
    'index': 'v4_piletrain_llama',
    'query_type': 'count',
    'query_ids': query
    }
    response = requests.post('https://api.infini-gram.io/', json=payload)
    if response.status_code == 200:
        result = response.json()
        if "count" in result:
            count = result["count"]
            return count
    return -1

def count_subsequences(input_file_path, output_file_path):
    config = load_config()
    QUERY_TOKEN_LENGTH = config["query_token_length"]
    PROPORTION_DOC_TOKENS = config["proportion_of_document_tokens"]
    
    with open(input_file_path, 'r') as infile, open(output_file_path, 'a') as outfile:
        json_dump_count = 0
        for i, line in enumerate(infile, start=1):
            if json_dump_count > 700:
                break
            
            json_data = json.loads(line)
            document_id = json_data["document_id"] 
            document = json_data["document"]

            document_infinigram_tokens = infinigram_tokenizer.encode(document)
            num_document_infinigram_tokens = len(document_infinigram_tokens)
            
            num_queries = int((num_document_infinigram_tokens * PROPORTION_DOC_TOKENS) // QUERY_TOKEN_LENGTH)
            if num_queries < 5:
                num_queries = 5  # minimum num_queries
            if num_queries > 20:
                num_queries = 20 # maximum num_queries
                
            last_possible_start_index = num_document_infinigram_tokens - QUERY_TOKEN_LENGTH
            start_indices = set()
            while len(start_indices) < num_queries:
                start_index = random.randint(1, last_possible_start_index)
                start_indices.add(start_index)

            subsequence_counts = []
            subsequences_chars = []
            subsequences_tokens = []

            for start in start_indices:
                end = start + QUERY_TOKEN_LENGTH
                subsequence = document_infinigram_tokens[start:end]
                subsequences_tokens.append(subsequence)
                subsequence_chars = infinigram_tokenizer.decode(subsequence)
                subsequences_chars.append(subsequence_chars)
                count = get_count(subsequence)
                subsequence_counts.append(count)

            average_subcount = np.mean(subsequence_counts)
            variance_subcount = np.var(subsequence_counts)
            mode_output = mode(subsequence_counts)
            mode_subcount = int(mode_output.mode)
            mode_frequency_subcount = int(mode_output.count)
            max_subcount = max(subsequence_counts)
            min_subcount = min(subsequence_counts)

            document_num_duplicates = json_data["num_duplicates"]

            converged = False
            if mode_frequency_subcount > (num_queries // 2):
                converged = True

            json_info = {
                'document_id': document_id,
                'document': document,
                'document_infinigram_tokens': document_infinigram_tokens,
                'document_infinigram_tokens_count': num_document_infinigram_tokens,
                'num_queries': num_queries,
                'start_indices': list(start_indices),
                'subsequence_chars': subsequences_chars,
                'subsequence_tokens': subsequences_tokens,
                'subsequence_counts': subsequence_counts,
                'average_subcount': average_subcount,
                'variance_subcount': variance_subcount,
                'mode_subcount': mode_subcount,
                'mode_frequency_subcount': mode_frequency_subcount,
                'converged': converged,
                'max_subcount': max_subcount,
                'min_subcount': min_subcount,
                'num_duplicates': document_num_duplicates
            }

            outfile.write(json.dumps(json_info) + "\n")
            outfile.flush()
            
            json_dump_count += 1

            if i % 50 == 0:
                print(f'Flushing: {i}')
                print(f'Number of documents dumped: {json_dump_count}')



component_paths = {
    "Pile-CC": "/home/maggiew/JSONL Collection/components/pile_cc.jsonl",
    "PubMed Central": "/home/maggiew/JSONL Collection/components/pubmed_central.jsonl",
    "Books3": "/home/maggiew/JSONL Collection/components/books3.jsonl",
    "OpenWebText2": "/home/maggiew/JSONL Collection/components//openwebtext2.jsonl",
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
for component in component_paths:
    print(component)
    input_file_path = component_paths[component]
    output_file_path = f"/home/maggiew/JSONL Fuzzy 3/{component}.jsonl"
    count_subsequences(input_file_path, output_file_path)