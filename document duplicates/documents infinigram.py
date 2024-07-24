# 1. IMPORTS
import json
import requests
import os
from transformers import AutoTokenizer
import pickle

#2. Load tokenizer
token = os.getenv('HF_READ_ONLY_KEY')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=token)

# 3. get_count()
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

# 5. get documents from shard
def first_5000_docs(input_file_path, component_paths, last_seen_line, last_seen_line_path, skipped_lines_path,
                    component_pairs_path, component_counts_path, done_components_path, 
                    batch_size=100):
    # Initialize variables

    # Initialize in-memory buffers for batch processing
    batch_data = {component: [] for component in component_paths.keys()}

    def flush_batches():
        for component, data_list in batch_data.items():
            if data_list:
                path = component_paths[component]
                with open(path, 'a') as output_file:
                    for data in data_list:
                        json.dump(data, output_file)
                        output_file.write('\n')
                batch_data[component] = []

    with open(input_file_path, 'r') as input_file:
        for i, line in enumerate(input_file):
            # Check if all done
            if len(done_components) == len(component_paths):
                break

            if i <= last_seen_line:
                continue

            # Update seen_lines
            last_seen_line = i
            with open(last_seen_line_path, 'w') as output_file:
                output_file.write(str(last_seen_line))

            json_data = json.loads(line)
            component = json_data['meta']['pile_set_name']
            count = component_counts[component]
            if count >= 5000:
                continue

            document = json_data['text']
            document_length = len(document)
            tokenized_document = tokenizer.encode(document)
            query_token_ids = tokenized_document[1:501]
            query_num_tokens = len(query_token_ids)
            if query_num_tokens < 100:
                skipped_lines.append(i)
                with open(skipped_lines_path, 'wb') as output_file:
                    pickle.dump(skipped_lines, output_file)
                    
                continue
            num_duplicates = get_count(query_token_ids)
            query_chars = tokenizer.decode(query_token_ids)

            json_info = {
                'document_id': i,
                'document': document,
                'document_length': document_length,
                'query_token_ids': query_token_ids,
                'query_num_tokens': query_num_tokens,
                'query_chars': query_chars,
                'num_duplicates': num_duplicates,
            }

            batch_data[component].append(json_info)

            pair = (i, num_duplicates)
            component_pairs[component].append(pair)
            component_counts[component] = count + 1

            with open(component_pairs_path, 'wb') as output_file:
                pickle.dump(component_pairs, output_file)

            with open(component_counts_path, 'wb') as output_file:
                pickle.dump(component_counts, output_file)

            if component_counts[component] == 5000:
                done_components.add(component)
                with open(done_components_path, 'wb') as output_file:
                    pickle.dump(done_components, output_file)

            if (i + 1) % batch_size == 0:
                print('flushing: ' + str(i))
                print('\n')
                print(component_counts)
                print('\n')

                flush_batches()

    # Flush remaining data in batches
    flush_batches()

    # Final save for seen_lines, skipped_lines, component_pairs, component_counts, and done_components
    with open(last_seen_line_path, 'w') as output_file:
        output_file.write(str(last_seen_line))
        
    with open(skipped_lines_path, 'wb') as output_file:
        pickle.dump(skipped_lines, output_file)
        
    with open(component_pairs_path, 'wb') as output_file:
        pickle.dump(component_pairs, output_file)

    with open(component_counts_path, 'wb') as output_file:
        pickle.dump(component_counts, output_file)

    with open(done_components_path, 'wb') as output_file:
        pickle.dump(done_components, output_file)

# Initialize file paths and variables
input_file_path = '/home/johnny/data/00.jsonl'  # 45 GB

component_paths = {
    "Pile-CC": "/home/maggiew/JSONL Components/pile_cc.jsonl",
    "PubMed Central": "/home/maggiew/JSONL Components/pubmed_central.jsonl",
    "Books3": "/home/maggiew/JSONL Components/books3.jsonl",
    "OpenWebText2": "/home/maggiew/JSONL Components/openwebtext2.jsonl",
    "ArXiv": "/home/maggiew/JSONL Components/arxiv.jsonl",
    "Github": "/home/maggiew/JSONL Components/github.jsonl",
    "FreeLaw": "/home/maggiew/JSONL Components/freelaw.jsonl",
    "StackExchange": "/home/maggiew/JSONL Components/stack_exchange.jsonl",
    "USPTO Backgrounds": "/home/maggiew/JSONL Components/uspto_backgrounds.jsonl",
    "PubMed Abstracts": "/home/maggiew/JSONL Components/pubmed_abstracts.jsonl",
    "Gutenberg (PG-19)": "/home/maggiew/JSONL Components/gutenberg.jsonl",
    "OpenSubtitles": "/home/maggiew/JSONL Components/opensubtitles.jsonl",
    "Wikipedia (en)": "/home/maggiew/JSONL Components/wikipedia.jsonl",
    "DM Mathematics": "/home/maggiew/JSONL Components/dm_mathematics.jsonl",
    "Ubuntu IRC": "/home/maggiew/JSONL Components/ubuntu_irc.jsonl",
    "BookCorpus2": "/home/maggiew/JSONL Components/bookcorpus2.jsonl",
    "EuroParl": "/home/maggiew/JSONL Components/europarl.jsonl",
    "HackerNews": "/home/maggiew/JSONL Components/hackernews.jsonl",
    "YoutubeSubtitles": "/home/maggiew/JSONL Components/youtubesubtitles.jsonl",
    "PhilPapers": "/home/maggiew/JSONL Components/philpapers.jsonl",
    "NIH ExPorter": "/home/maggiew/JSONL Components/nih_exporter.jsonl",
    "Enron Emails": "/home/maggiew/JSONL Components/enron_emails.jsonl"
}

done_components = set()
done_components_path = "done_components.pkl"
component_pairs = {key: [] for key in component_paths}
component_pairs_path = "component_pairs.pkl"
component_counts = {key: 0 for key in component_paths}
component_counts_path = "component_counts.pkl"
last_seen_line = -1
last_seen_line_path = "last_seen_line.txt"
skipped_lines = []
skipped_lines_path = "skipped_lines_components.pkl"

# call function
first_5000_docs(input_file_path, component_paths, last_seen_line,
                last_seen_line_path, skipped_lines_path, 
                component_pairs_path, component_counts_path, done_components_path)