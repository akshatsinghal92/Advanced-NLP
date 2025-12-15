# from huggingface_hub import login
# your_token = "INPUT YOUR TOKEN HERE"
# login(your_token)

import sys
import os

# Set HuggingFace mirror for faster model download (especially useful in China)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Set custom cache directory for LLM models (large models like Qwen)
# Embedding models will still use default cache location (~/.cache/huggingface)
LLM_CACHE_DIR = '/root/autodl-tmp/cache'
os.makedirs(LLM_CACHE_DIR, exist_ok=True)
os.environ['HF_LLM_CACHE_DIR'] = LLM_CACHE_DIR  # Custom env var for LLM only

# OpenAI proxy setup (match evaluation notebook settings)
os.environ["OPENAI_API_KEY"] = "your key"
os.environ["OPENAI_API_BASE"] = "https://api.openai-proxy.org/v1"
BASE_URL = os.getenv("OPENAI_BASE_URL", os.environ["OPENAI_API_BASE"])
API_KEY = os.getenv("OPENAI_API_KEY")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import csv
from tqdm import trange
from minirag import MiniRAG, QueryParam
from minirag.llm import hf_embed
from minirag.llm.openai import openai_complete
from minirag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Pre-load embedding model ONCE to avoid repeated loading
print("Loading embedding model...")
EMBED_TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
EMBED_MODEL = AutoModel.from_pretrained(EMBEDDING_MODEL)
print("Embedding model loaded successfully!")

import argparse


def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG")
    parser.add_argument("--model", type=str, default="gpt5.1")
    parser.add_argument("--outputpath", type=str, default="./logs/GPT51_output.csv")
    parser.add_argument("--workingdir", type=str, default="./LiHua-World")
    parser.add_argument("--datapath", type=str, default="./dataset/LiHua-World/data/")
    parser.add_argument(
        "--querypath", type=str, default="./dataset/LiHua-World/qa/query_set.csv"
    )
    args = parser.parse_args()
    return args


args = get_args()


if args.model.lower() == "gpt5.1":
    LLM_MODEL = "gpt-5.1"
else:
    print("Invalid model name. Please use --model gpt5.1 for this script.")
    exit(1)

WORKING_DIR = args.workingdir
DATA_PATH = args.datapath
QUERY_PATH = args.querypath
OUTPUT_PATH = args.outputpath
print("USING LLM:", LLM_MODEL)
print("USING WORKING DIR:", WORKING_DIR)


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=openai_complete,
    llm_model_max_token_size=200,
    llm_model_name=LLM_MODEL,
    llm_model_kwargs={"base_url": BASE_URL, "api_key": API_KEY},
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=1000,
        func=lambda texts: hf_embed(
            texts,
            tokenizer=EMBED_TOKENIZER,  # Use pre-loaded model
            embed_model=EMBED_MODEL,     # Use pre-loaded model
        ),
    ),
)

# Now QA
QUESTION_LIST = []
GA_LIST = []
with open(QUERY_PATH, mode="r", encoding="utf-8") as question_file:
    reader = csv.DictReader(question_file)
    for row in reader:
        QUESTION_LIST.append(row["Question"])
        GA_LIST.append(row["Gold Answer"])


def run_experiment(output_path):
    headers = ["Question", "Gold Answer", "minirag"]

    q_already = []
    if os.path.exists(output_path):
        with open(output_path, mode="r", encoding="utf-8") as question_file:
            reader = csv.DictReader(question_file)
            for row in reader:
                q_already.append(row["Question"])

    row_count = len(q_already)
    print("row_count", row_count)

    with open(output_path, mode="a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        if row_count == 0:
            writer.writerow(headers)

        for QUESTIONid in trange(row_count, len(QUESTION_LIST)):  #
            QUESTION = QUESTION_LIST[QUESTIONid]
            Gold_Answer = GA_LIST[QUESTIONid]
            print()
            print("QUESTION", QUESTION)
            print("Gold_Answer", Gold_Answer)

            try:
                minirag_answer = (
                    rag.query(QUESTION, param=QueryParam(mode="mini"))
                    .replace("\n", "")
                    .replace("\r", "")
                )
            except Exception as e:
                print("Error in minirag_answer", e)
                minirag_answer = "Error"

            writer.writerow([QUESTION, Gold_Answer, minirag_answer])
            log_file.flush() 

    print(f"Experiment data has been recorded in the file: {output_path}")


# if __name__ == "__main__":

run_experiment(OUTPUT_PATH)
