#!/usr/bin/env python3
"""
CLI tool to evaluate generated MultiTurnSample JSON using ragas.AspectCritic.
Prints a pandas‐style summary of mean scores per metric.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import List

import yaml

import pandas as pd
from langchain_openai import ChatOpenAI

from ragas import evaluate
from ragas.dataset_schema import MultiTurnSample, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AspectCritic
from ragas.metrics import RubricsScore, InstanceRubrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_samples(path: Path) -> EvaluationDataset:
    """
    Load JSON file of serialized MultiTurnSample dicts.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    samples: List[MultiTurnSample] = [MultiTurnSample(**r) for r in raw]
    return EvaluationDataset(samples=samples)


def main() -> None:
    with open("config.yaml", "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    # Wrap ChatOpenAI for ragas evaluation
    llm_wrapper = LangchainLLMWrapper(ChatOpenAI(api_key=config['api_key'], base_url=config['base_url'], streaming=False))

    with open(config["path_to_eval_dataset"], 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    rubrics = config["rubrics"]

    dataset = []

    for item in json_data:
        # Extract the user query (human message)
        user_query = next((msg['content'] for msg in item['user_input'] if msg['type'] == 'human'), "")

        # Extract the AI response
        response = next((msg['content'] for msg in item['user_input'] if msg['type'] == 'ai'), "")

        # Add to dataset
        dataset.append({
            "user_query": user_query,
            "response": response,
            "rubrics": rubrics if item['rubrics'] is None else item['rubrics']
        })

    print(dataset)
    evaluation_dataset = EvaluationDataset.from_list(dataset)

    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[InstanceRubrics(llm=llm_wrapper)],
        llm=llm_wrapper,
    )

    df = result.to_pandas()

    # print table & mean
    print("\n=== EVALUATION SCORES ===")
    print(df.to_csv("result.csv"))
    mean_score = df["instance_rubrics"].mean()
    print(f"\nMean WebUI: {mean_score:.3f}")


if __name__ == "__main__":
    main()