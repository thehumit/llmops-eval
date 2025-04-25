#!/usr/bin/env python
"""
CLI for:
  1) Generating AI answers from a remote 'benchmark_endpoint'
  2) Evaluating those answers with ragas.evaluate(...)
"""
import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
import yaml

from langchain_openai import ChatOpenAI

from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AspectCritic
from ragas.dataset_schema import MultiTurnSample
from ragas.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BenchmarkTool:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

        # instantiate LLM wrappers for evaluation phase
        self.llms = self._init_llms(self.config.get("llms", []))
        self.metrics = self._init_metrics(self.config.get("metrics", []))

        # placeholder for dataset; filled in generate or evaluate
        self.dataset: EvaluationDataset = EvaluationDataset(samples=[])

    def _init_llms(self, llm_configs: List[Dict[str, Any]]) -> List[LangchainLLMWrapper]:
        wrappers: List[LangchainLLMWrapper] = []
        for c in llm_configs:
            name = c["name"]
            logger.info(f"Initializing LLM '{name}' of type '{c['type']}'")
            if c["type"] == "openai":
                wrappers.append(
                    LangchainLLMWrapper(
                        name=name,
                        model=ChatOpenAI(api_key=c["api_key"], base_url=c["base_url"]),
                    )
                )
            else:
                raise ValueError(f"Unknown LLM type: {c['type']}")
        return wrappers

    def _init_metrics(self, metrics_cfg: List[Dict[str, Any]]) -> List[Any]:
        mets = []
        for m in metrics_cfg:
            if m["type"] == "aspect_critic":
                mets.append(
                    AspectCritic(name=m["name"], definition=m["definition"], llm=None)
                )
            else:
                raise ValueError(f"Unsupported metric type: {m['type']}")
        return mets

    def load_generated_dataset(self, path: Path) -> None:
        """
        Load a JSON file of serialized MultiTurnSample dicts
        and wrap into EvaluationDataset.
        """
        logger.info(f"Loading generated dataset from {path}")
        raw = json.load(path.open("r", encoding="utf-8"))
        samples = [MultiTurnSample(**r) for r in raw]
        self.dataset = EvaluationDataset(samples=samples)

    async def _evaluate_single(self, llm: LangchainLLMWrapper) -> Dict[str, float]:
        # assign llm to each metric
        for m in self.metrics:
            m.llm = llm

        logger.info(f"Evaluating LLM '{llm.name}' …")
        result = evaluate(dataset=self.dataset, metrics=self.metrics)
        df = result.to_pandas()
        # compute mean per metric
        return {col: df[col].mean() for col in df.columns}

    def run_evaluation(self) -> None:
        """
        Run evaluation across all llms+metrics and print a summary table.
        """
        loop = asyncio.get_event_loop()
        tasks = [self._evaluate_single(llm) for llm in self.llms]
        all_results = loop.run_until_complete(asyncio.gather(*tasks))
        summary = pd.DataFrame(all_results, index=[llm.name for llm in self.llms])
        print("\n=== EVALUATION RESULTS ===")
        print(summary.to_markdown())
        print()


def generate_responses(
    loader_path: str, endpoint: str, output_path: Path
) -> None:
    """
    1) Dynamically import the user's dataset loader module.
    2) Call get_evaluation_samples() → List[MultiTurnSample] initially containing only HumanMessage turns.
    3) For each sample, for each human turn:
         a) send conversation so far to ChatOpenAI(endpoint)
         b) append returned AIMessage
    4) Write the fully‐populated samples to `output_path` as JSON.

    Args:
        loader_path: Filesystem path to a .py file defining `get_evaluation_samples()`.
        endpoint:     Base URL for the ChatOpenAI‐compatible API.
        output_path:  Where to write the JSON of filled MultiTurnSample.
    """
    import importlib.util
    from importlib import util

    # --- 1) Dynamically import loader module ---
    loader_file = Path(loader_path).resolve()
    spec = util.spec_from_file_location(loader_file.stem, loader_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {loader_file}")
    loader_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader_mod)  # type: ignore

    if not hasattr(loader_mod, "get_evaluation_samples"):
        raise ImportError(f"{loader_file} must define get_evaluation_samples()")

    # --- 2) Retrieve samples with only HumanMessage turns ---
    samples: List[MultiTurnSample] = loader_mod.get_evaluation_samples()
    filled: List[MultiTurnSample] = []

    # Instantiate ChatOpenAI pointing to your endpoint (reads OPENAI_API_KEY env var by default)
    chat = ChatOpenAI(api_key=None, base_url=endpoint)

    # --- 3) For each sample, generate a reply per human turn ---
    for idx, sample in enumerate(samples, start=1):
        logger.info(f"Generating turns for sample #{idx}")
        conversation_payload: List[dict] = []
        new_turns: List[HumanMessage | AIMessage] = []

        # We iterate over original human‐only turns
        for turn_idx, human_msg in enumerate(sample.user_input, start=1):
            if not isinstance(human_msg, HumanMessage):
                raise TypeError(f"Expected HumanMessage, got {type(human_msg)}")
            # a) Append user
            new_turns.append(human_msg)
            conversation_payload.append({
                "role": "user",
                "content": human_msg.content
            })

            # b) Call the LLM for this partial conversation
            try:
                # ChatOpenAI is callable; returns an object with `.choices` or `.content`
                llm_response = chat(conversation_payload)  # type: ignore
            except Exception as e:
                logger.exception(f"LLM API error on sample {idx} turn {turn_idx}")
                raise

            # Extract assistant content
            if hasattr(llm_response, "content"):
                ai_text = llm_response.content
            else:
                # fallback for OpenAI‐style JSON
                ai_text = llm_response.choices[0].message.content  # type: ignore

            # Append assistant turn
            ai_msg = AIMessage(content=ai_text)
            new_turns.append(ai_msg)
            conversation_payload.append({
                "role": "assistant",
                "content": ai_text
            })
            logger.debug(f"Sample #{idx} turn {turn_idx} → {len(ai_text)} chars")

        # Replace sample.user_input with full human+AI turn list
        sample.user_input = new_turns
        filled.append(sample)

    # --- 4) Serialize to JSON ---
    serial = [s.dict() for s in filled]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(serial, f, ensure_ascii=False, indent=2)
    logger.info(f"Written {len(filled)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="LLM Benchmarking: generate phase & evaluate phase"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # generate subcommand
    g = sub.add_parser(
        "generate", help="Call benchmark_endpoint to generate AI replies"
    )
    g.add_argument(
        "--config", type=Path, default=Path("config.yaml"), help="YAML config path"
    )
    g.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the JSON of filled MultiTurnSample",
    )

    # evaluate subcommand
    e = sub.add_parser("evaluate", help="Run ragas.evaluate on generated JSON")
    e.add_argument(
        "--config", type=Path, default=Path("config.yaml"), help="YAML config path"
    )
    e.add_argument(
        "--generated",
        type=Path,
        required=True,
        help="Path to JSON file produced by the generate command",
    )

    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.open())

    if args.mode == "generate":
        endpoint = cfg.get("benchmark_endpoint")
        if not endpoint:
            raise ValueError("Please set `benchmark_endpoint` in config.yaml")
        generate_responses(cfg["dataset"]["loader"], endpoint, args.output)

    elif args.mode == "evaluate":
        tool = BenchmarkTool(cfg)
        tool.load_generated_dataset(args.generated)
        tool.run_evaluation()


if __name__ == "__main__":
    main()