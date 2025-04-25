#!/usr/bin/env python3
"""
CLI tool to generate AI replies for human‐only MultiTurnSample dataset.
Produces a JSON file of fully populated MultiTurnSample instances.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

from langchain_openai import ChatOpenAI
from ragas.dataset_schema import MultiTurnSample
from ragas.messages import HumanMessage, AIMessage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_loader_module(loader_path: Path):
    """
    Dynamically import a .py file that defines get_evaluation_samples().
    Returns the module object.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(loader_path.stem, loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {loader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    if not hasattr(module, "get_evaluation_samples"):
        raise AttributeError(f"{loader_path} must define get_evaluation_samples()")
    return module


def generate_samples(
    loader_path: Path, endpoint: str, api_key: Optional[str]
) -> List[MultiTurnSample]:
    """
    1) Imports get_evaluation_samples() from loader_path
    2) Calls it to get List[MultiTurnSample] with only HumanMessage turns
    3) For each human turn, calls ChatOpenAI to get an AI reply
    4) Returns a list of fully populated MultiTurnSample
    """
    module = load_loader_module(loader_path)
    raw_samples: List[MultiTurnSample] = module.get_evaluation_samples()

    # instantiate LangChain Chat model
    chat = ChatOpenAI(api_key=api_key, base_url=endpoint)

    filled_samples: List[MultiTurnSample] = []
    for idx, sample in enumerate(raw_samples, start=1):
        logger.info(f"Generating sample {idx}/{len(raw_samples)}")
        conversation: List[dict] = []
        new_turns: List[HumanMessage | AIMessage] = []

        for turn_idx, human_msg in enumerate(sample.user_input, start=1):
            if not isinstance(human_msg, HumanMessage):
                raise TypeError(f"Expected HumanMessage, got {type(human_msg)}")
            # append human
            new_turns.append(human_msg)
            conversation.append({"role": "user", "content": human_msg.content})

            # call LLM
            try:
                resp = chat(conversation)  # type: ignore
            except Exception:
                logger.exception(f"LLM call failed on sample {idx}, turn {turn_idx}")
                raise

            # extract AI text
            ai_text = getattr(resp, "content", None) or resp.choices[0].message.content  # type: ignore
            ai_msg = AIMessage(content=ai_text)

            # append AI
            new_turns.append(ai_msg)
            conversation.append({"role": "assistant", "content": ai_text})
            logger.debug(f"Sample {idx} turn {turn_idx}: generated {len(ai_text)} chars")

        filled_samples.append(MultiTurnSample(user_input=new_turns))

    return filled_samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate AI replies for human‐only MultiTurnSample."
    )
    parser.add_argument(
        "--loader",
        type=Path,
        required=True,
        help="Path to .py file with get_evaluation_samples()",
    )
    parser.add_argument(
        "--endpoint", type=str, required=True, help="ChatOpenAI base_url"
    )
    parser.add_argument(
        "--api-key", type=str, default=None, help="OpenAI API key (or omit to use env)"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Where to write JSON samples"
    )
    args = parser.parse_args()

    samples = generate_samples(args.loader, args.endpoint, args.api_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fout:
        json.dump([s.dict() for s in samples], fout, ensure_ascii=False, indent=2)
    logger.info(f"Wrote {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()