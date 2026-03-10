"""
Evaluation utilities for the biomedical LightRAG pipeline.

A. RAGAS-based RAG quality metrics (faithfulness, answer relevance, etc.)
B. CID relation extraction F1 against BC5CDR gold standard

Usage:
    import asyncio
    from module.RAG_pipeline.evaluate import evaluate_cid_f1, evaluate_ragas

    # CID F1
    results = asyncio.run(evaluate_cid_f1(pipeline, split="Test"))
    print(results)

    # RAGAS (requires a QA dataset with ground-truth answers)
    scores = asyncio.run(evaluate_ragas(pipeline, qa_pairs))
    print(scores)
"""

import asyncio
import logging
import os
import sys

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# A. CID Relation Extraction F1
# ─────────────────────────────────────────────────────────────────────────────

_CID_PROMPT = (
    'Based on the provided biomedical literature, does {chemical} cause or '
    'induce {disease}? Answer with exactly "yes" or "no".'
)


def _parse_yes_no(answer: str) -> bool:
    """Return True if the answer indicates yes."""
    return answer.strip().lower().startswith("yes")


async def evaluate_cid_f1(pipeline, split: str = "Test", max_pairs: int | None = None):
    """
    Evaluate Chemical–Disease Induction (CID) relation extraction.

    For each gold (chemical, disease) pair from BC5CDR, query the pipeline
    and compare predicted yes/no against the gold label (all positives).

    Args:
        pipeline:  An initialised RAGPipeline instance.
        split:     'Training', 'Development', or 'Test'.
        max_pairs: Limit number of pairs for quick smoke-tests.

    Returns:
        dict with keys: precision, recall, f1, tp, fp, fn, total_gold
    """
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from module.data_processing.bc5cdr import BC5CDR

    parser = BC5CDR()
    gold_df = parser.extract_relations(file_type=split)
    gold_pairs = list(zip(gold_df["chemical"], gold_df["disease"]))

    if max_pairs is not None:
        gold_pairs = gold_pairs[:max_pairs]

    logger.info("CID F1 eval: %d gold pairs from BC5CDR %s", len(gold_pairs), split)

    tp = fp = fn = 0

    for chemical, disease in gold_pairs:
        query = _CID_PROMPT.format(chemical=chemical, disease=disease)
        try:
            answer = await pipeline.query(query, mode="hybrid")
            predicted_positive = _parse_yes_no(answer)
        except Exception as exc:
            logger.warning("Query failed for (%s, %s): %s", chemical, disease, exc)
            predicted_positive = False

        if predicted_positive:
            tp += 1   # gold=positive, predicted=positive
        else:
            fn += 1   # gold=positive, predicted=negative

    # All predicted positives come from the gold set in this setup;
    # to measure FP we'd need negative pairs — skip for now and report recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_gold": len(gold_pairs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# B. RAGAS Metrics
# ─────────────────────────────────────────────────────────────────────────────

async def evaluate_ragas(pipeline, qa_pairs: list[dict], mode: str = "hybrid") -> dict:
    """
    Compute RAGAS metrics over a list of QA pairs.

    Args:
        pipeline: An initialised RAGPipeline instance.
        qa_pairs: List of dicts with keys:
                    - 'question'       (str)
                    - 'ground_truth'   (str)  — reference answer
        mode:     LightRAG retrieval mode.

    Returns:
        dict with RAGAS metric scores.

    Requires:
        pip install ragas datasets
    """
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
    except ImportError:
        raise ImportError(
            "RAGAS not installed. Run: pip install ragas datasets"
        )

    questions    = []
    answers      = []
    contexts     = []
    ground_truths = []

    for item in qa_pairs:
        q  = item["question"]
        gt = item["ground_truth"]

        # Retrieve context separately in 'local' mode for the context fields
        try:
            context_text = await pipeline.query(q, mode="local")
            answer_text  = await pipeline.query(q, mode=mode)
        except Exception as exc:
            logger.warning("Query failed for '%s': %s", q[:60], exc)
            context_text = ""
            answer_text  = ""

        questions.append(q)
        answers.append(answer_text)
        contexts.append([context_text])
        ground_truths.append(gt)

    dataset = Dataset.from_dict(
        {
            "question":     questions,
            "answer":       answers,
            "contexts":     contexts,
            "ground_truth": ground_truths,
        }
    )

    result = ragas_evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    return result.to_pandas().mean(numeric_only=True).to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# C. Benchmark accuracy (MedQA / PubMedQA style)
# ─────────────────────────────────────────────────────────────────────────────

async def evaluate_mcqa(pipeline, items: list[dict], rag_enabled: bool = True) -> dict:
    """
    Evaluate multiple-choice QA accuracy (e.g. MedQA).

    Each item must have:
        - 'question' (str)
        - 'options'  (dict[str, str])  e.g. {"A": "...", "B": "...", ...}
        - 'answer'   (str)             e.g. "A"

    Args:
        pipeline:    An initialised RAGPipeline instance.
        items:       List of MC QA items.
        rag_enabled: If False, skip retrieval (raw-model baseline).

    Returns:
        dict with 'accuracy' and 'correct' / 'total'.
    """
    correct = 0

    for item in items:
        opts_text = "\n".join(
            f"  {k}. {v}" for k, v in item["options"].items()
        )
        prompt = (
            f"Question: {item['question']}\n"
            f"Options:\n{opts_text}\n"
            "Answer with the letter only (A/B/C/D)."
        )

        if rag_enabled:
            answer = await pipeline.query(prompt, mode="hybrid")
        else:
            # Call the LLM directly without retrieval
            from module.RAG_pipeline.config import llm_fn
            answer = await llm_fn(prompt)

        predicted = answer.strip()[0].upper() if answer.strip() else ""
        if predicted == item["answer"].strip().upper():
            correct += 1

    total = len(items)
    return {
        "accuracy": round(correct / total, 4) if total else 0.0,
        "correct":  correct,
        "total":    total,
    }
