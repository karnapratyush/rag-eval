# eval_retriever.py
from dotenv import load_dotenv
import json
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

from src.retriever import build_retriever

load_dotenv()

GOLDEN_PATH = "goldens/retriever_goldens.json"
JUDGE_MODEL = "gpt-6-luna"   # NOTE: differs from the other evals (mostly gpt-4o-mini)
THRESHOLD = 0.7



with open(GOLDEN_PATH, "r") as f:
    goldens = json.load(f)

retriever = build_retriever()

test_cases = []
for g in goldens:
    retrieved = retriever.invoke(g["query"])
    retrieval_context = [doc.page_content for doc in retrieved]

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            expected_output=g["ideal_answer"],
            retrieval_context=retrieval_context,
            actual_output="(generator not evaluated in this run)",
        )
    )

metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ]
result = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    hyperparameters={
        "retriever": "base_k5",          # vs "reranked" in eval_retriever_with_reranker.py
        "embedding_model": "text-embedding-3-large",
        "chunk_size": 10000,      # must match src/retriever.py
        "chunk_overlap": 500,
        "top_k": 5,
        "judge_model": JUDGE_MODEL,
        "golden_set": GOLDEN_PATH,
    },
)

