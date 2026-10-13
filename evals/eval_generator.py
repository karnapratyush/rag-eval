# eval_generator.py
import json

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, ContextualRecallMetric, ContextualPrecisionMetric, FaithfulnessMetric

# from src.reranker import RerankingRetriever
from src.generator import generate
# from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

GOLDEN_PATH = "goldens/faithfullness_dataset.json"
JUDGE_MODEL = "gpt-4o-mini"
THRESHOLD = 0.7


def run():
    # 1. LOAD the golden set --- the fixed, human-authored truth
    goldens=[]
    with open(GOLDEN_PATH) as f:
        goldens = json.load(f)

    # 2. RUN THE GENERATOR on the IDEAL context (retriever bypassed), so any
    #    unfaithfulness is the generator's fault, not the retriever's.
    test_cases = []
    for g in goldens:
        print(f"Processing query: {g['query']}, ideal_context: {g['ideal_context']}, typeof: {type(g['ideal_context'])}")
        answer = generate(g["query"], g["ideal_context"])
        print(f"Answer: {answer}")
        

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=answer,
                retrieval_context=g["ideal_context"]
            )
        )

    # 3. THE METRICS --- faithfulness (grounded in context?) and answer relevancy (on topic?)
    metrics = [
        FaithfulnessMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True,   # prints WHY each score --- shows which claims were unsupported
        ),
        AnswerRelevancyMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True,
        ),
    ]

    # 4. EVALUATE --- every metric on every case, batched + parallel, printed report.
    #    hyperparameters travel with the run so the report is tagged with the config.
    result = evaluate(test_cases=test_cases, metrics=metrics)
    # return summarize_by_metric(result)
    return result


# def run_local():
#     """Standalone convenience: build the retriever, then run."""
#     return run(RerankingRetriever())


# if __name__ == "__main__":
#     print_summary("retriever", run_local())

run()