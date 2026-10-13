from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric

from src.rag_pipeline import RagPipeline
import json

load_dotenv()

GOLDEN_PATH = "goldens/correctness_goldens.json"
JUDGE_MODEL = "gpt-4o-mini"
THRESHOLD = 0.7

goldens=[]

with open(GOLDEN_PATH) as f:
    goldens = json.load(f)

test_cases = []
rag=RagPipeline()
for g in  goldens:
    query = g["question"]
    expected = g["ideal_answer"]
    response = rag.invoke(query)
    print(response)
    print("-" * 80)
    test_case = LLMTestCase(
                input=g["question"],
                actual_output=response["answer"],
                expected_output=expected,
    )
    test_cases.append(test_case)


# 3a. CORRECTNESS --- reference-based, judges TRUTH (not coverage or length)
correctness = GEval(
    name="Correctness",
    evaluation_steps=[
        "Compare only the factual claims in the actual output against the expected output.",
        "A claim is wrong only if it CONTRADICTS the expected output or is factually false. Judge truth, not completeness.",
        "A factually accurate answer must score at least 0.9 even if it is shorter or covers fewer points than the expected output.",
        "Do NOT deduct for brevity, missing elaboration, or omitted points --- omissions are not errors here.",
        "Additional correct information must NEVER lower the score.",
    ],
    rubric=[
        Rubric(score_range=(9, 10), expected_outcome="All stated claims are factually correct and consistent. No contradictions. Brevity is fine."),
        Rubric(score_range=(5, 8),  expected_outcome="Mostly correct but one minor inaccuracy."),
        Rubric(score_range=(0, 4),  expected_outcome="Contains a clear factual error or a claim that contradicts the expected output."),
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    strict_mode=False,
)

result = evaluate(test_cases=test_cases, metrics=[correctness])