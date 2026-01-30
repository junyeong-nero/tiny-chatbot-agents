"""Prompts for LLM-as-a-Judge evaluation.

This module contains all prompts used for evaluating LLM-generated answers
against golden (reference) answers using various criteria.
"""

# Main judge system prompt
JUDGE_SYSTEM_PROMPT = """당신은 고객 서비스 AI 응답 평가 전문가입니다.
생성된 답변을 모범 답변(Golden Answer)과 비교하여 평가합니다.

평가 기준에 따라 1-5점 척도로 점수를 부여하고,
반드시 JSON 형식으로 응답하세요."""

# Individual criteria prompts
CORRECTNESS_PROMPT = """[정확성 평가]
생성된 답변이 모범 답변과 비교하여 사실적으로 정확한지 평가하세요.

[질문]
{question}

[모범 답변 (Golden Answer)]
{golden_answer}

[생성된 답변 (Generated Answer)]
{generated_answer}

평가 기준:
- 5점: 모범 답변과 동일하거나 더 정확한 정보 제공
- 4점: 핵심 정보가 정확하며 사소한 차이만 있음
- 3점: 대체로 정확하나 일부 정보 누락 또는 약간의 오류
- 2점: 상당한 정보 누락 또는 오류 존재
- 1점: 대부분 부정확하거나 잘못된 정보

JSON 형식으로 응답:
{{"score": <1-5>, "reasoning": "<평가 근거>"}}"""

HELPFULNESS_PROMPT = """[유용성 평가]
생성된 답변이 고객의 질문에 얼마나 도움이 되는지 평가하세요.

[질문]
{question}

[모범 답변 (Golden Answer)]
{golden_answer}

[생성된 답변 (Generated Answer)]
{generated_answer}

평가 기준:
- 5점: 질문에 완벽하게 답변하고 추가적인 유용한 정보도 제공
- 4점: 질문에 충분히 답변하며 실용적임
- 3점: 기본적인 답변은 제공하나 완전하지 않음
- 2점: 부분적으로만 도움이 되며 중요한 정보 누락
- 1점: 질문에 대한 답변이 되지 않거나 전혀 도움이 안 됨

JSON 형식으로 응답:
{{"score": <1-5>, "reasoning": "<평가 근거>"}}"""

FAITHFULNESS_PROMPT = """[충실성 평가]
생성된 답변이 제공된 컨텍스트에 충실한지 평가하세요.
환각(hallucination) 여부를 확인합니다.

[질문]
{question}

[모범 답변 (Golden Answer)]
{golden_answer}

[생성된 답변 (Generated Answer)]
{generated_answer}

평가 기준:
- 5점: 모든 정보가 컨텍스트에 근거하며 환각이 전혀 없음
- 4점: 대부분 컨텍스트에 충실하며 사소한 추론만 포함
- 3점: 일부 정보가 컨텍스트 외 출처에서 온 것으로 보임
- 2점: 상당 부분이 컨텍스트와 무관하거나 지어낸 정보
- 1점: 대부분 환각이거나 사실과 다른 정보

JSON 형식으로 응답:
{{"score": <1-5>, "reasoning": "<평가 근거>"}}"""

FLUENCY_PROMPT = """[유창성 평가]
생성된 답변의 문장 품질과 자연스러움을 평가하세요.

[질문]
{question}

[모범 답변 (Golden Answer)]
{golden_answer}

[생성된 답변 (Generated Answer)]
{generated_answer}

평가 기준:
- 5점: 완벽하게 자연스럽고 전문적인 톤, 문법 오류 없음
- 4점: 자연스럽고 읽기 쉬우며 사소한 문제만 있음
- 3점: 이해 가능하나 어색한 표현이 일부 있음
- 2점: 문법 오류나 어색한 표현이 많음
- 1점: 이해하기 어렵거나 문장이 깨짐

JSON 형식으로 응답:
{{"score": <1-5>, "reasoning": "<평가 근거>"}}"""

# Comprehensive evaluation prompt (all criteria at once - more efficient)
COMPREHENSIVE_JUDGE_PROMPT = """다음 생성된 답변을 모범 답변과 비교하여 평가하세요.

[질문]
{question}

[모범 답변 (Golden Answer)]
{golden_answer}

[생성된 답변 (Generated Answer)]
{generated_answer}

[평가 기준별 점수 부여 (1-5점)]

1. **정확성 (Correctness)**: 사실적 정확도
   - 5점: 모범 답변과 동일하거나 더 정확
   - 4점: 핵심 정보 정확, 사소한 차이만 있음
   - 3점: 대체로 정확하나 일부 누락/오류
   - 2점: 상당한 누락 또는 오류 존재
   - 1점: 대부분 부정확

2. **유용성 (Helpfulness)**: 질문 해결에 도움이 되는 정도
   - 5점: 완벽한 답변 + 추가 유용 정보
   - 4점: 충분하고 실용적인 답변
   - 3점: 기본 답변 제공, 완전하지 않음
   - 2점: 부분적으로만 도움, 중요 정보 누락
   - 1점: 답변이 안 되거나 무용

3. **충실성 (Faithfulness)**: 환각 없이 컨텍스트에 충실한 정도
   - 5점: 모든 정보가 근거 있음, 환각 없음
   - 4점: 대부분 충실, 사소한 추론만 포함
   - 3점: 일부 정보가 컨텍스트 외 출처
   - 2점: 상당 부분 무관하거나 지어낸 정보
   - 1점: 대부분 환각

4. **유창성 (Fluency)**: 문장 품질 및 자연스러움
   - 5점: 완벽하게 자연스럽고 전문적
   - 4점: 자연스럽고 읽기 쉬움
   - 3점: 이해 가능, 어색한 표현 일부
   - 2점: 문법 오류나 어색함 많음
   - 1점: 이해 어렵거나 문장 깨짐

JSON 형식으로 응답:
{{
    "correctness": {{"score": <1-5>, "reasoning": "<근거>"}},
    "helpfulness": {{"score": <1-5>, "reasoning": "<근거>"}},
    "faithfulness": {{"score": <1-5>, "reasoning": "<근거>"}},
    "fluency": {{"score": <1-5>, "reasoning": "<근거>"}},
    "overall_score": <1-5>,
    "summary": "<종합 평가 (1-2문장)>"
}}"""

CONTEXT_AWARE_JUDGE_PROMPT = """다음 생성된 답변을 평가하세요.
모델이 검색한 컨텍스트를 참고하여 답변의 품질을 평가합니다.

[질문]
{question}

[모범 답변 (Golden Answer)]
{golden_answer}

[검색된 컨텍스트 (Retrieved Context)]
{retrieved_context}

[생성된 답변 (Generated Answer)]
{generated_answer}

[평가 기준별 점수 부여 (1-5점)]

1. **정확성 (Correctness)**: 모범 답변 대비 사실적 정확도
   - 5점: 모범 답변과 동일하거나 더 정확
   - 4점: 핵심 정보 정확, 사소한 차이만 있음
   - 3점: 대체로 정확하나 일부 누락/오류
   - 2점: 상당한 누락 또는 오류 존재
   - 1점: 대부분 부정확

2. **유용성 (Helpfulness)**: 질문 해결에 도움이 되는 정도
   - 5점: 완벽한 답변 + 추가 유용 정보
   - 4점: 충분하고 실용적인 답변
   - 3점: 기본 답변 제공, 완전하지 않음
   - 2점: 부분적으로만 도움, 중요 정보 누락
   - 1점: 답변이 안 되거나 무용

3. **컨텍스트 충실성 (Context Faithfulness)**: 검색된 컨텍스트에 근거한 정도
   - 5점: 모든 정보가 검색된 컨텍스트에 명시적으로 근거함
   - 4점: 대부분 컨텍스트에 충실, 합리적 추론만 포함
   - 3점: 일부 정보가 컨텍스트에 없거나 과도한 추론
   - 2점: 상당 부분 컨텍스트와 무관하거나 지어낸 정보
   - 1점: 대부분 컨텍스트에 없는 환각 정보

4. **유창성 (Fluency)**: 문장 품질 및 자연스러움
   - 5점: 완벽하게 자연스럽고 전문적
   - 4점: 자연스럽고 읽기 쉬움
   - 3점: 이해 가능, 어색한 표현 일부
   - 2점: 문법 오류나 어색함 많음
   - 1점: 이해 어렵거나 문장 깨짐

JSON 형식으로 응답:
{{
    "correctness": {{"score": <1-5>, "reasoning": "<근거>"}},
    "helpfulness": {{"score": <1-5>, "reasoning": "<근거>"}},
    "context_faithfulness": {{"score": <1-5>, "reasoning": "<근거>"}},
    "fluency": {{"score": <1-5>, "reasoning": "<근거>"}},
    "overall_score": <1-5>,
    "context_utilization": "<검색 컨텍스트 활용 평가>",
    "summary": "<종합 평가 (1-2문장)>"
}}"""

CRITERIA_PROMPTS = {
    "correctness": CORRECTNESS_PROMPT,
    "helpfulness": HELPFULNESS_PROMPT,
    "faithfulness": FAITHFULNESS_PROMPT,
    "fluency": FLUENCY_PROMPT,
    "comprehensive": COMPREHENSIVE_JUDGE_PROMPT,
    "context_aware": CONTEXT_AWARE_JUDGE_PROMPT,
}

DEFAULT_CRITERIA = ["correctness", "helpfulness", "faithfulness", "fluency"]

CONTEXT_AWARE_CRITERIA = ["correctness", "helpfulness", "context_faithfulness", "fluency"]
