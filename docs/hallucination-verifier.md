# Hallucination Verifier (AnswerVerifier)

RAG 파이프라인에서 생성된 답변의 신뢰성을 검증하는 3계층 방어 시스템입니다.

## 개요

LLM이 생성한 답변이 실제 컨텍스트(약관, FAQ 등)에 기반하는지 검증하여 환각(hallucination)을 방지합니다.

```
┌─────────────────────────────────────────────────────────┐
│                    AnswerVerifier                       │
├─────────────────────────────────────────────────────────┤
│  Layer 1: Citation Check (출처 인용 검증)               │
│     ↓                                                   │
│  Layer 2: Rule-based Check (패턴 기반 환각 감지)        │
│     ↓                                                   │
│  Layer 3: LLM-based Check (LLM 심층 검증)              │
│     ↓                                                   │
│  → VerificationResult (검증 결과)                       │
└─────────────────────────────────────────────────────────┘
```

## 3계층 방어 시스템

### Layer 1: Citation Check (출처 인용 검증)

답변에 포함된 출처 인용이 실제 컨텍스트에 존재하는지 확인합니다.

**작동 방식:**
1. 정규식으로 `[참조: ...]` 또는 `[출처: ...]` 패턴 추출
2. 컨텍스트의 section_title과 비교
3. "제N조" 형식의 조항 번호도 별도로 매칭

**예시:**
```
답변: "... [참조: 제3조 2항]"
컨텍스트: [{"section_title": "제3조 이용조건", ...}]
→ 매칭 성공 (citations_valid = True)
```

### Layer 2: Rule-based Check (패턴 기반 환각 감지)

환각을 나타낼 수 있는 불확실한 표현을 패턴 매칭으로 감지합니다.

**감지 패턴:**
- `일반적으로`
- `보통`
- `아마도`
- `~할 수도 있습니다`
- `제 생각에는`
- `추측컨대`

**추가 검사:**
- 500자 이상의 긴 답변에 출처가 없는 경우 경고

### Layer 3: LLM-based Check (LLM 심층 검증)

LLM을 사용하여 답변이 컨텍스트에 충실한지 심층 검증합니다.

**작동 방식:**
1. 컨텍스트, 질문, 답변을 검증 프롬프트에 포함
2. LLM에게 JSON 형식으로 검증 결과 요청
3. 응답에서 JSON 파싱 (마크다운 코드블록 등 다양한 형식 지원)

**검증 프롬프트:**
```
당신은 답변 검증 전문가입니다.
주어진 답변이 제공된 컨텍스트(약관/FAQ)에 정확히 기반하는지 검증합니다.

검증 기준:
1. 답변의 모든 사실이 컨텍스트에 명시적으로 존재하는가?
2. 컨텍스트에 없는 정보를 지어내지 않았는가?
3. 인용/참조가 정확한가?
```

## 신뢰도 점수 계산

각 계층의 결과를 종합하여 0.0 ~ 1.0 사이의 신뢰도 점수를 계산합니다.

```python
confidence = 1.0

# 출처 인용 문제 시 -0.3
if not citations_valid:
    confidence -= 0.3

# 환각 패턴 1개당 -0.15
confidence -= len(hallucination_signs) * 0.15

# LLM 검증 결과와 평균 (50% 가중치)
if llm_result:
    llm_confidence = llm_result.get("confidence", 0.5)
    confidence = (confidence + llm_confidence) / 2
```

**기본 임계값:** 0.7 (이 값 미만이면 검증 실패)

## 사용법

### 기본 사용

```python
from src.verifier import AnswerVerifier

verifier = AnswerVerifier(
    llm_client=llm_client,           # LLM 클라이언트 (Optional)
    confidence_threshold=0.7,        # 신뢰도 임계값
    require_citations=True,          # 출처 인용 필수 여부
    use_llm_verification=True,       # LLM 검증 사용 여부
)

result = verifier.verify(
    question="환불 규정이 어떻게 되나요?",
    answer="환불은 7일 이내 가능합니다. [참조: 제5조]",
    context=[
        {"section_title": "제5조 환불규정", "section_content": "..."}
    ]
)

print(result.verified)      # True/False
print(result.confidence)    # 0.0 ~ 1.0
print(result.issues)        # 발견된 문제점 목록
print(result.reasoning)     # 판단 근거
```

### 빠른 검증 (LLM 없이)

LLM 호출 없이 규칙 기반으로만 빠르게 검증합니다.

```python
is_valid = verifier.quick_verify(answer, context)
```

## VerificationResult 구조

```python
@dataclass
class VerificationResult:
    verified: bool           # 최종 검증 통과 여부
    confidence: float        # 신뢰도 점수 (0.0 ~ 1.0)
    issues: list[str]        # 발견된 문제점 목록
    reasoning: str           # 사람이 읽을 수 있는 판단 근거
    citations_valid: bool    # 출처 인용 유효성
```

**reasoning 예시:**
```
✓ 출처 인용이 유효합니다 | ✓ 불확실한 표현이 감지되지 않았습니다 | ✓ LLM 검증 통과
```

## 설계 철학

1. **다층 방어**: 단일 검증 방식에 의존하지 않고 여러 계층으로 검증
2. **빠른 실패**: 비용이 낮은 검사(규칙 기반)를 먼저 수행
3. **유연성**: LLM 없이도 기본 검증 가능 (`quick_verify`)
4. **투명성**: 검증 실패 이유를 명확하게 제공
