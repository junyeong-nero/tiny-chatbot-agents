# 데이터 크롤러

`playwright` 기반 크롤러로 QnA FAQ와 이용약관(ToS) 원시 데이터를 수집하고, 벡터 DB에 적재하는 워크플로를 설명합니다.

---

## 크롤러 구성

### QnA 크롤러 (`src/crawlers/qna_crawler.py`)

- **대상**: FAQ 페이지
- **방식**: CSS 셀렉터로 질문/답변 쌍 추출 (아코디언, 리스트 등 다양한 UI 패턴 지원)
- **출력 형식**: `data/raw/qna/` 아래 JSON 파일

```json
[
  {
    "question": "비밀번호를 변경하고 싶어요",
    "answer": "앱 > 설정 > 보안에서 변경 가능합니다.",
    "category": "보안",
    "sub_category": "계정",
    "source_url": "https://..."
  }
]
```

### ToS 크롤러 (`src/crawlers/tos_crawler.py`)

- **대상**: 이용약관 페이지
- **방식**: 문서 계층 구조(제목 → 조 → 항) 파싱
- **출력 형식**: `data/raw/tos/` 아래 JSON 파일

```json
[
  {
    "document_title": "서비스이용약관",
    "section_title": "제1조 (목적)",
    "section_content": "본 약관은...",
    "category": "서비스약관"
  }
]
```

---

## 데이터 적재 파이프라인

```
[1] 크롤링                     data/raw/ 에 JSON 저장
     python main.py crawl qna
     python main.py crawl tos
         │
         ▼
[2] (선택) 데이터 정제
     수동 검토 또는 자체 스크립트로 품질 필터링
         │
         ▼
[3] 벡터 DB 적재               data/vectordb/ 에 ChromaDB 저장
     python main.py ingest-qna
     python main.py ingest-tos
```

---

## CLI 명령 상세

### 크롤링

```bash
# QnA만 크롤링
python main.py crawl qna

# ToS만 크롤링
python main.py crawl tos

# 전체 크롤링
python main.py crawl all

# 특정 카테고리만, 브라우저 표시
python main.py crawl qna --categories CARD,LOAN --visible
```

### 벡터 DB 적재

```bash
# QnA 적재
python main.py ingest-qna

# 특정 파일로 QnA 적재, 기존 데이터 초기화 후 검색 확인
python main.py ingest-qna \
  --file data/raw/qna/example.json \
  --clear \
  --search "비밀번호"

# ToS 적재
python main.py ingest-tos

# 특정 파일로 ToS 적재, 카테고리 지정 후 검색 확인
python main.py ingest-tos \
  --file data/raw/tos/example.json \
  --category 약관 \
  --search "제1조"
```

### 주요 ingest 옵션

| 옵션 | 설명 |
|------|------|
| `--file <경로>` | 특정 JSON 파일 지정 (기본: `data/raw/` 전체) |
| `--clear` | 적재 전 기존 컬렉션 초기화 |
| `--search <쿼리>` | 적재 후 검색 결과 확인 |
| `--top-k <N>` | 검색 결과 수 (기본: 5) |
| `--category <이름>` | ToS 카테고리 레이블 지정 |

---

## 디렉토리 구조

```
data/
├── raw/
│   ├── qna/          # 크롤링된 QnA JSON
│   └── tos/          # 크롤링된 ToS JSON
└── vectordb/
    ├── qna/          # QnA ChromaDB 파일
    └── tos/          # ToS ChromaDB 파일
```
