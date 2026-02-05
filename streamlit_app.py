#!/usr/bin/env python3
"""Streamlit UI for RAG Pipeline.

Usage:
    streamlit run streamlit_app.py
"""

import os
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import streamlit as st

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def get_available_providers() -> list[str]:
    """Get list of available LLM providers based on environment."""
    providers = []

    if os.getenv("OPENAI_API_KEY"):
        providers.append("openai")
    if os.getenv("VLLM_API_BASE"):
        providers.append("vllm")
    if os.getenv("SGLANG_API_BASE"):
        providers.append("sglang")

    # ollama is typically available locally
    providers.append("ollama")

    return providers if providers else ["openai", "vllm", "sglang", "ollama"]


def load_pipeline(
    provider: str,
    model: str | None,
    qna_db_path: str,
    tos_db_path: str,
    enable_verification: bool,
    verification_threshold: float,
):
    """Load RAG Pipeline."""
    import traceback

    from src.llm import create_llm_client
    from src.pipeline import RAGPipeline

    try:
        llm_kwargs = {}
        if model:
            llm_kwargs["model"] = model

        llm = create_llm_client(provider=provider, **llm_kwargs)

        pipeline = RAGPipeline(
            llm=llm,
            qna_db_path=qna_db_path,
            tos_db_path=tos_db_path,
            enable_verification=enable_verification,
            verification_threshold=verification_threshold,
        )

        return pipeline
    except Exception as e:
        st.error(f"파이프라인 로드 중 에러: {e}")
        st.code(traceback.format_exc())
        raise


def render_sidebar():
    """Render sidebar with settings."""
    st.sidebar.title("⚙️ 설정")

    # LLM Settings
    st.sidebar.subheader("LLM 설정")

    providers = get_available_providers()
    default_provider = os.getenv("LLM_PROVIDER", providers[0] if providers else "openai")

    provider = st.sidebar.selectbox(
        "LLM Provider",
        options=providers,
        index=providers.index(default_provider) if default_provider in providers else 0,
        help="사용할 LLM 서비스를 선택하세요",
    )

    model = st.sidebar.text_input(
        "Model Name (optional)",
        value="",
        help="특정 모델을 사용하려면 입력하세요",
    )

    # Verification Settings
    st.sidebar.subheader("검증 설정")

    enable_verification = st.sidebar.checkbox(
        "할루시네이션 검증 활성화",
        value=False,
        help="응답의 사실 여부를 검증합니다 (처리 시간 증가)",
    )

    verification_threshold = st.sidebar.slider(
        "검증 임계값",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="이 값 미만이면 검증 실패로 표시",
        disabled=not enable_verification,
    )

    # Database Paths
    st.sidebar.subheader("데이터베이스")

    qna_db_path = st.sidebar.text_input(
        "QnA DB 경로",
        value="data/vectordb/qna",
    )

    tos_db_path = st.sidebar.text_input(
        "ToS DB 경로",
        value="data/vectordb/tos",
    )

    return {
        "provider": provider,
        "model": model if model else None,
        "qna_db_path": qna_db_path,
        "tos_db_path": tos_db_path,
        "enable_verification": enable_verification,
        "verification_threshold": verification_threshold,
    }


def render_response_info(response):
    """Render response metadata in expandable section."""
    source_emoji = {
        "qna": "📚",
        "tos": "📜",
        "no_context": "❓",
    }

    source_label = {
        "qna": "QnA 데이터베이스",
        "tos": "약관 데이터베이스",
        "no_context": "컨텍스트 없음",
    }

    source_value = response.source.value

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="출처",
            value=f"{source_emoji.get(source_value, '📄')} {source_label.get(source_value, source_value)}",
        )

    with col2:
        st.metric(
            label="신뢰도",
            value=f"{response.confidence:.0%}",
        )

    with col3:
        verification_status = "✅ 검증됨" if response.verified else "⚠️ 검증 실패"
        st.metric(
            label="검증 상태",
            value=verification_status,
            delta=f"{response.verification_score:.0%}",
        )

    # Citations
    if response.citations:
        st.info(f"📎 참조: {', '.join(response.citations)}")

    # Verification issues
    if response.verification_issues:
        with st.expander("⚠️ 검증 경고", expanded=True):
            for issue in response.verification_issues:
                st.warning(issue)

    # Context details
    if response.context:
        with st.expander("📄 참조된 컨텍스트"):
            for i, ctx in enumerate(response.context, 1):
                st.markdown(f"**[{i}]** {ctx.get('document_title', ctx.get('question', 'N/A'))}")
                content = ctx.get("section_content", ctx.get("answer", ""))
                if len(content) > 500:
                    content = content[:500] + "..."
                st.text(content)
                st.divider()

    # Metadata
    if response.metadata:
        with st.expander("🔧 메타데이터"):
            if response.metadata.get("verification_reasoning"):
                st.markdown(f"**검증 상세**: {response.metadata['verification_reasoning']}")
            if response.metadata.get("tokens_used"):
                st.markdown(f"**토큰 사용량**: {response.metadata['tokens_used']}")


def render_search_results(results: list[dict], search_type: str):
    """Render search results."""
    if not results:
        st.warning("검색 결과가 없습니다.")
        return

    st.success(f"검색 결과: {len(results)}개")

    for i, r in enumerate(results, 1):
        score = r.get("score", 0)
        score_color = "green" if score >= 0.8 else "orange" if score >= 0.6 else "red"

        with st.expander(
            f"[{i}] Score: :{score_color}[{score:.3f}]",
            expanded=i == 1,
        ):
            if search_type == "qna":
                st.markdown(f"**Q:** {r.get('question', 'N/A')}")
                st.markdown(f"**A:** {r.get('answer', 'N/A')}")
            else:
                st.markdown(f"**문서:** {r.get('document_title', 'N/A')}")
                st.markdown(f"**섹션:** {r.get('section_title', 'N/A')}")
                st.markdown(f"**내용:**")
                content = r.get("section_content", "")
                if len(content) > 500:
                    content = content[:500] + "..."
                st.text(content)


def load_evaluation_dataset(path: str) -> list[dict]:
    """Load evaluation dataset from JSON file."""
    import json

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("evaluation dataset 형식이 올바르지 않습니다. (list 필요)")

    cases = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if not item.get("question") or not item.get("expected_answer"):
            continue
        cases.append(item)

    return cases


def compute_lightweight_eval(expected: str, generated: str) -> dict[str, float | bool]:
    """Compute lightweight comparison metrics without heavy model dependencies."""
    from src.evaluation.evaluator import LLMEvaluator

    evaluator = LLMEvaluator(embedding_model=None, verifier=None)
    bleu = evaluator.compute_bleu(expected, generated)
    normalized_expected = expected.strip()
    normalized_generated = generated.strip()

    return {
        "bleu": bleu,
        "exact_match": normalized_expected == normalized_generated,
        "string_similarity": SequenceMatcher(None, normalized_expected, normalized_generated).ratio(),
    }


def format_eval_case_label(case: dict) -> str:
    """Format evaluation case label for selection UI."""
    case_id = case.get("id", "no-id")
    category = case.get("category", "uncategorized")
    question = case.get("question", "")
    return f"[{case_id}] ({category}) {question}"


def main():
    st.set_page_config(
        page_title="RAG 챗봇",
        page_icon="🤖",
        layout="wide",
    )

    st.title("🤖 RAG 파이프라인 챗봇")
    st.caption("QnA 및 약관 데이터베이스 기반 고객 서비스 챗봇")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pipeline_loaded" not in st.session_state:
        st.session_state.pipeline_loaded = False
    if "eval_dataset" not in st.session_state:
        st.session_state.eval_dataset = []
    if "eval_dataset_path" not in st.session_state:
        st.session_state.eval_dataset_path = "data/evaluation/evaluation_dataset.json"

    # Render sidebar and get settings
    settings = render_sidebar()

    # Mode selection tabs
    tab_chat, tab_qna_search, tab_tos_search, tab_eval = st.tabs([
        "💬 채팅",
        "🔍 QnA 검색",
        "📜 약관 검색",
        "🧪 Evaluation 테스트",
    ])

    # Load pipeline button in sidebar
    if st.sidebar.button("🔄 파이프라인 로드", use_container_width=True):
        try:
            with st.spinner("파이프라인을 로드하는 중..."):
                pipeline = load_pipeline(
                    provider=settings["provider"],
                    model=settings["model"],
                    qna_db_path=settings["qna_db_path"],
                    tos_db_path=settings["tos_db_path"],
                    enable_verification=settings["enable_verification"],
                    verification_threshold=settings["verification_threshold"],
                )
                st.session_state.pipeline = pipeline
                st.session_state.pipeline_loaded = True
                st.sidebar.success("파이프라인 로드 완료!")
                st.sidebar.info(f"QnA: {pipeline.qna_store.count()}개 | ToS: {pipeline.tos_store.count()}개")
        except Exception as e:
            st.sidebar.error(f"로드 실패: {e}")
            st.session_state.pipeline_loaded = False

    # Show pipeline status
    if st.session_state.pipeline_loaded:
        st.sidebar.success("✅ 파이프라인 활성화됨")
    else:
        st.sidebar.warning("⚠️ 파이프라인을 먼저 로드하세요")

    # Chat tab
    with tab_chat:
        if not st.session_state.pipeline_loaded:
            st.info("👈 사이드바에서 파이프라인을 먼저 로드하세요.")
        else:
            # Chat history in scrollable container
            chat_container = st.container(height=500)

            with chat_container:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                        if message["role"] == "assistant" and "response_data" in message:
                            render_response_info(message["response_data"])

            # Chat input (outside container, stays at bottom)
            if prompt := st.chat_input("질문을 입력하세요..."):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})

                # Generate response
                with st.spinner("답변 생성 중..."):
                    try:
                        response = st.session_state.pipeline.query(prompt)

                        # Save to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response.answer,
                            "response_data": response,
                        })
                    except Exception as e:
                        error_msg = f"오류가 발생했습니다: {e}"
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                        })

                st.rerun()

            # Clear chat button
            if st.button("🗑️ 대화 내역 지우기"):
                st.session_state.messages = []
                st.rerun()

    # QnA Search tab
    with tab_qna_search:
        if not st.session_state.pipeline_loaded:
            st.info("👈 사이드바에서 파이프라인을 먼저 로드하세요.")
        else:
            st.subheader("QnA 데이터베이스 검색")

            col1, col2 = st.columns([3, 1])
            with col1:
                qna_query = st.text_input(
                    "검색어",
                    key="qna_search_input",
                    placeholder="검색할 키워드를 입력하세요...",
                )
            with col2:
                qna_n_results = st.number_input(
                    "결과 수",
                    min_value=1,
                    max_value=20,
                    value=5,
                    key="qna_n_results",
                )

            if st.button("🔍 QnA 검색", key="qna_search_btn"):
                if qna_query:
                    with st.spinner("검색 중..."):
                        results = st.session_state.pipeline.search_qna(
                            qna_query, n_results=qna_n_results
                        )
                        render_search_results(results, "qna")
                else:
                    st.warning("검색어를 입력하세요.")

    # ToS Search tab
    with tab_tos_search:
        if not st.session_state.pipeline_loaded:
            st.info("👈 사이드바에서 파이프라인을 먼저 로드하세요.")
        else:
            st.subheader("약관 데이터베이스 검색")

            col1, col2 = st.columns([3, 1])
            with col1:
                tos_query = st.text_input(
                    "검색어",
                    key="tos_search_input",
                    placeholder="검색할 키워드를 입력하세요...",
                )
            with col2:
                tos_n_results = st.number_input(
                    "결과 수",
                    min_value=1,
                    max_value=20,
                    value=5,
                    key="tos_n_results",
                )

            if st.button("🔍 약관 검색", key="tos_search_btn"):
                if tos_query:
                    with st.spinner("검색 중..."):
                        results = st.session_state.pipeline.search_tos(
                            tos_query, n_results=tos_n_results
                        )
                        render_search_results(results, "tos")
                else:
                    st.warning("검색어를 입력하세요.")

    # Evaluation tab
    with tab_eval:
        st.subheader("Evaluation Query 테스트")

        dataset_path = st.text_input(
            "Evaluation Dataset 경로",
            value=st.session_state.eval_dataset_path,
            help="question/expected_answer가 포함된 JSON list 파일",
        )
        st.session_state.eval_dataset_path = dataset_path

        col_load, col_info = st.columns([1, 3])
        with col_load:
            if st.button("📥 Dataset 로드", key="eval_dataset_load_btn", use_container_width=True):
                try:
                    cases = load_evaluation_dataset(dataset_path)
                    st.session_state.eval_dataset = cases
                    st.success(f"{len(cases)}개 테스트 케이스 로드 완료")
                except Exception as e:
                    st.error(f"Dataset 로드 실패: {e}")
                    st.session_state.eval_dataset = []
        with col_info:
            st.caption(f"현재 로드된 케이스 수: {len(st.session_state.eval_dataset)}")

        if not st.session_state.pipeline_loaded:
            st.info("👈 사이드바에서 파이프라인을 먼저 로드하세요.")
        elif not st.session_state.eval_dataset:
            st.info("먼저 Evaluation dataset을 로드하세요.")
        else:
            all_cases = st.session_state.eval_dataset
            categories = sorted({c.get("category", "uncategorized") for c in all_cases})
            selected_categories = st.multiselect(
                "카테고리 필터",
                options=categories,
                default=categories,
                key="eval_category_filter",
            )

            filtered_cases = [
                c for c in all_cases if c.get("category", "uncategorized") in selected_categories
            ]
            st.caption(f"필터 결과: {len(filtered_cases)}개")

            if not filtered_cases:
                st.warning("필터 결과가 없습니다.")
            else:
                labels = [format_eval_case_label(c) for c in filtered_cases]
                selected_idx = st.selectbox(
                    "테스트할 질문 선택",
                    options=list(range(len(filtered_cases))),
                    format_func=lambda i: labels[i],
                    key="eval_case_select",
                )

                selected_case = filtered_cases[selected_idx]

                st.markdown("**Expected Answer**")
                st.write(selected_case.get("expected_answer", ""))

                col_run_one, col_run_batch = st.columns(2)
                with col_run_one:
                    run_one = st.button("▶️ 선택 질문 실행", key="eval_run_one_btn", use_container_width=True)
                with col_run_batch:
                    batch_size = st.number_input(
                        "일괄 실행 수",
                        min_value=1,
                        max_value=min(20, len(filtered_cases)),
                        value=min(5, len(filtered_cases)),
                        key="eval_batch_size",
                    )
                    run_batch = st.button(
                        "🚀 일괄 실행",
                        key="eval_run_batch_btn",
                        use_container_width=True,
                    )

                if run_one:
                    with st.spinner("질문 실행 중..."):
                        start = time.perf_counter()
                        response = st.session_state.pipeline.query(selected_case["question"])
                        latency_ms = (time.perf_counter() - start) * 1000
                        metrics = compute_lightweight_eval(
                            selected_case.get("expected_answer", ""),
                            response.answer,
                        )

                    st.markdown("**Generated Answer**")
                    st.write(response.answer)
                    render_response_info(response)

                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    col_m1.metric("BLEU", f"{metrics['bleu']:.3f}")
                    col_m2.metric("문자열 유사도", f"{metrics['string_similarity']:.3f}")
                    col_m3.metric("Exact Match", "Yes" if metrics["exact_match"] else "No")
                    col_m4.metric("Latency", f"{latency_ms:.1f} ms")

                if run_batch:
                    with st.spinner("일괄 실행 중..."):
                        rows = []
                        for case in filtered_cases[:batch_size]:
                            start = time.perf_counter()
                            response = st.session_state.pipeline.query(case["question"])
                            latency_ms = (time.perf_counter() - start) * 1000
                            metrics = compute_lightweight_eval(
                                case.get("expected_answer", ""),
                                response.answer,
                            )
                            rows.append({
                                "id": case.get("id", ""),
                                "category": case.get("category", ""),
                                "question": case.get("question", ""),
                                "source": response.source.value,
                                "confidence": round(response.confidence, 3),
                                "bleu": round(metrics["bleu"], 3),
                                "string_similarity": round(metrics["string_similarity"], 3),
                                "exact_match": metrics["exact_match"],
                                "latency_ms": round(latency_ms, 1),
                            })

                    st.success(f"{len(rows)}개 실행 완료")
                    st.dataframe(rows, use_container_width=True)


if __name__ == "__main__":
    main()
