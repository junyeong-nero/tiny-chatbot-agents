#!/usr/bin/env python3
"""Main entry point for the tiny-chatbot-agents.

Usage:
    python main.py              # Interactive mode
    python main.py --query "질문"  # Single query mode
"""

import argparse
import logging
from pathlib import Path

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/agent_config.yaml") -> dict:
    """Load agent configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_llm_client(config: dict):
    """Create LLM client based on configuration.
    
    Returns a simple wrapper for OpenAI-compatible API.
    """
    llm_config = config.get("llm", {})
    
    if not llm_config.get("base_url"):
        logger.warning("LLM not configured. Answer generation disabled.")
        return None
    
    try:
        from openai import OpenAI
        
        client = OpenAI(
            base_url=llm_config["base_url"],
            api_key="not-needed",  # Local LLM doesn't need API key
        )
        
        class LLMWrapper:
            def __init__(self, client, model, temperature, max_tokens):
                self.client = client
                self.model = model
                self.temperature = temperature
                self.max_tokens = max_tokens
            
            def generate(self, messages: list[dict]) -> dict:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return {"content": response.choices[0].message.content}
        
        return LLMWrapper(
            client=client,
            model=llm_config.get("model", ""),
            temperature=llm_config.get("temperature", 0.1),
            max_tokens=llm_config.get("max_tokens", 1024),
        )
        
    except ImportError:
        logger.warning("OpenAI package not installed. LLM features disabled.")
        return None
    except Exception as e:
        logger.warning(f"Failed to create LLM client: {e}")
        return None


def create_router(config: dict):
    """Create the query router with all components."""
    from src.vectorstore import QnAVectorStore, ToSVectorStore
    from src.retrieval import QnARetriever, ToSRetriever
    from src.verifier import AnswerVerifier
    from src.feedback import FeedbackHandler
    from src.router import QueryRouter
    
    # Create LLM client
    llm_client = create_llm_client(config)
    
    # Create QnA components
    qna_config = config.get("qna", {})
    qna_store = QnAVectorStore(
        persist_directory=qna_config.get("persist_directory", "data/vectordb/qna"),
    )
    qna_retriever = QnARetriever(
        store=qna_store,
        threshold=qna_config.get("threshold", 0.85),
    )
    
    # Create ToS components
    tos_config = config.get("tos", {})
    tos_store = ToSVectorStore(
        persist_directory=tos_config.get("persist_directory", "data/vectordb/tos"),
    )
    
    # Create verifier
    verifier_config = config.get("verifier", {})
    verifier = None
    if verifier_config.get("enabled", True):
        verifier = AnswerVerifier(
            llm_client=llm_client,
            confidence_threshold=verifier_config.get("confidence_threshold", 0.7),
            require_citations=verifier_config.get("require_citations", True),
            use_llm_verification=verifier_config.get("use_llm_verification", True) and llm_client is not None,
        )
    
    tos_retriever = ToSRetriever(
        store=tos_store,
        threshold=tos_config.get("threshold", 0.7),
        llm_client=llm_client,
        verifier=verifier,
    )
    
    # Create feedback handler
    feedback_config = config.get("feedback", {})
    feedback_handler = None
    if feedback_config.get("enabled", True):
        feedback_handler = FeedbackHandler(
            qna_store=qna_store,
            duplicate_threshold=feedback_config.get("duplicate_threshold", 0.95),
            require_quality_check=feedback_config.get("require_quality_check", False),
        )
    
    # Create router
    router = QueryRouter(
        qna_retriever=qna_retriever,
        tos_retriever=tos_retriever,
        qna_threshold=qna_config.get("threshold", 0.85),
        tos_threshold=tos_config.get("threshold", 0.7),
        feedback_handler=feedback_handler,
    )
    
    return router


def handle_single_query(router, query: str) -> None:
    """Handle a single query and print the result."""
    response = router.handle_query(query)
    
    print("\n" + "=" * 60)
    print(f"질문: {response.query}")
    print("-" * 60)
    print(f"답변: {response.answer}")
    print("-" * 60)
    print(f"출처: {response.source.value}")
    print(f"신뢰도: {response.confidence:.2f}")
    if response.citations:
        print(f"참조: {', '.join(response.citations)}")
    if response.needs_human:
        print("⚠️  상담원 연결이 필요합니다")
    print("=" * 60 + "\n")


def interactive_mode(router) -> None:
    """Run the chatbot in interactive mode."""
    print("\n" + "=" * 60)
    print("🤖 tiny-chatbot-agents")
    print("약관 및 QnA CS 챗봇")
    print("=" * 60)
    print("질문을 입력하세요 (종료: 'quit' 또는 'exit')")
    print("-" * 60 + "\n")
    
    while True:
        try:
            query = input("👤 질문: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ("quit", "exit", "q"):
                print("\n👋 이용해 주셔서 감사합니다!")
                break
            
            response = router.handle_query(query)
            
            print(f"\n🤖 답변: {response.answer}")
            print(f"   [출처: {response.source.value} | 신뢰도: {response.confidence:.2f}]")
            
            if response.citations:
                print(f"   [참조: {', '.join(response.citations)}]")
            
            if response.needs_human:
                # Simulate human agent response
                print("\n📞 상담원 연결 중... (시뮬레이션)")
                human_response = input("👨‍💼 상담원 답변: ").strip()
                if human_response:
                    router.on_human_response(query, human_response)
                    print("   ✅ 답변이 QnA DB에 저장되었습니다.")
            
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 이용해 주셔서 감사합니다!")
            break
        except Exception as e:
            logger.error(f"Error handling query: {e}")
            print(f"❌ 오류가 발생했습니다: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="tiny-chatbot-agents: Local LLM 기반 CS 챗봇"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single query to process",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/agent_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if config exists
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        return 1
    
    # Load config
    config = load_config(args.config)
    
    # Create router
    logger.info("Initializing chatbot...")
    router = create_router(config)
    logger.info("Chatbot initialized successfully")
    
    # Handle query
    if args.query:
        handle_single_query(router, args.query)
    else:
        interactive_mode(router)
    
    return 0


if __name__ == "__main__":
    exit(main())
