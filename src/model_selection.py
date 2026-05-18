"""
Model Selection Analysis — Legal AI Assistant
Phase 6: HELM, Artificial Analysis & LLM Arena
"""

MODEL_SELECTION_REPORT = {
    "project": "Legal AI Assistant — Agentic RAG",
    "date": "2026",
    "methodology": ["HELM LegalBench", "Artificial Analysis", "LLM Arena Elo"],

    "models_evaluated": [
        {
            "name": "GPT-4o",
            "provider": "OpenAI",
            "helm_legalbench_f1": 0.84,
            "helm_legalbench_exact_match": 0.71,
            "artificial_analysis_quality_index": 82,
            "output_speed_tokens_per_sec": 95,
            "context_window_k": 128,
            "cost_per_1m_input_usd": 5.00,
            "cost_per_1m_output_usd": 15.00,
            "ttft_seconds": 0.8,
            "llm_arena_elo": 1287,
            "recommendation": "PRIMARY — Best instruction following for structured legal output"
        },
        {
            "name": "Gemini 1.5 Pro",
            "provider": "Google AI Studio",
            "helm_legalbench_f1": 0.81,
            "helm_legalbench_exact_match": 0.68,
            "artificial_analysis_quality_index": 79,
            "output_speed_tokens_per_sec": 82,
            "context_window_k": 1000,
            "cost_per_1m_input_usd": 3.50,
            "cost_per_1m_output_usd": 10.50,
            "ttft_seconds": 1.1,
            "llm_arena_elo": 1266,
            "recommendation": "LONG CONTEXT — Use for full contract review (1M token window)"
        },
        {
            "name": "DeepSeek-V3",
            "provider": "DeepSeek",
            "helm_legalbench_f1": 0.78,
            "helm_legalbench_exact_match": 0.64,
            "artificial_analysis_quality_index": 74,
            "output_speed_tokens_per_sec": 120,
            "context_window_k": 64,
            "cost_per_1m_input_usd": 0.27,
            "cost_per_1m_output_usd": 1.10,
            "ttft_seconds": 0.6,
            "llm_arena_elo": 1245,
            "recommendation": "COST-EFFICIENT — 20x cheaper; use for batch summarization tasks"
        },
        {
            "name": "Mistral-Large",
            "provider": "Mistral AI",
            "helm_legalbench_f1": 0.76,
            "helm_legalbench_exact_match": 0.61,
            "artificial_analysis_quality_index": 71,
            "output_speed_tokens_per_sec": 88,
            "context_window_k": 128,
            "cost_per_1m_input_usd": 4.00,
            "cost_per_1m_output_usd": 12.00,
            "ttft_seconds": 0.9,
            "llm_arena_elo": 1221,
            "recommendation": "BACKUP — Competitive quality; EU data residency option"
        }
    ],

    "decision_matrix": {
        "task_primary_qa": "GPT-4o — highest F1 on LegalBench contract interpretation",
        "task_long_contract_review": "Gemini 1.5 Pro — 1M context window handles full contracts",
        "task_batch_summarization": "DeepSeek-V3 — 20x cost reduction acceptable for non-interactive",
        "task_eu_deployment": "Mistral-Large — EU infrastructure, strong GDPR compliance story",
    },

    "evaluation_criteria_weights": {
        "legal_reasoning_quality": 0.40,
        "faithfulness_no_hallucination": 0.30,
        "latency_interactive": 0.15,
        "cost_per_query": 0.15
    },

    "final_recommendation": "GPT-4o as primary for interactive Q&A (best LegalBench F1=0.84). "
                             "Gemini 1.5 Pro for full contract ingestion (1M context). "
                             "DeepSeek-V3 for high-volume batch tasks (cost $0.27/1M tokens).",

    "helm_sources": "https://crfm.stanford.edu/helm/ — LegalBench scenario",
    "artificial_analysis_sources": "https://artificialanalysis.ai",
    "llm_arena_sources": "https://lmarena.ai"
}

if __name__ == "__main__":
    import json
    from pathlib import Path
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/model_selection_report.json", "w") as f:
        json.dump(MODEL_SELECTION_REPORT, f, indent=2)
    print("Model selection report saved to outputs/model_selection_report.json")
    print(f"\n🏆 Final Recommendation:")
    print(f"   {MODEL_SELECTION_REPORT['final_recommendation']}")
