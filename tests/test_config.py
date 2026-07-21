"""Unit tests for src/config.py — all tunable parameters and path constants."""

from pathlib import Path

from src import config


class TestPathConstants:
    """Verify all path constants resolve to valid absolute paths under PROJECT_ROOT."""

    def test_project_root_is_absolute(self):
        assert isinstance(config.PROJECT_ROOT, Path)
        assert config.PROJECT_ROOT.is_absolute()

    def test_data_dir_resolves_under_root(self):
        assert config.DATA_DIR == config.PROJECT_ROOT / "data"

    def test_store_dir_resolves_under_root(self):
        assert config.STORE_DIR == config.PROJECT_ROOT / "store"

    def test_quiver_store_path_under_store(self):
        assert config.QUIVER_STORE_PATH == config.STORE_DIR / "quiver"

    def test_bm25_state_path_under_store(self):
        assert config.BM25_STATE_PATH == config.STORE_DIR / "bm25.json"

    def test_ingest_state_path_under_store(self):
        assert config.INGEST_STATE_PATH == config.STORE_DIR / "ingest_state.json"

    def test_eval_queries_path_under_data(self):
        assert config.EVAL_QUERIES_PATH == config.DATA_DIR / "eval_queries.json"


class TestNumericDefaults:
    """Verify all numeric constants match expected values."""

    def test_chunk_size_default(self):
        assert config.CHUNK_SIZE == 512

    def test_chunk_overlap_default(self):
        assert config.CHUNK_OVERLAP == 64

    def test_top_k_default(self):
        assert config.TOP_K == 3

    def test_embedding_dim_default(self):
        assert config.EMBEDDING_DIM == 768

    def test_chunk_size_pdf(self):
        assert config.CHUNK_SIZE_PDF == 1024

    def test_chunk_overlap_pdf(self):
        assert config.CHUNK_OVERLAP_PDF == 128

    def test_embed_batch_size(self):
        assert config.EMBED_BATCH_SIZE == 32

    def test_top_k_rerank(self):
        assert config.TOP_K_RERANK == 30

    def test_transform_min_length(self):
        assert config.TRANSFORM_MIN_LENGTH == 5

    def test_transform_min_entities(self):
        assert config.TRANSFORM_MIN_ENTITIES == 1

    def test_hyde_fusion_top_k(self):
        assert config.HYDE_FUSION_TOP_K == 10

    def test_hyde_fusion_rrf_k(self):
        assert config.HYDE_FUSION_RRF_K == 60


class TestStringConstants:
    """Verify all string constants match expected values."""

    def test_ollama_embedding_model(self):
        assert config.OLLAMA_EMBEDDING_MODEL == "nomic-embed-text"

    def test_ollama_gen_model(self):
        assert config.OLLAMA_GEN_MODEL == "gemma3:1b"

    def test_ollama_base_url(self):
        assert config.OLLAMA_BASE_URL == "http://localhost:11434"

    def test_reranker_model(self):
        assert config.RERANKER_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_query_transform_mode_default(self):
        assert config.QUERY_TRANSFORM_MODE == "none"


class TestTupleListConstants:
    """Verify tuple and list constants have expected structure and values."""

    def test_supported_extensions_is_tuple(self):
        assert isinstance(config.SUPPORTED_EXTENSIONS, tuple)

    def test_supported_extensions_values(self):
        assert config.SUPPORTED_EXTENSIONS == (".txt", ".pdf")

    def test_chunk_separators_is_list(self):
        assert isinstance(config.CHUNK_SEPARATORS, list)

    def test_chunk_separators_values(self):
        assert config.CHUNK_SEPARATORS == ["\n\n", "\n", ". ", " ", ""]

    def test_chunk_separators_pdf_values(self):
        assert config.CHUNK_SEPARATORS_PDF == ["\n\n", "\n", " ", ""]


class TestFloatConstants:
    """Verify all float constants match expected values."""

    def test_hybrid_dense_weight(self):
        assert config.HYBRID_DENSE_WEIGHT == 0.5

    def test_hybrid_sparse_weight(self):
        assert config.HYBRID_SPARSE_WEIGHT == 0.5

    def test_bm25_k1(self):
        assert config.BM25_K1 == 1.5

    def test_bm25_b(self):
        assert config.BM25_B == 0.75


class TestPromptConstants:
    """Verify prompt templates exist and contain expected placeholders."""

    def test_hyde_prompt_en_has_placeholder(self):
        assert "{question}" in config.HYDE_PROMPT_EN

    def test_hyde_prompt_ar_has_placeholder(self):
        assert "{question}" in config.HYDE_PROMPT_AR

    def test_hyde_prompt_fr_has_placeholder(self):
        assert "{question}" in config.HYDE_PROMPT_FR

    def test_rewrite_prompt_en_has_placeholder(self):
        assert "{question}" in config.REWRITE_PROMPT_EN

    def test_rewrite_prompt_ar_has_placeholder(self):
        assert "{question}" in config.REWRITE_PROMPT_AR

    def test_rewrite_prompt_fr_has_placeholder(self):
        assert "{question}" in config.REWRITE_PROMPT_FR

    def test_all_prompts_have_content(self):
        prompts = [
            config.HYDE_PROMPT_EN,
            config.HYDE_PROMPT_AR,
            config.HYDE_PROMPT_FR,
            config.REWRITE_PROMPT_EN,
            config.REWRITE_PROMPT_AR,
            config.REWRITE_PROMPT_FR,
        ]
        for prompt in prompts:
            assert len(prompt) > 50, f"Prompt too short: {prompt[:30]}..."


class TestHybridWeightsSum:
    """Verify hybrid dense and sparse weights sum to 1.0."""

    def test_hybrid_weights_sum_to_one(self):
        total = config.HYBRID_DENSE_WEIGHT + config.HYBRID_SPARSE_WEIGHT
        assert abs(total - 1.0) < 1e-9
