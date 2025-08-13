"""Tests for GenAI Entry Detection in Ollama instrumentation."""
import os
import pytest
from opentelemetry import trace
from opentelemetry.semconv_ai.genai_entry import (
    is_genai_entry_enabled,
    GENAI_ENTRY_ATTRIBUTE,
    GENAI_ENTRY_ENV_VAR,
    _reset_thread_local_state,
    _circuit_breaker,
    _circuit_breaker_lock,
    with_genai_entry_detection,
)


class TestOllamaGenAIEntryDetection:
    """Test GenAI Entry Detection for basic Ollama operations."""

    def setup_method(self):
        """Reset state before each test."""
        _reset_thread_local_state()
        with _circuit_breaker_lock:
            _circuit_breaker['failures'] = 0
            _circuit_breaker['is_open'] = False
            _circuit_breaker['last_failure_time'] = 0

    def test_genai_entry_env_var_enabled_by_default(self):
        """Test that GenAI entry detection is enabled by default."""
        # Remove environment variable if it exists
        original_value = os.environ.pop(GENAI_ENTRY_ENV_VAR, None)
        try:
            assert is_genai_entry_enabled() is True
        finally:
            if original_value:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value

    def test_genai_entry_env_var_can_be_disabled(self):
        """Test that GenAI entry detection can be disabled via environment variable."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            os.environ[GENAI_ENTRY_ENV_VAR] = "false"
            assert is_genai_entry_enabled() is False
        finally:
            if original_value:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)

    @pytest.mark.vcr
    def test_ollama_chat_has_genai_entry_attribute(
        self, ollama_client, instrument_legacy, span_exporter, tracer_provider
    ):
        """Test that ollama chat operation gets GenAI entry attribute."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            os.environ[GENAI_ENTRY_ENV_VAR] = "true"
            trace.set_tracer_provider(tracer_provider)
            ollama_client.chat(
                model="gemma3:1b",
                messages=[
                    {"role": "user", "content": "Say hello in one word"}
                ]
            )
            spans = span_exporter.get_finished_spans()
            assert len(spans) >= 1
            chat_span = next((span for span in spans if "ollama.chat" in span.name), None)
            assert chat_span is not None
            assert chat_span.attributes.get(GENAI_ENTRY_ATTRIBUTE) is True
        finally:
            _reset_thread_local_state()
            if original_value:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)

    @pytest.mark.vcr
    def test_ollama_generation_has_genai_entry_attribute(
        self, ollama_client, instrument_legacy, span_exporter, tracer_provider
    ):
        """Test that ollama generation operation gets GenAI entry attribute."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            os.environ[GENAI_ENTRY_ENV_VAR] = "true"
            trace.set_tracer_provider(tracer_provider)
            ollama_client.generate(
                model="gemma3:1b",
                prompt="Say hello"
            )
            spans = span_exporter.get_finished_spans()
            assert len(spans) >= 1
            generation_span = next((span for span in spans if "ollama.completion" in span.name), None)
            assert generation_span is not None
            assert generation_span.attributes.get(GENAI_ENTRY_ATTRIBUTE) is True
        finally:
            _reset_thread_local_state()
            if original_value:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)

    @pytest.mark.vcr
    def test_ollama_embeddings_has_genai_entry_attribute(
        self, ollama_client, instrument_legacy, span_exporter, tracer_provider
    ):
        """Test that ollama embeddings operation gets GenAI entry attribute."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            os.environ[GENAI_ENTRY_ENV_VAR] = "true"
            trace.set_tracer_provider(tracer_provider)
            ollama_client.embeddings(
                model="nomic-embed-text:latest",
                prompt="Hello world"
            )
            spans = span_exporter.get_finished_spans()
            assert len(spans) >= 1
            embedding_span = next((span for span in spans if "ollama.embeddings" in span.name), None)
            assert embedding_span is not None
            assert embedding_span.attributes.get(GENAI_ENTRY_ATTRIBUTE) is True
        finally:
            _reset_thread_local_state()
            if original_value:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)

    @pytest.mark.vcr
    def test_ollama_no_genai_entry_when_disabled(
        self, ollama_client, instrument_legacy, span_exporter, tracer_provider
    ):
        """Test that ollama operations don't get GenAI entry attribute when disabled."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            os.environ[GENAI_ENTRY_ENV_VAR] = "false"
            trace.set_tracer_provider(tracer_provider)
            ollama_client.chat(
                model="gemma3:1b",
                messages=[
                    {"role": "user", "content": "Say hello in one word"}
                ]
            )
            spans = span_exporter.get_finished_spans()
            assert len(spans) >= 1
            chat_span = next((span for span in spans if "ollama.chat" in span.name), None)
            assert chat_span is not None
            assert GENAI_ENTRY_ATTRIBUTE not in chat_span.attributes
        finally:
            _reset_thread_local_state()
            if original_value:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)

    @pytest.mark.vcr
    @pytest.mark.asyncio
    async def test_ollama_async_chat_has_genai_entry_attribute(
        self, ollama_client_async, instrument_legacy, span_exporter, tracer_provider
    ):
        """Test that ollama async chat operation gets GenAI entry attribute."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            os.environ[GENAI_ENTRY_ENV_VAR] = "true"
            trace.set_tracer_provider(tracer_provider)
            await ollama_client_async.chat(
                model="gemma3:1b",
                messages=[
                    {"role": "user", "content": "Say hello in one word"}
                ]
            )
            spans = span_exporter.get_finished_spans()
            assert len(spans) >= 1
            chat_span = next((span for span in spans if "ollama.chat" in span.name), None)
            assert chat_span is not None
            assert chat_span.attributes.get(GENAI_ENTRY_ATTRIBUTE) is True
        finally:
            _reset_thread_local_state()
            if original_value:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)


class TestOllamaGenAIEntryDetectionAdvanced:
    """Test advanced GenAI Entry Detection scenarios."""

    def setup_method(self):
        """Reset state before each test."""
        _reset_thread_local_state()
        with _circuit_breaker_lock:
            _circuit_breaker['failures'] = 0
            _circuit_breaker['is_open'] = False
            _circuit_breaker['last_failure_time'] = 0

    @pytest.mark.vcr
    def test_ollama_streaming_chat_has_genai_entry_attribute(
        self, ollama_client, instrument_legacy, span_exporter, tracer_provider
    ):
        """Test that ollama streaming chat operation gets GenAI entry attribute."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            os.environ[GENAI_ENTRY_ENV_VAR] = "true"
            trace.set_tracer_provider(tracer_provider)
            response = ollama_client.chat(
                model="gemma3:1b",
                messages=[
                    {"role": "user", "content": "Say hello in one word"}
                ],
                stream=True
            )
            # Consume the streaming response
            full_response = ""
            for chunk in response:
                full_response += chunk.get("message", {}).get("content", "")
            spans = span_exporter.get_finished_spans()
            assert len(spans) >= 1
            chat_span = next((span for span in spans if "ollama.chat" in span.name), None)
            assert chat_span is not None
            assert chat_span.attributes.get(GENAI_ENTRY_ATTRIBUTE) is True
        finally:
            _reset_thread_local_state()
            if original_value:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)

    def test_nested_genai_operations_only_outer_marked_as_entry(
        self, span_exporter, tracer_provider
    ):
        """Test that in nested GenAI operations, only the outer operation is marked as entry."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            os.environ[GENAI_ENTRY_ENV_VAR] = "true"
            trace.set_tracer_provider(tracer_provider)
            tracer = tracer_provider.get_tracer(__name__)

            @with_genai_entry_detection
            def mock_ollama_operation(tracer, *args, **kwargs):
                span = tracer.start_span("ollama.chat")
                span.end()
                return {"message": {"content": "Hello from Ollama"}}

            @with_genai_entry_detection
            def mock_openai_operation(tracer, *args, **kwargs):
                span = tracer.start_span("openai.chat")
                span.end()
                return {"choices": [{"message": {"content": "Hello from OpenAI"}}]}

            # Outer Ollama operation calls inner OpenAI operation
            @with_genai_entry_detection
            def outer_ollama_operation(tracer, *args, **kwargs):
                span = tracer.start_span("ollama.chat")
                # Simulate nested call to OpenAI
                mock_openai_operation(tracer)
                span.end()
                return {"message": {"content": "Processed response"}}
            # Execute nested operations
            outer_ollama_operation(tracer)
            spans = span_exporter.get_finished_spans()
            assert len(spans) >= 2
            ollama_span = next((span for span in spans if "ollama.chat" in span.name), None)
            openai_span = next((span for span in spans if "openai.chat" in span.name), None)
            assert ollama_span is not None
            assert openai_span is not None
            # Only the outer Ollama operation should be marked as entry
            assert ollama_span.attributes.get(GENAI_ENTRY_ATTRIBUTE) is True
            assert GENAI_ENTRY_ATTRIBUTE not in openai_span.attributes
        finally:
            _reset_thread_local_state()
            if original_value:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)
