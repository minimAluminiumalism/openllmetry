"""
Tests for GenAI entry detection functionality.
"""

import os
import pytest
from openai import OpenAI, AsyncOpenAI
from opentelemetry import trace
from opentelemetry.semconv_ai.genai_entry import (
    with_genai_entry_detection,
    is_genai_entry_enabled,
    GENAI_ENTRY_ATTRIBUTE,
    GENAI_ENTRY_ENV_VAR,
    GENAI_ENTRY_SAFE_MODE_VAR,
    get_genai_entry_detection_health,
    _get_genai_depth,
    _reset_thread_local_state,
    _record_circuit_breaker_failure,
)
import threading
import asyncio
from unittest.mock import patch


@pytest.fixture
def localhost_openai_client():
    """OpenAI client configured for localhost testing."""
    return OpenAI(
        base_url="http://localhost:5002/v1/",
        api_key="test-key-no-auth-required"
    )


@pytest.fixture
def async_localhost_openai_client():
    """Async OpenAI client configured for localhost testing."""
    return AsyncOpenAI(
        base_url="http://localhost:5002/v1/",
        api_key="test-key-no-auth-required"
    )


class TestGenAIEntryDetection:
    """Test cases for GenAI entry detection functionality."""

    def test_genai_entry_env_var_enabled_by_default(self):
        """Test that GenAI entry detection is enabled by default."""
        # Clear any existing env var
        original_value = os.environ.pop(GENAI_ENTRY_ENV_VAR, None)
        try:
            assert is_genai_entry_enabled() is True
        finally:
            if original_value is not None:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value

    def test_genai_entry_env_var_can_be_disabled(self):
        """Test that GenAI entry detection can be disabled via env var."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            os.environ[GENAI_ENTRY_ENV_VAR] = "false"
            assert is_genai_entry_enabled() is False
        finally:
            if original_value is not None:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)

    def test_genai_entry_env_var_case_insensitive(self):
        """Test that GenAI entry detection env var is case insensitive."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            for value in ["TRUE", "True", "true", "FALSE", "False", "false"]:
                os.environ[GENAI_ENTRY_ENV_VAR] = value
                expected = value.lower() == "true"
                assert is_genai_entry_enabled() is expected
        finally:
            if original_value is not None:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)

    @pytest.mark.vcr
    def test_chat_completion_has_genai_entry_attribute(
        self, instrument_legacy, span_exporter, localhost_openai_client
    ):
        """Test that chat completion spans are marked with GenAI entry attribute."""
        # Ensure GenAI entry detection is enabled
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            os.environ[GENAI_ENTRY_ENV_VAR] = "true"

            localhost_openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "Hello, test message for GenAI entry detection"}],
                max_tokens=10
            )

            spans = span_exporter.get_finished_spans()
            assert len(spans) == 1

            chat_span = spans[0]
            assert chat_span.name == "openai.chat"

            # Check that the span has the GenAI entry attribute
            assert chat_span.attributes.get(GENAI_ENTRY_ATTRIBUTE) is True

        finally:
            if original_value is not None:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)

    @pytest.mark.vcr
    def test_chat_completion_no_genai_entry_when_disabled(
        self, instrument_legacy, span_exporter, localhost_openai_client
    ):
        """Test that GenAI entry attribute is not added when detection is disabled."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            os.environ[GENAI_ENTRY_ENV_VAR] = "false"

            localhost_openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "Hello, test message for disabled GenAI entry"}],
                max_tokens=10
            )

            spans = span_exporter.get_finished_spans()
            assert len(spans) == 1

            chat_span = spans[0]
            assert chat_span.name == "openai.chat"

            # Check that the span does NOT have the GenAI entry attribute
            assert GENAI_ENTRY_ATTRIBUTE not in chat_span.attributes

        finally:
            if original_value is not None:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)

    @pytest.mark.vcr
    async def test_async_chat_completion_has_genai_entry_attribute(
        self, instrument_legacy, span_exporter, async_localhost_openai_client
    ):
        """Test that async chat completion spans are marked with GenAI entry attribute."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            os.environ[GENAI_ENTRY_ENV_VAR] = "true"

            await async_localhost_openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "Hello, async test message"}],
                max_tokens=10
            )

            spans = span_exporter.get_finished_spans()
            assert len(spans) == 1

            chat_span = spans[0]
            assert chat_span.name == "openai.chat"

            # Check that the span has the GenAI entry attribute
            assert chat_span.attributes.get(GENAI_ENTRY_ATTRIBUTE) is True

        finally:
            if original_value is not None:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)

    @pytest.mark.vcr
    def test_streaming_chat_completion_has_genai_entry_attribute(
        self, instrument_legacy, span_exporter, localhost_openai_client
    ):
        """Test that streaming chat completion spans are marked with GenAI entry attribute."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            os.environ[GENAI_ENTRY_ENV_VAR] = "true"

            stream = localhost_openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello, streaming test"}],
                max_tokens=10,
                stream=True
            )

            # Consume the stream
            for chunk in stream:
                pass

            spans = span_exporter.get_finished_spans()
            assert len(spans) == 1

            chat_span = spans[0]
            assert chat_span.name == "openai.chat"

            # Check that the span has the GenAI entry attribute
            assert chat_span.attributes.get(GENAI_ENTRY_ATTRIBUTE) is True

        finally:
            if original_value is not None:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)

    @pytest.mark.vcr
    async def test_async_streaming_chat_completion_has_genai_entry_attribute(
        self, instrument_legacy, span_exporter, async_localhost_openai_client
    ):
        """Test that async streaming chat completion spans are marked with GenAI entry attribute."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            os.environ[GENAI_ENTRY_ENV_VAR] = "true"

            stream = await async_localhost_openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "Hello, async streaming test for GenAI entry"}],
                max_tokens=10,
                stream=True
            )

            # Consume the async stream
            async for _ in stream:
                pass

            spans = span_exporter.get_finished_spans()
            assert len(spans) == 1

            chat_span = spans[0]
            assert chat_span.name == "openai.chat"

            # Check that the span has the GenAI entry attribute
            assert chat_span.attributes.get(GENAI_ENTRY_ATTRIBUTE) is True

        finally:
            if original_value is not None:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)

    @pytest.mark.vcr
    def test_embeddings_has_genai_entry_attribute(
        self, instrument_legacy, span_exporter, localhost_openai_client
    ):
        """Test that embeddings spans are marked with GenAI entry attribute."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            os.environ[GENAI_ENTRY_ENV_VAR] = "true"

            localhost_openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input="Hello, embeddings test for GenAI entry"
            )

            spans = span_exporter.get_finished_spans()
            assert len(spans) == 1

            embedding_span = spans[0]
            assert embedding_span.name == "openai.embeddings"

            # Check that the span has the GenAI entry attribute
            assert embedding_span.attributes.get(GENAI_ENTRY_ATTRIBUTE) is True

        finally:
            if original_value is not None:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)

    @pytest.mark.vcr
    def test_completions_has_genai_entry_attribute(
        self, instrument_legacy, span_exporter, localhost_openai_client
    ):
        """Test that completions spans are marked with GenAI entry attribute."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            os.environ[GENAI_ENTRY_ENV_VAR] = "true"

            localhost_openai_client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt="Hello, completions test for GenAI entry",
                max_tokens=10
            )

            spans = span_exporter.get_finished_spans()
            assert len(spans) == 1

            completion_span = spans[0]
            assert completion_span.name == "openai.completion"

            # Check that the span has the GenAI entry attribute
            assert completion_span.attributes.get(
                GENAI_ENTRY_ATTRIBUTE) is True

        finally:
            if original_value is not None:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)


class TestGenAIEntryDetectionNested:
    """Test cases for nested GenAI operations to verify only the outermost is marked as entry."""

    def test_single_genai_operation_marked_as_entry(self, span_exporter, tracer_provider):
        """Test that a single GenAI operation is correctly marked as entry."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            os.environ[GENAI_ENTRY_ENV_VAR] = "true"

            # Set the tracer provider to use the test setup
            trace.set_tracer_provider(tracer_provider)
            tracer = trace.get_tracer(__name__)

            @with_genai_entry_detection
            def mock_openai_wrapper(tracer_arg):
                span = tracer_arg.start_span("openai.chat")
                span.end()
                return "result"

            mock_openai_wrapper(tracer)

            spans = span_exporter.get_finished_spans()
            assert len(spans) == 1

            span = spans[0]
            assert span.name == "openai.chat"
            assert span.attributes.get(GENAI_ENTRY_ATTRIBUTE) is True

        finally:
            if original_value is not None:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)

    def test_nested_genai_operations_only_outer_marked_as_entry(self, span_exporter, tracer_provider):
        """Test that in nested GenAI operations, only the outer one is marked as entry."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            os.environ[GENAI_ENTRY_ENV_VAR] = "true"

            # Set the tracer provider to use the test setup
            trace.set_tracer_provider(tracer_provider)
            tracer = trace.get_tracer(__name__)

            @with_genai_entry_detection
            def mock_openai_wrapper(tracer_arg):
                span = tracer_arg.start_span("openai.chat")
                span.end()
                return "openai_result"

            @with_genai_entry_detection
            def mock_ollama_wrapper(tracer_arg):
                span = tracer_arg.start_span("ollama.chat")
                # Simulate calling OpenAI from within Ollama
                mock_openai_wrapper(tracer_arg)
                span.end()
                return "ollama_result"

            mock_ollama_wrapper(tracer)

            spans = span_exporter.get_finished_spans()
            assert len(spans) == 2

            # Find spans by name
            openai_span = next(s for s in spans if s.name == "openai.chat")
            ollama_span = next(s for s in spans if s.name == "ollama.chat")

            # Only the outer Ollama span should be marked as entry
            assert ollama_span.attributes.get(GENAI_ENTRY_ATTRIBUTE) is True
            assert GENAI_ENTRY_ATTRIBUTE not in openai_span.attributes

        finally:
            if original_value is not None:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)

    def test_multiple_independent_genai_operations_both_marked_as_entry(self, span_exporter, tracer_provider):
        """Test that multiple independent GenAI operations are both marked as entries."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            os.environ[GENAI_ENTRY_ENV_VAR] = "true"

            # Set the tracer provider to use the test setup
            trace.set_tracer_provider(tracer_provider)
            tracer = trace.get_tracer(__name__)

            @with_genai_entry_detection
            def mock_openai_wrapper(tracer_arg):
                span = tracer_arg.start_span("openai.chat")
                span.end()
                return "openai_result"

            @with_genai_entry_detection
            def mock_anthropic_wrapper(tracer_arg):
                span = tracer_arg.start_span("anthropic.chat")
                span.end()
                return "anthropic_result"

            # Two independent calls
            mock_openai_wrapper(tracer)
            mock_anthropic_wrapper(tracer)

            spans = span_exporter.get_finished_spans()
            assert len(spans) == 2

            # Both spans should be marked as entries since they are independent
            for span in spans:
                assert span.attributes.get(GENAI_ENTRY_ATTRIBUTE) is True

        finally:
            if original_value is not None:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)

    def test_deep_nested_genai_operations_only_outermost_marked(self, span_exporter, tracer_provider):
        """Test that in deeply nested GenAI operations, only the outermost is marked as entry."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            os.environ[GENAI_ENTRY_ENV_VAR] = "true"

            # Set the tracer provider to use the test setup
            trace.set_tracer_provider(tracer_provider)
            tracer = trace.get_tracer(__name__)

            @with_genai_entry_detection
            def mock_groq_wrapper(tracer_arg):
                span = tracer_arg.start_span("groq.chat")
                span.end()
                return "groq_result"

            @with_genai_entry_detection
            def mock_openai_wrapper(tracer_arg):
                span = tracer_arg.start_span("openai.chat")
                # Groq called from within OpenAI
                mock_groq_wrapper(tracer_arg)
                span.end()
                return "openai_result"

            @with_genai_entry_detection
            def mock_ollama_wrapper(tracer_arg):
                span = tracer_arg.start_span("ollama.chat")
                # OpenAI called from within Ollama
                mock_openai_wrapper(tracer_arg)
                span.end()
                return "ollama_result"

            # Start the chain: Ollama -> OpenAI -> Groq
            mock_ollama_wrapper(tracer)

            spans = span_exporter.get_finished_spans()
            assert len(spans) == 3

            # Find spans by name
            groq_span = next(s for s in spans if s.name == "groq.chat")
            openai_span = next(s for s in spans if s.name == "openai.chat")
            ollama_span = next(s for s in spans if s.name == "ollama.chat")

            # Only the outermost Ollama span should be marked as entry
            assert ollama_span.attributes.get(GENAI_ENTRY_ATTRIBUTE) is True
            assert GENAI_ENTRY_ATTRIBUTE not in openai_span.attributes
            assert GENAI_ENTRY_ATTRIBUTE not in groq_span.attributes

        finally:
            if original_value is not None:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)

    async def test_async_nested_genai_operations_only_outer_marked_as_entry(self, span_exporter, tracer_provider):
        """Test that in async nested GenAI operations, only the outer one is marked as entry."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            os.environ[GENAI_ENTRY_ENV_VAR] = "true"

            # Set the tracer provider to use the test setup
            trace.set_tracer_provider(tracer_provider)
            tracer = trace.get_tracer(__name__)

            @with_genai_entry_detection
            async def mock_openai_async_wrapper(tracer_arg):
                span = tracer_arg.start_span("openai.chat")
                span.end()
                return "openai_result"

            @with_genai_entry_detection
            async def mock_ollama_async_wrapper(tracer_arg):
                span = tracer_arg.start_span("ollama.chat")
                # Simulate async calling OpenAI from within Ollama
                await mock_openai_async_wrapper(tracer_arg)
                span.end()
                return "ollama_result"

            await mock_ollama_async_wrapper(tracer)

            spans = span_exporter.get_finished_spans()
            assert len(spans) == 2

            # Find spans by name
            openai_span = next(s for s in spans if s.name == "openai.chat")
            ollama_span = next(s for s in spans if s.name == "ollama.chat")

            # Only the outer Ollama span should be marked as entry
            assert ollama_span.attributes.get(GENAI_ENTRY_ATTRIBUTE) is True
            assert GENAI_ENTRY_ATTRIBUTE not in openai_span.attributes

        finally:
            if original_value is not None:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)

    def test_mixed_genai_and_non_genai_spans_correct_marking(self, span_exporter, tracer_provider):
        """Test that GenAI entry detection works correctly with mixed GenAI and non-GenAI spans."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            os.environ[GENAI_ENTRY_ENV_VAR] = "true"

            # Set the tracer provider to use the test setup
            trace.set_tracer_provider(tracer_provider)
            tracer = trace.get_tracer(__name__)

            def create_regular_span(tracer_arg, name):
                """Create a regular non-GenAI span."""
                span = tracer_arg.start_span(name)
                span.end()

            @with_genai_entry_detection
            def mock_openai_wrapper(tracer_arg):
                span = tracer_arg.start_span("openai.chat")
                span.end()
                return "openai_result"

            # Create a regular span first
            create_regular_span(tracer, "database.query")

            # Then a GenAI span (should be marked as entry)
            mock_openai_wrapper(tracer)

            # Then another regular span
            create_regular_span(tracer, "http.request")

            spans = span_exporter.get_finished_spans()
            assert len(spans) == 3

            # Find spans by name
            db_span = next(s for s in spans if s.name == "database.query")
            openai_span = next(s for s in spans if s.name == "openai.chat")
            http_span = next(s for s in spans if s.name == "http.request")

            # Only the GenAI span should have the entry attribute
            assert GENAI_ENTRY_ATTRIBUTE not in db_span.attributes
            assert openai_span.attributes.get(GENAI_ENTRY_ATTRIBUTE) is True
            assert GENAI_ENTRY_ATTRIBUTE not in http_span.attributes

        finally:
            if original_value is not None:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)


class TestGenAIEntryDetectionEdgeCases:
    """Test edge cases and error scenarios for GenAI entry detection."""

    def test_exception_during_genai_operation_cleanup(self, span_exporter, tracer_provider):
        """Test that exceptions during GenAI operations don't break depth management."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            os.environ[GENAI_ENTRY_ENV_VAR] = "true"
            trace.set_tracer_provider(tracer_provider)
            tracer = trace.get_tracer(__name__)

            @with_genai_entry_detection
            def failing_genai_wrapper(tracer_arg):
                span = tracer_arg.start_span("failing.span")
                span.end()
                raise RuntimeError("Simulated failure")

            # First call should fail but not break depth management
            with pytest.raises(RuntimeError):
                failing_genai_wrapper(tracer)

            # Second call should work normally
            @with_genai_entry_detection
            def normal_genai_wrapper(tracer_arg):
                span = tracer_arg.start_span("normal.span")
                span.end()
                return "success"

            result = normal_genai_wrapper(tracer)
            assert result == "success"

            # Check that the normal span is correctly marked as entry
            spans = span_exporter.get_finished_spans()
            normal_span = next(s for s in spans if s.name == "normal.span")
            assert normal_span.attributes.get(GENAI_ENTRY_ATTRIBUTE) is True

        finally:
            if original_value is not None:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)

    def test_deep_nesting_beyond_normal_limits(self, span_exporter, tracer_provider):
        """Test very deep nesting (5+ levels) to ensure depth management works."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            os.environ[GENAI_ENTRY_ENV_VAR] = "true"
            trace.set_tracer_provider(tracer_provider)
            tracer = trace.get_tracer(__name__)

            @with_genai_entry_detection
            def level_5(tracer_arg):
                span = tracer_arg.start_span("level5.span")
                span.end()
                return "level5"

            @with_genai_entry_detection
            def level_4(tracer_arg):
                span = tracer_arg.start_span("level4.span")
                result = level_5(tracer_arg)
                span.end()
                return result

            @with_genai_entry_detection
            def level_3(tracer_arg):
                span = tracer_arg.start_span("level3.span")
                result = level_4(tracer_arg)
                span.end()
                return result

            @with_genai_entry_detection
            def level_2(tracer_arg):
                span = tracer_arg.start_span("level2.span")
                result = level_3(tracer_arg)
                span.end()
                return result

            @with_genai_entry_detection
            def level_1(tracer_arg):
                span = tracer_arg.start_span("level1.span")
                result = level_2(tracer_arg)
                span.end()
                return result

            level_1(tracer)

            spans = span_exporter.get_finished_spans()
            assert len(spans) == 5

            # Only level1 should be marked as entry
            level1_span = next(s for s in spans if s.name == "level1.span")
            assert level1_span.attributes.get(GENAI_ENTRY_ATTRIBUTE) is True

            # All other levels should not be marked as entry
            for level in ["level2", "level3", "level4", "level5"]:
                level_span = next(s for s in spans if s.name == f"{level}.span")
                assert GENAI_ENTRY_ATTRIBUTE not in level_span.attributes

        finally:
            if original_value is not None:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)

    def test_concurrent_genai_operations_thread_safety(self, span_exporter, tracer_provider):
        """Test that concurrent GenAI operations in different threads work correctly."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            os.environ[GENAI_ENTRY_ENV_VAR] = "true"
            trace.set_tracer_provider(tracer_provider)
            tracer = trace.get_tracer(__name__)

            results = []
            errors = []

            @with_genai_entry_detection
            def thread_genai_wrapper(tracer_arg, thread_id):
                try:
                    span = tracer_arg.start_span(f"thread{thread_id}.span")
                    span.end()
                    results.append(f"thread{thread_id}_success")
                except Exception as e:
                    errors.append(f"thread{thread_id}_error: {e}")

            # Create multiple threads
            threads = []
            for i in range(5):
                thread = threading.Thread(
                    target=thread_genai_wrapper,
                    args=(tracer, i)
                )
                threads.append(thread)

            # Start all threads
            for thread in threads:
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # Check results
            assert len(errors) == 0, f"Errors occurred: {errors}"
            assert len(results) == 5

            # Each thread should have created one span marked as entry
            spans = span_exporter.get_finished_spans()
            thread_spans = [s for s in spans if s.name.startswith("thread")]
            assert len(thread_spans) == 5

            for span in thread_spans:
                assert span.attributes.get(GENAI_ENTRY_ATTRIBUTE) is True

        finally:
            if original_value is not None:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)

    def test_recursive_genai_calls(self, span_exporter, tracer_provider):
        """Test recursive GenAI calls to ensure depth management works correctly."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            os.environ[GENAI_ENTRY_ENV_VAR] = "true"
            trace.set_tracer_provider(tracer_provider)
            tracer = trace.get_tracer(__name__)

            call_count = 0

            @with_genai_entry_detection
            def recursive_genai(tracer_arg, depth):
                nonlocal call_count
                call_count += 1

                span = tracer_arg.start_span(f"recursive.span.{depth}")

                if depth > 0:
                    # Recursive call
                    recursive_genai(tracer_arg, depth - 1)

                span.end()
                return f"depth_{depth}"

            # Call with depth 3 (will create 4 spans: depth 3,2,1,0)
            recursive_genai(tracer, 3)

            spans = span_exporter.get_finished_spans()
            assert len(spans) == 4
            assert call_count == 4

            # Only the outermost call (depth 3) should be marked as entry
            depth3_span = next(s for s in spans if s.name == "recursive.span.3")
            assert depth3_span.attributes.get(GENAI_ENTRY_ATTRIBUTE) is True

            # All inner calls should not be marked as entry
            for depth in [0, 1, 2]:
                depth_span = next(s for s in spans if s.name == f"recursive.span.{depth}")
                assert GENAI_ENTRY_ATTRIBUTE not in depth_span.attributes

        finally:
            if original_value is not None:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)

    def test_safe_mode_behavior(self, span_exporter, tracer_provider):
        """Test safe mode behavior when enabled."""
        original_entry_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        original_safe_value = os.environ.get(GENAI_ENTRY_SAFE_MODE_VAR)
        try:
            os.environ[GENAI_ENTRY_ENV_VAR] = "true"
            os.environ[GENAI_ENTRY_SAFE_MODE_VAR] = "true"

            trace.set_tracer_provider(tracer_provider)
            tracer = trace.get_tracer(__name__)

            @with_genai_entry_detection
            def safe_mode_genai(tracer_arg):
                span = tracer_arg.start_span("safe.mode.span")
                span.end()
                return "safe_mode_result"

            result = safe_mode_genai(tracer)
            assert result == "safe_mode_result"

            # In safe mode, no spans should be marked as entry
            spans = span_exporter.get_finished_spans()
            safe_span = next(s for s in spans if s.name == "safe.mode.span")
            assert GENAI_ENTRY_ATTRIBUTE not in safe_span.attributes

            # Health check should show safe mode is enabled
            health = get_genai_entry_detection_health()
            assert health['safe_mode'] is True
            assert health['enabled'] is False

        finally:
            if original_entry_value is not None:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_entry_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)
            if original_safe_value is not None:
                os.environ[GENAI_ENTRY_SAFE_MODE_VAR] = original_safe_value
            else:
                os.environ.pop(GENAI_ENTRY_SAFE_MODE_VAR, None)

    def test_circuit_breaker_behavior(self, span_exporter, tracer_provider):
        """Test circuit breaker behavior after multiple failures."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            os.environ[GENAI_ENTRY_ENV_VAR] = "true"
            trace.set_tracer_provider(tracer_provider)
            tracer = trace.get_tracer(__name__)

            # Trigger multiple failures to open circuit breaker
            for _ in range(6):  # Threshold is 5, so 6 should open it
                _record_circuit_breaker_failure()

            health = get_genai_entry_detection_health()
            assert health['circuit_breaker']['is_open'] is True
            assert health['enabled'] is False

            @with_genai_entry_detection
            def circuit_breaker_test(tracer_arg):
                span = tracer_arg.start_span("circuit.breaker.span")
                span.end()
                return "circuit_breaker_result"

            result = circuit_breaker_test(tracer)
            assert result == "circuit_breaker_result"

            # With circuit breaker open, spans should not be marked as entry
            spans = span_exporter.get_finished_spans()
            cb_span = next(s for s in spans if s.name == "circuit.breaker.span")
            assert GENAI_ENTRY_ATTRIBUTE not in cb_span.attributes

        finally:
            if original_value is not None:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)
            # Reset circuit breaker state for other tests
            from opentelemetry.semconv_ai.genai_entry import _circuit_breaker
            _circuit_breaker['failures'] = 0
            _circuit_breaker['is_open'] = False
            _circuit_breaker['last_failure_time'] = 0

    async def test_async_concurrent_operations(self, span_exporter, tracer_provider):
        """Test concurrent async GenAI operations in same thread."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            # Reset state before test
            _reset_thread_local_state()

            os.environ[GENAI_ENTRY_ENV_VAR] = "true"
            trace.set_tracer_provider(tracer_provider)
            tracer = trace.get_tracer(__name__)

            @with_genai_entry_detection
            async def async_genai_operation(tracer_arg, op_id):
                span = tracer_arg.start_span(f"async.op.{op_id}")
                await asyncio.sleep(0.01)  # Small delay to simulate work
                span.end()
                return f"async_result_{op_id}"

            # Run multiple async operations concurrently in same thread
            tasks = [
                async_genai_operation(tracer, i)
                for i in range(3)
            ]
            results = await asyncio.gather(*tasks)

            assert len(results) == 3
            assert all("async_result_" in result for result in results)

            # In concurrent async operations in same thread, only the first should be marked as entry
            # because they share the same thread-local depth counter
            spans = span_exporter.get_finished_spans()
            async_spans = [s for s in spans if s.name.startswith("async.op.")]
            assert len(async_spans) == 3

            # Find spans and sort by name to ensure consistent ordering
            async_spans_sorted = sorted(async_spans, key=lambda s: s.name)

            # Only the first async operation should be marked as entry
            first_span = async_spans_sorted[0]  # async.op.0
            assert first_span.attributes.get(GENAI_ENTRY_ATTRIBUTE) is True

            # The rest should not be marked as entry (they're nested in the same thread)
            for span in async_spans_sorted[1:]:
                assert GENAI_ENTRY_ATTRIBUTE not in span.attributes

        finally:
            if original_value is not None:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)
            _reset_thread_local_state()

    async def test_independent_async_operations(self, span_exporter, tracer_provider):
        """Test independent async GenAI operations (not concurrent)."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            # Reset state before test
            _reset_thread_local_state()

            os.environ[GENAI_ENTRY_ENV_VAR] = "true"
            trace.set_tracer_provider(tracer_provider)
            tracer = trace.get_tracer(__name__)

            @with_genai_entry_detection
            async def async_genai_operation(tracer_arg, op_id):
                span = tracer_arg.start_span(f"independent.op.{op_id}")
                await asyncio.sleep(0.01)  # Small delay to simulate work
                span.end()
                return f"independent_result_{op_id}"

            # Run async operations independently (not concurrently)
            results = []
            for i in range(3):
                result = await async_genai_operation(tracer, i)
                results.append(result)

            assert len(results) == 3
            assert all("independent_result_" in result for result in results)

            # Each independent async operation should be marked as entry
            spans = span_exporter.get_finished_spans()
            independent_spans = [s for s in spans if s.name.startswith("independent.op.")]
            assert len(independent_spans) == 3

            # All independent operations should be marked as entry
            for span in independent_spans:
                assert span.attributes.get(GENAI_ENTRY_ATTRIBUTE) is True

        finally:
            if original_value is not None:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)
            _reset_thread_local_state()

    def test_tracer_restoration_after_exception(self, span_exporter, tracer_provider):
        """Test that tracer.start_span is properly restored even when exceptions occur."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            # Reset state before test
            _reset_thread_local_state()

            os.environ[GENAI_ENTRY_ENV_VAR] = "true"
            trace.set_tracer_provider(tracer_provider)
            tracer = trace.get_tracer(__name__)

            # Store original start_span method
            original_start_span = tracer.start_span

            @with_genai_entry_detection
            def exception_during_span_creation(tracer_arg):
                # This should fail during span creation
                with patch.object(tracer_arg, 'start_span', side_effect=RuntimeError("Span creation failed")):
                    tracer_arg.start_span("failing.span")

            # This should raise an exception
            with pytest.raises(RuntimeError, match="Span creation failed"):
                exception_during_span_creation(tracer)

            # Verify that tracer.start_span is restored to original
            assert tracer.start_span == original_start_span

            # Verify that subsequent operations work normally
            @with_genai_entry_detection
            def normal_operation_after_failure(tracer_arg):
                span = tracer_arg.start_span("recovery.span")
                span.end()
                return "recovered"

            result = normal_operation_after_failure(tracer)
            assert result == "recovered"

            spans = span_exporter.get_finished_spans()
            recovery_span = next(s for s in spans if s.name == "recovery.span")
            assert recovery_span.attributes.get(GENAI_ENTRY_ATTRIBUTE) is True

        finally:
            if original_value is not None:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)
            _reset_thread_local_state()

    def test_depth_reset_after_corruption(self, span_exporter, tracer_provider):
        """Test that depth can be reset if thread-local state gets corrupted."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            os.environ[GENAI_ENTRY_ENV_VAR] = "true"
            trace.set_tracer_provider(tracer_provider)
            tracer = trace.get_tracer(__name__)

            # Manually corrupt the depth state
            assert _get_genai_depth() == 0

            # Simulate corruption by manually setting an invalid state
            from opentelemetry.semconv_ai.genai_entry import _thread_local
            _thread_local.genai_operation_depth = 999

            assert _get_genai_depth() == 999

            # Reset should fix it
            _reset_thread_local_state()
            assert _get_genai_depth() == 0

            # Normal operations should work after reset
            @with_genai_entry_detection
            def operation_after_reset(tracer_arg):
                span = tracer_arg.start_span("reset.recovery.span")
                span.end()
                return "reset_recovery"

            result = operation_after_reset(tracer)
            assert result == "reset_recovery"

            spans = span_exporter.get_finished_spans()
            reset_span = next(s for s in spans if s.name == "reset.recovery.span")
            assert reset_span.attributes.get(GENAI_ENTRY_ATTRIBUTE) is True

        finally:
            if original_value is not None:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)
            # Clean up any remaining state
            _reset_thread_local_state()

    def test_disabled_feature_performance(self, span_exporter, tracer_provider):
        """Test that when feature is disabled, there's minimal overhead."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            os.environ[GENAI_ENTRY_ENV_VAR] = "false"
            trace.set_tracer_provider(tracer_provider)
            tracer = trace.get_tracer(__name__)

            @with_genai_entry_detection
            def disabled_feature_operation(tracer_arg):
                span = tracer_arg.start_span("disabled.feature.span")
                span.end()
                return "disabled_result"

            result = disabled_feature_operation(tracer)
            assert result == "disabled_result"

            # No entry attributes should be added when disabled
            spans = span_exporter.get_finished_spans()
            disabled_span = next(s for s in spans if s.name == "disabled.feature.span")
            assert GENAI_ENTRY_ATTRIBUTE not in disabled_span.attributes

            # Depth should remain 0 throughout
            assert _get_genai_depth() == 0

        finally:
            if original_value is not None:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)
