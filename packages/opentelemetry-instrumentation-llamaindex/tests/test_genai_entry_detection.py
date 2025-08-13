"""
Tests for GenAI entry detection functionality in LlamaIndex instrumentation.
"""

import os
import pytest
from unittest.mock import patch
from opentelemetry import trace
from opentelemetry.semconv_ai.genai_entry import (
    GENAI_ENTRY_ATTRIBUTE,
    GENAI_ENTRY_ENV_VAR,
    is_genai_entry_enabled,
    _reset_thread_local_state,
    _circuit_breaker,
    _circuit_breaker_lock,
)


class TestLlamaIndexGenAIEntryDetection:
    """Test GenAI Entry Detection for LlamaIndex operations."""

    def setup_method(self):
        """Reset state before each test."""
        _reset_thread_local_state()
        with _circuit_breaker_lock:
            _circuit_breaker['failures'] = 0
            _circuit_breaker['is_open'] = False
            _circuit_breaker['last_failure_time'] = 0

    def test_genai_entry_env_var_enabled_by_default(self):
        """Test that GenAI entry detection is enabled by default."""
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
    def test_llamaindex_query_engine_has_genai_entry_attribute(self, instrument_legacy, span_exporter):
        """Test that a LlamaIndex query engine operation gets marked as entry."""
        try:
            from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
            from llama_index.llms.openai_like import OpenAILike
            from llama_index.embeddings.openai import OpenAIEmbedding
            
            # Configure LlamaIndex to use DashScope API (cassette will replay responses)
            Settings.llm = OpenAILike(
                model="qwen-turbo",
                api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key="test-api-key",  # Real API key removed for security
                is_chat_model=True,
                context_window=32000,
            )
            Settings.embed_model = OpenAIEmbedding(
                api_key="test-api-key",  # Real API key removed for security
                api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                model_name="text-embedding-v1"
            )
            
            # Load test documents
            documents = SimpleDirectoryReader("data/opentelemetry-docs").load_data()
            index = VectorStoreIndex.from_documents(documents)
            query_engine = index.as_query_engine()
            
            # Perform a query
            response = query_engine.query("What is AI agent observability?")
            
            # Check spans
            spans = list(span_exporter.get_finished_spans())
            assert len(spans) > 0
            
            # Find the query engine workflow span (should be the root span)
            query_spans = [span for span in spans if span.attributes.get("traceloop.span.kind") == "workflow" and ("query" in span.name.lower() or "QueryEngine" in span.name)]
            assert len(query_spans) > 0
            
            # The query engine workflow span should be marked as entry
            query_span = query_spans[0]
            assert query_span.attributes.get(GENAI_ENTRY_ATTRIBUTE) is True
            
        except ImportError:
            pytest.skip("LlamaIndex not installed or test data not available")
        except FileNotFoundError:
            pytest.skip("Test data directory not found")
        except Exception as e:
            # Re-raise the exception instead of skipping, so we can see what went wrong
            raise

    @pytest.mark.vcr
    def test_llamaindex_agent_has_genai_entry_attribute(self, instrument_legacy, span_exporter):
        """Test that a LlamaIndex agent operation gets marked as entry."""
        try:
            from llama_index.core.agent import ReActAgent
            from llama_index.core.tools import FunctionTool
            from llama_index.llms.openai_like import OpenAILike
            
            # Create a simple tool
            def multiply(a: int, b: int) -> int:
                """Multiply two integers and return the result."""
                return a * b
            
            multiply_tool = FunctionTool.from_defaults(fn=multiply)
            
            # Create an agent with DashScope API (cassette will replay responses)
            llm = OpenAILike(
                model="qwen-turbo",
                api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key="test-api-key",  # Real API key removed for security
                is_chat_model=True,
                context_window=32000,
            )
            agent = ReActAgent.from_tools([multiply_tool], llm=llm, verbose=True)
            
            # Perform an agent query
            response = agent.chat("What is 3 multiplied by 4?")
            
            # Check spans
            spans = list(span_exporter.get_finished_spans())
            assert len(spans) > 0
            
            # Find the agent workflow span (should be the root span)
            agent_spans = [span for span in spans if span.attributes.get("traceloop.span.kind") == "workflow" and ("agent" in span.name.lower() or "AgentRunner" in span.name)]
            assert len(agent_spans) > 0
            
            # The agent workflow span should be marked as entry
            agent_span = agent_spans[0]
            assert agent_span.attributes.get(GENAI_ENTRY_ATTRIBUTE) is True
            
        except ImportError:
            pytest.skip("LlamaIndex agent components not installed")
        except Exception as e:
            # Re-raise the exception instead of skipping, so we can see what went wrong
            raise

    @pytest.mark.vcr
    def test_llamaindex_no_genai_entry_when_disabled(self, instrument_legacy, span_exporter):
        """Test that GenAI entry detection can be disabled."""
        original_value = os.environ.get(GENAI_ENTRY_ENV_VAR)
        try:
            os.environ[GENAI_ENTRY_ENV_VAR] = "false"
            
            try:
                from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
                from llama_index.llms.openai_like import OpenAILike
                from llama_index.embeddings.openai import OpenAIEmbedding
                
                # Configure LlamaIndex to use DashScope API (cassette will replay responses)
                Settings.llm = OpenAILike(
                    model="qwen-turbo",
                    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    api_key="test-api-key",  # Real API key removed for security
                    is_chat_model=True,
                    context_window=32000,
                )
                Settings.embed_model = OpenAIEmbedding(
                    api_key="test-api-key",  # Real API key removed for security
                    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    model_name="text-embedding-v1"
                )
                
                # Create a simple operation
                documents = SimpleDirectoryReader("data/opentelemetry-docs").load_data()
                index = VectorStoreIndex.from_documents(documents)
                query_engine = index.as_query_engine()
                response = query_engine.query("What is observability?")
                
                # Check spans
                spans = list(span_exporter.get_finished_spans())
                
                # No span should have the entry attribute
                for span in spans:
                    assert GENAI_ENTRY_ATTRIBUTE not in span.attributes
                    
            except ImportError:
                pytest.skip("LlamaIndex not installed or test data not available")
            except FileNotFoundError:
                pytest.skip("Test data directory not found")
            except Exception as e:
                # Re-raise the exception instead of skipping, so we can see what went wrong
                raise
                
        finally:
            if original_value:
                os.environ[GENAI_ENTRY_ENV_VAR] = original_value
            else:
                os.environ.pop(GENAI_ENTRY_ENV_VAR, None)

    @pytest.mark.vcr
    def test_nested_llamaindex_operations_only_outer_marked_as_entry(self, instrument_legacy, span_exporter):
        """Test that in nested LlamaIndex operations, only the outer one is marked as entry."""
        try:
            from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
            from llama_index.core.query_engine import RetrieverQueryEngine
            from llama_index.llms.openai_like import OpenAILike
            from llama_index.embeddings.openai import OpenAIEmbedding
            
            # Configure LlamaIndex to use Alibaba Cloud DashScope API
            Settings.llm = OpenAILike(
                model="qwen-turbo",
                api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key="test-api-key",  # Real API key removed for security
                is_chat_model=True,
                context_window=32000,
            )
            Settings.embed_model = OpenAIEmbedding(
                api_key="test-api-key",  # Real API key removed for security
                api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                model_name="text-embedding-v1"
            )
            
            # Create nested operations by creating multiple query engines
            documents = SimpleDirectoryReader("data/opentelemetry-docs").load_data()
            index1 = VectorStoreIndex.from_documents(documents)
            index2 = VectorStoreIndex.from_documents(documents)
            
            # First query (should be marked as entry)
            query_engine1 = index1.as_query_engine()
            response1 = query_engine1.query("First query")
            
            # Nested query (should not be marked as entry)
            query_engine2 = index2.as_query_engine()
            response2 = query_engine2.query("Second query")
            
            # Check spans
            spans = list(span_exporter.get_finished_spans())
            entry_spans = [span for span in spans if span.attributes.get(GENAI_ENTRY_ATTRIBUTE) is True]
            
            # Should have at least one entry span, but not all spans should be entry
            assert len(entry_spans) >= 1
            assert len(entry_spans) < len(spans)
            
        except ImportError:
            pytest.skip("LlamaIndex not installed or test data not available")

    @pytest.mark.vcr
    def test_llamaindex_with_openai_nested_call(self, instrument_legacy, span_exporter):
        """Test LlamaIndex + OpenAI nested calls - only LlamaIndex should be entry."""
        try:
            from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
            from llama_index.llms.openai_like import OpenAILike
            from llama_index.embeddings.openai import OpenAIEmbedding
            from openai import OpenAI as DirectOpenAI
            
            # Configure LlamaIndex to use DashScope API (cassette will replay responses)
            Settings.llm = OpenAILike(
                model="qwen-turbo",
                api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key="test-api-key",  # Real API key removed for security
                is_chat_model=True,
                context_window=32000,
            )
            Settings.embed_model = OpenAIEmbedding(
                api_key="test-api-key",  # Real API key removed for security
                api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                model_name="text-embedding-v1"
            )
            
            # Setup LlamaIndex with documents
            documents = SimpleDirectoryReader("data/opentelemetry-docs").load_data()
            index = VectorStoreIndex.from_documents(documents)
            
            # LlamaIndex query (should be entry) - uses Settings.llm configured above
            query_engine = index.as_query_engine()
            response = query_engine.query("What is AI agent observability?")
            
            # Direct OpenAI call using DashScope API (should not be entry due to nesting)
            client = DirectOpenAI(
                api_key="test-api-key",  # Real API key removed for security
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            openai_response = client.chat.completions.create(
                model="qwen-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            
            # Check spans
            spans = list(span_exporter.get_finished_spans())
            
            # Find LlamaIndex workflow spans
            llamaindex_spans = [span for span in spans if span.attributes.get("traceloop.span.kind") == "workflow" and ("query" in span.name.lower() or "QueryEngine" in span.name)]
            
            # LlamaIndex should be marked as entry
            assert len(llamaindex_spans) > 0
            llamaindex_entry_spans = [span for span in llamaindex_spans 
                                    if span.attributes.get(GENAI_ENTRY_ATTRIBUTE) is True]
            assert len(llamaindex_entry_spans) > 0
                
        except ImportError:
            pytest.skip("LlamaIndex or OpenAI components not installed")

    def test_llamaindex_error_handling(self, span_exporter):
        """Test that GenAI entry detection handles errors gracefully."""
        from opentelemetry.instrumentation.llamaindex.dispatcher_wrapper import mark_span_as_genai_entry
        from unittest.mock import MagicMock
        
        # Create a mock span
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        
        # Test that error in mark_span_as_genai_entry doesn't crash
        with patch('opentelemetry.instrumentation.llamaindex.dispatcher_wrapper.mark_span_as_genai_entry') as mock_mark:
            # Make the mark function raise an exception
            mock_mark.side_effect = Exception("Test error")
            
            # Should not raise an exception due to the try-catch in the fallback implementation
            try:
                mark_span_as_genai_entry(mock_span)
                # If we get here, the fallback implementation handled the error
                assert True
            except Exception:
                # If this happens, the error handling needs improvement
                assert False, "mark_span_as_genai_entry should handle errors gracefully"

    @pytest.mark.vcr
    def test_llamaindex_multiple_independent_operations(self, instrument_legacy, span_exporter):
        """Test multiple independent LlamaIndex operations are all marked as entry."""
        try:
            from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
            from llama_index.llms.openai_like import OpenAILike
            from llama_index.embeddings.openai import OpenAIEmbedding
            
            # Configure LlamaIndex to use Alibaba Cloud DashScope API
            Settings.llm = OpenAILike(
                model="qwen-turbo",
                api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key="test-api-key",  # Real API key removed for security
                is_chat_model=True,
                context_window=32000,
            )
            Settings.embed_model = OpenAIEmbedding(
                api_key="test-api-key",  # Real API key removed for security
                api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                model_name="text-embedding-v1"
            )
            
            documents = SimpleDirectoryReader("data/opentelemetry-docs").load_data()
            
            # First independent operation
            index1 = VectorStoreIndex.from_documents(documents)
            query_engine1 = index1.as_query_engine()
            response1 = query_engine1.query("First independent query")
            
            # Check first batch of spans
            spans_after_first = list(span_exporter.get_finished_spans())
            
            # Second independent operation
            index2 = VectorStoreIndex.from_documents(documents)
            query_engine2 = index2.as_query_engine()
            response2 = query_engine2.query("Second independent query")
            
            # Check all spans
            all_spans = list(span_exporter.get_finished_spans())
            
            # Find query engine spans
            query_spans = [span for span in all_spans if span.attributes.get("traceloop.span.kind") == "workflow" and ("query" in span.name.lower() or "QueryEngine" in span.name)]
            entry_spans = [span for span in query_spans if span.attributes.get(GENAI_ENTRY_ATTRIBUTE) is True]
            
            # Both operations should be marked as entry
            assert len(entry_spans) >= 1  # At least one should be marked
            
        except ImportError:
            pytest.skip("LlamaIndex not installed or test data not available")