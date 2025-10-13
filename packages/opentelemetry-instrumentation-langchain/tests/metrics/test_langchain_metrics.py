from unittest.mock import patch
import pytest
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from opentelemetry.semconv_ai import Meters, SpanAttributes
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics as GenAIMetrics


@pytest.fixture
def llm():
    return ChatOpenAI(temperature=0)


@pytest.fixture
def chain(llm):
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    return LLMChain(llm=llm, prompt=prompt)


@pytest.mark.vcr
def test_llm_chain_metrics(instrument_legacy, reader, chain):
    chain.run(product="colorful socks")

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_token_metric = False
    found_duration_metric = False

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.LLM_TOKEN_USAGE:
                    found_token_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes[SpanAttributes.LLM_TOKEN_TYPE] in [
                            "output",
                            "input",
                        ]
                        assert data_point.sum > 0
                        assert (
                            data_point.attributes[SpanAttributes.LLM_SYSTEM]
                            == "openai"
                        )

                if metric.name == Meters.LLM_OPERATION_DURATION:
                    found_duration_metric = True
                    assert any(
                        data_point.count > 0 for data_point in metric.data.data_points
                    )
                    assert any(
                        data_point.sum > 0 for data_point in metric.data.data_points
                    )
                    for data_point in metric.data.data_points:
                        assert (
                            data_point.attributes[SpanAttributes.LLM_SYSTEM]
                            == "openai"
                        )

    assert found_token_metric is True
    assert found_duration_metric is True


@pytest.mark.vcr
def test_llm_chain_streaming_metrics(instrument_legacy, reader, llm):
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    for _ in chain.stream({"product": "colorful socks"}):
        pass

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_token_metric = False
    found_duration_metric = False
    found_choices_metric = False
    # TTFT/streaming-time may not be emitted in all streaming paths for OpenAI via LangChain
    # They are validated explicitly in the DeepSeek streaming test below.

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.LLM_TOKEN_USAGE:
                    found_token_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes[SpanAttributes.LLM_TOKEN_TYPE] in [
                            "output",
                            "input",
                        ]
                        assert data_point.sum > 0
                        assert (
                            data_point.attributes[SpanAttributes.LLM_SYSTEM]
                            == "openai"
                        )

                if metric.name == Meters.LLM_OPERATION_DURATION:
                    found_duration_metric = True
                    assert any(
                        data_point.count > 0 for data_point in metric.data.data_points
                    )
                    assert any(
                        data_point.sum > 0 for data_point in metric.data.data_points
                    )
                    for data_point in metric.data.data_points:
                        assert (
                            data_point.attributes[SpanAttributes.LLM_SYSTEM]
                            == "openai"
                        )

                if metric.name == Meters.LLM_GENERATION_CHOICES:
                    found_choices_metric = True
                    assert any(
                        data_point.value >= 1 for data_point in metric.data.data_points
                    )

    assert found_token_metric is True
    assert found_duration_metric is True
    assert found_choices_metric is True


def verify_token_metrics(data_points):
    for data_point in data_points:
        assert data_point.attributes[SpanAttributes.LLM_TOKEN_TYPE] in [
            "output",
            "input",
        ]
        assert data_point.sum > 0
        assert data_point.attributes[SpanAttributes.LLM_SYSTEM] == "openai"


def verify_duration_metrics(data_points):
    assert any(data_point.count > 0 for data_point in data_points)
    assert any(data_point.sum > 0 for data_point in data_points)
    for data_point in data_points:
        assert data_point.attributes[SpanAttributes.LLM_SYSTEM] == "openai"


def verify_langchain_metrics(reader):
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_token_metric = False
    found_duration_metric = False

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.LLM_TOKEN_USAGE:
                    found_token_metric = True
                    verify_token_metrics(metric.data.data_points)

                if metric.name == Meters.LLM_OPERATION_DURATION:
                    found_duration_metric = True
                    verify_duration_metrics(metric.data.data_points)

    return found_token_metric, found_duration_metric


@pytest.mark.vcr
def test_llm_chain_metrics_with_none_llm_output(instrument_legacy, reader, chain, llm):
    """
    This test verifies that the metrics system correctly handles edge cases where the
    LLM response contains a None value in the llm_output field, ensuring that token
    usage and operation duration metrics are still properly recorded.
    """
    original_generate = llm._generate

    # Create a patched version that returns results with None llm_output
    def patched_generate(*args, **kwargs):
        result = original_generate(*args, **kwargs)
        result.llm_output = None
        return result

    with patch.object(llm, '_generate', side_effect=patched_generate):
        chain.run(product="colorful socks")

    found_token_metric, found_duration_metric = verify_langchain_metrics(reader)

    assert found_token_metric is True, "Token usage metrics not found"
    assert found_duration_metric is True, "Operation duration metrics not found"


@pytest.mark.vcr
def test_streaming_with_ttft_and_generation_time_metrics(instrument_legacy, reader):
    """Test streaming metrics with ChatDeepSeek to validate third-party model fixes."""
    from langchain_core.prompts import ChatPromptTemplate
    try:
        from langchain_deepseek import ChatDeepSeek
    except Exception:
        pytest.skip("langchain-deepseek not installed in this environment")

    llm = ChatDeepSeek(
        api_key="",
        api_base="https://api.deepseek.com/beta",
        model="deepseek-chat",
        temperature=0.7,
        streaming=True
    )

    prompt = ChatPromptTemplate.from_template("Tell me about {topic} in one sentence")
    chain = prompt | llm

    # Stream the response to trigger on_llm_new_token calls
    response_chunks = []
    for chunk in chain.stream({"topic": "machine learning"}):
        response_chunks.append(chunk)
    assert len(response_chunks) > 1

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_token_metric = False
    found_duration_metric = False
    found_choices_metric = False
    found_ttft_metric = False
    found_streaming_time_metric = False

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.LLM_TOKEN_USAGE:
                    found_token_metric = True
                elif metric.name == Meters.LLM_OPERATION_DURATION:
                    found_duration_metric = True
                elif metric.name == Meters.LLM_GENERATION_CHOICES:
                    found_choices_metric = True
                elif metric.name == GenAIMetrics.GEN_AI_SERVER_TIME_TO_FIRST_TOKEN:
                    found_ttft_metric = True
                elif metric.name == Meters.LLM_STREAMING_TIME_TO_GENERATE:
                    found_streaming_time_metric = True

    assert found_token_metric is True
    assert found_duration_metric is True
    assert found_choices_metric is True
    assert found_ttft_metric is True
    assert found_streaming_time_metric is True


def test_exception_metrics(instrument_legacy, reader):
    """Test that exception metrics are recorded when LLM calls fail."""
    from unittest.mock import patch

    llm = ChatOpenAI(model="gpt-3.5-turbo")
    chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["product"],
            template="What is a good name for a company that makes {product}?",
        )
    )

    # Mock the LLM to raise an exception
    with patch.object(llm, '_generate', side_effect=Exception("API Error")):
        with pytest.raises(Exception):
            chain.run(product="test")

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_exception_metric = False

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == "llm.langchain.completions.exceptions":
                    found_exception_metric = True
                    assert any(
                        data_point.value >= 1 for data_point in metric.data.data_points
                    )
                    # Check that error attributes are set
                    for data_point in metric.data.data_points:
                        assert "error.type" in data_point.attributes or ERROR_TYPE in data_point.attributes

    assert found_exception_metric is True
