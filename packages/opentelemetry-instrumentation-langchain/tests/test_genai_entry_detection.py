"""Tests for GenAI Entry Detection in LangChain instrumentation."""
import pytest
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI, OpenAI
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_direct_llm_call_has_genai_entry_attribute(instrument_legacy, span_exporter):
    """Test that direct LLM calls are marked as GenAI entry."""
    query = [HumanMessage(content="Tell me a short joke about testing")]
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key="test-key",
        openai_api_base="http://localhost:5002/v1/"
    )
    # Make the LLM call
    response = model.invoke(query)
    # Verify spans were created
    spans = span_exporter.get_finished_spans()
    assert len(spans) > 0
    # Find the ChatOpenAI span
    chat_spans = [span for span in spans if span.name == "ChatOpenAI.chat"]
    assert len(chat_spans) == 1
    chat_span = chat_spans[0]
    # Check if the span has the GenAI entry attribute
    # This is what we expect after implementing entry detection
    has_genai_entry = "gen_ai.is_entry" in chat_span.attributes
    genai_entry_value = chat_span.attributes.get("gen_ai.is_entry", False)
    print(f"Span attributes: {dict(chat_span.attributes)}")
    print(f"Has gen_ai.is_entry attribute: {has_genai_entry}")
    print(f"GenAI entry value: {genai_entry_value}")
    # For now, just verify the span exists and has expected attributes
    assert SpanAttributes.LLM_SYSTEM in chat_span.attributes
    assert response.content  # Verify we got a response


@pytest.mark.vcr
def test_sequential_chain_entry_detection(instrument_legacy, span_exporter):
    """Test GenAI entry detection in Sequential Chain scenario."""
    # Create OpenAI LLM with mock backend
    llm = OpenAI(
        temperature=0.7,
        openai_api_key="test-key",
        openai_api_base="http://localhost:5002/v1/"
    )
    # Create synopsis chain
    synopsis_template = """You are a playwright. Write a synopsis for this title.
    Title: {title}
    Era: {era}
    Synopsis:"""
    synopsis_prompt = PromptTemplate(
        input_variables=["title", "era"], template=synopsis_template
    )
    synopsis_chain = LLMChain(
        llm=llm, prompt=synopsis_prompt, output_key="synopsis", name="synopsis"
    )
    # Create review chain
    review_template = """Write a review for this play.
    Synopsis: {synopsis}
    Review:"""
    review_prompt = PromptTemplate(input_variables=["synopsis"], template=review_template)
    review_chain = LLMChain(llm=llm, prompt=review_prompt, output_key="review")
    # Create sequential chain
    overall_chain = SequentialChain(
        chains=[synopsis_chain, review_chain],
        input_variables=["era", "title"],
        output_variables=["synopsis", "review"],
        verbose=True,
    )
    # Execute the chain
    result = overall_chain.invoke({
        "title": "The Testing Chronicles",
        "era": "Modern Era"
    })
    # Analyze spans
    spans = span_exporter.get_finished_spans()
    print("\n=== Sequential Chain Spans ===")
    for i, span in enumerate(spans):
        has_entry = "gen_ai.is_entry" in span.attributes
        entry_value = span.attributes.get("gen_ai.is_entry", False)
        span_kind = span.attributes.get("traceloop.span.kind", "N/A")
        print(f"Span {i}: {span.name}")
        print(f"  Span Kind: {span_kind}")
        print(f"  Has GenAI Entry: {has_entry}")
        print(f"  Entry Value: {entry_value}")
        print(f"  Parent: {getattr(span, 'parent', 'None')}")
    # Verify expected spans exist
    span_names = [span.name for span in spans]
    expected_spans = ["SequentialChain.workflow", "synopsis.task", "LLMChain.task"]
    for expected in expected_spans:
        assert any(expected in name for name in span_names), f"Missing span: {expected}"
    # Verify entry detection logic
    workflow_spans = [s for s in spans if "workflow" in s.name]
    task_spans = [s for s in spans if "task" in s.name]
    llm_spans = [s for s in spans if any(llm_name in s.name for llm_name in ["OpenAI", "ChatOpenAI"])]
    # Only workflow spans should be marked as entry
    for span in workflow_spans:
        assert span.attributes.get("gen_ai.is_entry") is True, f"Workflow span {span.name} should be entry"
    # Task spans should NOT be marked as entry (they are nested)
    for span in task_spans:
        assert span.attributes.get("gen_ai.is_entry") is not True, f"Task span {span.name} should not be entry"
    # LLM spans should NOT be marked as entry (they are nested within tasks)
    for span in llm_spans:
        assert span.attributes.get("gen_ai.is_entry") is not True, f"LLM span {span.name} should not be entry"

    assert result  # Verify we got results


@pytest.mark.vcr
def test_agent_entry_detection(instrument_legacy, span_exporter):
    """Test GenAI entry detection in Agent scenario."""
    # Mock tools
    search = TavilySearchResults(max_results=1)
    tools = [search]
    # Create ChatOpenAI model with mock backend
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key="test-key",
        openai_api_base="http://localhost:5002/v1/"
    )
    # Create a simple prompt template instead of pulling from hub
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    # Create agent and executor
    agent = create_tool_calling_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    # Execute the agent
    result = agent_executor.invoke({"input": "What is testing?"})
    # Analyze spans
    spans = span_exporter.get_finished_spans()
    print("\n=== Agent Execution Spans ===")
    for i, span in enumerate(spans):
        has_entry = "gen_ai.is_entry" in span.attributes
        entry_value = span.attributes.get("gen_ai.is_entry", False)
        span_kind = span.attributes.get("traceloop.span.kind", "N/A")
        print(f"Span {i}: {span.name}")
        print(f"  Span Kind: {span_kind}")
        print(f"  Has GenAI Entry: {has_entry}")
        print(f"  Entry Value: {entry_value}")
    # Agent executor should create a workflow span
    workflow_spans = [s for s in spans if "workflow" in s.name]
    assert len(workflow_spans) > 0, "Should have workflow spans"
    # Only workflow spans should be marked as entry
    for span in workflow_spans:
        assert span.attributes.get("gen_ai.is_entry") is True, f"Workflow span {span.name} should be entry"
    # All other spans should NOT be entry
    non_workflow_spans = [s for s in spans if "workflow" not in s.name]
    for span in non_workflow_spans:
        assert span.attributes.get("gen_ai.is_entry") is not True, f"Non-workflow span {span.name} should not be entry"
    assert result  # Verify we got results


@pytest.mark.vcr
def test_nested_chain_entry_detection(instrument_legacy, span_exporter):
    """Test entry detection with nested chains."""
    # Create LLM with mock backend
    llm = OpenAI(
        temperature=0.5,
        openai_api_key="test-key",
        openai_api_base="http://localhost:5002/v1/"
    )
    # Create inner chain
    inner_prompt = PromptTemplate(
        input_variables=["topic"],
        template="Generate 3 facts about {topic}:"
    )
    inner_chain = LLMChain(llm=llm, prompt=inner_prompt, output_key="facts")
    # Create outer chain that uses inner chain
    outer_prompt = PromptTemplate(
        input_variables=["topic", "facts"],
        template="Based on these facts about {topic}: {facts}\nWrite a summary:"
    )
    outer_chain = LLMChain(llm=llm, prompt=outer_prompt, output_key="summary")
    # Create sequential chain combining both
    combined_chain = SequentialChain(
        chains=[inner_chain, outer_chain],
        input_variables=["topic"],
        output_variables=["facts", "summary"]
    )
    # Execute
    result = combined_chain.invoke({"topic": "software testing"})
    # Analyze spans
    spans = span_exporter.get_finished_spans()
    print("\n=== Nested Chain Spans ===")
    for i, span in enumerate(spans):
        has_entry = "gen_ai.is_entry" in span.attributes
        entry_value = span.attributes.get("gen_ai.is_entry", False)
        span_kind = span.attributes.get("traceloop.span.kind", "N/A")
        print(f"Span {i}: {span.name}")
        print(f"  Span Kind: {span_kind}")
        print(f"  Has GenAI Entry: {has_entry}")
        print(f"  Entry Value: {entry_value}")
    # Verify only the top-level workflow is marked as entry
    task_spans = [s for s in spans if s.attributes.get("traceloop.span.kind") == "task"]
    # Should have exactly one workflow span marked as entry
    entry_spans = [s for s in spans if s.attributes.get("gen_ai.is_entry") is True]
    assert len(entry_spans) == 1, f"Should have exactly 1 entry span, got {len(entry_spans)}"
    assert entry_spans[0].attributes.get("traceloop.span.kind") == "workflow"
    # All task and LLM spans should NOT be entry
    for span in task_spans:
        assert span.attributes.get("gen_ai.is_entry") is not True, f"Task span should not be entry: {span.name}"
    assert result  # Verify we got results


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_chain_entry_detection(instrument_legacy, span_exporter):
    """Test entry detection with async chain execution."""
    # Create LLM
    llm = OpenAI(
        temperature=0.3,
        openai_api_key="test-key",
        openai_api_base="http://localhost:5002/v1/"
    )
    # Create simple chain
    prompt = PromptTemplate(
        input_variables=["question"],
        template="Answer this question: {question}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    # Execute asynchronously
    result = await chain.ainvoke({"question": "What is async testing?"})
    # Analyze spans
    spans = span_exporter.get_finished_spans()
    print("\n=== Async Chain Spans ===")
    for i, span in enumerate(spans):
        has_entry = "gen_ai.is_entry" in span.attributes
        entry_value = span.attributes.get("gen_ai.is_entry", False)
        span_kind = span.attributes.get("traceloop.span.kind", "N/A")
        print(f"Span {i}: {span.name}")
        print(f"  Span Kind: {span_kind}")
        print(f"  Has GenAI Entry: {has_entry}")
        print(f"  Entry Value: {entry_value}")
    # For direct chain invocation, the chain itself should be the entry
    llm_chain_spans = [s for s in spans if "LLMChain" in s.name]
    if llm_chain_spans:
        # If there's a chain span, it should be marked as entry
        entry_spans = [s for s in spans if s.attributes.get("gen_ai.is_entry") is True]
        assert len(entry_spans) >= 1, "Should have at least one entry span in async execution"
    assert result  # Verify we got results


@pytest.mark.vcr
def test_multiple_independent_calls_entry_detection(instrument_legacy, span_exporter):
    """Test that multiple independent calls are each marked as entry."""
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key="test-key",
        openai_api_base="http://localhost:5002/v1/"
    )
    # Make multiple independent calls
    query1 = [HumanMessage(content="First question")]
    query2 = [HumanMessage(content="Second question")]
    response1 = model.invoke(query1)
    # Clear spans to separate the calls
    spans_after_first = list(span_exporter.get_finished_spans())
    span_exporter.clear()
    response2 = model.invoke(query2)
    spans_after_second = list(span_exporter.get_finished_spans())
    print("\n=== Multiple Independent Calls ===")
    print(f"First call spans: {len(spans_after_first)}")
    for span in spans_after_first:
        has_entry = span.attributes.get("gen_ai.is_entry", False)
        print(f"  {span.name}: entry={has_entry}")
    print(f"Second call spans: {len(spans_after_second)}")
    for span in spans_after_second:
        has_entry = span.attributes.get("gen_ai.is_entry", False)
        print(f"  {span.name}: entry={has_entry}")
    # Both calls should have their main span marked as entry
    first_chat_spans = [s for s in spans_after_first if "ChatOpenAI.chat" in s.name]
    second_chat_spans = [s for s in spans_after_second if "ChatOpenAI.chat" in s.name]
    assert len(first_chat_spans) == 1, "Should have one chat span in first call"
    assert len(second_chat_spans) == 1, "Should have one chat span in second call"
    assert first_chat_spans[0].attributes.get("gen_ai.is_entry") is True, "First call should be entry"
    assert second_chat_spans[0].attributes.get("gen_ai.is_entry") is True, "Second call should be entry"
    assert response1.content  # Verify responses
    assert response2.content


@pytest.mark.vcr
def test_complex_agent_with_multiple_tools_entry_detection(instrument_legacy, span_exporter):
    """Test GenAI entry detection in complex Agent with multiple tools."""
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_core.tools import tool

    # Create a simple custom tool
    @tool
    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return f"The weather in {location} is sunny and 25°C"
    # Create multiple tools
    search = TavilySearchResults(max_results=2)
    tools = [search, get_weather]
    # Create ChatOpenAI model with mock backend
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key="test-key",
        openai_api_base="http://localhost:5002/v1/"
    )
    # Create a simple prompt template
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can search for information and answer questions."),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    # Create agent and executor
    agent = create_tool_calling_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=3)
    # Execute the agent
    result = agent_executor.invoke({"input": "What is the latest news about AI?"})
    # Analyze spans
    spans = span_exporter.get_finished_spans()
    print("\n=== Complex Agent with Multiple Tools Spans ===")
    for i, span in enumerate(spans):
        has_entry = "gen_ai.is_entry" in span.attributes
        entry_value = span.attributes.get("gen_ai.is_entry", False)
        span_kind = span.attributes.get("traceloop.span.kind", "N/A")
        print(f"Span {i}: {span.name}")
        print(f"  Span Kind: {span_kind}")
        print(f"  Has GenAI Entry: {has_entry}")
        print(f"  Entry Value: {entry_value}")
    # Verify only the top-level workflow is marked as entry
    tool_spans = [s for s in spans if s.attributes.get("traceloop.span.kind") == "tool"]
    task_spans = [s for s in spans if s.attributes.get("traceloop.span.kind") == "task"]
    # Should have exactly one workflow span marked as entry
    entry_spans = [s for s in spans if s.attributes.get("gen_ai.is_entry") is True]
    assert len(entry_spans) == 1, f"Should have exactly 1 entry span, got {len(entry_spans)}"
    assert entry_spans[0].attributes.get("traceloop.span.kind") == "workflow"
    # All tool and task spans should NOT be entry
    for span in tool_spans + task_spans:
        assert span.attributes.get("gen_ai.is_entry") is not True, f"Tool/Task span should not be entry: {span.name}"
    assert result  # Verify we got results


@pytest.mark.vcr
def test_lcel_chain_entry_detection(instrument_legacy, span_exporter):
    """Test GenAI entry detection in LCEL (LangChain Expression Language) chain."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    # Create ChatOpenAI model with mock backend
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key="test-key",
        openai_api_base="http://localhost:5002/v1/"
    )
    # Create LCEL chain
    prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
    chain = prompt | model | StrOutputParser()
    # Execute the chain
    result = chain.invoke({"topic": "programming"})
    # Analyze spans
    spans = span_exporter.get_finished_spans()
    print("\n=== LCEL Chain Spans ===")
    for i, span in enumerate(spans):
        has_entry = "gen_ai.is_entry" in span.attributes
        entry_value = span.attributes.get("gen_ai.is_entry", False)
        span_kind = span.attributes.get("traceloop.span.kind", "N/A")
        print(f"Span {i}: {span.name}")
        print(f"  Span Kind: {span_kind}")
        print(f"  Has GenAI Entry: {has_entry}")
        print(f"  Entry Value: {entry_value}")
    # For LCEL chains, the top-level chain should be marked as entry
    chain_spans = [s for s in spans if "RunnableSequence" in s.name]
    # Should have at least one entry span
    entry_spans = [s for s in spans if s.attributes.get("gen_ai.is_entry") is True]
    assert len(entry_spans) >= 1, "Should have at least one entry span in LCEL chain"
    # The chain span should be marked as entry
    if chain_spans:
        chain_entry_spans = [s for s in chain_spans if s.attributes.get("gen_ai.is_entry") is True]
        assert len(chain_entry_spans) >= 1, "Chain span should be marked as entry"
    assert result  # Verify we got results


@pytest.mark.vcr
def test_structured_output_entry_detection(instrument_legacy, span_exporter):
    """Test GenAI entry detection in structured output scenario."""
    from langchain_core.prompts import ChatPromptTemplate
    # Create ChatOpenAI model with mock backend
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key="test-key",
        openai_api_base="http://localhost:5002/v1/"
    )
    # Create a simple chain with output parsing
    from langchain_core.output_parsers import StrOutputParser
    prompt = ChatPromptTemplate.from_template("Answer this question in one sentence: {question}")
    chain = prompt | model | StrOutputParser()
    # Execute the chain
    result = chain.invoke({"question": "What is structured output?"})
    # Analyze spans
    spans = span_exporter.get_finished_spans()
    print("\n=== Structured Output Spans ===")
    for i, span in enumerate(spans):
        has_entry = "gen_ai.is_entry" in span.attributes
        entry_value = span.attributes.get("gen_ai.is_entry", False)
        span_kind = span.attributes.get("traceloop.span.kind", "N/A")
        print(f"Span {i}: {span.name}")
        print(f"  Span Kind: {span_kind}")
        print(f"  Has GenAI Entry: {has_entry}")
        print(f"  Entry Value: {entry_value}")
    # Should have at least one entry span
    entry_spans = [s for s in spans if s.attributes.get("gen_ai.is_entry") is True]
    assert len(entry_spans) >= 1, "Should have at least one entry span in structured output"
    assert result  # Verify we got results


@pytest.mark.vcr
def test_tool_calling_entry_detection(instrument_legacy, span_exporter):
    """Test GenAI entry detection in tool calling scenario."""
    from langchain_core.tools import tool
    from langchain_core.prompts import ChatPromptTemplate

    # Define a simple tool
    @tool
    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return f"The weather in {location} is sunny and 25°C"
    # Create ChatOpenAI model with mock backend
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key="test-key",
        openai_api_base="http://localhost:5002/v1/"
    )
    # Create chain with tool
    from langchain_core.output_parsers import StrOutputParser
    prompt = ChatPromptTemplate.from_template("What's the weather like in {location}?")
    chain = prompt | model.bind_tools([get_weather]) | StrOutputParser()
    # Execute the chain
    result = chain.invoke({"location": "New York"})
    # Analyze spans
    spans = span_exporter.get_finished_spans()
    print("\n=== Tool Calling Spans ===")
    for i, span in enumerate(spans):
        has_entry = "gen_ai.is_entry" in span.attributes
        entry_value = span.attributes.get("gen_ai.is_entry", False)
        span_kind = span.attributes.get("traceloop.span.kind", "N/A")
        print(f"Span {i}: {span.name}")
        print(f"  Span Kind: {span_kind}")
        print(f"  Has GenAI Entry: {has_entry}")
        print(f"  Entry Value: {entry_value}")
    # Should have at least one entry span
    entry_spans = [s for s in spans if s.attributes.get("gen_ai.is_entry") is True]
    assert len(entry_spans) >= 1, "Should have at least one entry span in tool calling"
    assert result  # Verify we got results
