from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from traceloop.sdk.decorators import workflow
from traceloop.sdk import Traceloop
import os
from openai import OpenAI

# PostHog 分析现在默认禁用


client = OpenAI(
    base_url="http://localhost:5002/v1/",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Traceloop.init(app_name="story_service")

Traceloop.init(
    app_name="openai-instr",
    propagator=W3CBaggagePropagator(),
    exporter=OTLPSpanExporter(endpoint="http://127.0.0.1:4317"),
    metrics_exporter=OTLPMetricExporter(
        endpoint="http://127.0.0.1:4317",
    ),
)


@workflow(name="streaming_story")
def joke_workflow():
    stream = client.chat.completions.create(
        model="gpt-4o-2024-05-13",
        messages=[
            {"role": "user", "content": "Tell me a story about opentelemetry"}],
        stream=True,
    )

    for part in stream:
        print(part.choices[0].delta.content or "", end="")
    print()


joke_workflow()
