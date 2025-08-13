
This file shows the overall logic of genai span entry.

### Configuration

export OTEL_MARK_GENAI_ENTRY=true(default) 整个功能启用/禁用

export OTEL_GENAI_ENTRY_SAFE_MODE=false(default) 遇到错误时保护性禁用

### Overview

```mermaid
graph TB
    %% GenAI Entry Detection Architecture
    subgraph "User Application"
        UC[User Code]
        OC1[Ollama Call]
        OC2[OpenAI Call]
        OC3[Other GenAI Call]
    end
    
    subgraph "Instrumentation Layer"
        WD1["@with_genai_entry_detection<br/>Ollama Wrapper"]
        WD2["@with_genai_entry_detection<br/>OpenAI Wrapper"]
        WD3["@with_genai_entry_detection<br/>Other Wrapper"]
    end
    
    subgraph "GenAI Entry Detection Module"
        DC["Depth Counter<br/>Thread-Local Storage"]
        SI["Span Interceptor<br/>Monkey Patch tracer.start_span"]
        EL["Entry Logic<br/>depth == 1 ? mark : skip"]
    end
    
    subgraph "OpenTelemetry SDK"
        TR[Tracer]
        SP[Span Creation]
        SA[Span Attributes]
    end
    
    subgraph "Output"
        ES["Entry Span<br/>is_genai_entry=True"]
        NS["Nested Span<br/>no entry attribute"]
    end
    
    %% Flow connections
    UC --> OC1
    UC --> OC2
    UC --> OC3
    
    OC1 --> WD1
    OC2 --> WD2
    OC3 --> WD3
    
    WD1 --> DC
    WD2 --> DC
    WD3 --> DC
    
    DC --> SI
    SI --> EL
    EL --> TR
    
    TR --> SP
    SP --> SA
    
    SA --> ES
    SA --> NS
    
    %% Styling
    classDef userCode fill:#e1f5fe
    classDef wrapper fill:#fff3e0
    classDef detection fill:#f3e5f5
    classDef otel fill:#e8f5e8
    classDef output fill:#fce4ec
    
    class UC,OC1,OC2,OC3 userCode
    class WD1,WD2,WD3 wrapper
    class DC,SI,EL detection
    class TR,SP,SA otel
    class ES,NS output

```



```mermaid
flowchart TD
    Start([GenAI Function Called]) --> Check{Check if function<br/>has decorator}
    
    Check -->|No| Normal[Normal Execution]
    Check -->|Yes| Inc[Increment Depth Counter<br/>depth = increment_genai_depth]
    
    Inc --> Setup[Setup Span Interceptor<br/>Replace tracer.start_span<br/>with enhanced version]
    
    Setup --> Execute[Execute Original Function]
    
    Execute --> SpanCreated{Span Created?}
    
    SpanCreated -->|Yes| Intercept[enhanced_start_span Called]
    SpanCreated -->|No| Cleanup
    
    Intercept --> GetDepth[Get Current Depth<br/>current_depth = get_genai_depth]
    
    GetDepth --> DepthCheck{current_depth == 1?}
    
    DepthCheck -->|Yes| Mark[Mark as Entry Span<br/>Set is_genai_entry = True]
    DepthCheck -->|No| Skip[Skip Marking<br/>This is Nested Operation]
    
    Mark --> CreateSpan[Return Created Span]
    Skip --> CreateSpan
    
    CreateSpan --> MoreSpans{More Spans<br/>in Function?}
    
    MoreSpans -->|Yes| SpanCreated
    MoreSpans -->|No| Cleanup[Cleanup and Restore<br/>Original tracer.start_span]
    
    Cleanup --> Dec[Decrement Depth Counter<br/>decrement_genai_depth]
    
    Dec --> End([Function Complete])
    
    Normal --> End
    
    %% Styling
    classDef startEnd fill:#4CAF50,stroke:#2E7D32,color:#fff
    classDef decision fill:#FF9800,stroke:#F57C00,color:#fff
    classDef process fill:#2196F3,stroke:#1976D2,color:#fff
    classDef important fill:#E91E63,stroke:#C2185B,color:#fff
    
    class Start,End startEnd
    class Check,SpanCreated,DepthCheck,MoreSpans decision
    class Inc,Setup,Execute,Intercept,GetDepth,CreateSpan,Cleanup,Dec,Normal process
    class Mark,Skip important
```


## Function flow

```mermaid

graph LR
    subgraph "Thread-Local Depth Tracking"
        D0[Depth = 0<br/>No Operations]
        D1[Depth = 1<br/>Entry Operation]
        D2[Depth = 2<br/>Nested Operation]
        DN[Depth = N<br/>Deep Nesting]
    end
    
    subgraph "Span Marking Logic"
        MS[Mark Span<br/>is_genai_entry = True]
        SS[Skip Marking<br/>No Entry Attribute]
    end
    
    subgraph "Function Flow"
        FC[Function Called]
        FE[Function Ends]
    end
    
    %% Depth transitions
    D0 -->|increment| D1
    D1 -->|increment| D2
    D2 -->|increment| DN
    DN -->|decrement| D2
    D2 -->|decrement| D1
    D1 -->|decrement| D0
    
    %% Decision logic
    D1 -->|Span Created| MS
    D2 -->|Span Created| SS
    DN -->|Span Created| SS
    
    %% Function flow
    FC --> D0
    D0 --> FE
    
    %% Styling
    classDef depth fill:#e3f2fd,stroke:#1976d2
    classDef marking fill:#f3e5f5,stroke:#7b1fa2
    classDef flow fill:#e8f5e8,stroke:#388e3c
    
    class D0,D1,D2,DN depth
    class MS,SS marking
    class FC,FE flow
```


## Sequence

```mermaid
sequenceDiagram
    participant User
    participant OllamaWrapper as Ollama Wrapper
    participant OpenAIWrapper as OpenAI Wrapper
    participant DepthCounter as Depth Counter
    participant SpanInterceptor as Span Interceptor
    participant Tracer
    
    Note over User,Tracer: Nested GenAI Call Scenario: User -> Ollama -> OpenAI
    
    User->>OllamaWrapper: call ollama_function()
    OllamaWrapper->>DepthCounter: increment_depth() -> depth=1
    OllamaWrapper->>SpanInterceptor: setup tracer.start_span interception
    OllamaWrapper->>Tracer: start_span("ollama.chat")
    SpanInterceptor->>DepthCounter: get_current_depth() -> 1
    Note over SpanInterceptor: depth == 1, mark as entry
    SpanInterceptor->>Tracer: set_attribute("is_genai_entry", True)
    
    Note over OllamaWrapper: Ollama calls OpenAI internally
    OllamaWrapper->>OpenAIWrapper: call openai_function()
    OpenAIWrapper->>DepthCounter: increment_depth() -> depth=2
    OpenAIWrapper->>SpanInterceptor: setup tracer.start_span interception
    OpenAIWrapper->>Tracer: start_span("openai.chat")
    SpanInterceptor->>DepthCounter: get_current_depth() -> 2
    Note over SpanInterceptor: depth != 1, skip marking
    SpanInterceptor->>Tracer: return span (no entry attribute)
    
    OpenAIWrapper->>DepthCounter: decrement_depth() -> depth=1
    OpenAIWrapper-->>OllamaWrapper: return result
    
    OllamaWrapper->>DepthCounter: decrement_depth() -> depth=0
    OllamaWrapper-->>User: return result
    
    Note over User,Tracer: Result: Only Ollama span marked as entry
    
    rect rgb(200, 255, 200)
        Note over Tracer: Span 1: ollama.chat (is_genai_entry=True)
    end
    
    rect rgb(255, 200, 200)
        Note over Tracer: Span 2: openai.chat (no entry attribute)
    end

```