# langchain-qwq

**langchain-qwq** 提供了LangChain与阿里云QwQ模型及其他Qwen系列模型的集成，特别针对Qwen3模型进行了增强支持。

## 特性
该库包含了以下核心特性：

- **QwQ模型集成**: 原生支持QwQ模型
- **Qwen3模型集成**: 原生支持Qwen3系列模型，并提供切换混合推理模式的能力
- **其他Qwen模型**: 原生支持Qwen-Max, Qwen2.5, 以及其他Qwen系列模型
- **视觉模型**: 原生支持Qwen-VL系列视觉模型
- **流式支持**: 同步和异步流式支持
- **工具调用**: 支持并行执行的函数调用
- **结构化输出**: JSON模式和函数调用支持结构化响应
- **推理访问**: 直接访问模型推理/思考内容，提供两个基本函数，快速拼接推理与普通回复内容

## 安装

```bash
pip install -U langchain-qwq
```

或者如果你想在克隆仓库时安装额外的依赖项：

```bash
pip install -U langchain-qwq[docs]
pip install -U langchain-qwq[test]
pip install -U langchain-qwq[codespell]
pip install -U langchain-qwq[lint]
pip install -U langchain-qwq[typing]
```



但是对于开发来说，只需要安装最基本上的依赖即可。无法安装额外依赖项。

## 环境变量

通过设置以下环境变量来配置凭证：

* `DASHSCOPE_API_KEY`: 阿里云DashScope API密钥，用于访问QwQ或Qwen模型（必填）
* `DASHSCOPE_API_BASE`: (可选) API基础URL，默认为"https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

**注意**: 对于国内用户，通常需要将`DASHSCOPE_API_BASE`设置为国内端点，因为langchain-qwq默认使用国际阿里云的DashScope API。

## ChatQwQ

`ChatQwQ` 类提供了QwQ模型的聊天功能，并支持推理能力。

### 基本使用

```python
from langchain_qwq import ChatQwQ

model = ChatQwQ(model="qwq-32b")
response = model.invoke("你好，今天感觉怎么样？")
print(response.content)
```

### 访问推理内容

QwQ模型提供了推理/思考内容，可以通过`additional_kwargs`访问：

```python
response = model.invoke("你好，今天感觉怎么样？")
content = response.content
reasoning = response.additional_kwargs.get("reasoning_content", "")
print(f"Response: {content}")
print(f"Reasoning: {reasoning}")
```

### 流式输出

#### 同步流式

```python
model = ChatQwQ(model="qwq-32b")

is_first = True
is_end = True

for msg in model.stream("你好，今天感觉怎么样？"):
    if hasattr(msg, 'additional_kwargs') and "reasoning_content" in msg.additional_kwargs:
        if is_first:
            print("Starting to think...")
            is_first = False   
        print(msg.additional_kwargs["reasoning_content"], end="", flush=True)
    elif hasattr(msg, 'content') and msg.content:
        if is_end:
            print("\nThinking ended")
            is_end = False
        print(msg.content, end="", flush=True)
```

#### 异步流式

```python
is_first = True
is_end = True

async for msg in model.astream("你好，今天感觉怎么样？"):
    if hasattr(msg, 'additional_kwargs') and "reasoning_content" in msg.additional_kwargs:
        if is_first:
            print("开始思考...")
            is_first = False
        print(msg.additional_kwargs["reasoning_content"], end="", flush=True)
    elif hasattr(msg, 'content') and msg.content:
        if is_end:   
            print("\n思考结束")
            is_end = False
        print(msg.content, end="", flush=True)
```

### 方便的推理显示

使用工具函数来方便地显示推理内容：

```python
from langchain_qwq.utils import convert_reasoning_to_content

for msg in convert_reasoning_to_content(model.stream("你好，今天感觉怎么样？")):
    print(msg.content, end="", flush=True)

from langchain_qwq.utils import aconvert_reasoning_to_content

async for msg in aconvert_reasoning_to_content(model.astream("你好")):
    print(msg.content, end="", flush=True)
```

你也可以自定义思考标签：

```python
async for msg in aconvert_reasoning_to_content(
    model.astream("你好，今天感觉怎么样？"), 
    think_tag=("<Start>", "<End>")
):
    print(msg.content, end="", flush=True)
```

### 工具调用

#### 基本工具使用

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """查询某个地方的天气"""
    return f"{city}的天气是晴天"

bound_model = model.bind_tools([get_weather])
response = bound_model.invoke("西安的天气如何")
print(response.tool_calls)
```

#### 并行工具调用

```python
# Enable parallel tool calls
response = bound_model.invoke(
    "西安和温州的天气如何", 
    parallel_tool_calls=True
)
print(response.tool_calls)
```

### 结构化输出

#### JSON模式

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

struct_model = model.with_structured_output(User, method="json_mode")
response = struct_model.invoke("我叫张三，今年25岁")
print(response)  # User(name='John', age=25)
```

#### 函数调用模式

```python
struct_model = model.with_structured_output(User, method="function_calling")
response = struct_model.invoke("我叫李四，今年30岁")
print(response)  # User(name='Alice', age=30)
```

### 与LangChain Agents集成

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

agent = create_tool_calling_agent(
    model,
    [get_weather],
    prompt=ChatPromptTemplate.from_messages([
        ("system", "你是一个天气助手"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
)

agent_executor = AgentExecutor(agent=agent, tools=[get_weather])
result = agent_executor.invoke({"input": "西安的天气如何"})
print(result)
```

## ChatQwen

`ChatQwen` 提供了更好的Qwen3和其他Qwen系列模型的支持，包括增强的Qwen3思考功能的参数支持。

### 基本使用

```python
from langchain_qwq import ChatQwen

model = ChatQwen(model="qwen3-32b")
response = model.invoke("你好，今天感觉怎么样？")
print(response.content)

reasoning = response.additional_kwargs.get("reasoning_content", "")
print(f"Reasoning: {reasoning}")
```

### 思考控制（仅Qwen3）

#### 禁用思考模式

对于Qwen3模型，它支持混合推理模式，因此你可以关闭思考模式通过`enable_thinking`参数。

```python
model = ChatQwen(model="qwen3-32b", enable_thinking=False)
response = model.invoke("你好，今天感觉怎么样？")
print(response.content)  # No reasoning content
```

#### 启用思考模式

当然了也可以启用思考模式，但是开源模型默认是开启的，而闭源的模型则是默认关闭的。
```python
model = ChatQwen(model="qwen-plus-latest", enable_thinking=True)
response = model.invoke("你好，今天感觉怎么样？")
reasoning = response.additional_kwargs.get("reasoning_content", "")
print(f"Reasoning: {reasoning}")
```

#### 控制思考长度

```python
model = ChatQwen(model="qwen3-32b", thinking_budget=20)
response = model.invoke("你好，今天感觉怎么样？")
reasoning = response.additional_kwargs.get("reasoning_content", "")
print(f"Limited reasoning: {reasoning}")
```

### 其他Qwen模型

#### Qwen-Max

```python
model = ChatQwen(model="qwen-max")
print(model.invoke("你好，今天感觉怎么样？").content)


bound_model = model.bind_tools([get_weather])
response = bound_model.invoke("西安和温州的天气如何", parallel_tool_calls=True)
print(response.tool_calls)


struct_model = model.with_structured_output(User, method="json_mode")
result = struct_model.invoke("我叫王五，今年28岁")
print(result)
```

#### Qwen2.5-72B

```python
model = ChatQwen(model="qwen2.5-72b-instruct")
print(model.invoke("你好，今天感觉怎么样？").content)

bound_model = model.bind_tools([get_weather])
struct_model = model.with_structured_output(User, method="json_mode")
```

### 视觉模型

```python
from langchain_core.messages import HumanMessage

model = ChatQwen(model="qwen-vl-max-latest")

messages = [
    HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/image.jpg"
            },
        },
        {"type": "text", "text": "图中描绘的是什么景象?"},
    ])
]

response = model.invoke(messages)
print(response.content)
```

### 接入第三方/本地大模型

langchain-qwq的ChatQwen和ChatQwQ都支持接入第三方提供的开源Qwen大模型。

例如**硅基流动**
#### 硅基流动
```python
import os
os.environ["DASHSCOPE_API_BASE"] = "https://api.siliconflow.cn/v1"
os.environ["DASHSCOPE_API_KEY"] = os.getenv("SILICONFLOW_API_KEY") or ""
model = ChatQwen(model="Qwen/Qwen3-32B")
await model.ainvoke("你好")
```

流式输出也支持
```python
async for msg in model.astream("你好"):
    print(msg.content,end="",flush=True)
```

#### 本地大模型
同时支持自己本地部署的大模型，例如使用vllm。
```python
os.environ["DASHSCOPE_API_BASE"]="http://localhost:8000/v1"
os.environ["DASHSCOPE_API_KEY"]="sk-1234567890"
model=ChatQwen(model="Qwen/Qwen3-8B")
await model.ainvoke("你好")
```
vllm支持工具调用
```python
bind_model=model.bind_tools([get_weather])
await bind_model.ainvoke("查询温州和西安的天气")
```
混合推理特性也支持，但是不能使用ChatQwen提供的API
```python
model=ChatQwen(model="Qwen/Qwen3-8B",extra_body={
    "chat_template_kwargs":{
        "enable_thinking":False
    }
})
```


## 如何选择？
对于选择ChatQwQ还是ChatQwen，这里做出建议
如果接入的是QwQ模型建议使用ChatQwQ，因为这个类根据其QwQ模型本身做了改进
如果接入的是Qwen其它模型，尤其是Qwen3模型，建议使用ChatQwen,因为其能对这个模型提供更佳的使用体验
这两个类的对比表格如下所示：

| Feature | ChatQwQ | ChatQwen |
|---------|---------|----------|
| QwQ Models | ✅ Primary | ✅ Supported |
| Qwen3 Models | ✅ Basic | ✅ Enhanced |
| Other Qwen Models | ❌ | ✅ Full Support |
| Vision Models | ❌ | ✅ Supported |
| Thinking Control | ❌ | ✅ (Qwen3 only) |
| Thinking Budget | ❌ | ✅ (Qwen3 only) |

