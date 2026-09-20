from typing import TypedDict
from langgraph.graph import StateGraph, START, END
class State(TypedDict):
    message: str

def hello_node(state: State):
    return {
        "message": "Hello " + state["message"]
    }

graph = StateGraph(State)

graph.add_node("hello", hello_node)

graph.add_edge(START, "hello")
graph.add_edge("hello", END)

app = graph.compile()

result = app.invoke({
    "message": "Saurav"
})

print(result)