from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# -------------------------
# 1. Define State
# -------------------------

class State(TypedDict):
    number: int
    node: str


# -------------------------
# 2. Node 1
# -------------------------

def node1(state: State):
    print(f"Node 1: Current number = {state['number']}")

    return {
        "node": "node1"
    }


# -------------------------
# 3. Node 2
# -------------------------

def node2(state: State):
    number = state["number"]

    if number < 100:
        number = number * 2
        print(f"Node 2: Doubled number = {number}")

    return {
        "number": number,
        "node": "node2"
    }


# -------------------------
# 4. Node 3
# -------------------------

def node3(state: State):
    number = state["number"]

    print(f"Node 3: Checking number = {number}")

    if number < 100:
        return {
            "node": "node1"
        }

    else:
        return {
            "node": "finish"
        }


# -------------------------
# 5. Routing Function
# -------------------------

def route(state: State):

    if state["node"] == "node1":
        return "node1"

    elif state["node"] == "node2":
        return "node2"

    elif state["node"] == "finish":
        return "finish"


# -------------------------
# 6. Create Graph
# -------------------------

graph = StateGraph(State)


# Add nodes
graph.add_node("node1", node1)
graph.add_node("node2", node2)
graph.add_node("node3", node3)


# -------------------------
# 7. Add Edges
# -------------------------

graph.add_edge(START, "node1")

graph.add_edge("node1", "node2")

graph.add_edge("node2", "node3")


# Node 3 decides where to go
graph.add_conditional_edges(
    "node3",
    route,
    {
        "node1": "node1",
        "node2": "node2",
        "finish": END
    }
)


# Compile
app = graph.compile()


# -------------------------
# 8. Run Graph
# -------------------------

result = app.invoke({
    "number": 5,
    "node": ""
})


print("\nFinal State:")
print(result)