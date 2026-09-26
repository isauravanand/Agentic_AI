"""Restaurant order manager built with LangGraph and a Groq LLM."""

import os
import random
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

MENU = {
    "pizza": {"available_quantity": 2, "price": 12.00},
    "burger": {"available_quantity": 3, "price": 10.00},
    "pasta": {"available_quantity": 1, "price": 11.00},
    "salad": {"available_quantity": 4, "price": 7.00},
    "coffee": {"available_quantity": 5, "price": 3.00},
}


class OrderDetails(TypedDict):
    dish_name: str
    required_quantity: int
    available_quantity: str


# State is the shared memory passed through every LangGraph node.
class RestaurantState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    user_input: str
    order_details: OrderDetails
    status: Literal[
        "new", "confirmed", "partial", "not_available", "ready",
        "cook_failed", "served", "serve_failed", "complete", "failed",
    ]
    next_node: Literal[
        "order_confirm", "ask_user", "cook", "serve", "generate_bill", "respond",
    ]
    order_retry_attempts: int
    cook_retry_attempts: int
    serve_retry_attempts: int
    failure_plan: list[Literal["cook", "serve"]]
    simulated_inputs: list[str]
    bill: float
    final_result: str


class ManagerDecision(TypedDict, total=False):
    is_food_order: bool
    dish_name: str
    required_quantity: int
    accept_partial: bool
    next_node: str
    message: str


def get_manager() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to .env first.")
    return ChatGroq(model="openai/gpt-oss-20b", temperature=0, api_key=api_key)


# The LLM extracts orders and manages the next action after every node.
def manager(state: RestaurantState) -> dict:
    status = state.get("status", "new")
    current_order = state.get("order_details", {})
    decision = get_manager().with_structured_output(ManagerDecision).invoke(
        [
            SystemMessage(
                content=(
                    "You are a restaurant order manager, not a general-purpose assistant. "
                    f"Menu: {list(MENU)}. Current status: {status}. "
                    f"Current order: {current_order}. "
                    f"Order retries: {state.get('order_retry_attempts', 3)}, "
                    f"cook retries: {state.get('cook_retry_attempts', 2)}, "
                    f"serve retries: {state.get('serve_retry_attempts', 2)}. "
                    "Extract dish_name and positive required_quantity. "
                    "For unrelated input set is_food_order=false. "
                    "Set accept_partial=true only when the user clearly accepts the available quantity."
                )
            ),
            HumanMessage(content=state.get("user_input", "")),
        ]
    )

    updates: dict = {"next_node": "respond"}
    dish = decision.get("dish_name", "").strip().lower()
    quantity = decision.get("required_quantity")
    if dish and quantity:
        updates["order_details"] = {
            "dish_name": dish,
            "required_quantity": max(1, int(quantity)),
            "available_quantity": "unknown",
        }

    if status == "new":
        if decision.get("is_food_order", False) and dish and quantity:
            updates["next_node"] = "order_confirm"
        else:
            updates["final_result"] = decision.get(
                "message", "I am a restaurant order agent and can only help with food orders."
            )
    elif status == "confirmed":
        updates["next_node"] = "cook"
    elif status == "ready":
        updates["next_node"] = "serve"
    elif status == "served":
        updates["next_node"] = "generate_bill"
    elif status in {"partial", "not_available"}:
        if decision.get("accept_partial", False) and status == "partial":
            available = int(current_order["available_quantity"])
            updates["order_details"] = {**current_order, "required_quantity": available}
            updates.update(status="confirmed", next_node="cook")
        elif (
            dish
            and quantity
            and state.get("order_retry_attempts", 0) > 0
            and (
                dish != current_order.get("dish_name")
                or int(quantity) != current_order.get("required_quantity")
            )
        ):
            updates.update(
                status="new",
                order_retry_attempts=state.get("order_retry_attempts", 0) - 1,
                next_node="order_confirm",
            )
        elif state.get("order_retry_attempts", 0) > 0:
            updates["next_node"] = "ask_user"
        else:
            updates.update(
                status="failed",
                final_result="I am sorry, but the order retry limit has been reached.",
            )
    elif status == "cook_failed":
        if state.get("cook_retry_attempts", 0) > 0:
            updates["next_node"] = "cook"
        else:
            updates.update(
                status="failed",
                final_result="I am sorry, but the kitchen could not prepare your order.",
            )
    elif status == "serve_failed":
        if state.get("serve_retry_attempts", 0) > 0 and state.get("cook_retry_attempts", 0) > 0:
            updates["next_node"] = "cook"
        else:
            updates.update(
                status="failed",
                final_result="I am sorry, but we could not serve your order.",
            )
    return updates


# This node reads the menu and identifies full, partial, and unavailable orders.
def order_confirm(state: RestaurantState) -> dict:
    order = state["order_details"]
    dish = order["dish_name"].lower()
    requested = max(1, order["required_quantity"])
    available = MENU.get(dish, {}).get("available_quantity", 0)
    status = "not_available" if available == 0 else "confirmed" if requested <= available else "partial"
    updates = {
        "order_details": {
            "dish_name": dish,
            "required_quantity": requested,
            "available_quantity": str(available),
        },
        "status": status,
    }
    if status != "confirmed":
        updates["order_retry_attempts"] = state.get("order_retry_attempts", 3) - 1
    return updates


def ask_user(state: RestaurantState) -> dict:
    """Get a replacement order or partial-order decision from the user."""
    answers = state.get("simulated_inputs", [])
    answer = answers.pop(0) if answers else input("The order is partial/unavailable. What would you like to do? ")
    return {
        "user_input": answer,
        "simulated_inputs": answers,
        "messages": [HumanMessage(content=answer)],
    }


def should_fail(state: RestaurantState, station: Literal["cook", "serve"]) -> tuple[bool, list]:
    plan = list(state.get("failure_plan", []))
    if station in plan:
        plan.remove(station)
        return True, plan
    return random.random() < 0.4, plan


# Cooking has a 60% success probability and uses the cook retry counter on failure.
def cook(state: RestaurantState) -> dict:
    failed, plan = should_fail(state, "cook")
    if failed:
        return {
            "status": "cook_failed",
            "cook_retry_attempts": state.get("cook_retry_attempts", 2) - 1,
            "failure_plan": plan,
        }
    return {"status": "ready", "failure_plan": plan}


# Serving has a 60% success probability and uses the serve retry counter on failure.
def serve(state: RestaurantState) -> dict:
    failed, plan = should_fail(state, "serve")
    if failed:
        return {
            "status": "serve_failed",
            "serve_retry_attempts": state.get("serve_retry_attempts", 2) - 1,
            "failure_plan": plan,
        }
    return {"status": "served", "failure_plan": plan}


def generate_bill(state: RestaurantState) -> dict:
    order = state["order_details"]
    quantity = min(order["required_quantity"], int(order["available_quantity"]))
    total = MENU[order["dish_name"]]["price"] * quantity
    return {
        "status": "complete",
        "bill": total,
        "final_result": f"Your order is complete. {quantity} {order['dish_name']}(s) served. Total bill: ${total:.2f}.",
    }


def respond(state: RestaurantState) -> dict:
    return {"final_result": state.get("final_result", "The order could not be completed.")}


def route_from_manager(state: RestaurantState) -> str:
    return state.get("next_node", "respond")


# Conditional edges let the LLM manage the workflow after every action.
workflow = StateGraph(RestaurantState)
for name, node in {
    "manager": manager,
    "order_confirm": order_confirm,
    "ask_user": ask_user,
    "cook": cook,
    "serve": serve,
    "generate_bill": generate_bill,
    "respond": respond,
}.items():
    workflow.add_node(name, node)

workflow.add_edge(START, "manager")
workflow.add_conditional_edges(
    "manager",
    route_from_manager,
    {
        "order_confirm": "order_confirm",
        "ask_user": "ask_user",
        "cook": "cook",
        "serve": "serve",
        "generate_bill": "generate_bill",
        "respond": "respond",
    },
)
workflow.add_edge("order_confirm", "manager")
workflow.add_edge("ask_user", "manager")
workflow.add_edge("cook", "manager")
workflow.add_edge("serve", "manager")
workflow.add_edge("generate_bill", END)
workflow.add_edge("respond", END)
restaurant_graph = workflow.compile()


def run_order(
    user_input: str,
    *,
    simulated_inputs: list[str] | None = None,
    failure_plan: list[Literal["cook", "serve"]] | None = None,
) -> RestaurantState:
    """Run interactively or with scripted inputs/failures for the test cases."""
    return restaurant_graph.invoke(
        {
            "user_input": user_input,
            "messages": [HumanMessage(content=user_input)],
            "status": "new",
            "order_retry_attempts": 3,
            "cook_retry_attempts": 2,
            "serve_retry_attempts": 2,
            "simulated_inputs": simulated_inputs or [],
            "failure_plan": failure_plan or [],
        }
    )


def main() -> None:
    result = run_order(input("What would you like to order? "))
    print(result["final_result"])


if __name__ == "__main__":
    main()
    