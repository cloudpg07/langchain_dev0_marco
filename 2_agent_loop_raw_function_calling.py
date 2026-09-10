from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

import ollama
from ollama import chat
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL_NAME = "qwen3:8b"

# --- Tool (Langchain @tool decorator) ---


@traceable(name="get_product_price")
def get_product_price(product: str) -> float:
    """Loop up the price of a product in the Catalog."""

    print(f"   >> Executing get_product_price() for product: {product}")
    prices = {"laptop": 1299.00, "smartphone": 799.00, "headphones": 199.00}
    return prices.get(product, 0)


@traceable(name="apply_discount")
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold.
    """

    print(
        f"   >> Executing apply_discount() for price: {price}, discount: '{discount_tier}'"
    )
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# Difference 2: Without @tool, we must MANUALLY define the JSON schema for each function.
# This is exactly what LangChain's @tool decorator generates automatically
# from the function's type hints and docstring.
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

# NOTE: Ollama can also auto-generate these schemas if you pass the functions
# directly as tools (similar to LangChain's @tool decorator):
#   tools_for_llm = [get_product_price, apply_discount]
# However, this requires your docstrings to follow the Google docstring format
# so Ollama can parse parameter descriptions from the Args section. For example:
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """
# We keep the manual JSON version here so you can see what @tool hides from you.

# --- Helper: traced Ollama call ---
# Difference 3: Without LangChain, we must manually trace LLM calls for LangSmith.


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    """Wrapper for the Ollama chat function to make it traceable."""
    return ollama.chat(MODEL_NAME, tools=tools_for_llm, messages=messages)


# --- Agent Loop ---
@traceable(name="Langchain Agent Loop")
def run_agent(question: str):

    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"Question: {question}")
    print("=" * 60)

    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            ),
        },
        {"role": "user", "content": question},
    ]

    # Loop through iterations to allow the agent to call tools and process responses
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Difference 5: ollama.chat() directly instead of llm_with_tools.invoke()
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message

        tool_calls = ai_message.tool_calls

        # If no tools  calls, this is the final answer
        if not tool_calls:
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content

        #  Append the AI's message to the conversation history
        messages.append(
            {
                "role": ai_message.role,
                "content": ai_message.content,
                "tool_calls": ai_message.tool_calls,
            }
        )

        # Log the tool calls requested by the LLM
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments

            print(
                f" >> [Tool Call Requested] {tool_name} with args: {tool_args}"
            )

            print(f" >> [Tool Selected] {tool_name} with args: {tool_args}")
            tool_to_use = tools_dict.get(tool_name)

            if tool_to_use is None:
                observation = f"Error: Tool '{tool_name}' not found."
                print(f" >> [Tool Observation] {observation}")
            else:
                # Invoke the selected tool with the provided arguments
                observation = tool_to_use(**tool_args)
                print(f" >> [Tool Observation] {observation}")

            # Append the tool's observation to the conversation history so the LLM can see it in the next iteration
            messages.append({"role": "tool", "content": str(observation)})

    print("Error: Maximum iterations reached without a final answer.")
    return None


if __name__ == "__main__":
    print("=== Langchain Agent Loop with Tool Calling ===")
    print()
    result = run_agent("What is the price of a laptop with a silver discount?")
