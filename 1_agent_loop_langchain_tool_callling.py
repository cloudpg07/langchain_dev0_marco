from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL_NAME = "qwen3:8b"

# --- Tool (Langchain @tool decorator) ---


@tool("get_product_price")
def get_product_price(product: str) -> float:
    """Loop up the price of a product in the Catalog."""

    print(f"   >> Executing get_product_price() for product: {product}")
    prices = {"laptop": 1299.00, "smartphone": 799.00, "headphones": 199.00}
    return prices.get(product, 0)


@tool("apply_discount")
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


# --- Agent Loop ---
@traceable(name="Langchain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    # Initialize the chat model with tools
    llm = init_chat_model(MODEL_NAME, model_provider="ollama", temperature=0.0)
    
    # Bind the tools to the LLM so it can call them during the conversation
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(content=("""
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
            "ask them which tier to use — do NOT assume one.
        """)),
        HumanMessage(content=question),
    ]

    # Loop through iterations to allow the agent to call tools and process responses
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        
        # Invoke the LLM with the current messages. 
        # Send the entire conversation history so far, including any tool observations.
        ai_message = llm_with_tools.invoke(messages)
        tools_calls = ai_message.tool_calls

        # If no tools  calls, this is the final answer
        if not tools_calls:
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content
        
        #  Append the AI's message to the conversation history
        messages.append(ai_message)
        
        # Log the tool calls requested by the LLM
        for tool_call in tools_calls:
            print(f" >> [Tool Call Requested] {tool_call.get('name')} with args: {tool_call.get('args', {})}")
            
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id")
            
            print(f" >> [Tool Selected] {tool_name} with args: {tool_args}")
            tool_to_use = tools_dict.get(tool_name)
            
            if tool_to_use is None:
                observation = f"Error: Tool '{tool_name}' not found."
                print(f" >> [Tool Observation] {observation}")
            else:
                # Invoke the selected tool with the provided arguments
                observation = tool_to_use.invoke(tool_args)
                print(f" >> [Tool Observation] {observation}")
            
            # Append the tool's observation to the conversation history so the LLM can see it in the next iteration
            messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    print("Error: Maximum iterations reached without a final answer.")
    return None


if __name__ == "__main__":
    print("=== Langchain Agent Loop with Tool Calling ===")
    print()
    result = run_agent("What is the price of a laptop with a silver discount?")
