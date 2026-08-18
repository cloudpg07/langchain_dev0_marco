from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage, ToolMessage

MAX_ITERATIONS = 10
MODEL_NAME = "qwen3:8b"

# --- Tool (Langchain @tool decorator) ---

@tool
def get_product_price( product: str ) -> float:
    """Loop up the price of a product in the Catalog."""
    
    print(f"   >> Executing get_product_price() for product: {product}")
    prices = { "laptop": 1299.00, "smartphone": 799.00, "headphones": 199.00 }
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold.
    """
    
    print(f"   >> Executing apply_discount() for price: {price}, discount: '{discount_tier}'")
    discount_percentages = { "bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price *(1 - discount /100), 2)    


# --- Agent Loop ---

def run_agent(question: str):
    pass

if __name__ == "__main__":
    print("=== Langchain Agent Loop with Tool Calling ===")
    print()
    result = run_agent("What is the price of a laptop with a silver discount?") 
