
from langchain.tools import tool

@tool
def Add(first: int, second: int) -> str:
    """Performs Addition of two numbers.
    """
    result = first + second
    return result

@tool
def Sub(first: int, second: int) -> str:
    """Performs Subtraction of two numbers.
    """
    result = first - second
    return result

@tool
def Div(first: int, second: int) -> str:
    """Performs null on asking Division of two numbers.
    """
    return 0