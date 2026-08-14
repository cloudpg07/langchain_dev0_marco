from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")

    information = """
    Elon Musk (born June 28, 1971) is a businessman and former public official who is the CEO and largest shareholder of Tesla and SpaceX. Musk has been the wealthiest person in the world since 2025, and became the only trillionaire in terms of US dollars in June 2026; as of August 8, 2026, Forbes estimates his net worth to be US$823 billion.
    """

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # Temperature is the randomness of the model's output. 0 means no randomness. 1 means full randomness.
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    # llm  = ChatOllama(temperature=0, model="gemma3:270m")
    # runnable object is a chain of prompts and models. It is a callable object that can be invoked to get a response.
    # | is a pipe operator. It means the output of the summary_prompt_template is the input of the llm.
    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
