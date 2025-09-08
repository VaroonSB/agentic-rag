from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)


class GradeHallucinations(BaseModel):
    """Binary score to indicate the generation answer is grounded in and supported by the set of facts."""

    binary_score: str = Field(
        description="LLM generation is relevant to the available set of facts, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no' to indicate the answer is grounded in and supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader  # type: ignore
