from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)
prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()
