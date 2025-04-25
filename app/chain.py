# from langchain_core.runnables import RunnablePassthrough, RunnableParallel,RunnableLambda
# from .core import initialize_chain
# from .prompt import prompt
# from .memory import get_session_memory, memory_runnable

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# llm, retriever = initialize_chain()

# chain = (
#     RunnableParallel({
#         "context": retriever | format_docs,
#         "question": RunnablePassthrough(),
#         "chat_history": RunnableLambda(get_session_memory)
#     })
#     | prompt
#     | llm
#     | memory_runnable
# )



from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from .core import initialize_chain
from .prompt import prompt
from .memory import conversation_memory

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

llm, retriever = initialize_chain()

def rag_with_memory(input: str):
    # Get relevant documents
    docs = retriever.get_relevant_documents(input)
    context = format_docs(docs)
    
    # Get conversation history
    chat_history = conversation_memory.get_memory()
    
    # Generate response
    result = prompt.invoke({
        "context": context,
        "question": input,
        "chat_history": chat_history
    })
    
    response = llm.invoke(result)
    
    # Update memory
    conversation_memory.add_exchange(input, response.content)
    
    return response

chain = RunnableLambda(rag_with_memory)