from typing import List, Dict, Tuple
from collections import deque
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda

class ConversationMemory:
    def __init__(self, max_length: int = 5):
        self.memory = deque(maxlen=max_length)
    
    def add_exchange(self, user_query: str, ai_response: str):
        self.memory.extend([
            HumanMessage(content=user_query),
            AIMessage(content=ai_response)
        ])
    
    def get_memory(self) -> List[Dict]:
        return list(self.memory)
    
    def clear(self):
        self.memory.clear()

# Initialize global memory
conversation_memory = ConversationMemory()

def get_session_memory():
    return conversation_memory.get_memory()

def update_memory(input: str, output: str):
    conversation_memory.add_exchange(input, output)
    return output

memory_runnable = RunnableLambda(update_memory)