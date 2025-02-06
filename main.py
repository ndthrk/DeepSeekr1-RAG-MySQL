from doc_process import DocumentProcessor
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import re

# Chuẩn bị dữ liệu
path = "data/"
processor = DocumentProcessor(path, clear_db=True) 
processor.process_and_store_documents()

# Tạo prompt:
RAG_TEMPLATE = """
You are an assistant for answering questions related to Vietnamese history. Use the following pieces of retrieved context to think and answer the question in Vietnamese. If you don't know the answer, just say that you don't know. Use a maximum of three sentences and keep the answer concise.

<context>
{context}
</context>

Answer the following question in Vietnamese:

{question}
"""

def get_prompt(question):
    top_results = processor.search_cosine_similarity(question, top_k=2)
    context = '\n<or>\n'.join(top_results)
    print(f"############# CONTEXT: ##########\n{context}")
    prompt_template = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    prompt = prompt_template.format(context=context, question=question)
    return prompt


# Model
model = ChatOllama(
    model="deepseek-r1:8b",
    temperature=0.3,
)

# Lấy câu trả lời
remove_thinking = False
question = "Việt Nam thống nhất lúc nào?"
prompt = get_prompt(question)
response_text = model.invoke(prompt).content
if remove_thinking:
    response_text = re.sub(r'<think>*?</think>', '', response_text)
formatted_response = f"Response:\n {response_text}\n"
print(formatted_response)

# thêm lịch sử chat với bot, sửa prompt để hiểu rõ hơn các lần chat trước đó