# 这是一个示例 Python 脚本。
import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.embeddings import Embeddings
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

chat_model = ChatOpenAI(
    model="deepseek-chat",
    api_key= os.getenv("deepseek_api_key"),
    base_url= os.getenv("deepseek_base_url")
)


system_message = SystemMessage(
    content=(
        "You are an expert in English vocabulary explanation. "
        "The user will give you an English word. "
        "Please output the following information:\n"
        "1. Part of speech (e.g., n., v., adj., etc.)\n"
        "2. English definition\n"
        "3. Chinese translation\n"
        "4. Three common phrases using this word (with English and Chinese meanings)"
        "Pay attention to the following:\n"
        "1. Do not include any language that sounds like an AI introduction; directly present your answer without saying things like 'As an AI...'.\n"
        "2. Keep your explanations concise and effective.\n"
        "3. Do not provide Chinese pinyin, as the users are native Chinese speakers."
    )
)

file_path = "data/english_text.txt"

loader = TextLoader(file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        r"\n{2,}",  # ① 空行：表示一个说话人结束了
        r"(?=^[A-Za-z]+:\s)",  # ② 说话人开头（Todd: / Rachel:）作为断点
        r"(?<=[.!?])\s+",  # ③ 句号、问号、感叹号后面断句（保留标点）
        r"(?<=[.。！？])"  # ④ 中文句末（可选）
    ],
    is_separator_regex=True,  # 告诉程序这些是正则表达式
    keep_separator=True,  # 保留句号、感叹号、冒号等标点
    chunk_size=500,  # 每段最大长度（字数或字符数）
    chunk_overlap=150
)

split_docs = text_splitter.split_documents(docs)

# for i, d in enumerate(split_docs[:5]):
#     print(f"\n--- Chunk {i+1} ---\n")
#     print(d.page_content)

embedding_model= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector = FAISS.from_documents(split_docs, embedding_model)

vector.save_local("data/english_text_faiss")

word = input("Enter an English word: ")

human_message = HumanMessage(content=f"Explain the word: {word}")

response = chat_model.invoke([system_message, human_message])

print("\n📘 DeepSeek Output:\n")
print(response.content)
print("hello")

