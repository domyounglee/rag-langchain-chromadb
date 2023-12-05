#!/usr/bin/env python
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langserve import add_routes
from rag import FinanceRAG


app = FastAPI(
  title="HIT Retrieval App with LLM  ",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)


if __name__ == "__main__":
    
    import uvicorn

    dict_path = '/home/hanati/hit/output_assets/주식명_동의어사전_new.xlsx'

    smodel_name = '/home/hanati/hit/sentence_embedding_plm/ko-sbert-multitask'

    vectordb_path = '/home/hanati/hit/db_old'

    question = '2023년도 전기차 시장 동향 알려줘'

    llm_model = '/home/hanati/hit/llm/polyglot-ko-1.3b'

    llm_hyp = {}

    fin_rag = FinanceRAG(dict_path, smodel_name, vectordb_path, llm_model, llm_model, llm_hyp)
    
    chain = fin_rag.qa

    # Add routes for the chain
    add_routes(app, chain, path='/hit_rag')

    uvicorn.run(app, host="localhost", port=8888)
