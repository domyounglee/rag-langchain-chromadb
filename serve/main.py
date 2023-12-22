#!/usr/bin/env python

# Imports
from fastapi import FastAPI
import sys
import uvicorn
import argparse
from langserve import add_routes

from langchain.pydantic_v1 import BaseModel
from src.rag import FinanceRAG

from src.utils import read_json, setup_logging

class Question(BaseModel):
    __root__: str

app = FastAPI(
    title="HIT Retrieval App with LLM",
    version="1.0",
    description="A simple API server using Langchain's Runnable interfaces"
)


logger = setup_logging()
configs = read_json('rag_config.json')

fin_rag = FinanceRAG(configs, logger)
chain = fin_rag.chain.with_types(input_type=Question) #  types for playground input field

# Add routes for the chain
add_routes(app, chain, path='/hit_rag')

if __name__ == "__main__":

  uvicorn.run(app, host="localhost", port=8888)
