# rag-langchain-chromadb

## **Dependency**

```


langchain==0.0.278

chromadb==0.4.3

simsimd

langserve[all]==0.0.32

sse_starlette

uvicorn

sqlite3==2.6.0

pysqlite3-binary

gradio_client==0.3.0

gradio==3.32.0 

```

**특이사항**

chromadb 코드 변경 

```python
site-packages/chromadb/__init__.py 
```

sqlite3 버전 안맞아서 raise 하는 부분에 raise 지우고 아래코드 삽입 

```python
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

```



https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/retrievers/document_compressors/embeddings_filter.py

line 164 바꿈