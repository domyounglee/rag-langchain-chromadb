# rag-langchain-chromadb

## **Dependency**

```
langchain==0.0.278

chromadb==0.4.3

sqlite3==2.6.0

pysqlite3-binary
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

