import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers import ContextualCompressionRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


# Wrap our vectorstore
from langchain.retrievers.document_compressors import DocumentCompressorPipeline,EmbeddingsFilter, LLMChainExtractor
from langchain.chains import RetrievalQAWithSourcesChain



from langchain.prompts import PromptTemplate
import os 
from pprint import pprint 

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.vectorstores import Chroma


import sys
sys.path.append("../")

os.environ['ANONYMIZED_TELEMETRY']='false'


class FinanceRAG:
    def __init__(
        self,
        synonym_dict_path,
        smodel_path,
        vectordb_path,
        llm4qe_path,
        llm_path,
        llm_hyperp,
    ):
        self.device = 'cpu'

        self.logger = self.setup_logging()

        # Load synonym dictionary for company extraction
        self.load_synonym_dictionary(synonym_dict_path)

        # Load sentence embedding model
        self.load_sentence_embedding_model(smodel_path)

        # Load vectordb
        self.load_vectordb(vectordb_path, 'hit_QA_new')

        # Load query expansion LLM
        #self.load_query_expansion_llm(llm4qe_path)

        # Load generation LLM
        self.load_generation_llm(llm_path)

        #set llm hyperparameter
        self.llm_hyperp = llm_hyperp

        # Define templates
        self.define_templates()

        #set pipeline 
        self.set_compressor_pipline()

        #set llm pipeline 
        self.set_pipeline()

        #set RetrievalQAchain for langserve
        self.setRetrievalQAChain()


    def setup_logging(self):
        import logging

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        logger = logging.getLogger("FinanceRAG")
        

        return logger
            
    def extract_company(self, question):
        comp = None
        question_temp = ''.join(question.split())
        for syn in self.syn2comp:
            if syn in question_temp:
                comp = self.syn2comp[syn]
                break
        return comp

    
    def set_compressor_pipline(self):
        self.logger.info("set compressor pippline")
        self.splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0, separator=". ")
        self.redundant_filter = EmbeddingsRedundantFilter(embeddings=self.smodel)
        self.relevant_filter = EmbeddingsFilter(embeddings = self.smodel, similarity_threshold=0.1, k=2)
        self.pipeline_compressor = DocumentCompressorPipeline(
            transformers=[self.splitter, self.redundant_filter, self.relevant_filter]
            
        )


    def load_synonym_dictionary(self, synonym_dict_path):
        self.logger.info("Loading synonym dictionary...")
        self.synonym_df = pd.read_excel(synonym_dict_path)
        self.company_list = list(set(self.synonym_df["회사명"].tolist()))
        self.syn2comp = {
            syn: comp for syn, comp in zip(self.synonym_df["동의어"], self.synonym_df["회사명"])
        }

    def load_sentence_embedding_model(self, smodel_path):
        self.logger.info("Loading sentence embedding model...")
        self.smodel = HuggingFaceEmbeddings(
            model_name=smodel_path, model_kwargs={"device": self.device}, encode_kwargs={"normalize_embeddings": False}
        )


    def load_vectordb(self, vectordb_path, collection_name):
        self.logger.info("Loading vectordb...")

        self.vectordb = Chroma(
            collection_name=collection_name, 
            persist_directory=vectordb_path, embedding_function=self.smodel
        )
        self.logger.info("Loading vectordb number of chunks : "+str(self.vectordb._collection.count()))

    def load_query_expansion_llm(self, llm4qe_path):
        self.logger.info("Loading query expansion LLM...")
        self.tokenizer4qe = AutoTokenizer.from_pretrained(llm4qe_path, repo_type=True)
        self.llm_model4qe = AutoModelForCausalLM.from_pretrained(llm4qe_path)

    def load_generation_llm(self, llm_path):
        self.logger.info("Loading generation LLM...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, repo_type=True)
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_path)


    def define_templates(self):

        #templates for query expansions
        self.query_prompt_template = (
            "아래는 지식 베이스에서 검색하기위해 질문 바탕으로 검색쿼리를 생성해야하는 Task입니다.\n"
            "검색 쿼리를 생성할때 증권 이름, 회사명은 필수로 생성하고 관련 키워드를 검색쿼리로 생성합니다.\n\n"
            "###########################################\n\n"
            "입력 문장:\n"
            "21 4Q 삼성전자의 영업이익은?\n\n"
            "검색 쿼리:\n"
            "2021년, 4Q, 4분기, 삼성전자, 영업이익, PER\n\n"
            "###########################################\n\n"
            "입력 문장\n"
            "{question}\n"
            "검색 쿼리:\n"
        )


        #templates for lm generation

        self.system_chat_template="""Use the following pieces of context to answer the users question shortly.
        Given the following summaries of a long document and a question, create a final answer with references ("SOURCES"), use "SOURCES" in capital letters regardless of the number of sources.
        If you don't know the answer, just say that "I don't know", don't try to make up an answer.
        ----------------
        {summaries}

        You MUST answer in Korean and in Markdown format:"""

        self.system_chat_template2="""Use the following pieces of context to answer the users question shortly.
        Given the following summaries of a long document and a question, create a final answer with references ("SOURCES"), use "SOURCES" in capital letters regardless of the number of sources.
        If you don't know the answer, just say that "I don't know", don't try to make up an answer.
        ----------------
        {context}

        You MUST answer in Korean and in Markdown format:"""


        self.user_content = """\n\n질문:{question}\n\n답변:"""

        messages = [
            SystemMessagePromptTemplate.from_template(self.system_chat_template),
            HumanMessagePromptTemplate.from_template(self.user_content)
        ]

        messages2 = [
            SystemMessagePromptTemplate.from_template(self.system_chat_template2),
            HumanMessagePromptTemplate.from_template(self.user_content)
        ]
        self.template = ChatPromptTemplate.from_messages(messages)

        self.template_lcel = ChatPromptTemplate.from_messages(messages2)



    def set_pipeline(self):
        self.pipe = pipeline(
            "text-generation",
            model=self.llm_model,
            tokenizer=self.tokenizer,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.0001,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            bad_words_ids=[[6], [13]],
        )


    def _load_selfquery_retriever(self):
        """
        description : only retrieve chunks that has metadata related to the question 

        In progress : json parsing error with huggingface llm models ?? the reason is because the retriever relys on the llm's quality 

        ref : https://python.langchain.com/docs/modules/data_connection/retrievers/self_query/
              https://github.com/langchain-ai/langchain/issues/5882
        useage : 
        docs = retriever.get_relevant_documents(question)

        """
        document_content_description = "애널리스트가 증권 거래소에서 발간하는 증권 리포트"

        metadata_field_info = [
                        AttributeInfo(
                            name="company",
                            description="회사명, 회사 이름, 주식회사",
                            type="string",
                        ),
                        AttributeInfo(
                            name="exchange",
                            description="거래소, 증권거래소, 주식 거래소",
                            type="string",
                        ),
                        AttributeInfo(
                            name="source",
                            description="PDF 파일 이름 ",
                            type="string",
                        ),
                    ]


        self.retriever = SelfQueryRetriever.from_llm(
            self.llm_model,
            self.vectordb,
            document_content_description,
            metadata_field_info,
            verbose=True
        )

    def query_expansion(self, question):

        with torch.no_grad():
            input_text = self.query_prompt_template.format_map({"question": question})

            tokens = self.tokenizer4qe.encode(
                input_text, return_tensors="pt", return_token_type_ids=False
            )#.to(device="cuda:0", non_blocking=True)
            gen_tokens = self.llm_model4qe.generate(
                tokens,
                do_sample=True,
                top_p=0.95,
                temperature=0.00001,
                eos_token_id=self.tokenizer4qe.eos_token_id,
                pad_token_id=self.tokenizer4qe.pad_token_id,
                early_stopping=True,
                max_new_tokens=16,
                bad_words_ids=[[6], [13]],
            )
            generated = self.tokenizer4qe.batch_decode(gen_tokens)[0][
                len(input_text) :
            ].strip()
        return ''.join(generated.split('\n')[0].split(','))


    def retrieve_generate(self, question, company, use_compressor=True, top_k_docs = 2):
        if company is None:
            search_kwargs = {"k": top_k_docs}
        elif company in self.company_list:
            search_kwargs = {"k": top_k_docs} #{"filter": {"company": {"$eq": company}}, "k": 2}
        else:
            search_kwargs = {"k": top_k_docs}

        self.base_retriever = self.vectordb.as_retriever( search_type='similarity', search_kwargs = search_kwargs)

        if use_compressor:
            self.compressed_retriever = ContextualCompressionRetriever( base_compressor = self.pipeline_compressor, base_retriever = self.base_retriever, search_kwargs = search_kwargs)
            retriever = self.compressed_retriever
        else:
            retriever = self.base_retriever

        with torch.no_grad():
            self.qa = RetrievalQAWithSourcesChain.from_chain_type(
                llm=HuggingFacePipeline(pipeline=self.pipe),
                chain_type="stuff",
                retriever = retriever,
                chain_type_kwargs={
                    "prompt":self.template
                },
                return_source_documents=True,
            )

        return self.qa(question)



    def setRetrievalQAChain(self, use_compressor=True, top_k_docs = 2):
        search_kwargs = {"k": top_k_docs}

        self.base_retriever = self.vectordb.as_retriever( search_type='similarity', search_kwargs = search_kwargs)

        if use_compressor:
            self.compressed_retriever = ContextualCompressionRetriever( base_compressor = self.pipeline_compressor, base_retriever = self.base_retriever, search_kwargs = search_kwargs)
            retriever = self.compressed_retriever
        else:
            retriever = self.base_retriever

        with torch.no_grad():
            self.RetrievalQAChain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=HuggingFacePipeline(pipeline=self.pipe),
                chain_type="stuff",
                retriever = retriever,
                chain_type_kwargs={
                    "prompt":self.template
                },
                return_source_documents=True,
            )

    

    def generate(self, q):
        self.logger.info("Processing question: %s", q)

        # 1. extract_company
        comp = self.extract_company(q)
        self.logger.info("Extracted company: %s", comp)

        # 2. query expansion
        #expanded_q = self.query_expansion(q)
        #self.logger.info("Expanded query: %s", expanded_q)

        # 3. retrieve and generate
        generated_answer = self.retrieve_generate(q, comp)

        return generated_answer


dict_path = '/home/hanati/hit/output_assets/주식명_동의어사전_new.xlsx'

smodel_name = '/home/hanati/hit/sentence_embedding_plm/ko-sbert-multitask'

vectordb_path = '/home/hanati/hit/db_old'

question = '2023년도 전기차 시장 동향 알려줘'

llm_model = '/home/hanati/hit/llm/polyglot-ko-1.3b'

llm_hyp = {}

fin_rag = FinanceRAG(dict_path, smodel_name, vectordb_path, llm_model, llm_model, llm_hyp)




# RAG prompt

# LLM
model = HuggingFacePipeline(pipeline=fin_rag.pipe)

# RAG chain
chain = fin_rag.RetrievalQAChain

# Add typing for input
#class Question(BaseModel):
#    __root__: str
#chain = chain.with_types(input_type=Question)


#print(chain.invoke({"question": "2020년도 현대차 목표주가 얼마야?"}))
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

template = """다음 문맥을 사용하여 사용자의 질문에 짧게 답하십시오.  긴 문서와 질문에 대한 다음 요약이 주어졌을 때 참고 문헌("출처")이 포함된 최종 답변을 작성하세요. 답을 모르는 경우, 답을 만들려고 하지 말고 "모름"이라고 말하세요.
----------------
출처 : {context}

질문: {question}
도움 되는 답변:"""
rag_prompt_custom = PromptTemplate.from_template(template)

rag_chain = (
    {"context": fin_rag.compressed_retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt_custom
    | model
    | StrOutputParser()
)

from operator import itemgetter

from langchain.schema.runnable import RunnableParallel

rag_chain_from_docs = (
    {
        "context": lambda input: format_docs(input["documents"]),
        "question": itemgetter("question"),
    }
    | rag_prompt_custom
    | model
    | StrOutputParser()
)
rag_chain_with_source = RunnableParallel(
    {"documents": fin_rag.compressed_retriever, "question": RunnablePassthrough()}
) | {
    "documents": lambda input: [doc.page_content for doc in input["documents"]],
    "result": lambda input: [doc for doc in input["documents"]],
    "metadata": lambda input: [doc.metadata for doc in input["documents"]],
    "similarity_score": lambda input: [doc.state for doc in input["documents"]],
    "answer": rag_chain_from_docs,
}

print(rag_chain_with_source.invoke("기아차 2020년도 엽업이익 얼마야?"))



