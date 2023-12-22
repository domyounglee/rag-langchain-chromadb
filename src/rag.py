import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
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
from langchain.retrievers.document_compressors import DocumentCompressorPipeline,EmbeddingsFilter



from langchain.prompts import PromptTemplate
import os 

from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.vectorstores import Chroma


from operator import itemgetter
from langchain.schema import StrOutputParser
import argparse

from .utils import format_docs, read_json, setup_logging

os.environ['ANONYMIZED_TELEMETRY']='false'

class FinanceRAG:
    """
    FinanceRAG is a class for handling finance-related queries using RAG and LLMs.
    """

    def __init__(self, configs, logger):
        self.device = 'cpu'
        self.logger = logger

        # Load resources
        self.load_synonym_dictionary(configs['dict_path'])
        self.load_sentence_embedding_model(configs['smodel_name'])
        self.load_vectordb(configs['vectordb_path'], 'hit_QA_new')
        self.load_generation_llm(configs['llm_model'])
        self.llm_hyperp = configs['generation_hyp']

        self.define_templates()
        self.set_compressor_pipeline()
        self.set_retriever_pipeline()
        self.set_pipeline(configs['generation_hyp'])
        self.set_keyword_pipeline(configs['keyword_hyp'])
        self.set_chain()
            
    def extract_company(self, question):
        comp = None
        question_temp = ''.join(question.split())
        for syn in self.syn2comp:
            if syn in question_temp:
                comp = self.syn2comp[syn]
                break
        return comp
    
    def set_compressor_pipeline(self):
        self.logger.info("set compressor pipeline")
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


    def load_generation_llm(self, llm_path):
        self.logger.info("Loading generation LLM...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, repo_type=True)
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_path)


    def define_templates(self):

        #templates for query expansions
        self.query_prompt_template = (
            "아래는 지식 베이스에서 검색하기위해 질문 바탕으로 검색쿼리를 생성해야하는 Task입니다.\n"
            "검색 쿼리를 생성할때 증권 이름, 회사명은 필수로 생성하고 관련 키워드를 검색쿼리로 생성합니다.\n\n"
            "\n\n"
            "입력 문장:21년도 4Q 삼성전자의 영업이익은?\n\n"
            "검색 쿼리:2021년, 4분기, 삼성전자, 영업이익 \n\n"
            "\n\n"
            "입력 문장:{question}\n\n"
            "검색 쿼리:"
        )


        #templates for lm generation
        self.template = (
            "다음 문맥을 사용하여 사용자의 질문에 짧게 답하십시오.  긴 문서와 질문에 대한 다음 요약이 주어졌을 때 참고 문헌('출처')이 포함된 최종 답변을 작성하세요. 답을 모르는 경우, 답을 만들려고 하지 말고 '모름'이라고 말하세요.\n"
            "----------------\n"
            "출처 : {context}\n\n"
            "질문: {question}"
            "도움 되는 답변:"
        )


    def set_pipeline(self, hp):
        self.logger.info("set lm pipe")

        self.pipe = pipeline(
            "text-generation",
            model=self.llm_model,
            tokenizer=self.tokenizer,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **hp
        )

    def set_keyword_pipeline(self, hp):
        self.logger.info("set keyword lm pipe")

        self.keyword_pipe = pipeline(
            "text-generation",
            model=self.llm_model,
            tokenizer=self.tokenizer,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **hp
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


    def set_retriever_pipeline(self, use_compressor=True, top_k_docs = 2):

        self.logger.info("set retriever pipe")

        search_kwargs = {"k": top_k_docs}

        self.base_retriever = self.vectordb.as_retriever( search_type='similarity', search_kwargs = search_kwargs)

        if use_compressor:
            self.compressed_retriever = ContextualCompressionRetriever( base_compressor = self.pipeline_compressor, base_retriever = self.base_retriever, search_kwargs = search_kwargs)
            retriever = self.compressed_retriever
        else:
            retriever = self.base_retriever


    def set_chain(self):
        self.logger.info("set chain")

        # out parser
        output_parser = StrOutputParser()
        # LLM
        model = HuggingFacePipeline(pipeline=self.pipe)
        keyword_model = HuggingFacePipeline(pipeline=self.keyword_pipe)

        # RAG prompt
        rag_query_promp = PromptTemplate.from_template(self.query_prompt_template)
        rag_prompt = PromptTemplate.from_template(self.template)

        #LCEL
        rag_chain_extract_keywords = (

            { "question" : RunnablePassthrough() } | rag_query_promp  | keyword_model | output_parser | self.compressed_retriever 
        )

        rag_chain_from_docs = (
            {
                "context": lambda input: format_docs(input["documents"]),
                "question": itemgetter("question") , 
            }
            | rag_prompt
            | model
            | output_parser
        )

        self.chain = RunnableParallel(
            { "documents": rag_chain_extract_keywords, "question": RunnablePassthrough()  }
        ) | {
            "documents": lambda input: [doc.page_content for doc in input["documents"]],
            "metadata": lambda input: [doc.metadata for doc in input["documents"]],
            "similarity_score": lambda input: [str(doc.state['query_similarity_score']) for doc in input["documents"]],
            "answer": rag_chain_from_docs,
        }

     
if __name__ == "__main__":

    #setup logger
    logger = setup_logging()

    #setup argparser
    logger.info("read arguments")
    parser = argparse.ArgumentParser(description='Parse a YAML configuration file.')
    parser.add_argument('config_path', help='Path to the YAML configuration file')
    args = parser.parse_args()

    #read config
    configs = read_json(args.config_path)

    #call the chain
    fin_rag = FinanceRAG(configs, logger)
    chain = fin_rag.chain
