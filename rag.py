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
        self.relevant_filter = EmbeddingsFilter(embeddings = self.smodel, similarity_threshold=0.3, k=2)
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

        self.user_content = """\n\n질문:{question}\n\n답변:"""

        messages = [
            SystemMessagePromptTemplate.from_template(self.system_chat_template),
            HumanMessagePromptTemplate.from_template(self.user_content)
        ]
        self.template = ChatPromptTemplate.from_messages(messages)



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
            self.qa = RetrievalQAWithSourcesChain.from_chain_type(
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


if __name__ == '__main__':

    dict_path = '/home/hanati/hit/output_assets/주식명_동의어사전_new.xlsx'

    smodel_name = '/home/hanati/hit/sentence_embedding_plm/ko-sbert-multitask'

    vectordb_path = '/home/hanati/hit/db_old'

    question = '2023년도 전기차 시장 동향 알려줘'

    llm_model = '/home/hanati/hit/llm/polyglot-ko-1.3b'

    llm_hyp = {}

    fin_rag = FinanceRAG(dict_path, smodel_name, vectordb_path, llm_model, llm_model, llm_hyp)

    from fastapi import FastAPI
    from langserve import add_routes
    import uvicorn
    chain = fin_rag.qa
    app = FastAPI(title="Retrieval App")

    # Add routes for the chain
    add_routes(app, chain, path='/hit')

    uvicorn.run(app, host="localhost", port=8000)

    
    """   
    import gradio as gr
    def respond(message, chat_history):  # 채팅봇의 응답을 처리하는 함수를 정의합니다.

        result = fin_rag.generate(message)

        bot_message = result['answer']

        for i, doc in enumerate(result['source_documents']):
            bot_message += '[' + str(i+1) + '] ' + doc.metadata['source'] + '(' + str(doc.metadata['exchange']) + ') '

        chat_history.append((message, bot_message))  # 채팅 기록에 사용자의 메시지와 봇의 응답을 추가합니다.

        return "", chat_history  # 수정된 채팅 기록을 반환합니다.

    with gr.Blocks() as demo:  # gr.Blocks()를 사용하여 인터페이스를 생성합니다.
        chatbot = gr.Chatbot(label="채팅창")  # '채팅창'이라는 레이블을 가진 채팅봇 컴포넌트를 생성합니다.
        msg = gr.Textbox(label="입력")  # '입력'이라는 레이블을 가진 텍스트박스를 생성합니다.
        clear = gr.Button("초기화")  # '초기화'라는 레이블을 가진 버튼을 생성합니다.

        msg.submit(respond, [msg, chatbot], [msg, chatbot])  # 텍스트박스에 메시지를 입력하고 제출하면 respond 함수가 호출되도록 합니다.
        clear.click(lambda: None, None, chatbot, queue=False)  # '초기화' 버튼을 클릭하면 채팅 기록을 초기화합니다.

    demo.launch(debug=True)  # 인터페이스를 실행합니다. 실행하면 사용자는 '입력' 텍스트박스에 메시지를 작성하고 제출할 수 있으며, '초기화' 버튼을 통해 채팅 기록을 초기화 할 수 있습니다.

    """