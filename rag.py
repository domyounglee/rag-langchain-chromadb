import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate

from pprint import pprint 

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
        self.logger = self.setup_logging()

        # Load synonym dictionary for company extraction
        self.load_synonym_dictionary(synonym_dict_path)

        # Load sentence embedding model
        self.load_sentence_embedding_model(smodel_path)

        # Load vectordb
        self.load_vectordb(vectordb_path)

        # Load query expansion LLM
        #self.load_query_expansion_llm(llm4qe_path)

        # Load generation LLM
        self.load_generation_llm(llm_path)

        #set llm hyperparameter
        self.llm_hyperp = llm_hyperp

        # Define templates
        self.define_templates()

        #set pipeline 
        self.set_pipeline()


    def setup_logging(self):
        import logging

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        logger = logging.getLogger("FinanceRAG")
        

        return logger

    def load_synonym_dictionary(self, synonym_dict_path):
        self.logger.info("Loading synonym dictionary...")
        self.synonym_df = pd.read_excel(synonym_dict_path)
        self.company_list = list(set(self.synonym_df["회사명"].tolist()))
        self.syn2comp = {
            syn: comp for syn, comp in zip(self.synonym_df["동의어"], self.synonym_df["회사명"])
        }

    def load_sentence_embedding_model(self, smodel_path):
        self.logger.info("Loading sentence embedding model...")
        self.hf = HuggingFaceEmbeddings(
            model_name=smodel_path, model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": False}
        )

    def load_vectordb(self, vectordb_path):
        self.logger.info("Loading vectordb...")
        self.vectordb = Chroma(
            collection_name='hit_QA_poc', 
            persist_directory=vectordb_path, embedding_function=self.hf
        )
        self.logger.info("Loading vectordb...")

    def load_query_expansion_llm(self, llm4qe_path):
        self.logger.info("Loading query expansion LLM...")
        self.tokenizer4qe = AutoTokenizer.from_pretrained(llm4qe_path, repo_type=True)
        self.llm_model4qe = AutoModelForCausalLM.from_pretrained(llm4qe_path)

    def load_generation_llm(self, llm_path):
        self.logger.info("Loading generation LLM...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, repo_type=True)
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_path)


    def define_templates(self):
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

        self.system_chat_template = (
            "당신은 증권 리포트를 보고 증권에 관한 질문에 올바른 답변을 해주는 지능형 비서입니다. 대답은 간결하고 명확하게 합니다."
        )

        user_content = """시작!\n\n출처:\n{context}\n\n질문:{question}\n\n답변:"""

        self.template = self.system_chat_template + "\n\n" + user_content


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
            early_stopping=True,
            bad_words_ids=[[6], [13]],
        )
        
    def extract_company(self, question):
        comp = None
        question_temp = ''.join(question.split())
        for syn in self.syn2comp:
            if syn in question_temp:
                comp = self.syn2comp[syn]
                break
        return comp

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


    def retrieve_generate(self, question, company):
        if company is None:
            search_kwargs = {"k": 2}
        elif company in self.company_list:
            search_kwargs = {"k": 2} #{"filter": {"company": {"$eq": company}}, "k": 2}
        else:
            search_kwargs = {"k": 2}

        with torch.no_grad():
            self.qa = RetrievalQA.from_chain_type(
                llm=HuggingFacePipeline(pipeline=self.pipe),
                chain_type="stuff",
                retriever=self.vectordb.as_retriever(
                    search_type="similarity", search_kwargs=search_kwargs
                ),
                chain_type_kwargs={
                    "prompt": PromptTemplate(
                        template=self.template, input_variables=["context", "question"]
                    )
                },
                return_source_documents=True,
            )

        return self.qa(question)


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
        self.logger.info("Generated answer: %s", generated_answer)

        return generated_answer


if __name__ == '__main__':

    dict_path = 'output_assets/주식명_동의어사전_new.xlsx'

    smodel_name = 'sentence_embedding_plm/ko-sbert-multitask'

    vectordb_path = '/home/hanati/hit/db_old'

    question = '2023년도 전기차 시장 동향 알려줘'

    llm_model = '/home/hanati/hit/llm/polyglot-ko-1.3b'

    llm_hyp = {}

    fin_rag = FinanceRAG(dict_path, smodel_name, vectordb_path, llm_model, llm_model, llm_hyp)

    while(True):
        question = input('질문을 입력하세요:')
        result = fin_rag.generate(question)
        
        pprint(result)
