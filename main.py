import asyncio
from langgraph.graph import StateGraph
from graph_function2 import *
from langgraph.graph import END, START
import os
import json
import pandas as pd
from langchain_ollama import ChatOllama
from datasets import Dataset
from ragas import SingleTurnSample
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage
from ragas.metrics import LLMContextPrecisionWithoutReference
from langchain_openai import ChatOpenAI
from langchain.smith import RunEvalConfig
from ragas.integrations.langchain import EvaluatorChain

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, ResponseRelevancy
from ragas import EvaluationDataset
os.environ["OPENAI_API_KEY"] =""
gpt_llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)
evaluator_llm = LangchainLLMWrapper(gpt_llm)
# Define the workflow
workflow = StateGraph(GraphState)
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate

workflow.add_edge(START, "retrieve")
workflow.add_edge("websearch", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
graph = workflow.compile()

async def generate_responses(questions):
    test_schema = []

    # Generate responses for each question asynchronously
    for question in questions:
        inputs = {"question": question, "max_retries": 3}
        result = await graph.invoke(inputs) if asyncio.iscoroutinefunction(graph.invoke) else graph.invoke(inputs)
        generation = result.get("generation").content if "generation" in result else None
        documents_retrieved = result.get("documents", [])
        documents = [doc.page_content for doc in documents_retrieved if hasattr(doc, 'page_content')]

        # Append the generated result to test_schema
        test_schema.append({
            "question": question,
            "answer": generation,
            "contexts": documents
        })

    return test_schema

async def main():
    task = ""
    sample_queries = [
    "Under the Residential Tenancy (Smoke Alarms) Regulations 2022 (Tas), what are the power source requirements for smoke alarms in tenanted premises?",
    "According to the Explosives Act 2012 (Tas), what constitutes handling an explosive?",
    "What is the main purpose of the Radiation Protection Act 2005 (Tas)?",
    "When did the Conveyancing Amendment Act (No. 2) 2012 (Tas) commence, according to its proclamation?",
    "What happened to the Road Safety (Alcohol and Drugs) Amendment Act 2005 (Tas)?",
    "What authority governs the Presbyterian Church of Tasmania under the Presbyterian Church Act 1896 (Tas)?",
    "What date was fixed for the commencement of the Homes Tasmania (Consequential Amendments) Act 2022 (Tas)?",
    "Under the Passenger Transport Services Regulations 2023 (Tas), what vehicles are exempt from the Act under incidental passenger services?",
    "What amount was appropriated under the Appropriation (Further Supplementary Appropriation for 2022-23) Act 2023 (Tas)?",
    "According to the Local Government (Casual Vacancies) Order 2014 (Tas), when is a recount or by-election not required for a vacant councillor position?"
        ]

    expected_responses = [
            "Smoke alarms must be permanently connected to the premises’ power supply with a backup alternative power supply, or powered by a 10-year non-replaceable battery if permanent connection was not required by building regulations at installation time, according to the Residential Tenancy (Smoke Alarms) Regulations 2022 (Tas).",
            "Handling an explosive under the Explosives Act 2012 (Tas) includes activities like manufacturing, storing, using, selling, disposing, or controlling explosives, but does not normally include transportation unless moved by pipes.",
            "The Radiation Protection Act 2005 (Tas) aims to protect people's health and safety and the environment from the harmful effects of radiation.",
            "The Conveyancing Amendment Act (No. 2) 2012 (Tas) commenced on 1 May 2015, as fixed by the Proclamation published on 25 March 2015.",
            "The Road Safety (Alcohol and Drugs) Amendment Act 2005 (Tas) was repealed by the Redundant Legislation Repeal Act 2016.",
            "The governing authority for the Presbyterian Church of Tasmania under the Presbyterian Church Act 1896 (Tas) is the General Assembly or Supreme Judicatory constituted under the Church’s internal laws and usages.",
            "The Homes Tasmania (Consequential Amendments) Act 2022 (Tas) commenced on 1 December 2022, as stated in its proclamation.",
            "Under the Passenger Transport Services Regulations 2023 (Tas), vans transporting prisoners and emergency response vehicles are exempt from the Act as incidental passenger services.",
            "An appropriation of $340,543,000 was made under the Appropriation (Further Supplementary Appropriation for 2022-23) Act 2023 (Tas) for government services.",
            "Under the Local Government (Casual Vacancies) Order 2014 (Tas), a recount or by-election for a vacant councillor position is not held if filling the vacancy would exceed the new reduced number of councillors set by the Councillors Order."
        ]
    dataset = []
    if task == "evaluate_rag":
        
        
        for query,reference in zip(sample_queries,expected_responses):
            test_schema = await generate_responses([query])
            for data in test_schema:
                relevant_docs = data["contexts"]
                response = data["answer"]
            dataset.append(
                {
                    "user_input":query,
                    "retrieved_contexts":relevant_docs,
                    "response":response,
                    "reference":reference
                }
            )
        evaluation_dataset = EvaluationDataset.from_list(dataset)
        result = evaluate(dataset=evaluation_dataset,metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(), ResponseRelevancy()],llm=evaluator_llm)
        print("Start to evaluate")
        print(result)
    elif task == "evaluate_llm":
        for query, reference in zip(sample_queries, expected_responses):
            print("Answering questions")
            # Use LLM to generate a response directly without retrieval
            response = gpt_llm.invoke(query)  # Replace with your actual LLM call

            dataset.append(
                {
                    "user_input": query,
                    "retrieved_contexts": [],  # No contexts
                    "response": response.content,
                    "reference": reference,
                }
            )

        evaluation_dataset = EvaluationDataset.from_list(dataset)

        # Evaluate only FactualCorrectness (others are optional)
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=[FactualCorrectness(), ResponseRelevancy()],
            llm=evaluator_llm
        )
        print(result)
    else:
        questions=["Under the Local Government Act 1993 (Tas), who has the authority to issue a council proclamation?"]
        # Step 1: Generation Phase
        test_schema = await generate_responses(questions)
        for data in test_schema:
            print(f"Question: {data['question']}")
            print(f"Answer: {data['answer']}")
            print(f"Document: {data['contexts']}")
# Run the async workflow
asyncio.run(main())
