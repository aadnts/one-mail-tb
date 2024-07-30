import logging
from langchain.docstore.document import Document
import os
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_groq import ChatGroq
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_anthropic import ChatAnthropic
from langchain_fireworks import ChatFireworks
from langchain_aws import ChatBedrock
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import boto3
import google.auth
from src.shared.constants import MODEL_VERSIONS


def get_llm(model_version: str):
    """Retrieve the specified language model based on the model name."""
    env_key = "LLM_MODEL_CONFIG_" + model_version
    env_value = os.environ.get(env_key)
    logging.info("Model: {}".format(env_key))
    if "gemini" in model_version:
        credentials, project_id = google.auth.default()
        model_name = MODEL_VERSIONS[model_version]
        llm = ChatVertexAI(
            model_name=model_name,
            convert_system_message_to_human=True,
            credentials=credentials,
            project=project_id,
            temperature=0,
            safety_settings={
                HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            },
        )
    elif "openai" in model_version:
        model_name = MODEL_VERSIONS[model_version]
        llm = ChatOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=model_name,
            temperature=0,
        )

    elif "azure" in model_version:
        model_name, api_endpoint, api_key, api_version = env_value.split(",")
        llm = AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=api_endpoint,
            azure_deployment=model_name,  # takes precedence over model parameter
            api_version=api_version,
            temperature=0,
            max_tokens=None,
            timeout=None,
        )

    elif "anthropic" in model_version:
        model_name, api_key = env_value.split(",")
        llm = ChatAnthropic(
            api_key=api_key, model=model_name, temperature=0, timeout=None
        )

    elif "fireworks" in model_version:
        model_name, api_key = env_value.split(",")
        llm = ChatFireworks(api_key=api_key, model=model_name)

    elif "groq" in model_version:
        model_name, base_url, api_key = env_value.split(",")
        llm = ChatGroq(api_key=api_key, model_name=model_name, temperature=0)

    elif "bedrock" in model_version:
        model_name, aws_access_key, aws_secret_key, region_name = env_value.split(",")
        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region_name,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
        )

        llm = ChatBedrock(
            client=bedrock_client, model_id=model_name, model_kwargs=dict(temperature=0)
        )

    elif "ollama" in model_version:
        model_name, base_url = env_value.split(",")
        llm = ChatOllama(base_url=base_url, model=model_name)

    else:
        model_name = "diffbot"
        llm = DiffbotGraphTransformer(
            diffbot_api_key=os.environ.get("DIFFBOT_API_KEY"),
            extract_types=["entities", "facts"],
        )
    logging.info(f"Model created - Model Version: {model_version}")
    return llm, model_name


def get_combined_chunks(chunkId_chunkDoc_list):
    chunks_to_combine = int(os.environ.get("NUMBER_OF_CHUNKS_TO_COMBINE"))
    logging.info(f"Combining {chunks_to_combine} chunks before sending request to LLM")
    combined_chunk_document_list = []
    combined_chunks_page_content = [
        "".join(
            document["chunk_doc"].page_content
            for document in chunkId_chunkDoc_list[i : i + chunks_to_combine]
        )
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]
    combined_chunks_ids = [
        [
            document["chunk_id"]
            for document in chunkId_chunkDoc_list[i : i + chunks_to_combine]
        ]
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]

    for i in range(len(combined_chunks_page_content)):
        combined_chunk_document_list.append(
            Document(
                page_content=combined_chunks_page_content[i],
                metadata={"combined_chunk_ids": combined_chunks_ids[i]},
            )
        )
    return combined_chunk_document_list


def get_graph_document_list(
    llm, combined_chunk_document_list, allowedNodes, allowedRelationship
):
    futures = []
    graph_document_list = []
    if llm.get_name() == "ChatOllama":
        node_properties = False
    else:
        node_properties = ["description"]
    llm_transformer = LLMGraphTransformer(
        llm=llm,
        prompt=ChatPromptTemplate.from_messages(
        [(
          "system",
          f"""#KKKK Knowledge Graph Instructions for GPT-4
## 1. Overview
You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
- **Nodes** represent entities and concepts. They're akin to Wikipedia nodes.
- The aim is to achieve simplicity and clarity in the knowledge graph, making it accessible for a vast audience.
## 2. Labeling Nodes
- **Consistency**: Ensure you use basic or elementary types for node labels.
  - For example, when you identify an entity representing a person, always label it as **"person"**. Avoid using more specific terms like "mathematician" or "scientist".
- **Node IDs**: Never utilize integers as node IDs. Node IDs should be names or human-readable identifiers found in the text.
{'- **Allowed Node Labels:**' + ", ".join(allowedNodes) if allowedNodes else ""}
{'- **Allowed Relationship Types**:' + ", ".join(allowedRelationship) if allowedRelationship else ""}
## 3. Handling Numerical Data and Dates
- Numerical data, like age or other related information, should be incorporated as attributes or properties of the respective nodes.
- **No Separate Nodes for Dates/Numbers**: Do not create separate nodes for dates or numerical values. Always attach them as attributes or properties of nodes.
- **Property Format**: Properties must be in a key-value format.
- **Quotation Marks**: Never use escaped single or double quotes within property values.
- **Naming Convention**: Use camelCase for property keys, e.g., `birthDate`.
## 4. Coreference Resolution
- **Maintain Entity Consistency**: When extracting entities, it's vital to ensure consistency.
If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"),
always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the entity ID.
Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.
## 5. Constraints on the relationships
When extracting the relationships, please strictly follow these constraints :
Entities
Client
Quotation
Moving
Address
Service
Vehicle
Item to be moved
Payment
Visit/Appointment
Commercial
Relationships
Client - requests -> Quotation

Subject: Client
Object: Quotation
Quotation - pertains to -> Moving

Subject: Quotation
Object: Moving
Moving - originates from -> Address

Subject: Moving
Object: Address
Moving - destined for -> Address

Subject: Moving
Object: Address
Moving - utilizes -> Vehicle

Subject: Moving
Object: Vehicle
Moving - includes -> Service

Subject: Moving
Object: Service
Moving - concerns -> Item to be moved

Subject: Moving
Object: Item to be moved
Client - makes -> Payment

Subject: Client
Object: Payment
Commercial - manages -> Quotation

Subject: Commercial
Object: Quotation
Commercial - conducts -> Visit/Appointment

Subject: Commercial
Object: Visit/Appointment
Client - resides at -> Address

Subject: Client
Object: Address
Client Attributes
Client - has ID
Client - has name
Client - has first name
Client - has phone
Client - has email
Client - has type (private/company)
Client - has nationality
Quotation Attributes
Quotation - has number
Quotation - has issue date
Quotation - has validity date
Quotation - has price
Quotation - has estimated volume (m3)
Quotation - has moving type (NAT/INT)
Quotation - has service type (ECO GROUPÉ, ECO, STANDARD, LUXE, PREMIUM)
Moving Attributes
Moving - has loading date
Moving - has unloading date
Moving - has actual volume (m3)
Address Attributes
Address - has street
Address - has postal code
Address - has city
Address - has country
Address - has housing type (house/apartment)
Address - has floor
Address - has elevator
Address - has parking distance
Address - requires parking reservation
Service Attributes
Service - has type (dismantling, reassembly, packing, unpacking, storage, cleaning, material delivery, customs formalities)
Service - has specific details
Vehicle Attributes
Vehicle - has capacity (m3)
Vehicle - has type
Item to be moved Attributes
Item to be moved - has type (furniture, box, appliance, art)
Item to be moved - has quantity
Item to be moved - has weight
Item to be moved - has dimensions
Item to be moved - has volume
Item to be moved - has current floor
Payment Attributes
Payment - has amount
Payment - has type (deposit, balance)
Payment - has method (transfer, cash)
Payment - has date
Visit/Appointment Attributes
Visit/Appointment - has date
Visit/Appointment - has time
Visit/Appointment - has status (confirmed, refused, pending, conducted, canceled)
Commercial Attributes
Commercial - has name
Commercial - has phone number
## 6. Strict Compliance
Adhere to the rules strictly. Non-compliance will result in termination.
          """),
            ("human", "Use the given format to extract information from the following input: {input}"),
            ("human", "Tip: Make sure to answer in the correct format"),
        ]),
        node_properties=node_properties,
        allowed_nodes=allowedNodes,
        allowed_relationships=allowedRelationship,
    )
    with ThreadPoolExecutor(max_workers=10) as executor:
        for chunk in combined_chunk_document_list:
            chunk_doc = Document(
                page_content=chunk.page_content.encode("utf-8"), metadata=chunk.metadata
            )
            futures.append(
                executor.submit(llm_transformer.convert_to_graph_documents, [chunk_doc])
            )

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            graph_document = future.result()
            graph_document_list.append(graph_document[0])

    return graph_document_list


def get_graph_from_llm(model, chunkId_chunkDoc_list, allowedNodes, allowedRelationship):
    llm, model_name = get_llm(model)
    combined_chunk_document_list = get_combined_chunks(chunkId_chunkDoc_list)
    graph_document_list = get_graph_document_list(
        llm, combined_chunk_document_list, allowedNodes, allowedRelationship
    )
    return graph_document_list
