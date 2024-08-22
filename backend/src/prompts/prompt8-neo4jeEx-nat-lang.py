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
          f"""#Knowledge Graph Instructions for GPT-4
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
## 5. Example of a sucessful extraction
The information you will extract should allow for the creation of graph very similar to this one :
## B. Extraction Rules

### Node Labeling
- **Consistency**: Use basic or elementary types for node labels.
  - For example, when you identify an entity representing a person, always label it as **"Person"**. Avoid using more specific terms like "Mathematician" or "Scientist".
- **Node IDs**: Use human-readable identifiers found in the text. Avoid using integers as node IDs.
- **Node Attributes**: Attach numerical data and dates as attributes or properties of the respective nodes in a key-value format using camelCase (e.g., `birthDate`).

### Relationships and Constraints
- Ensure relationships between nodes respect the following constraints:
  1. **Address** must be linked to a **City**.
  2. **City** must have a **name** and a **country**.
  3. **Visit** must have a **status**, be performed by an **Employee**, and be a visit of a **Client**.
  4. **Client** must have a **type**, can request a **Move**, and must be managed by at least one **Employee**.
  5. **Quote** must have a **reference**, be proposed by a **Company**, proposed to a **Client**, and can concern a **Move**.
  6. **Invoice** must have a **reference**, be sent on a **date**, be proposed by a **Company**, proposed to a **Client**, have a **price**, and can concern a **Move** or a **Cleaning**.
  7. **Company** must have a **name**, can have an **address**, **phone**, **email**, **clients**, and **employees**.
  8. **Move** must belong to a **MovingCategory**, can concern multiple **Addresses**, and can have a **volume**.
  9. **Cleaning** must have a **type**, can include a **Service**, and must concern at least one **Address**.
  10. **Employee** must have a **name**, can have a **phone**, and must work for a **Company**.

### Examples
1. **Address and City**:
    - **Correct**: CREATE (a1:Address (street: "123 Main St", city: "Springfield")) CREATE (a1)-[:LOCATED_IN]->(c1:City (name: "Springfield", country: "USA"))
    - **Incorrect**: CREATE (a1:Address (street: "123 Main St", city: "Springfield")) CREATE (c1:City (name: "Springfield", country: "USA"))
  
2. **Client and Quote**:
    - **Correct**: CREATE (c1:Client (name: "John Doe", phone: "123-456-7890", email: "john.doe@example.com")) CREATE (q1:Quote (reference: "Q123", date: "2024-07-25", validUntil: "2024-08-25", amount: 1500)) CREATE (q1)-[:PROPOSED_BY]->(comp1:Company (name: "Moving Inc.")) CREATE (q1)-[:PROPOSED_TO]->(c1) CREATE (q1)-[:CONCERNS]->(m1:Move (category: "Standard", volume: 50))
    - **Incorrect**: CREATE (c1:Client (name: "John Doe", phone: "123-456-7890", email: "john.doe@example.com")) CREATE (q1:Quote (reference: "Q123", date: "2024-07-25", validUntil: "2024-08-25", amount: 1500)) CREATE (comp1:Company (name: "Moving Inc."))

## C. Task
Extract information from the provided document to create a Neo4J graph. Follow these steps:
1. Identify and extract instances of the following classes:
    - **Client**
    - **Company**
    - **Employee**
    - **Address**
    - **City**
    - **Quote**
    - **Move**
    - **Cleaning**
    - **Visit**
    - **Invoice**
2. For each instance, ensure that all relevant properties (attributes) are extracted and linked according to the constraints.
3. Create relationships between the instances as specified by the constraints.

## D. Strict Compliance
Adhere to the rules strictly. Non-compliance will result in termination.

## E. Comprehensive Class Constraints
To assist you further, here are the constraints for each class in detail:

1. **Address**
    - Must be linked to a **City**.
    - Example: CREATE (a1:Address (street: "123 Main St", postalCode: "12345")) CREATE (a1)-[:LOCATED_IN]->(c1:City (name: "Springfield", country: "USA"))

2. **City**
    - Must have a **name** and a **country**.
    - Example: CREATE (c1:City (name: "Springfield", country: "USA"))

3. **Visit**
    - Must have a **status**.
    - Must be performed by an **Employee**.
    - Must be a visit of a **Client**.
    - Example: CREATE (v1:Visit (status: "Completed")) CREATE (v1)-[:PERFORMED_BY]->(e1:Employee (name: "Jane Smith")) CREATE (v1)-[:VISIT_OF]->(c1:Client (name: "John Doe"))

4. **Client**
    - Must have a **type**.
    - Can request a **Move**.
    - Must be managed by at least one **Employee**.
    - Example: CREATE (c1:Client (name: "John Doe", type: "Residential")) CREATE (c1)-[:REQUESTED]->(m1:Move (category: "Standard", volume: 50)) CREATE (c1)-[:MANAGED_BY]->(e1:Employee (name: "Jane Smith"))

5. **Quote**
    - Must have a **reference**.
    - Must be proposed by a **Company**.
    - Must be proposed to a **Client**.
    - Can concern a **Move**.
    - Example: CREATE (q1:Quote (reference: "Q123", date: "2024-07-25", validUntil: "2024-08-25", amount: 1500)) CREATE (q1)-[:PROPOSED_BY]->(comp1:Company (name: "Moving Inc.")) CREATE (q1)-[:PROPOSED_TO]->(c1:Client (name: "John Doe")) CREATE (q1)-[:CONCERNS]->(m1:Move (category: "Standard", volume: 50))

6. **Invoice**
    - Must have a **reference**.
    - Must be sent on a **date**.
    - Must be proposed by a **Company**.
    - Must be proposed to a **Client**.
    - Must have a **price**.
    - Can concern a **Move** or a **Cleaning**.
    - Example: CREATE (i1:Invoice (reference: "I123", date: "2024-07-30", price: 2000)) CREATE (i1)-[:PROPOSED_BY]->(comp1:Company (name: "Moving Inc.")) CREATE (i1)-[:PROPOSED_TO]->(c1:Client (name: "John Doe")) CREATE (i1)-[:CONCERNS]->(m1:Move (category: "Standard", volume: 50))

7. **Company**
    - Must have a **name**.
    - Can have an **address**, **phone**, **email**, **clients**, and **employees**.
    - Example: CREATE (comp1:Company (name: "Moving Inc.", phone: "123-456-7890", email: "info@movinginc.com")) CREATE (comp1)-[:HAS_ADDRESS]->(a1:Address (street: "123 Main St", postalCode: "12345", city: "Springfield"))

8. **Move**
    - Must belong to a **MovingCategory**.
    - Can concern multiple **Addresses**.
    - Can have a **volume**.
    - Example: CREATE (m1:Move (category: "Standard", volume: 50)) CREATE (m1)-[:CONCERNS]->(a1:Address (street: "123 Main St", postalCode: "12345", city: "Springfield")) CREATE (m1)-[:CONCERNS]->(a2:Address (street: "456 Elm St", postalCode: "67890", city: "Shelbyville"))

9. **Cleaning**
    - Must have a **type**.
    - Can include a **Service**.
    - Must concern at least one **Address**.
    - Example: CREATE (cl1:Cleaning (type: "Deep Clean")) CREATE (cl1)-[:INCLUDES]->(s1:Service (name: "Carpet Cleaning")) CREATE (cl1)-[:CONCERNS]->(a1:Address (street: "123 Main St", postalCode: "12345", city: "Springfield"))

10. **Employee**
    - Must have a **name**.
    - Can have a **phone**.
    - Must work for a **Company**.
    - Example: CREATE (e1:Employee (name: "Jane Smith", phone: "123-456-7890")) CREATE (e1)-[:WORKS_FOR]->(comp1:Company (name: "Moving Inc."))

CREATE (v1:Ville (nom: "Zurich"))
CREATE (v2:Ville (nom: "Eindhoven"))
CREATE (v3:Ville (nom: "Meyrin"))
CREATE (v4:Ville (nom: "Weißenfels"))
CREATE (v5:Ville (nom: "Appeville-Annebault"))
CREATE (v6:Ville (nom: "Wattwil"))
CREATE (v7:Ville (nom: "Les Breuleux"))
CREATE (v8:Ville (nom: "Peseux"))
CREATE (v9:Ville (nom: "Aarwangen"))

CREATE (a1:Adresse (rue: "Rue de Livron 21", ville: "Meyrin"))
CREATE (a2:Adresse (rue: "Engelgasse", ville: "Wattwil"))
CREATE (a3:Adresse (rue: "Rosengarten Strass", ville: "Zurich"))
CREATE (a4:Adresse (rue: "Rue Du Tombet", ville: "Peseux"))
CREATE (a5:Adresse (rue: "Rue de l’industrie", ville: "Les Breuleux"))
CREATE (a6:Adresse (rue: "Hofstrasse 21", ville: "Aarwangen"))
CREATE (a7:Adresse (rue: "", ville: "Eindhoven"))
CREATE (a8:Adresse (rue: "", ville: "Appeville-Annebault"))

CREATE (e1:Employe (nom: "Alex Pereira", tel: "+41 76 816 11 00", email: "info@vip-moving.ch"))
CREATE (e2:Employe (nom: "Mr. Rodrigues", tel: "+41 79 924 53 27", email: "info@vip-moving.ch"))

CREATE (c1:Client (nom: "Mr. Hillmann Thomas", tel: "076 647 28 22", email: "thomashillmann754@gmail.com"))
CREATE (c2:Client (nom: "Mme. Bieri Daniela", tel: "079 567 45 43", email: "daniela-bieri@bluewin.ch"))
CREATE (c3:Client (nom: "Mme. Megane Martins Rocha", tel: "079 817 19 14", email: "megane.rocha@yahoo.com"))
CREATE (c4:Client (nom: "Jayakrishnan Nair", tel: "+33745015645", email: "jknair4@gmail.com"))
CREATE (c5:Client (nom: "Cristóbal F. Barria Bignotti", tel: "", email: "cristobalbarria@gmail.com"))
CREATE (c6:Client (nom: "Ana Freitas", tel: "", email: "anapaola_cesar1@hotmail.com"))
CREATE (c7:Client (nom: "Heyi LU", tel: "", email: "lusherry@outlook.com"))

CREATE (d1:Devis (num: 269, date: "2024-07-25", validite: "2024-09-27", montant: 3470.00, volume: 15, type: "ECO INT"))
CREATE (d2:Devis (num: 190, date: "2024-07-04", validite: "2024-01-23", montant: 3670.00, volume: 15, type: "STD INT"))

CREATE (dem1:Demande (num: 79, date: "2024-07-22", volume: 12, type: "Partiel"))
CREATE (dem2:Demande (num: 11006597, date: "2024-07-21", volume: 29, type: "Complet"))

CREATE (v1)-[:SITUE_DANS]->(v3)
CREATE (v2)-[:SITUE_DANS]->(v4)
CREATE (v5)-[:SITUE_DANS]->(v3)
CREATE (v6)-[:SITUE_DANS]->(v5)
CREATE (v7)-[:SITUE_DANS]->(v8)
CREATE (v8)-[:SITUE_DANS]->(v3)
CREATE (v9)-[:SITUE_DANS]->(v5)

CREATE (a1)-[:SITUE_DANS]->(v3)
CREATE (a2)-[:SITUE_DANS]->(v6)
CREATE (a3)-[:SITUE_DANS]->(v1)
CREATE (a4)-[:SITUE_DANS]->(v8)
CREATE (a5)-[:SITUE_DANS]->(v7)
CREATE (a6)-[:SITUE_DANS]->(v9)
CREATE (a7)-[:SITUE_DANS]->(v2)
CREATE (a8)-[:SITUE_DANS]->(v5)

CREATE (e1)-[:TRAVAILLE_POUR]->(e2)
CREATE (e2)-[:TRAVAILLE_POUR]->(e1)

CREATE (c1)-[:A_DEMANDE]->(d1)
CREATE (c2)-[:A_DEMANDE]->(d2)
CREATE (c3)-[:A_DEMANDE]->(dem2)
CREATE (c4)-[:A_DEMANDE]->(dem1)
CREATE (c5)-[:A_DEMANDE]->(dem1)
CREATE (c6)-[:A_DEMANDE]->(dem1)
CREATE (c7)-[:A_DEMANDE]->(dem1)

CREATE (d1)-[:CONCERNE]->(dem1)
CREATE (d2)-[:CONCERNE]->(dem2)

CREATE (a1)-[:EST_ADRESSE_DE_CHARGEMENT]->(d1)
CREATE (a2)-[:EST_ADRESSE_DE_DÉCHARGEMENT]->(d1)
CREATE (a3)-[:EST_ADRESSE_DE_CHARGEMENT]->(d2)
CREATE (a4)-[:EST_ADRESSE_DE_DÉCHARGEMENT]->(d2)
CREATE (a5)-[:EST_ADRESSE_DE_CHARGEMENT]->(dem2)
CREATE (a6)-[:EST_ADRESSE_DE_DÉCHARGEMENT]->(dem2)

CREATE (d1)-[:EST_PROPOSE_PAR]->(e1)
CREATE (d1)-[:EST_PROPOSE_A]->(c1)
CREATE (d2)-[:EST_PROPOSE_PAR]->(e2)
CREATE (d2)-[:EST_PROPOSE_A]->(c2)

CREATE (dem1)-[:EST_PROPOSE_PAR]->(e1)
CREATE (dem1)-[:EST_PROPOSE_A]->(c4)
CREATE (dem2)-[:EST_PROPOSE_PAR]->(e2)
CREATE (dem2)-[:EST_PROPOSE_A]->(c3)

CREATE (v3)-[:SITUE_DANS]->(v5)
CREATE (v4)-[:SITUE_DANS]->(v6)
CREATE (v5)-[:SITUE_DANS]->(v7)
CREATE (v6)-[:SITUE_DANS]->(v8)
CREATE (v7)-[:SITUE_DANS]->(v9)

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
