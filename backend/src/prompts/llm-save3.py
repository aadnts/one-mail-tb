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
- **"NotreEntreprise"** always refers to the company VIP Moving.
  "classRelations": [
    (
      "className": "Date_De_Chargement",
      "relations": [
        ("type": "A", "targetClass": "Jour", "maxRelations": 1),
        ("type": "A", "targetClass": "Mois", "maxRelations": 1),
        ("type": "A", "targetClass": "Annee", "maxRelations": 1)
      ]
    ),
    (
      "className": "Date_De_Dechargement",
      "relations": [
        ("type": "A", "targetClass": "Jour", "maxRelations": 1),
        ("type": "A", "targetClass": "Mois", "maxRelations": 1),
        ("type": "A", "targetClass": "Annee", "maxRelations": 1)
      ]
    ),
    (
      "className": "Date_Envoi_Devis",
      "relations": [
        ("type": "A", "targetClass": "Jour", "maxRelations": 1),
        ("type": "A", "targetClass": "Mois", "maxRelations": 1),
        ("type": "A", "targetClass": "Annee", "maxRelations": 1)
      ]
    ),
    (
      "className": "Date_Signature",
      "relations": [
        ("type": "A", "targetClass": "Jour", "maxRelations": 1),
        ("type": "A", "targetClass": "Mois", "maxRelations": 1),
        ("type": "A", "targetClass": "Annee", "maxRelations": 1)
      ]
    ),
    (
      "className": "Date_Envoi_Demande_Devis",
      "relations": [
        ("type": "A", "targetClass": "Jour", "maxRelations": 1),
        ("type": "A", "targetClass": "Mois", "maxRelations": 1),
        ("type": "A", "targetClass": "Annee", "maxRelations": 1)
      ]
    ),
    (
      "className": "Date_Visite",
      "relations": [
        ("type": "A", "targetClass": "Jour", "maxRelations": 1),
        ("type": "A", "targetClass": "Mois", "maxRelations": 1),
        ("type": "A", "targetClass": "Annee", "maxRelations": 1)
      ]
    ),
    (
      "className": "Adresse_Chargement_Client",
      "relations": [
        ("type": "A", "targetClass": "Rue", "maxRelations": 1),
        ("type": "A", "targetClass": "Numero_Rue", "maxRelations": 1),
        ("type": "A", "targetClass": "Ville", "maxRelations": 1),
        ("type": "A", "targetClass": "Code_Postal", "maxRelations": 1),
        ("type": "A", "targetClass": "Region", "maxRelations": 1),
        ("type": "A", "targetClass": "Pays", "maxRelations": 1)
      ]
    ),
    (
      "className": "Adresse_Dechargement_Client",
      "relations": [
        ("type": "A", "targetClass": "Rue", "maxRelations": 1),
        ("type": "A", "targetClass": "Numero_Rue", "maxRelations": 1),
        ("type": "A", "targetClass": "Ville", "maxRelations": 1),
        ("type": "A", "targetClass": "Code_Postal", "maxRelations": 1),
        ("type": "A", "targetClass": "Region", "maxRelations": 1),
        ("type": "A", "targetClass": "Pays", "maxRelations": 1)
      ]
    ),
    (
      "className": "Adresse_Entreprise",
      "relations": [
        ("type": "A", "targetClass": "Rue", "maxRelations": 1),
        ("type": "A", "targetClass": "Numero_Rue", "maxRelations": 1),
        ("type": "A", "targetClass": "Ville", "maxRelations": 1),
        ("type": "A", "targetClass": "Code_Postal", "maxRelations": 1),
        ("type": "A", "targetClass": "Region", "maxRelations": 1),
        ("type": "A", "targetClass": "Pays", "maxRelations": 1)
      ]
    ),
    (
      "className": "Visite_Confirmee",
      "relations": [
        ("type": "A", "targetClass": "Date_Visite", "maxRelations": 1),
        ("type": "A", "targetClass": "Adresse", "maxRelations": 1),
        ("type": "Est_Effectue_Par", "targetClass": "Employe", "minRelations": 1),
        ("type": "Est_Visite_De", "targetClass": "Client", "minRelations": 1)
      ]
    ),
    (
      "className": "Visite_Annulee",
      "relations": [
        ("type": "A", "targetClass": "Date_Visite", "maxRelations": 1),
        ("type": "A", "targetClass": "Adresse", "maxRelations": 1),
        ("type": "Est_Effectue_Par", "targetClass": "Employe", "minRelations": 1),
        ("type": "Est_Visite_De", "targetClass": "Client", "minRelations": 1)
      ]
    ),
    (
      "className": "Visite_Non_Confirmee",
      "relations": [
        ("type": "A", "targetClass": "Date_Visite", "maxRelations": 1),
        ("type": "A", "targetClass": "Adresse", "maxRelations": 1),
        ("type": "Est_Effectue_Par", "targetClass": "Employe", "minRelations": 1),
        ("type": "Est_Visite_De", "targetClass": "Client", "minRelations": 1)
      ]
    ),
    (
      "className": "Notre_Entreprise",
      "relations": [
        ("type": "A", "targetClass": "Nom", "maxRelations": 1),
        ("type": "A", "targetClass": "Adresse_Entreprise", "minRelations": 1),
        ("type": "A", "targetClass": "Telephone", "minRelations": 1),
        ("type": "A", "targetClass": "Email", "minRelations": 1),
        ("type": "A", "targetClass": "Client", "minRelations": 1),
        ("type": "A", "targetClass": "Employe", "minRelations": 1)
      ]
    ),
    (
      "className": "Autres_Entreprises",
      "relations": [
        ("type": "A", "targetClass": "Nom", "maxRelations": 1),
        ("type": "A", "targetClass": "Adresse_Entreprise", "minRelations": 1),
        ("type": "A", "targetClass": "Telephone", "minRelations": 1),
        ("type": "A", "targetClass": "Email", "minRelations": 1),
        ("type": "A", "targetClass": "Client", "minRelations": 1),
        ("type": "A", "targetClass": "Employe", "minRelations": 1)
      ]
    ),
    (
      "className": "Client_Prive",
      "relations": [
        ("type": "A", "targetClass": "Nom", "maxRelations": 1),
        ("type": "A", "targetClass": "Adresse_Chargement_Client", "minRelations": 1),
        ("type": "A", "targetClass": "Adresse_Dechargement_Client", "minRelations": 1),
        ("type": "A", "targetClass": "Telephone", "minRelations": 1),
        ("type": "A", "targetClass": "Email", "minRelations": 1),
        ("type": "A", "targetClass": "Demenagement", "minRelations": 1),
        ("type": "Est_Gere_Par", "targetClass": "Employe", "minRelations": 1)
      ]
    ),
    (
      "className": "Client_Entreprise",
      "relations": [
        ("type": "A", "targetClass": "Nom", "maxRelations": 1),
        ("type": "A", "targetClass": "Adresse_Chargement_Client", "minRelations": 1),
        ("type": "A", "targetClass": "Adresse_Dechargement_Client", "minRelations": 1),
        ("type": "A", "targetClass": "Telephone", "minRelations": 1),
        ("type": "A", "targetClass": "Email", "minRelations": 1),
        ("type": "A", "targetClass": "Demenagement", "minRelations": 1),
        ("type": "Est_Gere_Par", "targetClass": "Employe", "minRelations": 1)
      ]
    ),
    (
      "className": "Service_Eco_Groupe",
      "relations": [
        ("type": "A", "targetClass": "Prestation", "minRelations": 1)
      ]
    ),
    (
      "className": "Service_Eco",
      "relations": [
        ("type": "A", "targetClass": "Prestation", "minRelations": 1)
      ]
    ),
    (
      "className": "Service_Standard",
      "relations": [
        ("type": "A", "targetClass": "Prestation", "minRelations": 1)
      ]
    ),
    (
      "className": "Service_Luxe",
      "relations": [
        ("type": "A", "targetClass": "Prestation", "minRelations": 1)
      ]
    ),
    (
      "className": "Service_Premium",
      "relations": [
        ("type": "A", "targetClass": "Prestation", "minRelations": 1)
      ]
    ),
    (
      "className": "Devis_Accepte",
      "relations": [
        ("type": "A", "targetClass": "ID_Reference_Devis", "maxRelations": 1),
        ("type": "A", "targetClass": "Date_Envoi_Devis", "maxRelations": 1),
        ("type": "Est_Propose_Par", "targetClass": "Entreprise", "maxRelations": 1),
        ("type": "Est_Propose_A", "targetClass": "Client", "maxRelations": 1),
        ("type": "A", "targetClass": "Montant", "maxRelations": 1),
        ("type": "Concerne", "targetClass": "Demenagement", "maxRelations": 1),
        ("type": "Concerne", "targetClass": "Nettoyage", "minRelations": 1)
      ]
    ),
    (
      "className": "Devis_Refuse",
      "relations": [
        ("type": "A", "targetClass": "ID_Reference_Devis", "maxRelations": 1),
        ("type": "A", "targetClass": "Date_Envoi_Devis", "maxRelations": 1),
        ("type": "Est_Propose_Par", "targetClass": "Entreprise", "maxRelations": 1),
        ("type": "Est_Propose_A", "targetClass": "Client", "maxRelations": 1),
        ("type": "A", "targetClass": "Montant", "maxRelations": 1),
        ("type": "Concerne", "targetClass": "Demenagement", "maxRelations": 1),
        ("type": "Concerne", "targetClass": "Nettoyage", "minRelations": 1)
      ]
    ),
    (
      "className": "Devis_En_Attente",
      "relations": [
        ("type": "A", "targetClass": "ID_Reference_Devis", "maxRelations": 1),
        ("type": "A", "targetClass": "Date_Envoi_Devis", "maxRelations": 1),
        ("type": "Est_Propose_Par", "targetClass": "Entreprise", "maxRelations": 1),
        ("type": "Est_Propose_A", "targetClass": "Client", "maxRelations": 1),
        ("type": "A", "targetClass": "Montant", "maxRelations": 1),
        ("type": "Concerne", "targetClass": "Demenagement", "maxRelations": 1),
        ("type": "Concerne", "targetClass": "Nettoyage", "minRelations": 1)
      ]
    ),
    (
      "className": "Facture_Payee",
      "relations": [
        ("type": "A", "targetClass": "ID_Reference_Facture", "maxRelations": 1),
        ("type": "A", "targetClass": "Date_Envoi_Devis", "maxRelations": 1),
        ("type": "Est_Propose_Par", "targetClass": "Entreprise", "maxRelations": 1),
        ("type": "Est_Propose_A", "targetClass": "Client", "maxRelations": 1),
        ("type": "A", "targetClass": "Montant", "maxRelations": 1),
        ("type": "Concerne", "targetClass": "Demenagement", "maxRelations": 1),
        ("type": "Concerne", "targetClass": "Nettoyage", "minRelations": 1)
      ]
    ),
    (
      "className": "Facture_Non_Payee",
      "relations": [
        ("type": "A", "targetClass": "ID_Reference_Facture", "maxRelations": 1),
        ("type": "A", "targetClass": "Date_Envoi_Devis", "maxRelations": 1),
        ("type": "Est_Propose_Par", "targetClass": "Entreprise", "maxRelations": 1),
        ("type": "Est_Propose_A", "targetClass": "Client", "maxRelations": 1),
        ("type": "A", "targetClass": "Montant", "maxRelations": 1),
        ("type": "Concerne", "targetClass": "Demenagement", "maxRelations": 1),
        ("type": "Concerne", "targetClass": "Nettoyage", "minRelations": 1)
      ]
    ),
    (
      "className": "Devant_Entree",
      "relations": [
        ("type": "A", "targetClass": "Besoin_Reservation_Parking", "maxRelations": 1),
        ("type": "A", "targetClass": "Besoin_Reservation_Monte_Charges", "maxRelations": 1)
      ]
    ),
    (
      "className": "A10m_Entree",
      "relations": [
        ("type": "A", "targetClass": "Besoin_Reservation_Parking", "maxRelations": 1),
        ("type": "A", "targetClass": "Besoin_Reservation_Monte_Charges", "maxRelations": 1)
      ]
    ),
    (
      "className": "A20m_Entree",
      "relations": [
        ("type": "A", "targetClass": "Besoin_Reservation_Parking", "maxRelations": 1),
        ("type": "A", "targetClass": "Besoin_Reservation_Monte_Charges", "maxRelations": 1)
      ]
    ),
    (
      "className": "A30m",
      "relations": [
        ("type": "A", "targetClass": "Besoin_Reservation_Parking", "maxRelations": 1),
        ("type": "A", "targetClass": "Besoin_Reservation_Monte_Charges", "maxRelations": 1)
      ]
    ),
    (
      "className": "Metres",
      "relations": [
        ("type": "A", "targetClass": "Besoin_Reservation_Parking", "maxRelations": 1),
        ("type": "A", "targetClass": "Besoin_Reservation_Monte_Charges", "maxRelations": 1)
      ]
    ),
    (
      "className": "Maison",
      "relations": [
        ("type": "A", "targetClass": "Adresse_Chargement_Client", "maxRelations": 1),
        ("type": "A", "targetClass": "Adresse_Dechargement_Client", "maxRelations": 1),
        ("type": "A", "targetClass": "Etage", "maxRelations": 1),
        ("type": "A", "targetClass": "Accessibilite", "maxRelations": 1),
        ("type": "A", "targetClass": "Visite", "maxRelations": 1),
        ("type": "A", "targetClass": "Nombre_De_Pieces", "maxRelations": 1),
        ("type": "A", "targetClass": "Surface_M2", "maxRelations": 1)
      ]
    ),
    (
      "className": "Appartement",
      "relations": [
        ("type": "A", "targetClass": "Adresse_Chargement_Client", "maxRelations": 1),
        ("type": "A", "targetClass": "Adresse_Dechargement_Client", "maxRelations": 1),
        ("type": "A", "targetClass": "Etage", "maxRelations": 1),
        ("type": "A", "targetClass": "Accessibilite", "maxRelations": 1),
        ("type": "A", "targetClass": "Visite", "maxRelations": 1),
        ("type": "A", "targetClass": "Nombre_De_Pieces", "maxRelations": 1),
        ("type": "A", "targetClass": "Surface_M2", "maxRelations": 1)
      ]
    ),
    (
      "className": "Chalet",
      "relations": [
        ("type": "A", "targetClass": "Adresse_Chargement_Client", "maxRelations": 1),
        ("type": "A", "targetClass": "Adresse_Dechargement_Client", "maxRelations": 1),
        ("type": "A", "targetClass": "Etage", "maxRelations": 1),
        ("type": "A", "targetClass": "Accessibilite", "maxRelations": 1),
        ("type": "A", "targetClass": "Visite", "maxRelations": 1),
        ("type": "A", "targetClass": "Nombre_De_Pieces", "maxRelations": 1),
        ("type": "A", "targetClass": "Surface_M2", "maxRelations": 1)
      ]
    ),
    (
      "className": "Container",
      "relations": [
        ("type": "A", "targetClass": "Adresse_Chargement_Client", "maxRelations": 1),
        ("type": "A", "targetClass": "Adresse_Dechargement_Client", "maxRelations": 1),
        ("type": "A", "targetClass": "Etage", "maxRelations": 1),
        ("type": "A", "targetClass": "Accessibilite", "maxRelations": 1),
        ("type": "A", "targetClass": "Visite", "maxRelations": 1),
        ("type": "A", "targetClass": "Nombre_De_Pieces", "maxRelations": 1),
        ("type": "A", "targetClass": "Surface_M2", "maxRelations": 1)
      ]
    ),
    (
      "className": "Inconnu",
      "relations": [
        ("type": "A", "targetClass": "Adresse_Chargement_Client", "maxRelations": 1),
        ("type": "A", "targetClass": "Adresse_Dechargement_Client", "maxRelations": 1),
        ("type": "A", "targetClass": "Etage", "maxRelations": 1),
        ("type": "A", "targetClass": "Accessibilite", "maxRelations": 1),
        ("type": "A", "targetClass": "Visite", "maxRelations": 1),
        ("type": "A", "targetClass": "Nombre_De_Pieces", "maxRelations": 1),
        ("type": "A", "targetClass": "Surface_M2", "maxRelations": 1)
      ]
    ),
    (
      "className": "Meuble",
      "relations": [
        ("type": "A", "targetClass": "Poids", "maxRelations": 1),
        ("type": "A", "targetClass": "Largeur", "maxRelations": 1),
        ("type": "A", "targetClass": "Longueur", "maxRelations": 1),
        ("type": "A", "targetClass": "Profondeur", "maxRelations": 1),
        ("type": "A", "targetClass": "Volume", "maxRelations": 1),
        ("type": "Est_Dans", "targetClass": "Type_Lieu", "maxRelations": 1)
      ]
    ),
    (
      "className": "Appareil",
      "relations": [
        ("type": "A", "targetClass": "Poids", "maxRelations": 1),
        ("type": "A", "targetClass": "Largeur", "maxRelations": 1),
        ("type": "A", "targetClass": "Longueur", "maxRelations": 1),
        ("type": "A", "targetClass": "Profondeur", "maxRelations": 1),
        ("type": "A", "targetClass": "Volume", "maxRelations": 1),
        ("type": "Est_Dans", "targetClass": "Type_Lieu", "maxRelations": 1)
      ]
    ),
    (
      "className": "Carton",
      "relations": [
        ("type": "A", "targetClass": "Poids", "maxRelations": 1),
        ("type": "A", "targetClass": "Largeur", "maxRelations": 1),
        ("type": "A", "targetClass": "Longueur", "maxRelations": 1),
        ("type": "A", "targetClass": "Profondeur", "maxRelations": 1),
        ("type": "A", "targetClass": "Volume", "maxRelations": 1),
        ("type": "Est_Dans", "targetClass": "Type_Lieu", "maxRelations": 1)
      ]
    ),
    (
      "className": "Oeuvre_Art",
      "relations": [
        ("type": "A", "targetClass": "Poids", "maxRelations": 1),
        ("type": "A", "targetClass": "Largeur", "maxRelations": 1),
        ("type": "A", "targetClass": "Longueur", "maxRelations": 1),
        ("type": "A", "targetClass": "Profondeur", "maxRelations": 1),
        ("type": "A", "targetClass": "Volume", "maxRelations": 1),
        ("type": "Est_Dans", "targetClass": "Type_Lieu", "maxRelations": 1)
      ]
    ),
    (
      "className": "Vehicule",
      "relations": [
        ("type": "A", "targetClass": "Poids", "maxRelations": 1),
        ("type": "A", "targetClass": "Largeur", "maxRelations": 1),
        ("type": "A", "targetClass": "Longueur", "maxRelations": 1),
        ("type": "A", "targetClass": "Profondeur", "maxRelations": 1),
        ("type": "A", "targetClass": "Volume", "maxRelations": 1),
        ("type": "Est_Dans", "targetClass": "Type_Lieu", "maxRelations": 1)
      ]
    ),
    (
      "className": "Demenagement_Nat",
      "relations": [
        ("type": "A", "targetClass": "Categorie_Dem", "maxRelations": 1),
        ("type": "A", "targetClass": "Type_Lieu", "minRelations": 2),
        ("type": "A", "targetClass": "Volume", "minRelations": 1)
      ]
    ),
    (
      "className": "Demenagement_Int",
      "relations": [
        ("type": "A", "targetClass": "Categorie_Dem", "maxRelations": 1),
        ("type": "A", "targetClass": "Type_Lieu", "minRelations": 2),
        ("type": "A", "targetClass": "Volume", "minRelations": 1)
      ]
    ),
    (
      "className": "Nettoyage_Fin_Bail",
      "relations": [
        ("type": "A", "targetClass": "Prestation", "minRelations": 1),
        ("type": "A", "targetClass": "Type_Lieu", "maxRelations": 1),
        ("type": "A", "targetClass": "Surface_M2", "maxRelations": 1)
      ]
    ),
    (
      "className": "Nettoyage_Fin_Chantier",
      "relations": [
        ("type": "A", "targetClass": "Prestation", "minRelations": 1),
        ("type": "A", "targetClass": "Type_Lieu", "maxRelations": 1),
        ("type": "A", "targetClass": "Surface_M2", "maxRelations": 1)
      ]
    )
  ]
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
