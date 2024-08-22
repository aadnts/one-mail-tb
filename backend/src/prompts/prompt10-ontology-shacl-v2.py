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

Nodes represent entities and concepts defined in the ontology.
The aim is to achieve simplicity and clarity in the knowledge graph, making it accessible for a vast audience.
## 2. Labeling Nodes
Consistency: Ensure you use basic or elementary types for node labels as defined in the ontology.
For example, when you identify an entity representing a person, always label it using the appropriate class such as "Employe" or "Client".
{'- **Allowed Node Labels:**' + ", ".join(allowedNodes) if allowedNodes else ""}
{'- **Allowed Relationship Types**:' + ", ".join(allowedRelationship) if allowedRelationship else ""}
## 3. Handling Numerical Data and Dates
- Numerical data, like age or other related information, should be incorporated as attributes or properties of the respective nodes.
- **No Separate Nodes for Dates/Numbers**: Do not create separate nodes for dates or numerical values. Always attach them as attributes or properties of nodes.
- **Property Format**: Properties must be in a key-value format.
- **Quotation Marks**: Never use escaped single or double quotes within property values.
- **Naming Convention**: Use camelCase for property keys, e.g., `birthDate`.
## 4. Extracting Entities and Relationships
Classes: Extract all instances of the classes defined in the ontology.
Properties: Extract all properties and relationships as defined, ensuring they adhere to the SHACL constraints.
For example, an instance of the class "DevisAccepte" should have properties like IDReferenceDevis, DateEnvoiDevis, Montant, etc., with the appropriate data types and constraints.
Relationships: Establish relationships between entities as defined by object properties in the ontology.
For instance, use the EstEffectuePar property to link a Visite to an Employe.
## 5. Coreference Resolution
Maintain Entity Consistency: Ensure consistency in entity references. If an entity is mentioned multiple times but referred to by different names or pronouns, always use the most complete identifier for that entity throughout the knowledge graph.
## 6. Coreference Resolution
Adhere strictly to the SHACL constraints defined in the ontology.
Ensure the cardinality constraints (minCount, maxCount) are respected.
Ensure the domain and range of properties are correctly implemented.
Ensure data types for properties are correctly assigned.   
## 7. Strict Compliance
Adhere to the rules strictly. Non-compliance will result in termination.

Example Extraction
For an entity of class DateDeChargement, extract properties Jour, Mois, and Annee with their values, ensuring compliance with the cardinality constraints.
For an entity of class AdresseChargementClient, extract properties Rue, NumeroRue, Ville, CodePostal, Region, and Pays with their values, ensuring compliance with the cardinality constraints.
Establish relationships using object properties like EstEffectuePar, EstVisiteDe, EstProposePar, etc., linking the appropriate entities.
With these instructions, extract all instances of the classes and properties defined in the ontology while ensuring compliance with all SHACL constraints.      
## 7. Ontology
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@base <http://www.semanticweb.org/ontologie_de_demonstration/> .

<http://www.semanticweb.org/ontologie_de_demonstration/> rdf:type owl:Ontology ;
    owl:versionIRI <http://www.semanticweb.org/ontologie_de_demonstration/1.0.0> ;
    rdfs:comment "Ontologie pour la démonstration des relations entre les entités d'un déménagement." .

#########################################################
#
# Déclaration des classes
#
#########################################################

<Status> rdf:type owl:Class .
<Confirmé> rdf:type owl:Class ;
    rdfs:subClassOf <Status> .
<Annulé> rdf:type owl:Class ;
    rdfs:subClassOf <Status> .
<Non-confirmé> rdf:type owl:Class ;
    rdfs:subClassOf <Status> .

<Date> rdf:type owl:Class .
<Date_de_chargement> rdf:type owl:Class ;
    rdfs:subClassOf <Date> .
<Date_de_déchargement> rdf:type owl:Class ;
    rdfs:subClassOf <Date> .
<Date_d_envoi_du_devis> rdf:type owl:Class ;
    rdfs:subClassOf <Date> .
<Date_de_signature> rdf:type owl:Class ;
    rdfs:subClassOf <Date> .
<Date_d_envoi_de_la_demande_de_devis> rdf:type owl:Class ;
    rdfs:subClassOf <Date> .
<Date_de_la_visite> rdf:type owl:Class ;
    rdfs:subClassOf <Date> .

<Adresse> rdf:type owl:Class .
<Ville> rdf:type owl:Class .
<Code_postal> rdf:type owl:Class .
<Région> rdf:type owl:Class .
<Pays> rdf:type owl:Class .
<Détails_du_lieu> rdf:type owl:Class .

<Employé> rdf:type owl:Class .
<Entreprise> rdf:type owl:Class .
<Client> rdf:type owl:Class .
<Visite> rdf:type owl:Class .
<Devis> rdf:type owl:Class .
<Facture> rdf:type owl:Class .
<Catégorie_de_déménagement> rdf:type owl:Class .
<Prestation> rdf:type owl:Class .
<Accessibilité> rdf:type owl:Class .
<Bien> rdf:type owl:Class .
<Déménagement> rdf:type owl:Class .
<Nettoyage> rdf:type owl:Class .

#########################################################
#
# Déclaration des sous-classes et propriétés
#
#########################################################

<Client_privé> rdf:type owl:Class ;
    rdfs:subClassOf <Client> .
<Client_entreprise> rdf:type owl:Class ;
    rdfs:subClassOf <Client> .

<Service_ECO_GROUPÉ> rdf:type owl:Class ;
    rdfs:subClassOf <Catégorie_de_déménagement> .
<Service_ECO> rdf:type owl:Class ;
    rdfs:subClassOf <Catégorie_de_déménagement> .
<Service_STD> rdf:type owl:Class ;
    rdfs:subClassOf <Catégorie_de_déménagement> .
<Service_LUX> rdf:type owl:Class ;
    rdfs:subClassOf <Catégorie_de_déménagement> .
<Service_PREM> rdf:type owl:Class ;
    rdfs:subClassOf <Catégorie_de_déménagement> .

<Devant_entrée> rdf:type owl:Class ;
    rdfs:subClassOf <Accessibilité> .
<A_10m_entrée> rdf:type owl:Class ;
    rdfs:subClassOf <Accessibilité> .
<A_20m_entrée> rdf:type owl:Class ;
    rdfs:subClassOf <Accessibilité> .
<A_30m_entrée> rdf:type owl:Class ;
    rdfs:subClassOf <Accessibilité> .
<A_plus_30m_entrée> rdf:type owl:Class ;
    rdfs:subClassOf <Accessibilité> .

#########################################################
#
# Déclaration des propriétés d'objets
#
#########################################################

<Situé_à> rdf:type owl:ObjectProperty ;
    rdfs:domain <Adresse> ;
    rdfs:range <Rue> .

<Situé_dans> rdf:type owl:ObjectProperty ;
    rdfs:domain <Adresse> ;
    rdfs:range <Ville> .

<A_les_détails> rdf:type owl:ObjectProperty ;
    rdfs:domain <Adresse> ;
    rdfs:range <Détails_du_lieu> .

<A_le_statut> rdf:type owl:ObjectProperty ;
    rdfs:domain <Visite> ;
    rdfs:range <Status> .

<Est_effectuée_par> rdf:type owl:ObjectProperty ;
    rdfs:domain <Visite> ;
    rdfs:range <Employé> .

<Est_la_visite_de> rdf:type owl:ObjectProperty ;
    rdfs:domain <Visite> ;
    rdfs:range <Client> .

<A_la_référence> rdf:type owl:ObjectProperty ;
    rdfs:domain <Devis> ;
    rdfs:range <ID_de_réference> .

<Est_proposé_par> rdf:type owl:ObjectProperty ;
    rdfs:domain <Devis> ;
    rdfs:range <Entreprise> .

<Est_proposé_à> rdf:type owl:ObjectProperty ;
    rdfs:domain <Devis> ;
    rdfs:range <Client> .

<Concerne> rdf:type owl:ObjectProperty ;
    rdfs:domain <Devis> ;
    rdfs:range <Déménagement> .

<Est_du_Type> rdf:type owl:ObjectProperty ;
    rdfs:domain <Client> ;
    rdfs:range <Type_client> .

<A_Demandé> rdf:type owl:ObjectProperty ;
    rdfs:domain <Client> ;
    rdfs:range <Déménagement> .

<Est_géré_par> rdf:type owl:ObjectProperty ;
    rdfs:domain <Client> ;
    rdfs:range <Employé> .

<Est_de_la_catégorie> rdf:type owl:ObjectProperty ;
    rdfs:domain <Déménagement> ;
    rdfs:range <Catégorie_de_déménagement> .

<Concerne_l_adresse> rdf:type owl:ObjectProperty ;
    rdfs:domain <Déménagement> ;
    rdfs:range <Adresse> .

#########################################################
#
# Déclaration des propriétés de données
#
#########################################################

<type_tel> rdf:type owl:DatatypeProperty ;
    rdfs:domain <Téléphone> ;
    rdfs:range xsd:string .

<type_mail> rdf:type owl:DatatypeProperty ;
    rdfs:domain <Email> ;
    rdfs:range xsd:string .

<type_url> rdf:type owl:DatatypeProperty ;
    rdfs:domain <Site_internet> ;
    rdfs:range xsd:anyURI .

<type_string> rdf:type owl:DatatypeProperty ;
    rdfs:domain <Prestation> ;
    rdfs:range xsd:string .

<nombre_de_pieces> rdf:type owl:DatatypeProperty ;
    rdfs:domain <Détails_du_lieu> ;
    rdfs:range xsd:float .

<surface_en_m2> rdf:type owl:DatatypeProperty ;
    rdfs:domain <Détails_du_lieu> ;
    rdfs:range xsd:float .

<type_float> rdf:type owl:DatatypeProperty ;
    rdfs:domain <Bien> ;
    rdfs:range xsd:float .

#########################################################
#
# Déclaration des contraintes SHACL
#
#########################################################

<AdresseShape> a sh:NodeShape ;
    sh:targetClass <Adresse> ;
    sh:property [
        sh:path <Situé_à> ;
        sh:maxCount 1 ;
        sh:class <Rue>
    ] ;
    sh:property [
        sh:path <Situé_dans> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Ville>
    ] ;
    sh:property [
        sh:path <A_les_détails> ;
        sh:maxCount 1 ;
        sh:class <Détails_du_lieu>
    ] .

<VilleShape> a sh:NodeShape ;
    sh:targetClass <Ville> ;
    sh:property [
        sh:path <Situé_dans> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Région>
    ] ;
    sh:property [
        sh:path <Situé_dans> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Pays>
    ] .

<VisiteShape> a sh:NodeShape ;
    sh:targetClass <Visite> ;
    sh:property [
        sh:path <A_le_statut> ;
        sh:maxCount 1 ;
        sh:class <Status>
    ] ;
    sh:property [
        sh:path <Est_effectuée_par> ;
        sh:class <Employé>
    ] ;
    sh:property [
        sh:path <Est_la_visite_de> ;
        sh:class <Client>
    ] .

<ClientShape> a sh:NodeShape ;
    sh:targetClass <Client> ;
    sh:property [
        sh:path <Est_du_Type> ;
        sh:maxCount 1 ;
        sh:class <Type_client>
    ] ;
    sh:property [
        sh:path <A_Demandé> ;
        sh:class <Déménagement>
    ] ;
    sh:property [
        sh:path <Est_géré_par> ;
        sh:minCount 1 ;
        sh:class <Employé>
    ] .

<DevisShape> a sh:NodeShape ;
    sh:targetClass <Devis> ;
    sh:property [
        sh:path <A_la_référence> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <ID_de_réference>
    ] ;
    sh:property [
        sh:path <Est_proposé_par> ;
        sh:minCount 1 ;
        sh:class <Entreprise>
    ] ;
    sh:property [
        sh:path <Est_proposé_à> ;
        sh:minCount 1 ;
        sh:class <Client>
    ] ;
    sh:property [
        sh:path <Concerne> ;
        sh:maxCount 1 ;
        sh:class <Déménagement>
    ] .

<AccessibilitéShape> a sh:NodeShape ;
    sh:targetClass <Accessibilité> ;
    sh:property [
        sh:path <Necessite_réservation_de_parking> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Bool>
    ] ;
    sh:property [
        sh:path <Necessite_réservation_de_monte_charges> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Necessite_réservation_de_monte_charges>
    ] .
<FactureShape> a sh:NodeShape ;
    sh:targetClass <Facture> ;
    sh:property [
        sh:path <A_la_référence> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <ID_de_réference>
    ] ;
    sh:property [
        sh:path <A_été_envoyé_le> ;
        sh:minCount 1 ;
        sh:class <Date>
    ] ;
    sh:property [
        sh:path <Est_proposé_par> ;
        sh:minCount 1 ;
        sh:class <Entreprise>
    ] ;
    sh:property [
        sh:path <Est_proposé_à> ;
        sh:minCount 1 ;
        sh:class <Client>
    ] ;
    sh:property [
        sh:path <A_le_prix> ;
        sh:minCount 1 ;
        sh:class <Montant>
    ] ;
    sh:property [
        sh:path <Concerne> ;
        sh:maxCount 1 ;
        sh:class <Déménagement>
    ] ;
    sh:property [
        sh:path <Concerne> ;
        sh:maxCount 1 ;
        sh:class <Nettoyage>
    ] .

<EntrepriseShape> a sh:NodeShape ;
    sh:targetClass <Entreprise> ;
    sh:property [
        sh:path <A> ;
        sh:minCount 1 ;
        sh:class <Nom>
    ] ;
    sh:property [
        sh:path <A> ;
        sh:class <Adresse_de_l_entreprise>
    ] ;
    sh:property [
        sh:path <A> ;
        sh:class <Téléphone>
    ] ;
    sh:property [
        sh:path <A> ;
        sh:class <Email>
    ] ;
    sh:property [
        sh:path <A> ;
        sh:class <Client>
    ] ;
    sh:property [
        sh:path <Employe> ;
        sh:class <Employé>
    ] .

<DéménagementShape> a sh:NodeShape ;
    sh:targetClass <Déménagement> ;
    sh:property [
        sh:path <Est_de_la_catégorie> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Catégorie_de_déménagement>
    ] ;
    sh:property [
        sh:path <Est_du_type> ;
        sh:maxCount 1 ;
        sh:class <Type_de_déménagement>
    ] ;
    sh:property [
        sh:path <Concerne_l_adresse> ;
        sh:minCount 2 ;
        sh:class <Adresse>
    ] ;
    sh:property [
        sh:path <A_le_volume> ;
        sh:datatype xsd:float
    ] .

<NettoyageShape> a sh:NodeShape ;
    sh:targetClass <Nettoyage> ;
    sh:property [
        sh:path <Est_du_type> ;
        sh:maxCount 1 ;
        sh:class <Type_De_nettoyage>
    ] ;
    sh:property [
        sh:path <Inclus_la_prestation> ;
        sh:class <Prestation>
    ] ;
    sh:property [
        sh:path <Concerne_l_adresse> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Adresse>
    ] .

<EmployéShape> a sh:NodeShape ;
    sh:targetClass <Employé> ;
    sh:property [
        sh:path <A> ;
        sh:minCount 1 ;
        sh:class <Nom>
    ] ;
    sh:property [
        sh:path <A> ;
        sh:class <Téléphone>
    ] ;
    sh:property [
        sh:path <Travaille_pour> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Entreprise>
    ] .

<ClientPrivéShape> a sh:NodeShape ;
    sh:targetClass <Client_privé> ;
    sh:property [
        sh:path <A> ;
        sh:minCount 1 ;
        sh:class <Nom>
    ] ;
    sh:property [
        sh:path <Est_adresse_de_chargement> ;
        sh:class <Adresse>
    ] ;
    sh:property [
        sh:path <Est_adresse_de_déchargement> ;
        sh:class <Adresse>
    ] ;
    sh:property [
        sh:path <A> ;
        sh:class <Téléphone>
    ] ;
    sh:property [
        sh:path <A> ;
        sh:class <Email>
    ] ;
    sh:property [
        sh:path <A_Demandé> ;
        sh:class <Déménagement>
    ] ;
    sh:property [
        sh:path <Est_géré_par> ;
        sh:minCount 1 ;
        sh:class <Employé>
    ] .

<ClientEntrepriseShape> a sh:NodeShape ;
    sh:targetClass <Client_entreprise> ;
    sh:property [
        sh:path <A> ;
        sh:minCount 1 ;
        sh:class <Nom>
    ] ;
    sh:property [
        sh:path <Est_adresse_de_chargement> ;
        sh:class <Adresse>
    ] ;
    sh:property [
        sh:path <Est_adresse_de_déchargement> ;
        sh:class <Adresse>
    ] ;
    sh:property [
        sh:path <A> ;
        sh:class <Téléphone>
    ] ;
    sh:property [
        sh:path <A> ;
        sh:class <Email>
    ] ;
    sh:property [
        sh:path <A_Demandé> ;
        sh:class <Déménagement>
    ] ;
    sh:property [
        sh:path <Est_géré_par> ;
        sh:minCount 1 ;
        sh:class <Employé>
    ] .

<BienShape> a sh:NodeShape ;
    sh:targetClass <Bien> ;
    sh:property [
        sh:path <Est_du_Type> ;
        sh:maxCount 1 ;
        sh:class <Type_de_bien>
    ] ;
    sh:property [
        sh:path <A_le_poids> ;
        sh:maxCount 1 ;
        sh:datatype xsd:float
    ] ;
    sh:property [
        sh:path <A_la_largeur> ;
        sh:maxCount 1 ;
        sh:datatype xsd:float
    ] ;
    sh:property [
        sh:path <A_la_longueur> ;
        sh:maxCount 1 ;
        sh:datatype xsd:float
    ] ;
    sh:property [
        sh:path <A_la_profondeur> ;
        sh:maxCount 1 ;
        sh:datatype xsd:float
    ] ;
    sh:property [
        sh:path <A_le_volume> ;
        sh:maxCount 1 ;
        sh:datatype xsd:float
    ] ;
    sh:property [
        sh:path <A_les_dimensions> ;
        sh:maxCount 1 ;
        sh:datatype xsd:string
    ] ;
    sh:property [
        sh:path <Est_dans> ;
        sh:maxCount 1 ;
        sh:class <Adresse>
    ] .

<DétailsDuLieuShape> a sh:NodeShape ;
    sh:targetClass <Détails_du_lieu> ;
    sh:property [
        sh:path <Est_du_Type> ;
        sh:maxCount 1 ;
        sh:class <Type_de_lieu>
    ] ;
    sh:property [
        sh:path <Est_a_l_étage> ;
        sh:maxCount 1 ;
        sh:datatype xsd:int
    ] ;
    sh:property [
        sh:path <Est_accessible> ;
        sh:maxCount 1 ;
        sh:class <Accessibilité>
    ] ;
    sh:property [
        sh:path <Contient> ;
        sh:maxCount 1 ;
        sh:datatype xsd:float ;
        sh:name "nombre de pièces"
    ] ;
    sh:property [
        sh:path <A_la_surface> ;
        sh:maxCount 1 ;
        sh:datatype xsd:float ;
        sh:name "surface en m2"
    ] .

<FactureShape> a sh:NodeShape ;
    sh:targetClass <Facture> ;
    sh:property [
        sh:path <A_la_référence> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <ID_de_réference>
    ] ;
    sh:property [
        sh:path <A_été_envoyé_le> ;
        sh:minCount 1 ;
        sh:class <Date>
    ] ;
    sh:property [
        sh:path <Est_proposé_par> ;
        sh:minCount 1 ;
        sh:class <Entreprise>
    ] ;
    sh:property [
        sh:path <Est_proposé_à> ;
        sh:minCount 1 ;
        sh:class <Client>
    ] ;
    sh:property [
        sh:path <A_le_prix> ;
        sh:minCount 1 ;
        sh:class <Montant>
    ] ;
    sh:property [
        sh:path <Concerne> ;
        sh:maxCount 1 ;
        sh:class <Déménagement>
    ] ;
    sh:property [
        sh:path <Concerne> ;
        sh:maxCount 1 ;
        sh:class <Nettoyage>
    ] .

<EntrepriseShape> a sh:NodeShape ;
    sh:targetClass <Entreprise> ;
    sh:property [
        sh:path <A> ;
        sh:minCount 1 ;
        sh:class <Nom>
    ] ;
    sh:property [
        sh:path <A> ;
        sh:class <Adresse_de_l_entreprise>
    ] ;
    sh:property [
        sh:path <A> ;
        sh:class <Téléphone>
    ] ;
    sh:property [
        sh:path <A> ;
        sh:class <Email>
    ] ;
    sh:property [
        sh:path <A> ;
        sh:class <Client>
    ] ;
    sh:property [
        sh:path <Employe> ;
        sh:class <Employé>
    ] .

<DéménagementShape> a sh:NodeShape ;
    sh:targetClass <Déménagement> ;
    sh:property [
        sh:path <Est_de_la_catégorie> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Catégorie_de_déménagement>
    ] ;
    sh:property [
        sh:path <Est_du_type> ;
        sh:maxCount 1 ;
        sh:class <Type_de_déménagement>
    ] ;
    sh:property [
        sh:path <Concerne_l_adresse> ;
        sh:minCount 2 ;
        sh:class <Adresse>
    ] ;
    sh:property [
        sh:path <A_le_volume> ;
        sh:datatype xsd:float
    ] .

<NettoyageShape> a sh:NodeShape ;
    sh:targetClass <Nettoyage> ;
    sh:property [
        sh:path <Est_du_type> ;
        sh:maxCount 1 ;
        sh:class <Type_De_nettoyage>
    ] ;
    sh:property [
        sh:path <Inclus_la_prestation> ;
        sh:class <Prestation>
    ] ;
    sh:property [
        sh:path <Concerne_l_adresse> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Adresse>
    ] .

<EmployéShape> a sh:NodeShape ;
    sh:targetClass <Employé> ;
    sh:property [
        sh:path <A> ;
        sh:minCount 1 ;
        sh:class <Nom>
    ] ;
    sh:property [
        sh:path <A> ;
        sh:class <Téléphone>
    ] ;
    sh:property [
        sh:path <Travaille_pour> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Entreprise>
    ] .

<ClientPrivéShape> a sh:NodeShape ;
    sh:targetClass <Client_privé> ;
    sh:property [
        sh:path <A> ;
        sh:minCount 1 ;
        sh:class <Nom>
    ] ;
    sh:property [
        sh:path <Est_adresse_de_chargement> ;
        sh:class <Adresse>
    ] ;
    sh:property [
        sh:path <Est_adresse_de_déchargement> ;
        sh:class <Adresse>
    ] ;
    sh:property [
        sh:path <A> ;
        sh:class <Téléphone>
    ] ;
    sh:property [
        sh:path <A> ;
        sh:class <Email>
    ] ;
    sh:property [
        sh:path <A_Demandé> ;
        sh:class <Déménagement>
    ] ;
    sh:property [
        sh:path <Est_géré_par> ;
        sh:minCount 1 ;
        sh:class <Employé>
    ] .

<ClientEntrepriseShape> a sh:NodeShape ;
    sh:targetClass <Client_entreprise> ;
    sh:property [
        sh:path <A> ;
        sh:minCount 1 ;
        sh:class <Nom>
    ] ;
    sh:property [
        sh:path <Est_adresse_de_chargement> ;
        sh:class <Adresse>
    ] ;
    sh:property [
        sh:path <Est_adresse_de_déchargement> ;
        sh:class <Adresse>
    ] ;
    sh:property [
        sh:path <A> ;
        sh:class <Téléphone>
    ] ;
    sh:property [
        sh:path <A> ;
        sh:class <Email>
    ] ;
    sh:property [
        sh:path <A_Demandé> ;
        sh:class <Déménagement>
    ] ;
    sh:property [
        sh:path <Est_géré_par> ;
        sh:minCount 1 ;
        sh:class <Employé>
    ] .

<BienShape> a sh:NodeShape ;
    sh:targetClass <Bien> ;
    sh:property [
        sh:path <Est_du_Type> ;
        sh:maxCount 1 ;
        sh:class <Type_de_bien>
    ] ;
    sh:property [
        sh:path <A_le_poids> ;
        sh:maxCount 1 ;
        sh:datatype xsd:float
    ] ;
    sh:property [
        sh:path <A_la_largeur> ;
        sh:maxCount 1 ;
        sh:datatype xsd:float
    ] ;
    sh:property [
        sh:path <A_la_longueur> ;
        sh:maxCount 1 ;
        sh:datatype xsd:float
    ] ;
    sh:property [
        sh:path <A_la_profondeur> ;
        sh:maxCount 1 ;
        sh:datatype xsd:float
    ] ;
    sh:property [
        sh:path <A_le_volume> ;
        sh:maxCount 1 ;
        sh:datatype xsd:float
    ] ;
    sh:property [
        sh:path <A_les_dimensions> ;
        sh:maxCount 1 ;
        sh:datatype xsd:string
    ] ;
    sh:property [
        sh:path <Est_dans> ;
        sh:maxCount 1 ;
        sh:class <Adresse>
    ] .

<DétailsDuLieuShape> a sh:NodeShape ;
    sh:targetClass <Détails_du_lieu> ;
    sh:property [
        sh:path <Est_du_Type> ;
        sh:maxCount 1 ;
        sh:class <Type_de_lieu>
    ] ;
    sh:property [
        sh:path <Est_a_l_étage> ;
        sh:maxCount 1 ;
        sh:datatype xsd:int
    ] ;
    sh:property [
        sh:path <Est_accessible> ;
        sh:maxCount 1 ;
        sh:class <Accessibilité>
    ] ;
    sh:property [
        sh:path <Contient> ;
        sh:maxCount 1 ;
        sh:datatype xsd:float ;
        sh:name "nombre de pièces"
    ] ;
    sh:property [
        sh:path <A_la_surface> ;
        sh:maxCount 1 ;
        sh:datatype xsd:float ;
        sh:name "surface en m2"
    ] .

<AccessibilitéShape> a sh:NodeShape ;
    sh:targetClass <Accessibilité> ;
    sh:property [
        sh:path <Necessite_réservation_de_parking> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Bool>
    ] ;
    sh:property [
        sh:path <Necessite_réservation_de_monte_charges> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Necessite_réservation_de_monte_charges>
    ] .

         
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
