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
- **Nodes** represent entities and concepts of a moving company, they are solely instances of the classes defined in the ontology under point 3.
Do not extract classes as nodes, you must solely find instances of the classes defined in the ontology and extract these instances as nodes.
## 2. Coreference Resolution
- **Maintain Entity Consistency**: When extracting entities, it's vital to ensure consistency.
If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"),
always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the entity ID.
Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.
## 3. Ontology : Classes, subclasses, relations and constraints
When extracting the nodes and relations, please strictly follow this Ontology :

#########################################################
#
# Déclaration des classes
#
#########################################################

<Date> rdf:type owl:Class ; 
    rdfs:comment "Classe représentant les différentes dates importantes" .

<DateDeChargement> rdfs:subClassOf <Date> ;
    rdfs:comment "Date de chargement" .

<DateDeDechargement> rdfs:subClassOf <Date> ;
    rdfs:comment "Date de déchargement" .

<DateEnvoiDevis> rdfs:subClassOf <Date> ;
    rdfs:comment "Date d'envoi du devis" .

<DateSignature> rdfs:subClassOf <Date> ;
    rdfs:comment "Date de signature" .

<DateEnvoiDemandeDevis> rdfs:subClassOf <Date> ;
    rdfs:comment "Date d'envoi de la demande de devis" .

<DateVisite> rdfs:subClassOf <Date> ;
    rdfs:comment "Date de la visite" .

<Adresse> rdf:type owl:Class ; 
    rdfs:comment "Classe représentant les différentes adresses" .

<AdresseChargementClient> rdfs:subClassOf <Adresse> ;
    rdfs:comment "Adresse de chargement du client" .

<AdresseDechargementClient> rdfs:subClassOf <Adresse> ;
    rdfs:comment "Adresse de déchargement du client" .

<AdresseEntreprise> rdfs:subClassOf <Adresse> ;
    rdfs:comment "Adresse de l'entreprise" .

<IDReference> rdf:type owl:Class ; 
    rdfs:comment "Classe représentant les ID de référence" .

<IDReferenceDevis> rdfs:subClassOf <IDReference> ;
    rdfs:comment "ID de référence du devis" .

<IDReferenceDemande> rdfs:subClassOf <IDReference> ;
    rdfs:comment "ID de référence de la demande" .

<IDReferenceFacture> rdfs:subClassOf <IDReference> ;
    rdfs:comment "ID de référence de la facture" .

<Telephone> rdf:type owl:Class ; 
    rdfs:comment "Classe représentant le numéro de téléphone" .

<Email> rdf:type owl:Class ; 
    rdfs:comment "Classe représentant l'adresse email" .

<SiteInternet> rdf:type owl:Class ; 
    rdfs:comment "Classe représentant le site internet" .

<Employe> rdf:type owl:Class ; 
    rdfs:comment "Classe représentant un employé" .

<Visite> rdf:type owl:Class ; 
    rdfs:comment "Classe représentant les visites" .

<VisiteConfirmee> rdfs:subClassOf <Visite> ;
    rdfs:comment "Visite confirmée" .

<VisiteAnnulee> rdfs:subClassOf <Visite> ;
    rdfs:comment "Visite annulée" .

<VisiteNonConfirmee> rdfs:subClassOf <Visite> ;
    rdfs:comment "Visite non-confirmée" .

<Entreprise> rdf:type owl:Class ; 
    rdfs:comment "Classe représentant les entreprises" .

<NotreEntreprise> rdfs:subClassOf <Entreprise> ;
    rdfs:comment "Notre entreprise" .

<AutresEntreprises> rdfs:subClassOf <Entreprise> ;
    rdfs:comment "Autres entreprises" .

<Client> rdf:type owl:Class ; 
    rdfs:comment "Classe représentant les clients" .

<ClientPrive> rdfs:subClassOf <Client> ;
    rdfs:comment "Client privé" .

<ClientEntreprise> rdfs:subClassOf <Client> ;
    rdfs:comment "Client entreprise" .

<CategorieDem> rdf:type owl:Class ; 
    rdfs:comment "Classe représentant les catégories de déménagement" .

<ServiceEconomiqueGroupe> rdfs:subClassOf <CategorieDem> ;
    rdfs:comment "Service économique groupé" .

<ServiceEconomique> rdfs:subClassOf <CategorieDem> ;
    rdfs:comment "Service économique" .

<ServiceStandard> rdfs:subClassOf <CategorieDem> ;
    rdfs:comment "Service standard" .

<ServiceLuxe> rdfs:subClassOf <CategorieDem> ;
    rdfs:comment "Service luxe" .

<ServicePremium> rdfs:subClassOf <CategorieDem> ;
    rdfs:comment "Service premium" .

<Prestation> rdf:type owl:Class ;
    rdfs:comment "Classe représentant une prestation" .

<Devis> rdf:type owl:Class ; 
    rdfs:comment "Classe représentant les devis" .

<DevisAccepte> rdfs:subClassOf <Devis> ;
    rdfs:comment "Devis accepté" .

<DevisRefuse> rdfs:subClassOf <Devis> ;
    rdfs:comment "Devis refusé" .

<DevisEnAttente> rdfs:subClassOf <Devis> ;
    rdfs:comment "Devis en attente" .

<Facture> rdf:type owl:Class ; 
    rdfs:comment "Classe représentant les factures" .

<FacturePayee> rdfs:subClassOf <Facture> ;
    rdfs:comment "Facture payée" .

<FactureNonPayee> rdfs:subClassOf <Facture> ;
    rdfs:comment "Facture non-payée" .

<Accessibilite> rdf:type owl:Class ; 
    rdfs:comment "Classe représentant l'accessibilité" .

<DevantEntree> rdfs:subClassOf <Accessibilite> ;
    rdfs:comment "Devant l'entrée" .

<A10mEntree> rdfs:subClassOf <Accessibilite> ;
    rdfs:comment "À 10m de l'entrée" .

<A20mEntree> rdfs:subClassOf <Accessibilite> ;
    rdfs:comment "À 20m de l'entrée" .

<A30m> rdfs:subClassOf <Accessibilite> ;
    rdfs:comment "À 30m" .

<Metres> rdfs:subClassOf <Accessibilite> ;
    rdfs:comment "N mètres" .

<TypeLieu> rdf:type owl:Class ; 
    rdfs:comment "Classe représentant les types de lieux" .

<Maison> rdfs:subClassOf <TypeLieu> ;
    rdfs:comment "Maison" .

<Appartement> rdfs:subClassOf <TypeLieu> ;
    rdfs:comment "Appartement" .

<Chalet> rdfs:subClassOf <TypeLieu> ;
    rdfs:comment "Chalet" .

<Container> rdfs:subClassOf <TypeLieu> ;
    rdfs:comment "Container" .

<Inconnu> rdfs:subClassOf <TypeLieu> ;
    rdfs:comment "Inconnu" .

<Bien> rdf:type owl:Class ; 
    rdfs:comment "Classe représentant les biens" .

<Meuble> rdfs:subClassOf <Bien> ;
    rdfs:comment "Meuble" .

<Appareil> rdfs:subClassOf <Bien> ;
    rdfs:comment "Appareil" .

<Carton> rdfs:subClassOf <Bien> ;
    rdfs:comment "Carton" .

<OeuvreArt> rdfs:subClassOf <Bien> ;
    rdfs:comment "Œuvre d'art" .

<Vehicule> rdfs:subClassOf <Bien> ;
    rdfs:comment "Véhicule" .

<Demenagement> rdf:type owl:Class ; 
    rdfs:comment "Classe représentant les déménagements" .

<DemenagementNational> rdfs:subClassOf <Demenagement> ;
    rdfs:comment "Déménagement national" .

<DemenagementInternational> rdfs:subClassOf <Demenagement> ;
    rdfs:comment "Déménagement international" .

<Nettoyage> rdf:type owl:Class ; 
    rdfs:comment "Classe représentant les nettoyages" .

<NettoyageFinBail> rdfs:subClassOf <Nettoyage> ;
    rdfs:comment "Nettoyage de fin de bail" .

<NettoyageFinChantier> rdfs:subClassOf <Nettoyage> ;
    rdfs:comment "Nettoyage de fin de chantier" .

#########################################################
#
# Déclaration des propriétés d'objets
#
#########################################################

<A> rdf:type owl:ObjectProperty ; 
    rdfs:comment "Propriété de relation générique" ;
    rdfs:domain owl:Class ;
    rdfs:range owl:Class .

<EstEffectuePar> rdf:type owl:ObjectProperty ; 
    rdfs:comment "Propriété liant une visite à un employé" ;
    rdfs:domain <Visite> ;
    rdfs:range <Employe> .

<EstVisiteDe> rdf:type owl:ObjectProperty ; 
    rdfs:comment "Propriété liant une visite à un client" ;
    rdfs:domain <Visite> ;
    rdfs:range <Client> .

<EstProposePar> rdf:type owl:ObjectProperty ; 
    rdfs:comment "Propriété liant un devis/facture à une entreprise" ;
    rdfs:domain [ owl:unionOf (<Devis> <Facture>) ] ;
    rdfs:range <Entreprise> .

<EstProposeA> rdf:type owl:ObjectProperty ; 
    rdfs:comment "Propriété liant un devis/facture à un client" ;
    rdfs:domain [ owl:unionOf (<Devis> <Facture>) ] ;
    rdfs:range <Client> .

<Concerne> rdf:type owl:ObjectProperty ; 
    rdfs:comment "Propriété liant un devis/facture à un déménagement/nettoyage" ;
    rdfs:domain [ owl:unionOf (<Devis> <Facture>) ] ;
    rdfs:range [ owl:unionOf (<Demenagement> <Nettoyage>) ] .

<EstDans> rdf:type owl:ObjectProperty ; 
    rdfs:comment "Propriété liant un bien à un type de lieu" ;
    rdfs:domain <Bien> ;
    rdfs:range <TypeLieu> .

<EstGerePar> rdf:type owl:ObjectProperty ; 
    rdfs:comment "Propriété liant un client à un employé" ;
    rdfs:domain <Client> ;
    rdfs:range <Employe> .

#########################################################
#
# Déclaration des propriétés de données
#
#########################################################

<Nom> rdf:type owl:DatatypeProperty ;
    rdfs:domain [ owl:unionOf (<Employe> <Client> <Entreprise>) ] ;
    rdfs:range rdfs:Literal ;
    rdfs:comment "Propriété représentant le nom d'un employé, client ou entreprise" .

<Montant> rdf:type owl:DatatypeProperty ;
    rdfs:domain [ owl:unionOf (<Devis> <Facture>) ] ;
    rdfs:range xsd:decimal ;
    rdfs:comment "Propriété représentant le montant d'un devis ou d'une facture" .

<Jour> rdf:type owl:DatatypeProperty ;
    rdfs:domain <Date> ;
    rdfs:range xsd:integer ;
    rdfs:comment "Propriété représentant le jour d'une date" .

<Mois> rdf:type owl:DatatypeProperty ;
    rdfs:domain <Date> ;
    rdfs:range xsd:integer ;
    rdfs:comment "Propriété représentant le mois d'une date" .

<Annee> rdf:type owl:DatatypeProperty ;
    rdfs:domain <Date> ;
    rdfs:range xsd:integer ;
    rdfs:comment "Propriété représentant l'année d'une date" .

<Rue> rdf:type owl:DatatypeProperty ;
    rdfs:domain <Adresse> ;
    rdfs:range rdfs:Literal ;
    rdfs:comment "Propriété représentant la rue d'une adresse" .

<NumeroRue> rdf:type owl:DatatypeProperty ;
    rdfs:domain <Adresse> ;
    rdfs:range xsd:integer ;
    rdfs:comment "Propriété représentant le numéro de rue d'une adresse" .

<Ville> rdf:type owl:DatatypeProperty ;
    rdfs:domain <Adresse> ;
    rdfs:range rdfs:Literal ;
    rdfs:comment "Propriété représentant la ville d'une adresse" .

<CodePostal> rdf:type owl:DatatypeProperty ;
    rdfs:domain <Adresse> ;
    rdfs:range xsd:string ;
    rdfs:comment "Propriété représentant le code postal d'une adresse" .

<Region> rdf:type owl:DatatypeProperty ;
    rdfs:domain <Adresse> ;
    rdfs:range rdfs:Literal ;
    rdfs:comment "Propriété représentant la région d'une adresse" .

<Pays> rdf:type owl:DatatypeProperty ;
    rdfs:domain <Adresse> ;
    rdfs:range rdfs:Literal ;
    rdfs:comment "Propriété représentant le pays d'une adresse" .

#########################################################
#
# Déclaration des contraintes SHACL
#
#########################################################

@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

<DateDeChargementShape> a sh:NodeShape ;
    sh:targetClass <DateDeChargement> ;
    sh:property [
        sh:path <Jour> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Mois> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Annee> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] .

<DateDeDechargementShape> a sh:NodeShape ;
    sh:targetClass <DateDeDechargement> ;
    sh:property [
        sh:path <Jour> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Mois> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Annee> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] .

<DateEnvoiDevisShape> a sh:NodeShape ;
    sh:targetClass <DateEnvoiDevis> ;
    sh:property [
        sh:path <Jour> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Mois> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Annee> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] .

<DateSignatureShape> a sh:NodeShape ;
    sh:targetClass <DateSignature> ;
    sh:property [
        sh:path <Jour> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Mois> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Annee> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] .

<DateEnvoiDemandeDevisShape> a sh:NodeShape ;
    sh:targetClass <DateEnvoiDemandeDevis> ;
    sh:property [
        sh:path <Jour> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Mois> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Annee> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] .

<DateVisiteShape> a sh:NodeShape ;
    sh:targetClass <DateVisite> ;
    sh:property [
        sh:path <Jour> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Mois> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Annee> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] .

<AdresseChargementClientShape> a sh:NodeShape ;
    sh:targetClass <AdresseChargementClient> ;
    sh:property [
        sh:path <Rue> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <NumeroRue> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Ville> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <CodePostal> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Region> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Pays> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] .

<AdresseDechargementClientShape> a sh:NodeShape ;
    sh:targetClass <AdresseDechargementClient> ;
    sh:property [
        sh:path <Rue> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <NumeroRue> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Ville> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <CodePostal> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Region> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Pays> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] .

<AdresseEntrepriseShape> a sh:NodeShape ;
    sh:targetClass <AdresseEntreprise> ;
    sh:property [
        sh:path <Rue> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <NumeroRue> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Ville> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <CodePostal> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Region> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Pays> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] .

<VisiteConfirmeeShape> a sh:NodeShape ;
    sh:targetClass <VisiteConfirmee> ;
    sh:property [
        sh:path <DateVisite> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Adresse> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <EstEffectuePar> ;
        sh:minCount 1 ;
        sh:class <Employe> ;
    ] ;
    sh:property [
        sh:path <EstVisiteDe> ;
        sh:minCount 1 ;
        sh:class <Client> ;
    ] .

<VisiteAnnuleeShape> a sh:NodeShape ;
    sh:targetClass <VisiteAnnulee> ;
    sh:property [
        sh:path <DateVisite> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Adresse> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <EstEffectuePar> ;
        sh:minCount 1 ;
        sh:class <Employe> ;
    ] ;
    sh:property [
        sh:path <EstVisiteDe> ;
        sh:minCount 1 ;
        sh:class <Client> ;
    ] .

<VisiteNonConfirmeeShape> a sh:NodeShape ;
    sh:targetClass <VisiteNonConfirmee> ;
    sh:property [
        sh:path <DateVisite> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Adresse> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <EstEffectuePar> ;
        sh:minCount 1 ;
        sh:class <Employe> ;
    ] ;
    sh:property [
        sh:path <EstVisiteDe> ;
        sh:minCount 1 ;
        sh:class <Client> ;
    ] .

<NotreEntrepriseShape> a sh:NodeShape ;
    sh:targetClass <NotreEntreprise> ;
    sh:property [
        sh:path <Nom> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <AdresseEntreprise> ;
        sh:minCount 1 ;
        sh:class <Adresse> ;
    ] ;
    sh:property [
        sh:path <Telephone> ;
        sh:class <Telephone> ;
    ] ;
    sh:property [
        sh:path <Email> ;
        sh:class <Email> ;
    ] ;
    sh:property [
        sh:path <Client> ;
        sh:class <Client> ;
    ] ;
    sh:property [
        sh:path <Employe> ;
        sh:class <Employe> ;
    ] .

<AutresEntreprisesShape> a sh:NodeShape ;
    sh:targetClass <AutresEntreprises> ;
    sh:property [
        sh:path <Nom> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <AdresseEntreprise> ;
        sh:minCount 1 ;
        sh:class <Adresse> ;
    ] ;
    sh:property [
        sh:path <Telephone> ;
        sh:class <Telephone> ;
    ] ;
    sh:property [
        sh:path <Email> ;
        sh:class <Email> ;
    ] ;
    sh:property [
        sh:path <Client> ;
        sh:class <Client> ;
    ] ;
    sh:property [
        sh:path <Employe> ;
        sh:class <Employe> ;
    ] .

<ClientPriveShape> a sh:NodeShape ;
    sh:targetClass <ClientPrive> ;
    sh:property [
        sh:path <Nom> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <AdresseChargementClient> ;
        sh:class <Adresse> ;
    ] ;
    sh:property [
        sh:path <AdresseDechargementClient> ;
        sh:class <Adresse> ;
    ] ;
    sh:property [
        sh:path <Telephone> ;
        sh:class <Telephone> ;
    ] ;
    sh:property [
        sh:path <Email> ;
        sh:class <Email> ;
    ] ;
    sh:property [
        sh:path <Demenagement> ;
        sh:class <Demenagement> ;
    ] ;
    sh:property [
        sh:path <EstGerePar> ;
        sh:minCount 1 ;
        sh:class <Employe> ;
    ] .

<ClientEntrepriseShape> a sh:NodeShape ;
    sh:targetClass <ClientEntreprise> ;
    sh:property [
        sh:path <Nom> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <AdresseChargementClient> ;
        sh:class <Adresse> ;
    ] ;
    sh:property [
        sh:path <AdresseDechargementClient> ;
        sh:class <Adresse> ;
    ] ;
    sh:property [
        sh:path <Telephone> ;
        sh:class <Telephone> ;
    ] ;
    sh:property [
        sh:path <Email> ;
        sh:class <Email> ;
    ] ;
    sh:property [
        sh:path <Demenagement> ;
        sh:class <Demenagement> ;
    ] ;
    sh:property [
        sh:path <EstGerePar> ;
        sh:minCount 1 ;
        sh:class <Employe> ;
    ] .

<ServiceEconomiqueGroupeShape> a sh:NodeShape ;
    sh:targetClass <ServiceEconomiqueGroupe> ;
    sh:property [
        sh:path <Prestation> ;
        sh:class <Prestation> ;
    ] .

<ServiceEconomiqueShape> a sh:NodeShape ;
    sh:targetClass <ServiceEconomique> ;
    sh:property [
        sh:path <Prestation> ;
        sh:class <Prestation> ;
    ] .

<ServiceStandardShape> a sh:NodeShape ;
    sh:targetClass <ServiceStandard> ;
    sh:property [
        sh:path <Prestation> ;
        sh:class <Prestation> ;
    ] .

<ServiceLuxeShape> a sh:NodeShape ;
    sh:targetClass <ServiceLuxe> ;
    sh:property [
        sh:path <Prestation> ;
        sh:class <Prestation> ;
    ] .

<ServicePremiumShape> a sh:NodeShape ;
    sh:targetClass <ServicePremium> ;
    sh:property [
        sh:path <Prestation> ;
        sh:class <Prestation> ;
    ] .

<DevisAccepteShape> a sh:NodeShape ;
    sh:targetClass <DevisAccepte> ;
    sh:property [
        sh:path <IDReferenceDevis> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <DateEnvoiDevis> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <EstProposePar> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Entreprise> ;
    ] ;
    sh:property [
        sh:path <EstProposeA> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Client> ;
    ] ;
    sh:property [
        sh:path <Montant> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Concerne> ;
        sh:maxCount 1 ;
        sh:class <Demenagement> ;
    ] ;
    sh:property [
        sh:path <Concerne> ;
        sh:minCount 1 ;
        sh:class <Nettoyage> ;
    ] .

<DevisRefuseShape> a sh:NodeShape ;
    sh:targetClass <DevisRefuse> ;
    sh:property [
        sh:path <IDReferenceDevis> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <DateEnvoiDevis> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <EstProposePar> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Entreprise> ;
    ] ;
    sh:property [
        sh:path <EstProposeA> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Client> ;
    ] ;
    sh:property [
        sh:path <Montant> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Concerne> ;
        sh:maxCount 1 ;
        sh:class <Demenagement> ;
    ] ;
    sh:property [
        sh:path <Concerne> ;
        sh:minCount 1 ;
        sh:class <Nettoyage> ;
    ] .

<DevisEnAttenteShape> a sh:NodeShape ;
    sh:targetClass <DevisEnAttente> ;
    sh:property [
        sh:path <IDReferenceDevis> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <DateEnvoiDevis> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <EstProposePar> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Entreprise> ;
    ] ;
    sh:property [
        sh:path <EstProposeA> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Client> ;
    ] ;
    sh:property [
        sh:path <Montant> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Concerne> ;
        sh:maxCount 1 ;
        sh:class <Demenagement> ;
    ] ;
    sh:property [
        sh:path <Concerne> ;
        sh:minCount 1 ;
        sh:class <Nettoyage> ;
    ] .

<FacturePayeeShape> a sh:NodeShape ;
    sh:targetClass <FacturePayee> ;
    sh:property [
        sh:path <IDReferenceFacture> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <DateEnvoiDevis> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <EstProposePar> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Entreprise> ;
    ] ;
    sh:property [
        sh:path <EstProposeA> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Client> ;
    ] ;
    sh:property [
        sh:path <Montant> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Concerne> ;
        sh:maxCount 1 ;
        sh:class <Demenagement> ;
    ] ;
    sh:property [
        sh:path <Concerne> ;
        sh:minCount 1 ;
        sh:class <Nettoyage> ;
    ] .

<FactureNonPayeeShape> a sh:NodeShape ;
    sh:targetClass <FactureNonPayee> ;
    sh:property [
        sh:path <IDReferenceFacture> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <DateEnvoiDevis> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <EstProposePar> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Entreprise> ;
    ] ;
    sh:property [
        sh:path <EstProposeA> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Client> ;
    ] ;
    sh:property [
        sh:path <Montant> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Concerne> ;
        sh:maxCount 1 ;
        sh:class <Demenagement> ;
    ] ;
    sh:property [
        sh:path <Concerne> ;
        sh:minCount 1 ;
        sh:class <Nettoyage> ;
    ] .

<DevantEntreeShape> a sh:NodeShape ;
    sh:targetClass <DevantEntree> ;
    sh:property [
        sh:path <BesoinReservationParking> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <BesoinReservationMonteCharges> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] .

<A10mEntreeShape> a sh:NodeShape ;
    sh:targetClass <A10mEntree> ;
    sh:property [
        sh:path <BesoinReservationParking> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <BesoinReservationMonteCharges> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] .

<A20mEntreeShape> a sh:NodeShape ;
    sh:targetClass <A20mEntree> ;
    sh:property [
        sh:path <BesoinReservationParking> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <BesoinReservationMonteCharges> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] .

<A30mShape> a sh:NodeShape ;
    sh:targetClass <A30m> ;
    sh:property [
        sh:path <BesoinReservationParking> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <BesoinReservationMonteCharges> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] .

<MetresShape> a sh:NodeShape ;
    sh:targetClass <Metres> ;
    sh:property [
        sh:path <BesoinReservationParking> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <BesoinReservationMonteCharges> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] .

<MaisonShape> a sh:NodeShape ;
    sh:targetClass <Maison> ;
    sh:property [
        sh:path <AdresseChargementClient> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Adresse> ;
    ] ;
    sh:property [
        sh:path <AdresseDechargementClient> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Adresse> ;
    ] ;
    sh:property [
        sh:path <Etage> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Accessibilite> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Visite> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <NombreDePieces> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <SurfaceM2> ;
        sh:maxCount 1 ;
    ] .

<AppartementShape> a sh:NodeShape ;
    sh:targetClass <Appartement> ;
    sh:property [
        sh:path <AdresseChargementClient> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Adresse> ;
    ] ;
    sh:property [
        sh:path <AdresseDechargementClient> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Adresse> ;
    ] ;
    sh:property [
        sh:path <Etage> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Accessibilite> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Visite> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <NombreDePieces> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <SurfaceM2> ;
        sh:maxCount 1 ;
    ] .

<ChaletShape> a sh:NodeShape ;
    sh:targetClass <Chalet> ;
    sh:property [
        sh:path <AdresseChargementClient> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Adresse> ;
    ] ;
    sh:property [
        sh:path <AdresseDechargementClient> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Adresse> ;
    ] ;
    sh:property [
        sh:path <Etage> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Accessibilite> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Visite> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <NombreDePieces> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <SurfaceM2> ;
        sh:maxCount 1 ;
    ] .

<ContainerShape> a sh:NodeShape ;
    sh:targetClass <Container> ;
    sh:property [
        sh:path <AdresseChargementClient> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Adresse> ;
    ] ;
    sh:property [
        sh:path <AdresseDechargementClient> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Adresse> ;
    ] ;
    sh:property [
        sh:path <Etage> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Accessibilite> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Visite> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <NombreDePieces> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <SurfaceM2> ;
        sh:maxCount 1 ;
    ] .

<InconnuShape> a sh:NodeShape ;
    sh:targetClass <Inconnu> ;
    sh:property [
        sh:path <AdresseChargementClient> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Adresse> ;
    ] ;
    sh:property [
        sh:path <AdresseDechargementClient> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <Adresse> ;
    ] ;
    sh:property [
        sh:path <Etage> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Accessibilite> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Visite> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <NombreDePieces> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <SurfaceM2> ;
        sh:maxCount 1 ;
    ] .

<MeubleShape> a sh:NodeShape ;
    sh:targetClass <Meuble> ;
    sh:property [
        sh:path <Poids> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Largeur> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Longueur> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Profondeur> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Volume> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <EstDans> ;
        sh:maxCount 1 ;
        sh:class <TypeLieu> ;
    ] .

<AppareilShape> a sh:NodeShape ;
    sh:targetClass <Appareil> ;
    sh:property [
        sh:path <Poids> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Largeur> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Longueur> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Profondeur> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Volume> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <EstDans> ;
        sh:maxCount 1 ;
        sh:class <TypeLieu> ;
    ] .

<CartonShape> a sh:NodeShape ;
    sh:targetClass <Carton> ;
    sh:property [
        sh:path <Poids> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Largeur> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Longueur> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Profondeur> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Volume> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <EstDans> ;
        sh:maxCount 1 ;
        sh:class <TypeLieu> ;
    ] .

<OeuvreArtShape> a sh:NodeShape ;
    sh:targetClass <OeuvreArt> ;
    sh:property [
        sh:path <Poids> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Largeur> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Longueur> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Profondeur> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Volume> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <EstDans> ;
        sh:maxCount 1 ;
        sh:class <TypeLieu> ;
    ] .

<VehiculeShape> a sh:NodeShape ;
    sh:targetClass <Vehicule> ;
    sh:property [
        sh:path <Poids> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Largeur> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Longueur> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Profondeur> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <Volume> ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <EstDans> ;
        sh:maxCount 1 ;
        sh:class <TypeLieu> ;
    ] .

<DemenagementNationalShape> a sh:NodeShape ;
    sh:targetClass <DemenagementNational> ;
    sh:property [
        sh:path <CategorieDem> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <TypeLieu> ;
        sh:minCount 2 ;
        sh:class <TypeLieu> ;
    ] ;
    sh:property [
        sh:path <Volume> ;
        sh:class <Volume> ;
    ] .

<DemenagementInternationalShape> a sh:NodeShape ;
    sh:targetClass <DemenagementInternational> ;
    sh:property [
        sh:path <CategorieDem> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path <TypeLieu> ;
        sh:minCount 2 ;
        sh:class <TypeLieu> ;
    ] ;
    sh:property [
        sh:path <Volume> ;
        sh:class <Volume> ;
    ] .

<NettoyageFinBailShape> a sh:NodeShape ;
    sh:targetClass <NettoyageFinBail> ;
    sh:property [
        sh:path <Prestation> ;
        sh:class <Prestation> ;
    ] ;
    sh:property [
        sh:path <TypeLieu> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <TypeLieu> ;
    ] ;
    sh:property [
        sh:path <SurfaceM2> ;
        sh:maxCount 1 ;
    ] .

<NettoyageFinChantierShape> a sh:NodeShape ;
    sh:targetClass <NettoyageFinChantier> ;
    sh:property [
        sh:path <Prestation> ;
        sh:class <Prestation> ;
    ] ;
    sh:property [
        sh:path <TypeLieu> ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class <TypeLieu> ;
    ] ;
    sh:property [
        sh:path <SurfaceM2> ;
        sh:maxCount 1 ;
    ] .

## 4. Strict Compliance
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
