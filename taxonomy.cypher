// Création des classes
CREATE (date:Class {name: 'Date'});
CREATE (dateDeChargement:Class {name: 'Date de chargement'})-[:SUBCLASS_OF]->(date);
CREATE (dateDeDechargement:Class {name: 'Date de déchargement'})-[:SUBCLASS_OF]->(date);
CREATE (dateEnvoiDevis:Class {name: 'Date d’envoi du devis'})-[:SUBCLASS_OF]->(date);
CREATE (dateSignature:Class {name: 'Date de signature'})-[:SUBCLASS_OF]->(date);
CREATE (dateEnvoiDemandeDevis:Class {name: 'Date d’envoi de la demande de devis'})-[:SUBCLASS_OF]->(date);
CREATE (dateVisite:Class {name: 'Date de la visite'})-[:SUBCLASS_OF]->(date);

CREATE (adresse:Class {name: 'Adresse'});
CREATE (adresseChargementClient:Class {name: 'Adresse de chargement du client'})-[:SUBCLASS_OF]->(adresse);
CREATE (adresseDechargementClient:Class {name: 'Adresse de déchargement du client'})-[:SUBCLASS_OF]->(adresse);
CREATE (adresseEntreprise:Class {name: 'Adresse de l’entreprise'})-[:SUBCLASS_OF]->(adresse);

CREATE (idReference:Class {name: 'ID de référence'});
CREATE (idReferenceDevis:Class {name: 'ID de référence du devis'})-[:SUBCLASS_OF]->(idReference);
CREATE (idReferenceDemande:Class {name: 'ID de référence de la demande'})-[:SUBCLASS_OF]->(idReference);
CREATE (idReferenceFacture:Class {name: 'ID de référence de la facture'})-[:SUBCLASS_OF]->(idReference);

CREATE (telephone:Class {name: 'Téléphone'});
CREATE (email:Class {name: 'Email'});
CREATE (siteInternet:Class {name: 'Site internet'});

CREATE (employe:Class {name: 'Employé'});

CREATE (visite:Class {name: 'Visite'});
CREATE (visiteConfirmee:Class {name: 'Visite confirmée'})-[:SUBCLASS_OF]->(visite);
CREATE (visiteAnnulee:Class {name: 'Visite annulée'})-[:SUBCLASS_OF]->(visite);
CREATE (visiteNonConfirmee:Class {name: 'Visite non-confirmée'})-[:SUBCLASS_OF]->(visite);

CREATE (entreprise:Class {name: 'Entreprise'});
CREATE (notreEntreprise:Class {name: 'Notre entreprise'})-[:SUBCLASS_OF]->(entreprise);
CREATE (autresEntreprises:Class {name: 'Autres entreprises'})-[:SUBCLASS_OF]->(entreprise);

CREATE (client:Class {name: 'Client'});
CREATE (clientPrive:Class {name: 'Client Privé'})-[:SUBCLASS_OF]->(client);
CREATE (clientEntreprise:Class {name: 'Client entreprise'})-[:SUBCLASS_OF]->(client);

CREATE (categorieDem:Class {name: 'Catégorie de déménagement'});
CREATE (serviceEconomiqueGroupe:Class {name: 'Service économique groupé'})-[:SUBCLASS_OF]->(categorieDem);
CREATE (serviceEconomique:Class {name: 'Service économique'})-[:SUBCLASS_OF]->(categorieDem);
CREATE (serviceStandard:Class {name: 'Service standard'})-[:SUBCLASS_OF]->(categorieDem);
CREATE (serviceLuxe:Class {name: 'Service luxe'})-[:SUBCLASS_OF]->(categorieDem);
CREATE (servicePremium:Class {name: 'Service premium'})-[:SUBCLASS_OF]->(categorieDem);

CREATE (prestation:Class {name: 'Prestation'});

CREATE (devis:Class {name: 'Devis'});
CREATE (devisAccepte:Class {name: 'Devis accepté'})-[:SUBCLASS_OF]->(devis);
CREATE (devisRefuse:Class {name: 'Devis refusé'})-[:SUBCLASS_OF]->(devis);
CREATE (devisEnAttente:Class {name: 'Devis en attente'})-[:SUBCLASS_OF]->(devis);

CREATE (facture:Class {name: 'Facture'});
CREATE (facturePayee:Class {name: 'Facture payée'})-[:SUBCLASS_OF]->(facture);
CREATE (factureNonPayee:Class {name: 'Facture non-payée'})-[:SUBCLASS_OF]->(facture);

CREATE (accessibilite:Class {name: 'Accessibilité'});
CREATE (devantEntree:Class {name: 'Devant l’entrée'})-[:SUBCLASS_OF]->(accessibilite);
CREATE (a10mEntree:Class {name: 'À 10m de l’entrée'})-[:SUBCLASS_OF]->(accessibilite);
CREATE (a20mEntree:Class {name: 'À 20m de l’entrée'})-[:SUBCLASS_OF]->(accessibilite);
CREATE (a30m:Class {name: 'À 30m'})-[:SUBCLASS_OF]->(accessibilite);
CREATE (metres:Class {name: 'N mètres'})-[:SUBCLASS_OF]->(accessibilite);

CREATE (typeLieu:Class {name: 'Type de lieu'});
CREATE (maison:Class {name: 'Maison'})-[:SUBCLASS_OF]->(typeLieu);
CREATE (appartement:Class {name: 'Appartement'})-[:SUBCLASS_OF]->(typeLieu);
CREATE (chalet:Class {name: 'Chalet'})-[:SUBCLASS_OF]->(typeLieu);
CREATE (container:Class {name: 'Container'})-[:SUBCLASS_OF]->(typeLieu);
CREATE (inconnu:Class {name: 'Inconnu'})-[:SUBCLASS_OF]->(typeLieu);

CREATE (bien:Class {name: 'Bien'});
CREATE (meuble:Class {name: 'Meuble'})-[:SUBCLASS_OF]->(bien);
CREATE (appareil:Class {name: 'Appareil'})-[:SUBCLASS_OF]->(bien);
CREATE (carton:Class {name: 'Carton'})-[:SUBCLASS_OF]->(bien);
CREATE (oeuvreArt:Class {name: 'Œuvre d’art'})-[:SUBCLASS_OF]->(bien);
CREATE (vehicule:Class {name: 'Véhicule'})-[:SUBCLASS_OF]->(bien);

CREATE (demenagement:Class {name: 'Déménagement'});
CREATE (demenagementNational:Class {name: 'Déménagement national'})-[:SUBCLASS_OF]->(demenagement);
CREATE (demenagementInternational:Class {name: 'Déménagement international'})-[:SUBCLASS_OF]->(demenagement);

CREATE (nettoyage:Class {name: 'Nettoyage'});
CREATE (nettoyageFinBail:Class {name: 'Nettoyage de fin de bail'})-[:SUBCLASS_OF]->(nettoyage);
CREATE (nettoyageFinChantier:Class {name: 'Nettoyage de fin de chantier'})-[:SUBCLASS_OF]->(nettoyage);
// Propriétés de relations génériques
CREATE (dateDeChargement)-[:RELATION {name: 'A'}]->(jour:Class {name: 'Jour'});
CREATE (dateDeChargement)-[:RELATION {name: 'A'}]->(mois:Class {name: 'Mois'});
CREATE (dateDeChargement)-[:RELATION {name: 'A'}]->(annee:Class {name: 'Année'});

CREATE (adresseChargementClient)-[:RELATION {name: 'A'}]->(rue:Class {name: 'Rue'});
CREATE (adresseChargementClient)-[:RELATION {name: 'A'}]->(numeroRue:Class {name: 'Numéro de rue'});
CREATE (adresseChargementClient)-[:RELATION {name: 'A'}]->(ville:Class {name: 'Ville'});
CREATE (adresseChargementClient)-[:RELATION {name: 'A'}]->(codePostal:Class {name: 'Code postal'});
CREATE (adresseChargementClient)-[:RELATION {name: 'A'}]->(region:Class {name: 'Région'});
CREATE (adresseChargementClient)-[:RELATION {name: 'A'}]->(pays:Class {name: 'Pays'});

// Contraintes SHACL en Cypher
CREATE (visiteConfirmee)-[:RELATION {name: 'A'}]->(dateVisite);
CREATE (visiteConfirmee)-[:RELATION {name: 'A'}]->(adresse);
CREATE (visiteConfirmee)-[:RELATION {name: 'Est effectuée par'}]->(employe);
CREATE (visiteConfirmee)-[:RELATION {name: 'Est la visite de'}]->(client);

CREATE (visiteAnnulee)-[:RELATION {name: 'A'}]->(dateVisite);
CREATE (visiteAnnulee)-[:RELATION {name: 'A'}]->(adresse);
CREATE (visiteAnnulee)-[:RELATION {name: 'Est effectuée par'}]->(employe);
CREATE (visiteAnnulee)-[:RELATION {name: 'Est la visite de'}]->(client);

CREATE (visiteNonConfirmee)-[:RELATION {name: 'A'}]->(dateVisite);
CREATE (visiteNonConfirmee)-[:RELATION {name: 'A'}]->(adresse);
CREATE (visiteNonConfirmee)-[:RELATION {name: 'Est effectuée par'}]->(employe);
CREATE (visiteNonConfirmee)-[:RELATION {name: 'Est la visite de'}]->(client);

CREATE (notreEntreprise)-[:RELATION {name: 'A'}]->(nom:Class {name: 'Nom'});
CREATE (notreEntreprise)-[:RELATION {name: 'A'}]->(adresseEntreprise);
CREATE (notreEntreprise)-[:RELATION {name: 'A'}]->(telephone);
CREATE (notreEntreprise)-[:RELATION {name: 'A'}]->(email);
CREATE (notreEntreprise)-[:RELATION {name: 'A'}]->(client);
CREATE (notreEntreprise)-[:RELATION {name: 'A'}]->(employe);

CREATE (autresEntreprises)-[:RELATION {name: 'A'}]->(nom);
CREATE (autresEntreprises)-[:RELATION {name: 'A'}]->(adresseEntreprise);
CREATE (autresEntreprises)-[:RELATION {name: 'A'}]->(telephone);
CREATE (autresEntreprises)-[:RELATION {name: 'A'}]->(email);
CREATE (autresEntreprises)-[:RELATION {name: 'A'}]->(client);
CREATE (autresEntreprises)-[:RELATION {name: 'A'}]->(employe);

CREATE (clientPrive)-[:RELATION {name: 'A'}]->(nom);
CREATE (clientPrive)-[:RELATION {name: 'A'}]->(adresseChargementClient);
CREATE (clientPrive)-[:RELATION {name: 'A'}]->(adresseDechargementClient);
CREATE (clientPrive)-[:RELATION {name: 'A'}]->(telephone);
CREATE (clientPrive)-[:RELATION {name: 'A'}]->(email);
CREATE (clientPrive)-[:RELATION {name: 'A'}]->(demenagement);
CREATE (clientPrive)-[:RELATION {name: 'Est géré par'}]->(employe);

CREATE (clientEntreprise)-[:RELATION {name: 'A'}]->(nom);
CREATE (clientEntreprise)-[:RELATION {name: 'A'}]->(adresseChargementClient);
CREATE (clientEntreprise)-[:RELATION {name: 'A'}]->(adresseDechargementClient);
CREATE (clientEntreprise)-[:RELATION {name: 'A'}]->(telephone);
CREATE (clientEntreprise)-[:RELATION {name: 'A'}]->(email);
CREATE (clientEntreprise)-[:RELATION {name: 'A'}]->(demenagement);
CREATE (clientEntreprise)-[:RELATION {name: 'Est géré par'}]->(employe);

CREATE (serviceEconomiqueGroupe)-[:RELATION {name: 'A'}]->(prestation);
CREATE (serviceEconomique)-[:RELATION {name: 'A'}]->(prestation);
CREATE (serviceStandard)-[:RELATION {name: 'A'}]->(prestation);
CREATE (serviceLuxe)-[:RELATION {name: 'A'}]->(prestation);
CREATE (servicePremium)-[:RELATION {name: 'A'}]->(prestation);

CREATE (devisAccepte)-[:RELATION {name: 'A'}]->(idReferenceDevis);
CREATE (devisAccepte)-[:RELATION {name: 'A'}]->(dateEnvoiDevis);
CREATE (devisAccepte)-[:RELATION {name: 'Est proposé par'}]->(notreEntreprise);
CREATE (devisAccepte)-[:RELATION {name: 'Est proposé à'}]->(clientPrive);
CREATE (devisAccepte)-[:RELATION {name: 'A'}]->(montant:Class {name: 'Montant'});
CREATE (devisAccepte)-[:RELATION {name: 'Concerne'}]->(demenagement);
CREATE (devisAccepte)-[:RELATION {name: 'Concerne'}]->(nettoyage);

CREATE (devisRefuse)-[:RELATION {name: 'A'}]->(idReferenceDevis);
CREATE (devisRefuse)-[:RELATION {name: 'A'}]->(dateEnvoiDevis);
CREATE (devisRefuse)-[:RELATION {name: 'Est proposé par'}]->(notreEntreprise);
CREATE (devisRefuse)-[:RELATION {name: 'Est proposé à'}]->(clientPrive);
CREATE (devisRefuse)-[:RELATION {name: 'A'}]->(montant);
CREATE (devisRefuse)-[:RELATION {name: 'Concerne'}]->(demenagement);
CREATE (devisRefuse)-[:RELATION {name: 'Concerne'}]->(nettoyage);

CREATE (devisEnAttente)-[:RELATION {name: 'A'}]->(idReferenceDevis);
CREATE (devisEnAttente)-[:RELATION {name: 'A'}]->(dateEnvoiDevis);
CREATE (devisEnAttente)-[:RELATION {name: 'Est proposé par'}]->(notreEntreprise);
CREATE (devisEnAttente)-[:RELATION {name: 'Est proposé à'}]->(clientPrive);
CREATE (devisEnAttente)-[:RELATION {name: 'A'}]->(montant);
CREATE (devisEnAttente)-[:RELATION {name: 'Concerne'}]->(demenagement);
CREATE (devisEnAttente)-[:RELATION {name: 'Concerne'}]->(nettoyage);

CREATE (facturePayee)-[:RELATION {name: 'A'}]->(idReferenceFacture);
CREATE (facturePayee)-[:RELATION {name: 'A'}]->(dateEnvoiDevis);
CREATE (facturePayee)-[:RELATION {name: 'Est proposé par'}]->(notreEntreprise);
CREATE (facturePayee)-[:RELATION {name: 'Est proposé à'}]->(clientPrive);
CREATE (facturePayee)-[:RELATION {name: 'A'}]->(montant);
CREATE (facturePayee)-[:RELATION {name: 'Concerne'}]->(demenagement);
CREATE (facturePayee)-[:RELATION {name: 'Concerne'}]->(nettoyage);

CREATE (factureNonPayee)-[:RELATION {name: 'A'}]->(idReferenceFacture);
CREATE (factureNonPayee)-[:RELATION {name: 'A'}]->(dateEnvoiDevis);
CREATE (factureNonPayee)-[:RELATION {name: 'Est proposé par'}]->(notreEntreprise);
CREATE (factureNonPayee)-[:RELATION {name: 'Est proposé à'}]->(clientPrive);
CREATE (factureNonPayee)-[:RELATION {name: 'A'}]->(montant);
CREATE (factureNonPayee)-[:RELATION {name: 'Concerne'}]->(demenagement);
CREATE (factureNonPayee)-[:RELATION {name: 'Concerne'}]->(nettoyage);

CREATE (devantEntree)-[:RELATION {name: 'A'}]->(besoinReservationParking:Class {name: 'Besoin de réservation de parking'});
CREATE (devantEntree)-[:RELATION {name: 'A'}]->(besoinReservationMonteCharges:Class {name: 'Besoin de réservation de monte-charges'});
CREATE (a10mEntree)-[:RELATION {name: 'A'}]->(besoinReservationParking);
CREATE (a10mEntree)-[:RELATION {name: 'A'}]->(besoinReservationMonteCharges);
CREATE (a20mEntree)-[:RELATION {name: 'A'}]->(besoinReservationParking);
CREATE (a20mEntree)-[:RELATION {name: 'A'}]->(besoinReservationMonteCharges);
CREATE (a30m)-[:RELATION {name: 'A'}]->(besoinReservationParking);
CREATE (a30m)-[:RELATION {name: 'A'}]->(besoinReservationMonteCharges);
CREATE (metres)-[:RELATION {name: 'A'}]->(besoinReservationParking);
CREATE (metres)-[:RELATION {name: 'A'}]->(besoinReservationMonteCharges);

CREATE (maison)-[:RELATION {name: 'A'}]->(adresseChargementClient);
CREATE (maison)-[:RELATION {name: 'A'}]->(adresseDechargementClient);
CREATE (maison)-[:RELATION {name: 'A'}]->(etage:Class {name: 'Étage'});
CREATE (maison)-[:RELATION {name: 'A'}]->(accessibilite);
CREATE (maison)-[:RELATION {name: 'A'}]->(visite);
CREATE (maison)-[:RELATION {name: 'A'}]->(nombreDePieces:Class {name: 'Nombre de pièces'});
CREATE (maison)-[:RELATION {name: 'A'}]->(surfaceM2:Class {name: 'Surface en m2'});

CREATE (appartement)-[:RELATION {name: 'A'}]->(adresseChargementClient);
CREATE (appartement)-[:RELATION {name: 'A'}]->(adresseDechargementClient);
CREATE (appartement)-[:RELATION {name: 'A'}]->(etage);
CREATE (appartement)-[:RELATION {name: 'A'}]->(accessibilite);
CREATE (appartement)-[:RELATION {name: 'A'}]->(visite);
CREATE (appartement)-[:RELATION {name: 'A'}]->(nombreDePieces);
CREATE (appartement)-[:RELATION {name: 'A'}]->(surfaceM2);

CREATE (chalet)-[:RELATION {name: 'A'}]->(adresseChargementClient);
CREATE (chalet)-[:RELATION {name: 'A'}]->(adresseDechargementClient);
CREATE (chalet)-[:RELATION {name: 'A'}]->(etage);
CREATE (chalet)-[:RELATION {name: 'A'}]->(accessibilite);
CREATE (chalet)-[:RELATION {name: 'A'}]->(visite);
CREATE (chalet)-[:RELATION {name: 'A'}]->(nombreDePieces);
CREATE (chalet)-[:RELATION {name: 'A'}]->(surfaceM2);

CREATE (container)-[:RELATION {name: 'A'}]->(adresseChargementClient);
CREATE (container)-[:RELATION {name: 'A'}]->(adresseDechargementClient);
CREATE (container)-[:RELATION {name: 'A'}]->(etage);
CREATE (container)-[:RELATION {name: 'A'}]->(accessibilite);
CREATE (container)-[:RELATION {name: 'A'}]->(visite);
CREATE (container)-[:RELATION {name: 'A'}]->(nombreDePieces);
CREATE (container)-[:RELATION {name: 'A'}]->(surfaceM2);

CREATE (inconnu)-[:RELATION {name: 'A'}]->(adresseChargementClient);
CREATE (inconnu)-[:RELATION {name: 'A'}]->(adresseDechargementClient);
CREATE (inconnu)-[:RELATION {name: 'A'}]->(etage);
CREATE (inconnu)-[:RELATION {name: 'A'}]->(accessibilite);
CREATE (inconnu)-[:RELATION {name: 'A'}]->(visite);
CREATE (inconnu)-[:RELATION {name: 'A'}]->(nombreDePieces);
CREATE (inconnu)-[:RELATION {name: 'A'}]->(surfaceM2);

CREATE (meuble)-[:RELATION {name: 'A'}]->(poids:Class {name: 'Poids'});
CREATE (meuble)-[:RELATION {name: 'A'}]->(largeur:Class {name: 'Largeur'});
CREATE (meuble)-[:RELATION {name: 'A'}]->(longueur:Class {name: 'Longueur'});
CREATE (meuble)-[:RELATION {name: 'A'}]->(profondeur:Class {name: 'Profondeur'});
CREATE (meuble)-[:RELATION {name: 'A'}]->(volume:Class {name: 'Volume'});
CREATE (meuble)-[:RELATION {name: 'Est dans'}]->(typeLieu);

CREATE (appareil)-[:RELATION {name: 'A'}]->(poids);
CREATE (appareil)-[:RELATION {name: 'A'}]->(largeur);
CREATE (appareil)-[:RELATION {name: 'A'}]->(longueur);
CREATE (appareil)-[:RELATION {name: 'A'}]->(profondeur);
CREATE (appareil)-[:RELATION {name: 'A'}]->(volume);
CREATE (appareil)-[:RELATION {name: 'Est dans'}]->(typeLieu);

CREATE (carton)-[:RELATION {name: 'A'}]->(poids);
CREATE (carton)-[:RELATION {name: 'A'}]->(largeur);
CREATE (carton)-[:RELATION {name: 'A'}]->(longueur);
CREATE (carton)-[:RELATION {name: 'A'}]->(profondeur);
CREATE (carton)-[:RELATION {name: 'A'}]->(volume);
CREATE (carton)-[:RELATION {name: 'Est dans'}]->(typeLieu);

CREATE (oeuvreArt)-[:RELATION {name: 'A'}]->(poids);
CREATE (oeuvreArt)-[:RELATION {name: 'A'}]->(largeur);
CREATE (oeuvreArt)-[:RELATION {name: 'A'}]->(longueur);
CREATE (oeuvreArt)-[:RELATION {name: 'A'}]->(profondeur);
CREATE (oeuvreArt)-[:RELATION {name: 'A'}]->(volume);
CREATE (oeuvreArt)-[:RELATION {name: 'Est dans'}]->(typeLieu);

CREATE (vehicule)-[:RELATION {name: 'A'}]->(poids);
CREATE (vehicule)-[:RELATION {name: 'A'}]->(largeur);
CREATE (vehicule)-[:RELATION {name: 'A'}]->(longueur);
CREATE (vehicule)-[:RELATION {name: 'A'}]->(profondeur);
CREATE (vehicule)-[:RELATION {name: 'A'}]->(volume);
CREATE (vehicule)-[:RELATION {name: 'Est dans'}]->(typeLieu);

CREATE (demenagementNational)-[:RELATION {name: 'A'}]->(categorieDem);
CREATE (demenagementNational)-[:RELATION {name: 'A'}]->(typeLieu);
CREATE (demenagementNational)-[:RELATION {name: 'A'}]->(volume);

CREATE (demenagementInternational)-[:RELATION {name: 'A'}]->(categorieDem);
CREATE (demenagementInternational)-[:RELATION {name: 'A'}]->(typeLieu);
CREATE (demenagementInternational)-[:RELATION {name: 'A'}]->(volume);

CREATE (nettoyageFinBail)-[:RELATION {name: 'A'}]->(prestation);
CREATE (nettoyageFinBail)-[:RELATION {name: 'A'}]->(typeLieu);
CREATE (nettoyageFinBail)-[:RELATION {name: 'A'}]->(surfaceM2);

CREATE (nettoyageFinChantier)-[:RELATION {name: 'A'}]->(prestation);
CREATE (nettoyageFinChantier)-[:RELATION {name: 'A'}]->(typeLieu);
CREATE (nettoyageFinChantier)-[:RELATION {name: 'A'}]->(surfaceM2);
