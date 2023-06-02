# Titre de votre projet 

Description du projet
* programme moteur
* Explication de l'objectif du code
* qui sont les clients ?
* période du projet
* Niveau de confidentialité de la données C1, C2, C3
* ajouter des hashtags pour votre projet (https://gitlab.sofa.snm.snecma/documentation/gitlab-sofa/-/wikis/home#partage-de-projets) pour faciliter les recherches de votre projet
 
## Données d'entrée
 
description des données d'entrée
* emplacement de stockage
* format de la données (csv, hdf5, etc...)
* Si le projet n'utilise pas toutes les données disponibles, préciser le périmètre d'utilisation (ex : manque de labélisation, périmètre de maintenance, etc)
* garant de la donnée ou source si ça vient d'un service
* Niveau de confidentialité de la données C1, C2, C3 (ajouter le badge correspondant https://gitlab.sofa.snm.snecma/documentation/gitlab-sofa/-/wikis/home#sensibilitA9)
* dictionnaire de données
 
## Description du code
 
Explication des différents branches du projet
 
### Version Python
 
rappeler la version de Python utilisée pour créer la venv  
ne pas oublier de mettre à jour le requirements.txt dans votre environnement : 
```python
pip freeze > requirements.txt
```
 
### Ressources matérielles
Quel hardware a été utilisé pour faire tourner le projet ?  
Quel est le minimum de capacité de calcul nécessaire pour lancer le projet :
* CPU
* GPU (version CUDA minimum, cuDNN, etc)
* Mémoire
 
Préciser le temps de calcul avec le hardware utilisé
 
### Fonctionnement
Expliquer comment fonctionne votre code :
- quelles sont les étapes pour lancer votre code
- diagramme UML (optionnel)
 
### Arborescence
Expliquer les différents fichiers
 
### Tests
Comment on lance les tests unitaires ou des scénarios de tests
 
## Résultats
 
Si le projet est terminé, vous pouvez préciser les principaux résultats
* Brevets
* Thèse
* Performance de l'algorithme
* Les incompris restants
 
## TODO (facultatif)
 
Pour les librairies, vous pouvez inclure la liste des tâches à faire
 
## Pistes de développement explorées
 
Description des méthodes déjà testées même si elles n'ont pas fonctionné
 
## Documents projet
Lien vers les documents utiles du projet :
* DEC 
* Planches de présentation
* Bibliographie
 
## Outils externes
Préciser si vous avez utilisé des outils externes à votre projet :
* Dépendance à d'autres librairies (s'assurer qu'on a les accès au code)
* logiciels 
* PowerBI
* API
* etc
 
## Membres du projet
 
préciser les personnes qui ont réalisé le projet ainsi que les personnes avec qui vous avez été en interface
 
