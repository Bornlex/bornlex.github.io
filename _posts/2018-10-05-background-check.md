---
layout: post
title: "Background checking - Partie 1"
---

<div class="message">
     Vérifier les CV et s'assurer que les candidats aux entretiens d'embauche ont bien été dans les écoles qu'ils
     prétendent peut s'avérer difficile pour les départements de ressources humaines et les recruteurs.
     Voyons comment l'OSINT peut nous aider dans cette tâche.
</div>

## Classification

La manière dont nous allons nous y prendre dépend essentiellement du type d'étude du candidats.
Certaines écoles sont plus secrètes que d'autres et gardent secrète la liste de leurs étudiants, même après qu'ils
aient quitté l'école.

### Les ingénieurs

Les ingénieurs sont les plus simples à trouver. Pour une raison simple : il existe un registre national des diplômés des écoles d'ingénieurs. Autrement dit, nous avons accès à une base de données des anciens élèves des écoles d'ingénieurs.
La voici : <a href="https://repertoire.iesf.fr/">Répertoire des Ingénieurs et Scientifiques</a>.

N'hésitez pas à taper un nom dans cette barre de recherche pour essayer.

Voici les informations que l'on peut espérer obtenir grâce à une recherche sur ce site :
- Nom de l'élève
- Nom de l'école
- Année d'obtention du diplôme
- Adresse de l'école
- Contact de l'école (téléphone et adresse email)
- Contact de l'association des anciens élèves de l'école


C'est plus qu'il n'en faut pour s'assurer que le candidat est bien ce qu'il prétend être, ou non...

### Les écoles de commerce

Les étudiants en école de commerce sont difficiles à trouver pour plusieurs raisons :
- il n'existe pas de répertoire commun à toutes les écoles
- les écoles sont souvent privées et fonctionnent différemment

Cela dit, quelques petites techniques peuvent vous aider.

Etant souvent privées, les écoles fonctionnent recourent parfois aux dons. Et la plupart du temps, les listes de donnateurs (classés par taille de don) sont publiques.
Elles ne représentent évidemment pas tous les étudiants (loin de là évidemment), et des non-étudiants peuvent donner à une école s'ils le souhaitent, mais les petits dons sont majoritairement réalisés par d'anciens élèves de l'école. Trouver le nom du candidat dans la liste des dons de l'école qu'il affiche sur son CV est un signe qu'il dit probablement la vérité.