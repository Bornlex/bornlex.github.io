---
layout: post
title: "Google - Recherche avancée"
---

<div class="message">
     Le moteur de recherche Google est capable de bien plus que les simples requêtes dont nous avons l'habitude. Voyons comment en tirer profit au maximum.
</div>


## Google

Google est le plus important moteur de recherche du monde, sa part de marché est de près de 80%. Il traite plus de 3,5 milliards de recherches par jour, ce qui représente 40 000 requêtes par seconde. Autrement dit, c'est un outil très puissant...

Mais malgré tout ce qu'il est capable de faire, nous n'utilisons en réalité qu'une petite partie de ses fonctionnalités. Google est capable de traiter des requêtes plus compliquées que celles dont nous avons l'habitude.
Il est possible d'utiliser ce que l'on appelle les opérateurs afin d'affiner la recherche.

Regardons plus en détails ces fameux opérateurs.

## Les Opérateurs

### Guillemets

L'opérateur '"' sert à chercher un terme exact, une expression entière par exemple.

> "steve jobs"

cherchera uniquement l'expression complète 'steve jobs' dans les résultats, et ne retournera donc pas les pages qui contiennent uniquement 'steve' ou uniquement 'jobs' ni celles qui contiennent 'jobs steve'.

### -

Il est possible de retirer des mots de la recherche. Les pages retournées par Google ne contiendront aucun des mots que vous avez specifiés comme étant à enlever.

> steve -jobs

Les résultats contiendront le prénom Steve **mais pas** Jobs.

### AND

Utiliser l'opérateur 'AND' ('et') entre deux mots ou termes va specifier à Google de ne rechercher que les pages qui contiennent ces mots ou termes, peu importe l'ordre.

> steve AND jobs

retournera les pages contenant le mot 'steve' **et** le mot 'jobs' obligatoirement, mais ils peuvent se trouver n'importe ou dans la page, pas forcément l'un à coté de l'autre comme dans le cas des guillemets.

### OR

x OR y ('ou') renverra les pages qui contiennent x **ou** y, **ou les deux**. Il est aussi possible d'utiliser le 'pipe' ('|') à la place du mot OR.

> apple OR microsoft

retournera les pages contenant des références à Apple ou Microsoft.

### *

L'étoile '*' est utilisée pour capturer tous les résultats.
Il n'est pas utile dans une recherche simple, mais on verra comment s'en servir efficacement plus tard.

### ()

Les parenthèses permettent de grouper des termes.

> (ipad OR iphone) apple

retournera les résultats contenant apple et **soit** iphone **soit** ipad **soit** les deux. On aura donc les pages contenant 'apple' et 'ipad', les pages contenant 'apple' et 'iphone', et les pages contenant 'apple' et 'ipad' et 'iphone'.

### filetype: ou ext:

**filetype:** et **ext:** sont équivalents. Ils servent à spécifier les résultats d'un certain type de fichier, par exemple les fichiers PDF, TXT, DOCX...

> apple ext:pdf

renverra tous les PDF qui contiennent le mot 'apple'.

### site:

Cet opérateur est utile pour limiter la recherche à un **site internet** en particulier.
Si je cherche un article du Monde contenant certains mots, je peux utiliser cet opérateur pour que Google ne regarde que dans les pages du Monde.

> apple site:lemonde.fr

renverra toutes les références à Apple trouvées sur le site internet du Monde.

### related:

> related:apple.com

renverra tous les sites internet **en lien** avec celui d'Apple.

### intitle:

Cet opérateur permet de faire une recherche dans le **titre** des pages.

> intitle:apple iphone

renverra toutes les pages dont le titre contient 'apple' ou 'iphone', ou les deux.

### allintitle:

Celui-ci est similaire au précédent, sauf qu'il ne cherchera que les titres qui contiennent tous les mots de la recherche.

### inurl:

Comme vous pouvez l'imaginer, cet opérateur cherchera dans l'**url** des sites internet.

> inurl:apple iphone

renverra toutes les pages dont l'url contient apple, iphone ou les deux.

### allinurl:

Celui-ci est similaire au précédent, sauf qu'il ne cherchera que les urls qui contiennent tous les mots de la recherche.

### AROUND(X)

Cet opérateur effectue une recherche **par proximité**. Il indique à Google de ne renvoyer que les résultats qui contiennent les mots de la recherche mais qu'en plus ces mots ne doivent pas etre à une distance de plus de X mots dans le contenu.

> apple AROUND(10) iphone

renverra toutes les pages qui contiennent 'apple' et 'iphone' à une distance maximale de 10 mots.

### map:

Affiche une petite **carte** qui correspond à une localisation.

> map:paris

montre la carte de Paris.

### source:

Cet opérateur affiche les **news** ne provenant que de la **source** demandée.

> apple source:the_verge

affichera les news à propos d'Apple en provenance de The Verge.

### X..Y

Cet opérateur limite la recherche à une **plage de dates**.

> apple 2015..2018

renverra les pages contenant 'apple' qui ont ete publiees entre 2015 et 2018.

### inanchor:

Cet opérateur un peu particulier va renvoyer des résultats dont les **liens entrant** contiendront les mots specifiés dans la requête.

> inanchor:apple

renverra les pages pointées par des sites qui contiennents le mot 'apple'.

### allinanchor:

La version exclusive de l'opérateur précédent.

## Combiner les opérateurs

Bien entendu, pour des recherches encore plus poussées, nous pouvons combiner les opérateurs entre eux afin de n'obtenir que des résultats précisément ciblés.

Deux applications sont detaillées dans d'autres posts :
- une application positive, vérifier les CV des candidats aux entretients d'embauche : [Background checking](/2018/09/02/background-checking/)
- une application moins positive, chercher les fichiers qui ne devraient pas etre indexés par Google et qui peuvent contenir des informations confidentielles (normalement) : [Google Dorking](/2018/09/03/google-dorking/)