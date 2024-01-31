+++
title = 'Le problème des vendeurs de plage'
date = 2024-01-31T17:04:23+01:00
draft = false
+++

Pourquoi est-ce que ni l’extrême droite, ni l’extrême gauche ne gagne jamais l’élection présidentielle ?

On pourrait penser que la courbe du nombre de votants en fonction du spectre politique ressemble à une courbe gaussienne centrée sur le milieu de l’échiquier politique, avec quelques variations. Il y a plus de gens modérés que d’extrêmes.

![Loi normale](/plage/gaussian.webp)

Mais si l’on regarde les intentions de vote pour chaque candidat ou les résultats du premier tour de l’élection présidentielle, on réalise rapidement que cette réponse n’est pas satisfaisante.

![Intentions de vote](/plage/scores.webp)

Le graphique des résultats donne plutôt l’impression d’une distribution uniforme pour 4 grands candidats (qui se partagent 85% des votants), et quelques résidus pour les autres.

Comment expliquer alors que la victoire n’ait jamais été donnée à Jean-Luc Mélenchon ou Marine Le Pen, qui sont des candidats de longue date, et dont les scores sont élevés depuis plusieurs années déjà ?

Je suggère une piste qui mérite d’être explorée pour tenter d’expliquer cet apparent paradoxe.

## Petite précision d'abord

Avant d’introduire ma réflexion, je souhaite cependant prendre les devants d’une erreur qui pourrait être faite en voyant le graphique :

Obtenir 15% des voix n’est pas équivalent à avoir 15% de chance de gagner l’élection.

Si l’on s’intéresse au 2ème tour de l’élection, les résultats serrés ne sont pas rares. En observant un score de 60%-40%, on serait tentés de penser que l’élection s’est jouée au coude à coude.
Mais si l’on réfléchit bien à la signification de ces chiffres, il en est tout autre. Si un candidat obtient 600 voix sur 1000 et l’autre 400, le premier candidat a en fait 50% de voix de plus que son concurrent.

Autrement dit, il aurait fallu que le 2ème déploie 50% de ressources en plus pour rattraper le premier. Les chances de victoires étaient donc loin de 60% et 40%. En réalité, le 2ème avait très peu de chance de gagner, certainement moins d’une chance sur 10, ou 20.

On pourrait imaginer une courbe des probabilités de victoire (si tant est qu’on puisse la construire) qui soit légèrement différente des résultats, et qui accentue les différences entre les candidats. Comme une sorte de softmax.

# Réflexion

J’aimerais présenter maintenant la réflexion.

Il existe en statistiques économiques un problème qui s’appelle le problème des vendeurs de plage. Le problème met en scène deux vendeurs de glace qui se partagent une plage. Ils doivent choisir un emplacement pour leur stand afin, bien entendu, d’avoir le plus de clients possibles.

Lorsqu’un touriste souhaite acheter une glace, il se dirige naturellement vers le vendeur le plus proche de sa position.
De la même manière, un électeur vote pour le candidat qui se rapproche le plus de son identité politique.

On cherche donc à connaître la configuration des vendeurs sur la plage, où doivent-ils positionner leur stand, de sorte d’obtenir le maximum de clients.

## Formalisation

Afin de réfléchir rigoureusement, nous allons maintenant formaliser la situation.

On note :

- $A$ : le vendeur le plus à gauche
- $B$ : le vendeur le plus à droite
- $x_A$ : la position en abscisse du vendeur A
- $x_B$ : la position en abscisse du vendeur B
- $m$ : la position de la ligne médiane (une ligne imaginaire qui se trouve à égale distance des deux vendeurs)
- $L$ : la taille de la plage

Voici ces notations représentées sur un schéma :

![Plage](/plage/beach.webp)

Ici A est à gauche, et B à droite. Si jamais A “dépassait” B en se décalant à droite, alors on renommerait A en B et B en A de sorte que la situation soit exactement identique (par symétrie).

On suppose donc : $x_A \\leq x_B$.

La ligne médiane se trouve à équidistance de A et de B. On peut donc donner la formule de m :

$$
m = \frac{x_B - x_A}{2} + x_A
$$

On veut maintenant connaître les valeurs de xA et de xB de sorte que A et B maximisent respectivement leur nombre de clients.

Pour cela, nous devons mesurer le nombre de touristes qui vont se rendre au stand de chacun des vendeurs. On note :

- $c_A$ : le nombre de clients de A
- $c_B$ : le nombre de clients de B

- Puisque l’on sait que les touristes iront au stand le plus proche, on sait que :

- tous les touristes qui se trouvent à gauche de A iront vers A
- tous les touristes qui se trouvent à droite de B iront vers B
- les touristes situés entre $x_A$ et m iront vers A
- les touristes situés entre m et $x_B$ iront vers B

Voici la polarisation des touristes :

![Polarisation des touristes](/plage/beach2.webp)

On observe que A obtient en fait tous les touristes de la limite gauche de la plage jusqu’à la ligne médiane, et que B obtient les autres, ceux entre la ligne médiane et la limite droite de la plage.

En équation, cela donne très simplement :

$$
\\left\\{\\begin{matrix}
c_A = m &  \\\
c_B = L - m = L - c_A &  \\\
\\end{matrix}\\right.
$$

On observe bien l’antagonisme à la deuxième ligne : $c_B = L - c_A$. On constate que L est le nombre de clients maximal, et que B récupère ceux que ne prend pas A.

## Résolution

Il nous faut maintenant nous poser la question de la stratégie des deux vendeurs.

Elle est en réalité très simple.

Voyons la situation du point de vue de B :

![](/plage/beach3.webp)

- Étape 1 : B constate que A est plus à gauche que le centre de la plage
- Étape 2 : B se rapproche de A, et se met très légèrement plus à droite de lui
- Étape 3 : B récupère donc tous les clients à sa droite, ce qui correspond à plus de la moitié des clients
- Étape 4 : A a donc intérêt à effectuer la même opération que B, et se place légèrement plus à droite que B, c’est maintenant lui qui obtient la plus grosse part de clients
- Étape 5 : B recommence le raisonnement initial

Les deux vendeurs vont donc se déplacer petit à petit jusqu’à atteindre le centre de la plage, où se positionner légèrement à droite de son concurrent n’aura plus d’intérêt.

La situation est symétrique, si B est trop à droite, A viendra se mettre juste à sa gauche, etc.

Finalement, les deux vendeurs se trouveront au milieu, l’un à côté de l’autre et bénéficieront chacun d’un côté de la plage :

- $x_A = x_B$
- $x_B = L - x_A$
- $x_B + x_A = L$
- $2 x_B = L$
- $x_B = x_A = \\frac{L}{2}$

# Métaphore

Ce problème et sa résolution, sont en réalité une métaphore du véritable sujet qui nous occupe, la politique.

Une fois que le premier tour des élections présidentielles est terminé, il ne reste plus que deux candidats. Et à ce moment-là, la situation se rapproche de celle que nous venons d’évoquer.

Lors des dernières élections par exemple, en 2017, le duel du deuxième tour s’est joué entre Emmanuel Macron (LREM) et Marine Le Pen (FN).

Marine Le Pen étant considérée plus à droite qu’Emmanuel Macron, les électeurs de la France Insoumise ou du Parti Socialiste se reportent plutôt vers Emmanuel Macron, plus proche (ou plutôt moins loin) de leurs convictions. C’est le côté gauche de la plage.
Les électeurs plus à droite de Marine Le Pen iront voter pour elle, et ceux entre Marine Le Pen et Emmanuel Macron, le parti Les Républicains grossièrement, se scinderont en deux groupes. Un groupe ira voter pour Emmanuel Macron, et l’autre pour Marine Le Pen, suivant leurs sensibilités. C’est la partie qui se trouvait entre les deux vendeurs, dans la situation initiale.

Voici à quoi ressemble notre plage politique au 2ème tour :

![Elections](/plage/beach4.webp)

Emmanuel Macron gagne parce qu’il est considéré plus au centre. Le centre étant l’optimum de la stratégie des vendeurs de plage.

## Vote stratégique

Ce constat est le fondement de ce qui s’appelle le vote stratégique, dont on entend parfois parler dans les médias.

L’idée est, pour n’importe quel candidat du centre, ou proche du centre, d’augmenter autant que faire se peut le score du Rassemblement National (la stratégie pourrait fonctionner aussi la France Insoumise), de sorte de se retrouver face à lui au second tour. De cette manière, étant plus au centre, le candidat gagne l’élection.

# Limites théoriques

J’aimerais terminer en abordant les limites théoriques et les différences entre l’exemple des vendeurs et le sujet politique.

## Déplacement sur la plage

Autant un vendeur peut facilement se déplacer sur la plage, autant un politicien est affilié à un groupe, et ne voyage pas dans le paysage politique tout à fait librement, ce qui constitue une différence importante avec notre exemple.

## Déplacement de la plage

Dans la vie politique, c’est plutôt la plage, autrement dit le contexte politique, qui se déplace.

Un homme politique peut être déplacé par le contexte. Le centre du spectre de 2021 est différent du centre du spectre de 1910. Un même discours peut être considéré rétrospectivement comme plus à droite ou plus à gauche qu’il ne l’était à son époque, certaines choses inacceptables deviennent acceptables et inversement.

## Plusieurs plages

Nous pouvons même aller plus loin dans la métaphore. En politique, il est impossible de représenter tous les candidats le long d’une droite, un espace à une seule dimension. Certains candidats sont plus libéraux, donc plus à droite économiquement, mais peuvent être moins conservateurs socialement, donc plus à gauche.

On pourrait donc imaginer une droite par thème politique :

- une droite pour l’économie
- une droite pour l’immigration
- une droite pour l’aspect social
- une droite pour la question européenne
- ...

Chacun d’entre nous est sensible différemment aux sujets qui occupent la vie politique. **Pour un individu donné, toutes les plages ne se valent pas**.

## Cambridge Analytica

Le déplacement du contexte est en fait le point important. C’est ce qui fonde les tactiques politiciennes.

Lors de la campagne de Donald Trump contre Hillary Clinton, l’entreprise Cambridge Analytica a utilisé les informations des profils Facebook de 50 millions d’américains de sorte de leur présenter des informations qui les touchaient le plus.

Ils avaient bien compris que les électeurs qu’il fallait cibler étaient les indécis (il est beaucoup trop difficile de convaincre un électeur de Benoît Hamon de voter pour François Fillon, par exemple).
En revanche, les électeurs qui se trouvaient entre Clinton et Trump étaient susceptibles de changer d’avis.

Ils ont donc montré des publicités et des faits d’actualité aux indécis de sorte de présenter le candidat Trump comme la réponse la plus adaptée au contexte politique. Le discours de Donald Trump n’a pas changé, mais le contexte oui. C’est la plage qui s’est déplacée.

## Temps de marche

On peut supposer sans prendre de risque que certains clients allongés aux extrémités de la plage pourraient renoncer à leur glace s’il leur fallait marcher top longtemps.
Similairement, un électeur peut ne pas voter, ou voter blanc, si les candidats lui semblent trop lointain de sa conception de la politique.

## Limites

Évidemment, comme n’importe quelle métaphore, l’exemple des vendeurs de plage a ses limites. Les candidats sont choisis selon de nombreux critères, notamment de personnalité, de cohérence…

De plus, dans le cas des vendeurs, le vote dit “barrage” n’existe pas. Personne ne se rend chez Quick pour désavantager McDonald.

Mais elle permet d’approcher de manière intéressante un point de société.

# Origine des vendeurs de plage

Le problème des vendeurs de glace est un problème de théorie des jeux, qui modélise de manière simplifiée la Loi de Hotelling, en économie.

Cette loi stipule que, dans certains marchés économiques, la concurrence entraîne une diminution de la différenciation des produits. Dans ces cas, une concertation des acteurs entre eux pourrait permettre de résoudre le problème, ce qui contredit le principe de la main invisible.

# Edit

A la fin de l’écriture de l’article, je suis tombé par hasard sur un théorème qui s’appelle le théorème de l’électeur médian, et qui stipule qu’en politique, l’option centriste bat la gauche à l’aide de la droite, et bat la droite à l’aide de la gauche.

Autrement dit, l’option la plus au centre, si elle se retrouve au second tour contre une seule autre option, gagne l’élection.

C’est tout à fait le sens de ma réflexion. Je cite donc ce théorème par honnêteté intellectuelle.
