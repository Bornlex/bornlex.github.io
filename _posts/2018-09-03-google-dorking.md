---
layout: post
title: "Google Dorking"
---

<div class="message">
     Google indexe tout ce qu'il voit, et si un site internet ne restreint pas l'accès à des données confidentielles, elles peuvent être indexées par Google.
Regardons ensemble comment obtenir ce que Google n'aurait pas du voir.
</div>

## Vous avez dit confidentiel ?

### Fonctionnement de Google

Google, comme les autres moteurs de recherche, **visite** les pages internet et **suit** les liens qu'il trouve dans ces sites afin de se rendre sur d'autres sites pas encore visités et de continuer son périple.
Chaque page sur laquelle il se rend est analysée et **"indexée"**, c'est à dire insérée dans une base de données.
A chaque requête, Google cherche dans sa base de données les résultats les plus pertinents, les ordonne et les affiche à l'utilisateur.

### Documents indexés

Puisque Google ne fait que suivre les liens qu'il trouve dans les pages qu'il visite, il ne pourra pas voir ce qui n'est référencé par
personne, mais il pourra voir ce que vous ne voulez pas qu'il voit si par mégarde vous y faites référence dans une des pages de votre
site internet.
Autrement, si un document Word, PDF, Excel se trouve sur votre site internet et que vous y faites référence d'une manière ou d'une autre
dans une page, Google le visitera, le lira et l'indexera.
Une fois indexé, ce document sera disponible à la recherche, vous pourrez potentiellement le trouver via [google.com](google.com).


## Définition

Le Google Dorking (ou Google Hacking) se sert des opérateurs dont nous avons parlé ici [Google - Recherche avancée](/2018/09/02/google-advanced/) afin de déterrer spécifiquement les résultats qui ne devraient pas se trouver accessible aux moteurs de recherche.

Donnons quelques exemples concrets et expliquons-les.

## Mesdames et messieurs : les exemples

Voici trois exemples parmi bien d'autres qui vous donneront un apercu de ce que vous pouvez trouver
grace à notre cher Google. J'espère qu'ils vous sensibiliseront à l'importance de faire attention à
comment votre site internet est organisé.

Encore une fois, je vous conseille de lire l'article sur la [recherche avancée](/2018/09/02/google-advanced/) pour comprendre mieux comment fonctionnent ces 'hacks'.

### Excel

Comme nous en avons parlé un peu plus haut, commencons avec les fichiers de type tableur Excel.
Ce sont tous les fichiers dont l'extension fait partie de la liste suivante :
- xls
- xlsm
- xlsx
- xlsb

Excel et les logiciels de tableur gèrent bien d'autres formats mais ceux-la sont les plus susceptibles
d'avoir été référencés par Google, parce que ce sont ceux que l'on utilise le plus.

Souvent, ils contiennent des données de comptabilité, des listes d'utilisateurs, des notes etc.

Voici un squelette de commande pour trouver ces fichiers :

> ext:xls test

Il est important de préciser un terme à rechercher (ici 'test'), sinon Google ne vous renverra rien. En effet, comment
pourrait-il trier les fichiers sur la pertinence à partir d'aucune recherche ?

Vous pouvez voir que cette commande va retourner uniquement des fichiers au format **.xls** et qui
contiennent le mot **test**.
Vous pouvez maintenant utiliser n'importe quel format et n'importe quel terme pour trouver ce qui vous
intéresse.

PS : je vous laisse imaginer ce que vous pouvez trouver si vous utiliser **password** ou **username** à
la place de **test**...

### Mots de passe - usernames

Malheureusement, parfois, les sites internet sont attaqués, les bases de données sont volées et
vendues sur internet ou rendues publiques.

Une pratique courante est de rendre public des séries d'identifiants et les mots de passe qui leur
sont associés via le site [Pastebin](https://pastebin.com/).

Essayons de trouver toutes les adresses GMail rendues publiques sur Pastebin :
> site:pastebin.com intext:@gmail.com

Comme vous pouvez le voir, il y en a quelques unes...
La plupart d'entre elles sont relativement vieilles, elles datent de 2017. Utilisons un autre opérateur
afin de n'afficher que les résultats de 2018.

> site:pastebin.com intext:@gmail.com 2018..2018

Voila qui est mieux. N'hésitez pas à cliquer sur les liens. Vous pouvez voir que beaucoup ne contiennent
que des adresses mail. Mais certains résultats contiennent aussi des mots de passe. Libre à vous de savoir
s'ils sont justes, en votre âme et conscience...

**PS :** pourquoi ne chercheriez-vous pas votre adresse mail ?

### Webcams & caméras

Le titre n'est pas trompeur, on peut trouver des webcams publiques grace à Google.

> inurl:"ViewerFrame?Mode="

Cette recherche va renvoyer les résultats qui contiennent le terme entre guillemets dans l'url du site.
Ce terme n'est pas choisi au hasard, il correspond à des sites internets fournis avec des caméras.
Certains sites ne seront pas des webcams, ils contiendront ce terme pour d'autres raisons.

Essayez quelques liens, vous vous rendrez compte que vous pouvez effectivement voir ce que voient
les caméras dans les rues, et parfois meme les controller.

## Conclusion

Comme vous l'avez constaté, Google est un outil très puissant, et il peut servir à bien plus qu'à
renvoyer une page Wikipedia ou le site d'Apple (je n'ai aucun problème ni avec Wikipedia ni avec Apple bien entendu).

Expérimentez, changez les extensions de fichier, les format d'adresse email ou le site sur lequel des
combinaisons d'identifiant-mot de passe pourraient avoir été rendues publiques et faites-moi part de vos trouvailles !

**PS :** si vous cherchez à savoir si votre adresse mail a été corrompue ou non lors d'un hack, utilisez
plutot le site [HaveIBeenPwned](https://haveibeenpwned.com/).

**PS2 :** pour trouver les webcams et caméras, je vous conseille [Shodan](https://www.shodan.io/search?query=webcam).