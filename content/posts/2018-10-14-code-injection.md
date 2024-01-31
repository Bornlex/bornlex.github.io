+++
date = 2024-01-24T14:03:13+01:00
title = "Code injection"
draft = false
+++

<div class="message">
     Le premier article à propos de sécurité informatique à proprement parler. C'est un sujet passionnant et je vais partager quelques techniques entre deux articles sur l'OSINT.
</div>

## Introduction

J'ai découvert l'OpenSource Intelligence en m'intéressant de près à la sécurité informatique.

Lorsque des pirates préparent une attaque sur un système d'information (SI), ils ont besoin d'avoir autant d'informations que possible afin de mener à bien les opérations. En effet, les systèmes d'exploitation (OS), les serveurs utilisés, les versions de ces serveurs et de ces OS conditionnent énormément la manière dont va se dérouler l'attaque.

Il est donc primordial de disposer d'un arsenal de techniques qui vont nous permettre de préciser un peu le contexte de l'attaque.

C'est la raison pour laquelle je me suis intéressé à l'OSINT.

Mais je suis toujours un passionné de sécurité informatique, particulièrement sur Linux et je suis content de pouvoir partager une technique de **code injection** que je trouve instructive.

## Contexte

Lorsqu'un pirate prend le contrôle d'un serveur ou d'un ordinateur, il peut souhaite altérer certaines fonctionnalités du système sur lequel il se trouve pour conserver l'accès par exemple.
La difficulté à laquelle il doit faire face cependant, est de rester invisible autant que faire se peut.
Lancer un exécutable sur le système pourrait avoir l'air louche. En revanche, particulièrement sur les serveurs qui hébergent des sites internet, de nombreux programmes s'exécutent en permanence
pour supprimer les logs à intervale donné par exemple.

Ne pourrait-on pas utiliser ces programmes pour s'immiscer dans le système en ayant l'air légitime ?

Oui, on peut le faire, et on va voir comment.

## Prérequis

Cette technique est relativement simple, mais elle nécessite tout de même quelques connaissances préalables, notamment à propos de ce que l'on appelle les **relocations**.

La compilation est un processus qui se déroule en 3 étapes :
- la précompilation (préprocesseur) : c'est une phase de substitution de texte, où toutes les lignes commençant par le symbole '#' sont traitées (#define...).
- la compilation : un fichier binaire est créé pour chaque fichier source
- le linkage : le compilateur agrège chaque fichier binaire avec les bibliothèques auxquelles les fichiers font référence. Les bibliothèques statiques sont recopiées dans l'exécutable final contrairement aux bibliothèques dynamiques, qui doivent être présentes sur le système au moment de l'exécution. Le linker vérifie que chaque fonction est déclarée et implémentée (ce qui n'était pas le cas jusqu'à présent), et que les fonctions ne sont implémentées qu'une seule fois.

Finalement, le linker "édite les liens", on dit aussi qu'il "résout les symboles". Après la compilation, les appels de fonctions qui se trouvent dans les bibliothèques ne sont pas "résolus", c'est à dire que leurs adresses ne pointent pas encore sur la fonction en question, ils pointent vers l'instruction suivante dans l'exécutable. C'est le linker qui s'occupe de cette phase via la création d'une **table de relocation**.
Le linker va passer sur le code petit à petit et à chaque fois qu'il trouve un appel de fonction, il remplace cet appel par l'adresse de la fonction en question, si elle est connue.
La table des relocations indique comment chaque symbole présent dans le fichier doit être obtenu (par rapport au **program counter**, etc).
Nous allons voir à quoi elle ressemble dans la section suivante, lorsque nous en aurons besoin.

## Technique

Imaginons qu'un programme s'exécute en permanence sur le système. Pour des raisons de simplicité, nous allons utiliser un programme en boucle infinie qui appelle une fonction "print" depuis une bibliothèque
partagée puis qui marque une pause pendant 3 secondes, et qui recommence.

Voici le code de ce programme **app.cc** :
```c
#include <dlfcn.h>
#include <iostream>
#include <unistd.h>
#include "dynlib.h"

using namespace std;

int main()
{
  while (true)
    {
      print();
      cout << "going to sleep..." << endl;
      sleep(3);
      cout << "waked up!" << endl;
    }
  return 0;
}
```

Et voici celui de la bibliothèque **dynlib.cc** :
```c
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <iostream>
#include "dynlib.h"

using namespace std;

extern "C" void print()
{
  static unsigned int counter = 0;
  counter++;

  cout << counter << ": PID " << getpid() << ": in print()" << endl;
}
```

Et son header dynlib.h :
```c
extern "C" void print();
```

Pour terminer, voici le code injection.cc que l'on aimerait injecter à la place de la fonction **print** de la bibliothèque partagée dynlib :
```c
#include <stdlib.h>

extern "C" void print();

extern "C" void injection()
{
  print();
  system("date");
}
```

Et le Makefile pour les compiler tous :
```make
all:
	g++ -ggdb -Wall dynlib.cc -fPIC -shared -o libdynlib.so
	g++ -ggdb app.cc -ldynlib -L./ -o app
	gcc -Wall injection.cc -c -o injection.o
```

Une fois les programmes compilés, vous devriez avoir les ressources suivantes :
- app
- libdynlib.so
- injection.so


Premièrement, nous avons besoin de connaitre le PID du programme qui s'exécute. Dans mon cas, le PID est **13858**.

Utilisons GDB afin de nous attacher au programme qui s'exécute :
> gdb 13858 app

On a besoin de voir à quoi ressemble le mapping de la mémoire du processus. Les memory mappings se trouvent dans le dossier **/proc/pid/**.
Le fichier de mapping s'appelle **maps**.
Dans mon cas, je peux obtenir le memory mapping du processus simplement par la commande :
> cat /proc/13858/maps

Voilà à quoi cela ressemble :
```
563d5aba1000-563d5aba2000 r-xp 00000000 08:01 6816141                    /root/Documents/security/lab/code_injection/app
563d5ada1000-563d5ada2000 r--p 00000000 08:01 6816141                    /root/Documents/security/lab/code_injection/app
563d5ada2000-563d5ada3000 rw-p 00001000 08:01 6816141                    /root/Documents/security/lab/code_injection/app
563d5b4be000-563d5b4df000 rw-p 00000000 00:00 0                          [heap]
7f6704235000-7f67043e6000 r-xp 00000000 08:01 2230879                    /lib/x86_64-linux-gnu/libc-2.27.so
7f67043e6000-7f67045e5000 ---p 001b1000 08:01 2230879                    /lib/x86_64-linux-gnu/libc-2.27.so
7f67045e5000-7f67045e9000 r--p 001b0000 08:01 2230879                    /lib/x86_64-linux-gnu/libc-2.27.so
7f67045e9000-7f67045eb000 rw-p 001b4000 08:01 2230879                    /lib/x86_64-linux-gnu/libc-2.27.so
7f67045eb000-7f67045ef000 rw-p 00000000 00:00 0
7f67045ef000-7f6704606000 r-xp 00000000 08:01 2230279                    /lib/x86_64-linux-gnu/libgcc_s.so.1
7f6704606000-7f6704805000 ---p 00017000 08:01 2230279                    /lib/x86_64-linux-gnu/libgcc_s.so.1
7f6704805000-7f6704806000 r--p 00016000 08:01 2230279                    /lib/x86_64-linux-gnu/libgcc_s.so.1
7f6704806000-7f6704807000 rw-p 00017000 08:01 2230279                    /lib/x86_64-linux-gnu/libgcc_s.so.1
7f6704807000-7f6704999000 r-xp 00000000 08:01 2230887                    /lib/x86_64-linux-gnu/libm-2.27.so
7f6704999000-7f6704b98000 ---p 00192000 08:01 2230887                    /lib/x86_64-linux-gnu/libm-2.27.so
7f6704b98000-7f6704b99000 r--p 00191000 08:01 2230887                    /lib/x86_64-linux-gnu/libm-2.27.so
7f6704b99000-7f6704b9a000 rw-p 00192000 08:01 2230887                    /lib/x86_64-linux-gnu/libm-2.27.so
7f6704b9a000-7f6704d10000 r-xp 00000000 08:01 13238937                   /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.25
7f6704d10000-7f6704f10000 ---p 00176000 08:01 13238937                   /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.25
7f6704f10000-7f6704f1a000 r--p 00176000 08:01 13238937                   /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.25
7f6704f1a000-7f6704f1c000 rw-p 00180000 08:01 13238937                   /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.25
7f6704f1c000-7f6704f1f000 rw-p 00000000 00:00 0
7f6704f1f000-7f6704f20000 r-xp 00000000 08:01 6816036                    /root/Documents/security/lab/code_injection/libdynlib.so
7f6704f20000-7f670511f000 ---p 00001000 08:01 6816036                    /root/Documents/security/lab/code_injection/libdynlib.so
7f670511f000-7f6705120000 r--p 00000000 08:01 6816036                    /root/Documents/security/lab/code_injection/libdynlib.so
7f6705120000-7f6705121000 rw-p 00001000 08:01 6816036                    /root/Documents/security/lab/code_injection/libdynlib.so
7f6705121000-7f6705146000 r-xp 00000000 08:01 2230166                    /lib/x86_64-linux-gnu/ld-2.27.so
```


La partie de la mémoire allouée à la bibliothèque dynamique se trouve entre les adresses **0x7f6704f1f000** et **0x7f6705121000**.

La fonction print étant définie dans cette bibliothèque, on peut vérifier qu'elle s'y trouve bien grâce à GDB :
> p &print
(void (*)(void)) 0x7f6704f1f97a <print()>

C'est bien le cas.

Jetons un oeil à la table de relocations grâce à **readelf** :
> readelf -r app

```
Relocation section '.rela.dyn' at offset 0x6d8 contains 12 entries:
  Offset          Info           Type           Sym. Value    Sym. Name + Addend
000000200d90  000000000008 R_X86_64_RELATIVE                    a20
000000200d98  000000000008 R_X86_64_RELATIVE                    ad8
000000200da0  000000000008 R_X86_64_RELATIVE                    9e0
000000201050  000000000008 R_X86_64_RELATIVE                    201050
000000200fc8  000100000006 R_X86_64_GLOB_DAT 0000000000000000 __cxa_finalize@GLIBC_2.2.5 + 0
000000200fd0  000200000006 R_X86_64_GLOB_DAT 0000000000000000 _ZSt4endlIcSt11char_tr@GLIBCXX_3.4 + 0
000000200fd8  000900000006 R_X86_64_GLOB_DAT 0000000000000000 _ITM_deregisterTMClone + 0
000000200fe0  000a00000006 R_X86_64_GLOB_DAT 0000000000000000 __libc_start_main@GLIBC_2.2.5 + 0
000000200fe8  000b00000006 R_X86_64_GLOB_DAT 0000000000000000 __gmon_start__ + 0
000000200ff0  000c00000006 R_X86_64_GLOB_DAT 0000000000000000 _ITM_registerTMCloneTa + 0
000000200ff8  000d00000006 R_X86_64_GLOB_DAT 0000000000000000 _ZNSt8ios_base4InitD1E@GLIBCXX_3.4 + 0
000000201060  001300000005 R_X86_64_COPY     0000000000201060 _ZSt4cout@GLIBCXX_3.4 + 0

Relocation section '.rela.plt' at offset 0x7f8 contains 6 entries:
  Offset          Info           Type           Sym. Value    Sym. Name + Addend
000000201018  000300000007 R_X86_64_JUMP_SLO 0000000000000000 sleep@GLIBC_2.2.5 + 0
000000201020  000400000007 R_X86_64_JUMP_SLO 0000000000000000 __cxa_atexit@GLIBC_2.2.5 + 0
000000201028  000500000007 R_X86_64_JUMP_SLO 0000000000000000 _ZStlsISt11char_traits@GLIBCXX_3.4 + 0
000000201030  000600000007 R_X86_64_JUMP_SLO 0000000000000000 _ZNSolsEPFRSoS_E@GLIBCXX_3.4 + 0
000000201038  000700000007 R_X86_64_JUMP_SLO 0000000000000000 print + 0
000000201040  000800000007 R_X86_64_JUMP_SLO 0000000000000000 _ZNSt8ios_base4InitC1E@GLIBCXX_3.4 + 0
```

Dans cette table, on constate que la fonction print se trouve à l'offset **000000201038** dans notre exécutable.
Encore une fois, vérifions avec GDB :
> p/x *(0x563d5aba1000+0x000000201038)
0x4f1f97a

On retrouve bien le suffixe de l'adresse de print qui était : 0x7f670**4f1f97a**.

Maintenant que nous avons établi le contexte, passons à l'exploitation. Pour utiliser notre code et remplacer l'appel à print, nous devons d'abord mapper notre fichier injection.o en mémoire afin de pouvoir pointer dessus. Avec GDB :
> call open("injection.o", 2)

Puis :
> call mmap(0, 1088, 1\|2\|4, 1, 3, 0)

A présent, notre fichier injection.o est contenu dans la mémoire du processus :
```
563d5aba1000-563d5aba2000 r-xp 00000000 08:01 6816141                    /root/Documents/security/lab/code_injection/app
563d5ada1000-563d5ada2000 r--p 00000000 08:01 6816141                    /root/Documents/security/lab/code_injection/app
563d5ada2000-563d5ada3000 rw-p 00001000 08:01 6816141                    /root/Documents/security/lab/code_injection/app
563d5b4be000-563d5b4df000 rw-p 00000000 00:00 0                          [heap]
7f6704235000-7f67043e6000 r-xp 00000000 08:01 2230879                    /lib/x86_64-linux-gnu/libc-2.27.so
7f67043e6000-7f67045e5000 ---p 001b1000 08:01 2230879                    /lib/x86_64-linux-gnu/libc-2.27.so
7f67045e5000-7f67045e9000 r--p 001b0000 08:01 2230879                    /lib/x86_64-linux-gnu/libc-2.27.so
7f67045e9000-7f67045eb000 rw-p 001b4000 08:01 2230879                    /lib/x86_64-linux-gnu/libc-2.27.so
7f67045eb000-7f67045ef000 rw-p 00000000 00:00 0
7f67045ef000-7f6704606000 r-xp 00000000 08:01 2230279                    /lib/x86_64-linux-gnu/libgcc_s.so.1
7f6704606000-7f6704805000 ---p 00017000 08:01 2230279                    /lib/x86_64-linux-gnu/libgcc_s.so.1
7f6704805000-7f6704806000 r--p 00016000 08:01 2230279                    /lib/x86_64-linux-gnu/libgcc_s.so.1
7f6704806000-7f6704807000 rw-p 00017000 08:01 2230279                    /lib/x86_64-linux-gnu/libgcc_s.so.1
7f6704807000-7f6704999000 r-xp 00000000 08:01 2230887                    /lib/x86_64-linux-gnu/libm-2.27.so
7f6704999000-7f6704b98000 ---p 00192000 08:01 2230887                    /lib/x86_64-linux-gnu/libm-2.27.so
7f6704b98000-7f6704b99000 r--p 00191000 08:01 2230887                    /lib/x86_64-linux-gnu/libm-2.27.so
7f6704b99000-7f6704b9a000 rw-p 00192000 08:01 2230887                    /lib/x86_64-linux-gnu/libm-2.27.so
7f6704b9a000-7f6704d10000 r-xp 00000000 08:01 13238937                   /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.25
7f6704d10000-7f6704f10000 ---p 00176000 08:01 13238937                   /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.25
7f6704f10000-7f6704f1a000 r--p 00176000 08:01 13238937                   /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.25
7f6704f1a000-7f6704f1c000 rw-p 00180000 08:01 13238937                   /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.25
7f6704f1c000-7f6704f1f000 rw-p 00000000 00:00 0
7f6704f1f000-7f6704f20000 r-xp 00000000 08:01 6816036                    /root/Documents/security/lab/code_injection/libdynlib.so
7f6704f20000-7f670511f000 ---p 00001000 08:01 6816036                    /root/Documents/security/lab/code_injection/libdynlib.so
7f670511f000-7f6705120000 r--p 00000000 08:01 6816036                    /root/Documents/security/lab/code_injection/libdynlib.so
7f6705120000-7f6705121000 rw-p 00001000 08:01 6816036                    /root/Documents/security/lab/code_injection/libdynlib.so
7f6705121000-7f6705146000 r-xp 00000000 08:01 2230166                    /lib/x86_64-linux-gnu/ld-2.27.so
7f670530b000-7f6705310000 rw-p 00000000 00:00 0
7f6705340000-7f6705341000 rwxs 00000000 08:01 6816145                    /root/Documents/security/lab/code_injection/injection.o
7f6705341000-7f6705342000 rwxs 00000000 08:01 6816145                    /root/Documents/security/lab/code_injection/injection.o
7f6705342000-7f6705343000 rwxs 00000000 08:01 6816145                    /root/Documents/security/lab/code_injection/injection.o
7f6705343000-7f6705345000 rw-p 00000000 00:00 0
7f6705345000-7f6705346000 r--p 00024000 08:01 2230166                    /lib/x86_64-linux-gnu/ld-2.27.so
7f6705346000-7f6705347000 rw-p 00025000 08:01 2230166                    /lib/x86_64-linux-gnu/ld-2.27.so
7f6705347000-7f6705348000 rw-p 00000000 00:00 0
7ffef74c2000-7ffef74e3000 rw-p 00000000 00:00 0                          [stack]
7ffef75b2000-7ffef75b4000 r--p 00000000 00:00 0                          [vvar]
7ffef75b4000-7ffef75b6000 r-xp 00000000 00:00 0                          [vdso]
ffffffffff600000-ffffffffff601000 r-xp 00000000 00:00 0                  [vsyscall]
```

On obtient l'adresse de la fonction print que l'on souhaite appeler en trouvant l'adresse de la section .text d'injection.o et l'offset de print dans cette section grâce aux deux commandes suivantes :
- readelf -s injection.o
- readelf -S injection.o

Dans mon cas, cette adresse est : **0x7f6705340000**.
Et l'offset : **0x40**.

Remplaçons l'adresse de print avec GDB :
> set *(0x563d5aba1000+0x000000201038) = 0x05340040
> set *(0x563d5aba1000+0x000000201038-0x4) = 0x7f67

Où :
- 0x563d5aba1000+0x000000201038 est l'adresse de la relocation de la fonction print
- 0x7f6705340000+0x40 est l'adresse de la fonction injection contenue dans injection.o que l'on a mappé

Je dois la remplacer en 2 fois car je me trouve sur un système x86_64, les adresses ont une taille de 64 bits.

Bien que l'on ai fait le plus gros, le travail n'est pas terminé, nous devons à présent résoudre les relocations de la fonction injection. Pour être invisible, il faut faire pointer l'appel de print dans la fonction injection vers la fonction print initiale afin que le comportement ne soit pas altéré.
Avec GDB :
> set *(0x7f6705340000+0x40+0x5)=0x7f6704f1f97a-(0x7f6705340000+0x40+0x5)-0x4

Où :
- 0x7f6705340000 : adresse d'injection.o en mémoire
- 0X40 : offset de la section .text
- 0x5 : offset de la fonction print dans la section .text
- 0x7f6704f1f97a : adresse de la fonction print
- 0x4 : l'**addend** de la fonction print que l'on trouve dans la table de relocations d'injection.o

On soustrait 0x7f6704f1f97a car c'est une adresse et la relocation requiert un offset.

On effectue la même opération pour la fonction **system** :
> set *(0x7f6705340000+0x40+0x11)=0x7f6704277510-(0x7f6705340000+0x40+0x11)-0x4

Puis on le fait une dernière fois pour la section .rodata :
> set *(0x7f6705340000+0x40+0xc)=0x58-0x40-0xc-0x4

Où :
- 0x7f6705340000 : adresse d'injection.o en mémoire
- 0x40 : offset de la section .text
- 0xc : offset de l'instruction dans la section .text où la relocation est nécessaire
- 0x58 : offset de la section .rodata dans injection.o
- 0x4 : addend de la relocation de .rodata

Nous n'avons plus qu'à continuer l'exécution du programme avec l'instruction **continue** de GDB...

La date s'affiche ! C'est bien le comportement attendu de notre executable.

## Appendice

Pour comprendre un peu mieux les relocations :
Relocation section '.rela.text' at offset 0x230 contains 3 entries:
  Offset          Info           Type           Sym. Value    Sym. Name + Addend
000000000005  000b00000004 R_X86_64_PLT32    0000000000000000 print - 4
00000000000c  000500000002 R_X86_64_PC32     0000000000000000 .rodata - 4
000000000011  000c00000004 R_X86_64_PLT32    0000000000000000 system - 4

Relocation section '.rela.eh_frame' at offset 0x278 contains 1 entry:
  Offset          Info           Type           Sym. Value    Sym. Name + Addend
000000000020  000200000002 R_X86_64_PC32     0000000000000000 .text + 0

Voilà la table de relocations d'injection.o.
On remarque que .rodata est une relocation de type **R_X86_64_PC32**.
Cela signifie qu'elle est relative au program counter (PC), ici le registre **rip**.

En revanche, les fonctions print et system ont des relocations de type **R_X86_64_PLT32**.

Cela indique que nous devons utiliser leur adresse relativement à la Procedure Linkage Table (PLT).

## Conclusion

Voilà une technique intéressante qui nous en apprend plus sur le format d'exécutable ELF et la manière dont les compilateurs et linkers fonctionnent.