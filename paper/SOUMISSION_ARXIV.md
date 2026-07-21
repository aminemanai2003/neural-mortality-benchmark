# Préparation de la soumission arXiv

## Positionnement recommandé

- Catégorie principale : `stat.AP` (Statistics — Applications).
- Cross-list raisonnable : `q-fin.RM` (Risk Management), car l’étude traite du risque de longévité et du capital assurantiel.
- Ne pas demander `cs.LG` sauf si le manuscrit est renforcé par une contribution méthodologique machine-learning plus générale.

L’endorsement n’est ni une évaluation par les pairs ni une garantie d’acceptation. Les modérateurs vérifient notamment la pertinence pour la catégorie, le caractère autonome et référencable du manuscrit, ainsi que son intérêt académique.

## Obtenir l’endorsement

1. Créer ou compléter le compte arXiv avec une adresse institutionnelle réellement valide, si elle existe.
2. Commencer la soumission et sélectionner `stat.AP`. arXiv indique alors si un endorsement est requis et fournit le lien/code exact de la demande.
3. Utiliser la fonction arXiv « Which authors of this paper are endorsers? » afin d’identifier des auteurs pertinents dans les références.
4. Contacter individuellement un petit nombre de chercheurs proches du sujet : enseignant ayant relu le travail, auteur d’un article récent de prévision de mortalité, ou chercheur en statistique actuarielle.
5. Joindre le PDF, le lien GitHub et le lien/code d’endorsement. Demander d’abord une lecture rapide de l’adéquation à `stat.AP`, puis l’endorsement si la personne juge le manuscrit approprié.

Ne pas envoyer de message de masse et ne pas présenter l’endorsement comme une approbation scientifique.

## Modèle d’e-mail

**Objet :** Demande d’avis et, si approprié, endorsement arXiv `stat.AP`

> Bonjour Professeur/Docteur [Nom],
>
> Je prépare le dépôt arXiv d’un preprint intitulé « When Should Actuaries Trust Neural Networks? ». Le travail compare 14 implémentations de prévision de mortalité sur huit populations HMD avec validation rolling-origin et métriques actuarielles, et propose un hybride Lee-Carter–réseau résiduel.
>
> Votre travail sur [article/sujet précis] est directement lié à ce benchmark. Si vous avez la possibilité de regarder le résumé ou le manuscrit, votre avis sur son adéquation à `stat.AP` me serait très utile. Si vous le jugez approprié, accepteriez-vous également de m’endorser via ce lien arXiv : [lien/code généré par arXiv] ?
>
> PDF : [lien]
> Code et résultats : [lien GitHub]
>
> Je comprends que l’endorsement ne constitue pas une validation du contenu. Merci pour votre temps.
>
> Cordialement,
> Amine Manai
> [affiliation actuelle confirmée]
> [adresse e-mail valide]

## Métadonnées

**Titre**

```text
When Should Actuaries Trust Neural Networks? A Benchmark of Classical and
Neural Mortality Models with a Hybrid LC-ResNet and a Practical Decision Framework
```

**Commentaires**

Mettre à jour le nombre de pages et de tableaux après la compilation finale :

```text
[N] pages, [N] tables. Code and aggregated results available at
https://github.com/aminemanai2003/neural-mortality-benchmark
```

## Contrôles avant envoi aux endorsers

- [ ] Lire le PDF final en entier et pouvoir défendre chaque chiffre.
- [ ] Confirmer personnellement l’affiliation « ESPRIT School of Engineering » et l’adresse `amine.manai@esprit.tn`; les modifier si elles ne sont plus valides.
- [ ] Vérifier que le dépôt GitHub public correspond exactement au PDF.
- [ ] Exécuter `pytest -q` et `ruff check .`.
- [ ] Exécuter `python scripts/run_case_study.py` et comparer `results/case_study.csv` aux deux tableaux EUR.
- [ ] Vérifier l’aperçu compilé par arXiv, les références et tous les liens.
- [ ] Conserver la déclaration d’usage de l’IA générative dans le manuscrit.
- [ ] Choisir explicitement une licence ; CC BY 4.0 facilite la réutilisation, mais la décision appartient à l’auteur.

## Construire l’archive

Depuis `paper/`, compiler puis inclure au minimum les sources réellement nécessaires :

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
tar -czf arxiv-upload.tar.gz main.tex references.bib main.bbl
```

Le PDF final doit être généré par les sources présentes dans l’archive. Après le dépôt, parler de « preprint arXiv », pas d’article évalué par les pairs.

## Sources officielles

- Endorsement : https://info.arxiv.org/help/endorsement.html
- Modération : https://info.arxiv.org/help/moderation/index.html
- Cross-list : https://info.arxiv.org/help/cross.html

