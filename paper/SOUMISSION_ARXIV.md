# Guide de soumission arXiv

## Fichiers prêts

| Fichier | Rôle |
|---|---|
| `main.pdf` | Le papier compilé (12 pages) — pour relecture et pour ton CV/LinkedIn |
| `arxiv-upload.tar.gz` | L'archive à uploader sur arXiv (`main.tex` + `main.bbl` + `references.bib`) |

## Étapes de soumission

### 1. Créer un compte arXiv
- Va sur https://arxiv.org/user/register
- Utilise ton **email universitaire** (`@univ-lemans.fr`) — c'est important : un email
  académique augmente fortement les chances d'être auto-endorsé.

### 2. L'endorsement (le seul vrai obstacle)
arXiv exige qu'un premier auteur dans une catégorie soit « endorsé » par un
auteur établi. Trois options :

1. **Email universitaire** : parfois suffisant pour l'auto-endorsement dans
   certaines catégories.
2. **Demander à un enseignant-chercheur du Mans** (ton prof d'actuariat ou de
   statistique qui a déjà publié sur arXiv) : la demande d'endorsement se fait
   en 2 clics via un lien qu'arXiv génère pour toi lors de la soumission.
   C'est la voie normale et personne ne trouvera ça bizarre — c'est aussi une
   excellente occasion de faire relire le papier par un chercheur.
3. Contacter un des auteurs cités (Richman, Scognamiglio…) — moins simple.

### 3. Catégories recommandées
- **Catégorie principale** : `stat.AP` (Statistics – Applications)
- **Catégories croisées** : `q-fin.RM` (Risk Management), `cs.LG` (Machine Learning)

### 4. Métadonnées à copier-coller

**Title:**
```
When Should Actuaries Trust Neural Networks? A Comprehensive Benchmark of
Classical and Neural Mortality Models with a Hybrid LC-ResNet and Practical
Decision Framework
```

**Abstract:** reprendre l'abstract du PDF (page 1).

**Comments (champ facultatif mais recommandé):**
```
12 pages, 6 tables. Code available at
https://github.com/aminemanai2003/neural-mortality-benchmark
```

### 5. Upload
- « Start New Submission » → licence **CC BY 4.0** (recommandé) ou la licence
  arXiv par défaut
- Uploader `arxiv-upload.tar.gz`
- arXiv compile automatiquement — vérifier l'aperçu PDF
- Soumettre. L'annonce publique se fait sous 1–2 jours ouvrés.

## Points de vigilance avant de cliquer « Submit »

- [ ] Relire le papier en entier une fois toi-même (tu dois pouvoir défendre
      chaque chiffre en entretien)
- [ ] Vérifier ton email de correspondance (actuellement `amine.manai@esprit.tn`
      dans le .tex — mets ton email du Mans si tu préfères)
- [ ] Optionnel mais fortement conseillé : faire relire par un prof
      (qui peut aussi t'endorser — une pierre deux coups)
- [ ] La date « July 2026 » dans le .tex

## Après la publication

1. Ajouter le lien arXiv (`arxiv.org/abs/XXXX.XXXXX`) :
   - dans le README du repo GitHub (badge)
   - sur ton CV, section « Publications / Preprints »
   - sur ton profil LinkedIn (section Publications)
2. Le post LinkedIn du `report/cv_kit.md` peut annoncer le preprint.
3. Penser au prix mémoire / prix jeunes de l'Institut des Actuaires (dates sur
   institutdesactuaires.com).

## Recompiler le PDF localement

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
