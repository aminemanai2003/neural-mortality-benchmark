# Benchmark de modèles de mortalité classiques et neuronaux

**Établissements associés.** ESPRIT School of Engineering, Tunisie ; Institut du Risque et de l’Assurance (IRA), Le Mans Université, France — M1 Actuariat à partir de l’année universitaire 2026–2027.

## Résumé exécutif

Ce projet compare 14 implémentations de prévision de mortalité sur huit populations de la Human Mortality Database (HMD) : cinq modèles statistiques structurés, deux baselines simples, six architectures neuronales et un modèle hybride proposé, LC-ResNet. Les données couvrent les âges 0 à 100 et, selon le pays, les années 1950 à 2023.

Le protocole utilise une validation rolling-origin aux horizons 1, 5, 10 et 20 ans. Un résultat à l’horizon `h` mesure l’erreur sur toute la trajectoire allant de l’année 1 à l’année `h`, et non uniquement sur l’année terminale. Les critères sont l’erreur sur `log(mx)`, l’espérance de vie à la naissance et la valeur d’une rente à 65 ans. Les deux critères actuariels sont tronqués à l’âge 100.

Les principaux résultats sont les suivants :

- la marche aléatoire avec dérive par âge obtient la meilleure précision sur `log(mx)` à tous les horizons ;
- LC-ResNet est le meilleur modèle structuré à un an, mais Lee-Miller le dépasse à 20 ans ;
- le modèle fonctionnel H-U obtient la plus faible erreur sur l’espérance de vie à 20 ans ;
- le FFNN obtient la plus faible erreur sur la valeur de rente à 20 ans ;
- les taux gelés sont les plus robustes dans le test COVID cumulé 2020–2022 et avec seulement 20 ans d’historique ;
- dans l’étude de cas française, l’écart de provision entre cinq modèles atteint 6,26 M€, dont 1,22 M€ entre les quatre modèles couvrant tous les âges.

Ces résultats sont propres au jeu de données, au protocole et aux fonctions de perte retenues. Ils ne constituent pas un classement universel des modèles.

## 1. Contexte actuariel

Le risque de longévité correspond au risque que les assurés vivent plus longtemps que prévu. Il affecte directement les provisions de rentes et le capital de solvabilité. Dans la formule standard de Solvabilité II, le choc de longévité est une baisse instantanée et permanente de 20 % des taux de mortalité.

La difficulté pratique n’est donc pas seulement de prévoir correctement les taux de décès. Il faut aussi vérifier que les erreurs résiduelles n’entraînent pas un biais matériel sur les quantités utilisées en actuariat : espérance de vie, valeur de rente et besoin en capital.

## 2. Données

Le benchmark utilise les taux centraux de mortalité `mx`, les expositions et les décès de la HMD pour la France, l’Angleterre et le Pays de Galles, les États-Unis, le Japon, l’Italie, l’Espagne, la Suède et les Pays-Bas. La population totale est utilisée. Les taux nuls ou manquants sont remplacés par un plancher de `10^-6` avant le passage au logarithme. Les années terminales absentes ne sont pas imputées.

La HMD autorise l’utilisation de ses données après inscription, mais leur redistribution n’est pas permise. Le dépôt contient donc le code de téléchargement et de traitement, mais pas les fichiers HMD bruts.

## 3. Modèles comparés

### 3.1 Modèles structurés et baselines

1. Lee-Carter par SVD, avec marche aléatoire avec dérive sur l’indice temporel.
2. Lee-Miller, avec ajustement de l’indice sur l’espérance de vie observée.
3. Lee-Carter Poisson, estimé à partir des décès et des expositions.
4. CBD, modèle à deux facteurs limité aux âges 60–100.
5. Modèle fonctionnel de style Hyndman-Ullah : lissage par splines, six composantes fonctionnelles et marches aléatoires avec dérive sur les scores. Il s’agit d’une version simplifiée de la procédure H-U complète.
6. Marche aléatoire avec dérive, séparément pour chaque âge.
7. Taux gelés au dernier niveau observé.

### 3.2 Modèles neuronaux

Les six modèles PyTorch utilisent un budget commun : 200 époques au maximum, early stopping de patience 20, Adam, taux d’apprentissage `10^-3`, pénalisation `10^-4` et seed 42.

- LSTM, GRU, Bi-LSTM et Transformer prédisent l’indice temporel de Lee-Carter à partir de 20 années d’historique.
- Le FFNN combine un embedding d’âge et une variable temporelle continue.
- Le CNN traite la surface âge–année comme une image et produit les profils futurs de manière récursive.

### 3.3 LC-ResNet

LC-ResNet ajuste d’abord un Lee-Carter Poisson, puis apprend les résidus avec un MLP 64–32 à partir de l’âge, de l’année, de `bx` et de `kt` normalisés. La correction neuronale est multipliée par `exp(-0,1 h)`. Son poids vaut environ 0,90 à un an et 0,14 à 20 ans.

Le squelette Lee-Carter reste transparent, mais la correction neuronale demeure opaque. Le modèle est donc partiellement interprétable, pas totalement interprétable.

## 4. Protocole d’évaluation

Pour chaque origine 1990, 1992, …, 2018, le modèle est réajusté sur les observations disponibles puis évalué aux horizons 1, 5, 10 et 20 ans lorsque la vérité est disponible. Les résultats agrégés sont des moyennes non pondérées des RMSE par couple pays–origine valide.

Les scénarios complémentaires sont :

- historique court : une origine par pays, 10 années laissées pour l’évaluation, fenêtres d’apprentissage de 20, 30 et 50 ans ;
- choc COVID : entraînement jusqu’en 2019, puis erreurs cumulées sur 2020, 2020–2021 et 2020–2022 ;
- groupes d’âges : réajustement séparé de chaque modèle sur 0–19, 20–64 et 65–100 ans.

Le benchmark enregistré contient 27 312 lignes pour 14 implémentations. Il mesure la précision ponctuelle ; il ne produit ni test de Diebold-Mariano publié, ni couverture d’intervalles prédictifs.

## 5. Résultats

### 5.1 RMSE sur log(mx)

| Modèle | h=1 | h=5 | h=10 | h=20 |
|---|---:|---:|---:|---:|
| Marche aléatoire | 0,100 | 0,124 | 0,146 | 0,203 |
| Taux gelés | 0,100 | 0,133 | 0,186 | 0,314 |
| LC-ResNet | 0,120 | 0,152 | 0,184 | 0,247 |
| Lee-Miller | 0,145 | 0,166 | 0,187 | 0,232 |
| Lee-Carter | 0,156 | 0,177 | 0,199 | 0,248 |
| LSTM sur kt | 0,131 | 0,152 | 0,180 | 0,263 |
| Transformer sur kt | 0,141 | 0,177 | 0,225 | 0,327 |
| CNN surface | 0,147 | 0,191 | 0,240 | 0,448 |

CBD obtient des valeurs plus faibles sur les seuls âges 60–100, mais celles-ci ne sont pas comparables aux résultats sur la grille complète.

### 5.2 Critères actuariels

| Objectif | Meilleur modèle à 5 ans | Meilleur modèle à 20 ans |
|---|---|---|
| Espérance de vie e0 | Marche aléatoire, 0,315 an | H-U, 0,694 an |
| Rente à 65 ans | Marche aléatoire, 0,159 | FFNN, 0,381 |

Les modèles récurrents peuvent améliorer `log(mx)` à court horizon tout en détériorant fortement les critères actuariels à 20 ans. Par exemple, l’erreur de l’espérance de vie du LSTM atteint 2,059 ans à 20 ans, contre 0,742 pour Lee-Miller.

### 5.3 Scénarios

- COVID : les taux gelés obtiennent les meilleurs résultats sur les trois trajectoires cumulées complètes ; plusieurs modèles neuronaux sur `kt` restent proches.
- Historique de 20 ans : les taux gelés sont premiers avec 0,134 ; LC-ResNet est le meilleur modèle structuré avec 0,159.
- Jeunes âges : LC-ResNet et Bi-LSTM sont à égalité après arrondi, avec 0,228.
- Âges 65–100 : taux gelés 0,054 ; H-U et marche aléatoire 0,068 ; CBD 0,102.

## 6. Guide de décision

Le choix doit être lié à l’objectif :

- `log(mx)` sur la grille complète : toujours inclure la marche aléatoire comme baseline forte ;
- historique très court : tester d’abord les taux gelés ; utiliser LC-ResNet si un modèle structuré est exigé ;
- espérance de vie à 20 ans : H-U arrive premier dans ce benchmark, Lee-Miller étant l’option la plus forte de la famille LC ;
- valeur de rente à 20 ans : le FFNN arrive premier, mais il doit être accompagné de sensibilités classiques ;
- transparence : préférer une famille LC ; LC-ResNet conserve un squelette explicable mais pas une correction explicable ;
- choc : conserver plusieurs modèles, car une seule période COVID ne suffit pas à sélectionner un modèle universellement robuste.

## 7. Étude de cas française

Les modèles sont ajustés sur la France jusqu’en 2023 et leur projection 2033 est utilisée pour un portefeuille fictif de 1 000 rentiers âgés de 65 ans, rente annuelle 12 000 €, taux d’actualisation 2 %, paiements jusqu’à 100 ans.

| Modèle | Provision de base | Provision choquée | SCR longévité |
|---|---:|---:|---:|
| Lee-Carter | 220,05 M€ | 231,57 M€ | 11,52 M€ |
| Lee-Carter Poisson | 219,87 M€ | 231,39 M€ | 11,52 M€ |
| LC-ResNet | 219,22 M€ | 230,80 M€ | 11,58 M€ |
| Marche aléatoire | 218,83 M€ | 230,54 M€ | 11,71 M€ |
| CBD | 213,79 M€ | 226,12 M€ | 12,33 M€ |

L’écart de provision atteint 6,26 M€ en incluant CBD et 1,22 M€ entre modèles couvrant tous les âges. Le SCR varie de 11,52 à 12,33 M€, soit environ 7 % d’écart relatif. Cette étude est illustrative et ne remplace pas une valorisation propre à un assureur.

## 8. Limites

- modèles mono-population et absence d’effet cohorte explicite ;
- une seule seed et pas de tuning spécifique à chaque architecture ;
- prévisions ponctuelles seulement, sans calibration d’intervalles ;
- critères actuariels tronqués à 100 ans ;
- nombres d’origines valides différents selon la date de fin de chaque pays ;
- scénario COVID limité à un seul épisode ;
- étude de cas sensible à l’inclusion de CBD, qui ne couvre que les âges élevés.

## 9. Reproductibilité

```bash
pip install -e ".[dev]"
python scripts/run_benchmark.py
python scripts/run_case_study.py
pytest -q
```

Le dépôt contient le manuscrit LaTeX, 27 312 lignes de résultats, le CSV de l’étude de cas, les tests et le dashboard Streamlit. OpenAI Codex a été utilisé pour l’édition linguistique, les vérifications de cohérence entre code et manuscrit et l’assurance qualité des documents ; l’auteur reste responsable de l’étude et de son contenu.

## 10. Références principales

- Lee, R. D. et Carter, L. R. (1992). Modeling and Forecasting U.S. Mortality. JASA.
- Brouhns, N., Denuit, M. et Vermunt, J. K. (2002). Poisson log-bilinear mortality projection. IME.
- Cairns, A. J. G., Blake, D. et Dowd, K. (2006). A two-factor model for stochastic mortality. Journal of Risk and Insurance.
- Hyndman, R. J. et Ullah, M. S. (2007). Robust forecasting of mortality and fertility rates. CSDA.
- Barigou, K. et al. (2023). Bayesian model averaging for mortality forecasting. IJF.
- Li, L., Li, H. et Panagiotelis, A. (2025). Boosting domain-specific models with shrinkage. IJF.
- De Mori, L. et al. (2025). Mortality forecasting via multi-task neural networks. ASTIN Bulletin.
