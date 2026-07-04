# Quand faut-il faire confiance aux réseaux de neurones ? Benchmark complet des modèles Lee-Carter classiques et neuronaux pour la prévision de mortalité

**Amine Manai** — Juillet 2026

---

## Résumé

Ce projet propose un benchmark rigoureux comparant 14 modèles de prévision de mortalité — 8 classiques (Lee-Carter, Lee-Miller, Booth-Maindonald-Smith, Poisson Lee-Carter, CBD, Hyndman-Ullah, et deux baselines naïves) et 6 neuronaux (LSTM, GRU, Bi-LSTM, Transformer sur l'indice kt ; réseau feed-forward à embeddings ; CNN sur la surface de mortalité) — sur 8 pays de la Human Mortality Database (1950–2023).

L'originalité du travail réside dans trois contributions :

1. **Un cadre décisionnel pratique** : un arbre de décision « quel modèle utiliser quand » basé sur la longueur de l'historique, l'horizon de prévision, le segment d'âge visé et le besoin d'interprétabilité.
2. **Un modèle hybride original (LC-ResNet)** : un Lee-Carter Poisson comme squelette interprétable, augmenté d'un petit réseau de neurones qui corrige les résidus structurés (non-linéarités, effets cohorte), avec un shrinkage à l'horizon qui fait tendre la correction vers zéro aux horizons longs.
3. **Une étude de cas actuarielle chiffrée en euros** : pricing d'une rente viagère ä65 pour un portefeuille de 1 000 rentiers, quantification du risque de modèle en euros, et choc de longévité type Solvabilité II.

L'évaluation utilise la validation rolling-origin (origines 1990–2018, horizons 1–20 ans), des métriques actuarielles (erreur sur e0 et sur la valeur d'annuité ä65), des tests de Diebold-Mariano, et des scénarios ciblés (données courtes, choc COVID, groupes d'âges).

---

## 1. Introduction

### 1.1 Le risque de longévité

Le risque de longévité — le risque que les assurés vivent plus longtemps que prévu — est l'un des principaux risques auxquels font face les compagnies d'assurance vie, les fonds de pension et les réassureurs. Sous Solvabilité II, le capital requis pour le risque de longévité (SCR longevity) représente typiquement 5 à 15 % du best estimate des provisions pour rentes.

La qualité de la prévision de mortalité a un impact direct et quantifiable sur :
- Le provisionnement (best estimate)
- Le capital de solvabilité requis (SCR)
- La tarification des rentes viagères
- Les transferts de risque (ILS, swaps de longévité)

### 1.2 Objectif du projet

Les modèles Lee-Carter et leurs variantes dominent la pratique actuarielle depuis 30 ans. Parallèlement, les réseaux de neurones ont montré des résultats prometteurs dans la littérature récente. Mais **aucune étude ne fournit un guide pratique permettant à un actuaire de choisir le bon modèle pour son contexte spécifique**.

Ce projet comble ce manque en répondant à des questions concrètes :
- Quel modèle fonctionne le mieux avec un historique court (20 ans) ?
- Lequel est le plus robuste aux chocs de mortalité (COVID) ?
- Lequel est le plus performant à horizon 5, 10, 20 ans ?
- Lequel est le meilleur pour les âges élevés (65+) — le segment rentes/longévité ?
- Quand la complexité des réseaux de neurones est-elle justifiée ?

---

## 2. Revue de littérature

### 2.1 Modèles classiques

- **Lee-Carter (1992)** : le modèle fondateur. Décomposition SVD de la matrice de log-mortalité, avec un indice kt suivant une marche aléatoire avec dérive.
- **Lee-Miller (2001)** : ajustement de kt sur l'espérance de vie observée.
- **Booth-Maindonald-Smith (2002)** : sélection de la période de fit par test de linéarité de kt.
- **Brouhns et al. (2002)** : estimation par maximum de vraisemblance sous hypothèse de Poisson.
- **Cairns-Blake-Dowd (2006)** : modèle à deux facteurs pour les âges élevés, utilisé par les régulateurs.
- **Hyndman-Ullah (2007)** : analyse en données fonctionnelles avec lissage par splines et ACP multi-composantes.

### 2.2 Approches neuronales

- **Richman & Wüthrich (2019)** : réseau feed-forward avec embeddings pour âge, année, pays, sexe — la référence actuariat-ML.
- **Nigri et al. (2019)** : LSTM appliqué à l'indice kt de Lee-Carter.
- **Perla et al. (2021)** : revue systématique des méthodes de deep learning pour la mortalité (Cambridge Annals of Actuarial Science).

### 2.3 Positionnement

Notre contribution se distingue par (1) l'ampleur du benchmark (14 modèles, 8 pays, rolling-origin), (2) le cadre décisionnel pratique, et (3) le modèle hybride LC-ResNet qui combine interprétabilité et flexibilité.

---

## 3. Données

**Source** : Human Mortality Database (mortality.org), données non redistribuables.

**Pays** : France, Angleterre & Galles, États-Unis, Japon, Italie, Espagne, Suède, Pays-Bas.

**Période** : 1950–2023 (inclut le choc COVID 2020–2023).

**Âges** : 0–100+, par sexe.

**Prétraitement** : plancher des taux nuls à 10⁻⁶, pas de lissage des grands âges au-delà de 100.

---

## 4. Modèles

### 4.1 Modèles classiques (implémentés from scratch en Python)

*[Détail de chaque modèle avec les équations — ax + bx*kt, contraintes d'identifiabilité, estimation, prévision]*

### 4.2 Modèles neuronaux (PyTorch)

*[Architecture de chaque réseau, stratégie d'entraînement commune : early stopping, même budget d'hyperparamètres, deep ensembles pour l'incertitude]*

### 4.3 Modèle hybride LC-ResNet (contribution)

Le modèle combine :
- Un **squelette Lee-Carter Poisson** (ax, bx, kt) — interprétable, stable à long terme
- Un **réseau résiduel** qui apprend les résidus structurés du LC
- Un **shrinkage à l'horizon** : la correction est multipliée par exp(-λh), garantissant la convergence vers le LC pur aux horizons longs

---

## 5. Protocole d'évaluation

### 5.1 Validation rolling-origin
Origines de 1990 à 2018 (pas de 2), horizons de 1 à 20 ans, ré-entraînement complet à chaque origine.

### 5.2 Métriques
- **Statistiques** : RMSE et MAE sur log m(x,t)
- **Actuarielles** : RMSE sur e0 et sur la valeur d'annuité ä65

### 5.3 Tests statistiques
Tests de Diebold-Mariano par paires de modèles.

### 5.4 Scénarios
- Données courtes (20/30/50 ans d'historique)
- Choc de mortalité (entraînement avant 2020, évaluation 2020–2023)
- Groupes d'âges (0–19, 20–64, 65+)

---

## 6. Résultats

*[Tableaux et figures générés par le benchmark — à remplir après exécution complète]*

---

## 7. Cadre décisionnel

*[Arbre de décision et scorecard — à remplir après exécution complète]*

---

## 8. Étude de cas actuarielle

### 8.1 Pricing de rente viagère

Portefeuille fictif de 1 000 rentiers français, âge 65, rente annuelle de 12 000 EUR, taux d'actualisation 2 %.

*[Tableau comparatif des provisions par modèle]*

### 8.2 Choc de longévité Solvabilité II

Application d'un choc de -20 % sur les taux de mortalité (les assurés vivent plus longtemps). Quantification du SCR longévité et du risque de modèle.

*[Tableau SCR par modèle]*

---

## 9. Limites et perspectives

- Pas de modèle multi-population joint (Li-Lee, etc.)
- Pas de modélisation explicite des effets cohorte (Renshaw-Haberman)
- Données HMD limitées à 2023 — les effets post-COVID ne sont pas encore stabilisés
- Le modèle hybride LC-ResNet pourrait bénéficier d'une architecture multi-tâches

---

## 10. Conclusion

Ce benchmark montre que [conclusions à remplir après exécution complète]. Le cadre décisionnel proposé permet aux actuaires de choisir le modèle le plus adapté à leur contexte, en fonction de la longueur de l'historique, de l'horizon de prévision, du segment d'âge et du besoin d'interprétabilité.

Le modèle hybride LC-ResNet offre un compromis prometteur entre la stabilité du Lee-Carter et la flexibilité des réseaux de neurones, particulièrement aux horizons courts et moyens.

---

## Bibliographie

1. Lee, R.D. & Carter, L.R. (1992). Modeling and forecasting U.S. mortality. *JASA*, 87(419), 659–671.
2. Lee, R.D. & Miller, T. (2001). Evaluating the performance of the Lee-Carter method. *Demography*, 38(4), 537–549.
3. Booth, H., Maindonald, J. & Smith, L. (2002). Applying Lee-Carter under conditions of variable mortality decline. *Population Studies*, 56(3), 325–336.
4. Brouhns, N., Denuit, M. & Vermunt, J.K. (2002). A Poisson log-bilinear regression approach to the construction of projected lifetables. *Insurance: Mathematics and Economics*, 31(3), 373–393.
5. Cairns, A.J., Blake, D. & Dowd, K. (2006). A two-factor model for stochastic mortality. *North American Actuarial Journal*, 10(2), 1–22.
6. Hyndman, R.J. & Ullah, M.S. (2007). Robust forecasting of mortality and fertility rates. *Computational Statistics & Data Analysis*, 51(10), 4942–4956.
7. Richman, R. & Wüthrich, M.V. (2019). A neural network extension of the Lee-Carter model to multiple populations. *Annals of Actuarial Science*, 15(2), 346–366.
8. Perla, F., Richman, R., Scognamiglio, S. & Wüthrich, M.V. (2021). A brief review of deep learning methods in mortality forecasting. *Annals of Actuarial Science*, 18(1), 72–95.
