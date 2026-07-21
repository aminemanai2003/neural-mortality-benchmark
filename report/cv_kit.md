# Kit CV / LinkedIn — Mortality Benchmark

## Bullets CV

- Conçu un benchmark reproductible de 14 modèles de mortalité sur 8 populations HMD, avec validation rolling-origin, scénarios COVID/historique court et métriques actuarielles tronquées à 100 ans (`e0`, rente à 65 ans).
- Développé LC-ResNet, un hybride Lee-Carter Poisson + correction neuronale décroissant avec l’horizon ; meilleur modèle structuré à 1 an dans le benchmark.
- Traduit le risque de modèle en impact financier sur 1 000 rentes françaises : écart de provision de 6,26 M€ entre cinq modèles et écart relatif de 7 % sur le SCR longévité.

Version courte, si le CV manque d’espace :

> Benchmark Python/PyTorch de 14 modèles de mortalité sur 8 pays (HMD), validation rolling-origin et étude Solvabilité II ; développement de LC-ResNet et mise en évidence d’un écart de provision de 6,26 M€ entre modèles.

## Post LinkedIn — avant acceptation arXiv

**Quand les réseaux de neurones améliorent-ils vraiment une prévision de mortalité ?**

Je viens de finaliser un projet de recherche reproductible comparant 14 modèles — de Lee-Carter, CBD et Hyndman-Ullah aux LSTM, Transformer, FFNN et CNN — sur huit populations de la Human Mortality Database.

Quelques résultats qui m’ont surpris :

→ Une simple marche aléatoire avec dérive obtient la meilleure précision sur les taux de mortalité à tous les horizons testés.

→ Le meilleur modèle dépend de la décision : H-U arrive premier sur l’espérance de vie à 20 ans, alors que le FFNN arrive premier sur la valeur de rente à 20 ans.

→ Le modèle hybride que je propose, LC-ResNet, est le meilleur modèle structuré à un an, mais sa correction neuronale ne domine pas partout.

→ Sur un portefeuille fictif de 1 000 rentes françaises valorisé avec des taux 2033, l’écart de provision entre cinq modèles atteint 6,26 M€ ; parmi les modèles couvrant tous les âges, il reste de 1,22 M€.

Le code, les 27 312 résultats agrégés, les tests et le manuscrit sont disponibles ici :

🔗 [lien GitHub]

Le manuscrit est actuellement en préparation comme preprint. Je partagerai le lien arXiv après son dépôt.

#actuariat #mortalité #longevityrisk #machinelearning #datascience #solvencyII

## Post LinkedIn — après dépôt arXiv

Reprendre la version ci-dessus et remplacer les deux dernières phrases par :

> Le code et le manuscrit sont ouverts. Preprint arXiv : [lien arXiv]. Dépôt GitHub : [lien GitHub].

Ne pas écrire « publication scientifique » ou « article publié » : arXiv héberge un preprint et ne constitue pas une évaluation par les pairs.

## Pitch entretien — 90 secondes

« J’ai construit un benchmark reproductible de prévision de mortalité parce que la précision statistique seule ne suffit pas en actuariat. J’ai comparé 14 implémentations sur huit populations HMD, avec une validation rolling-origin et des critères directement liés aux provisions : espérance de vie et valeur de rente.

Le résultat principal est que le meilleur modèle dépend de l’objectif. La marche aléatoire domine sur les taux, H-U sur l’espérance de vie à long horizon, et un FFNN sur la valeur de rente. J’ai aussi proposé LC-ResNet, un Lee-Carter Poisson corrigé par un petit réseau dont l’effet décroît avec l’horizon. Il est le meilleur modèle structuré à un an, sans être présenté comme un gagnant universel.

Enfin, j’ai rendu le risque de modèle concret : sur 1 000 rentes françaises, l’écart de provision entre cinq modèles atteint 6,26 millions d’euros, et le SCR longévité varie d’environ 7 %. Le projet comprend le code Python/PyTorch, 43 tests, un dashboard et un manuscrit reproductible. »

## Compétences démontrées

| Compétence | Preuve |
|---|---|
| Mortalité stochastique | LC, Lee-Miller, Poisson LC, CBD, H-U-style |
| Deep learning | LSTM, GRU, Bi-LSTM, Transformer, FFNN, CNN |
| Validation | Rolling-origin, baselines, scénarios, pertes actuarielles |
| Risque de longévité | Provision de rente, choc permanent de -20 % sur `mx` |
| Reproductibilité | Scripts, résultats CSV, 43 tests, lint, documentation |
| Communication scientifique | Manuscrit, rapport, dashboard, limites explicites |

## Vérifications avant diffusion

- Remplacer les liens entre crochets.
- Ne revendiquer que l’affiliation et l’adresse e-mail actuellement valides.
- Employer « preprint » après dépôt arXiv, pas « publication évaluée par les pairs ».
- Pouvoir expliquer la différence entre l’écart de 6,26 M€ avec CBD et celui de 1,22 M€ entre modèles full-grid.
- Pouvoir expliquer que les critères actuariels et la rente sont tronqués à l’âge 100.

