# Kit CV / LinkedIn — Projet Mortality Benchmark

## Bullets CV (3 lignes prêtes à coller)

- Conçu et implémenté un benchmark comparant 14 modèles de prévision de mortalité (Lee-Carter, Poisson LC, CBD, LSTM, Transformer, CNN) sur 8 pays (HMD), avec validation rolling-origin et métriques actuarielles (e0, ä65)
- Développé un modèle hybride original (LC-ResNet) combinant Lee-Carter Poisson et correction neuronale des résidus, compétitif aux horizons courts/moyens tout en préservant l'interprétabilité
- Produit un cadre décisionnel pratique (« quel modèle quand ») et une étude de cas chiffrée en EUR (pricing rente viagère, choc de longévité Solvabilité II) démontrant le risque de modèle

---

## Post LinkedIn (FR)

**Quand faut-il faire confiance aux réseaux de neurones pour la mortalité ?**

Je viens de terminer un projet qui m'a passionné : un benchmark complet comparant 14 modèles de prévision de mortalité — des classiques (Lee-Carter, CBD, Hyndman-Ullah) aux réseaux de neurones (LSTM, Transformer, CNN).

Ce que j'ai appris :
→ Les réseaux de neurones ne dominent pas toujours. Avec moins de 25 ans d'historique, une simple marche aléatoire fait souvent mieux.
→ Le Lee-Carter classique reste très difficile à battre à horizon 20 ans.
→ Mon modèle hybride (Lee-Carter + correction neuronale des résidus) offre le meilleur compromis interprétabilité/performance aux horizons courts.

Le plus parlant : sur un portefeuille fictif de 1 000 rentes viagères, l'écart de provision entre modèles se chiffre en centaines de milliers d'euros. Le choix du modèle n'est pas académique — c'est un enjeu de capital.

Tout le code est en open source, implémenté from scratch en Python (PyTorch), avec un dashboard interactif Streamlit.

🔗 [lien GitHub]

#actuariat #mortalité #deeplearning #solvabilité #datascience

---

## Pitch 90 secondes (entretien)

« Mon projet porte sur la prévision de mortalité — un sujet central en actuariat vie puisqu'il détermine directement les provisions et le capital de solvabilité.

J'ai implémenté from scratch 14 modèles, des classiques comme Lee-Carter jusqu'aux réseaux de neurones — LSTM, Transformer, CNN — et je les ai comparés sur 8 pays de la Human Mortality Database avec un protocole rigoureux de validation rolling-origin.

Ce qui rend le projet différent d'une simple comparaison, c'est que j'ai construit un cadre décisionnel pratique : quel modèle utiliser quand, en fonction de l'historique disponible, de l'horizon de prévision, et du besoin d'interprétabilité réglementaire.

J'ai aussi développé un modèle hybride original qui combine le Lee-Carter Poisson — interprétable et stable — avec un petit réseau de neurones qui corrige les résidus structurés. Le réseau s'efface aux horizons longs grâce à un shrinkage, ce qui préserve la stabilité.

Enfin, j'ai traduit les résultats en euros : sur un portefeuille de rentes viagères, l'écart de provision entre modèles atteint plusieurs centaines de milliers d'euros, ce qui illustre concrètement le risque de modèle.

Le tout est en Python, avec tests, CI, et un dashboard Streamlit interactif. »

---

## Compétences démontrées

| Compétence | Preuve dans le projet |
|---|---|
| Modélisation stochastique de mortalité | 8 modèles classiques implémentés from scratch |
| Deep learning | 6 architectures PyTorch (LSTM, GRU, BiLSTM, Transformer, FFNN, CNN) |
| Validation de modèles | Rolling-origin, Diebold-Mariano, couverture des IC |
| Risque de longévité | Pricing rente ä65, choc Solvabilité II |
| Python avancé | Architecture propre, tests, CI GitHub Actions |
| Communication | Rapport, dashboard, visualisations |
