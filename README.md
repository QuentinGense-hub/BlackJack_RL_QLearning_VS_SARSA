# BlackJack_RL_QLearning_VS_SARSA

Modèle d'apprentissage par renforcement utilisant des modèles tel que le QLearning et le SARSA sur un jeu de BlackJack en utilisant un environnement Gymnasium.
Les deux modèles n'étaient pas précis à plus de 50% malgré le fait que le SARSA jouait par nature plus prudemment que le QLearning. Ceci découle du fait que le 
BlackJack est un jeu fait pour faire gagner la maison et non le joueur. Des méthodes liant la victoire, la défaite et la mise pourrait permettre de développer des méthodes de gagner de l'argent par le BlackJack (optimisation de mise), mais le jeu reste par lui-même plus dans le sens de la maison que du joueur.

Regarder le modèle jouer : python watch_blackjack.py --mode policy --games 5

Plot : python plot_learning_curves.py --episodes 1000000
