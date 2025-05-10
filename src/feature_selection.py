import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed

def fitness_function(selected_features, X, y):
    X_selected = X[:, selected_features]
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    accuracy = cross_val_score(clf, X_selected, y, cv=5, scoring='accuracy', n_jobs=-1).mean()
    return accuracy

def update_wolves(wolves, best_wolf):
    for i in range(len(wolves)):
        wolves[i] = best_wolf.copy()
    return wolves

def wolf_optimizer(X, y, num_features=20, num_wolves=10, max_iter=10):
    wolves = np.random.randint(0, 2, size=(num_wolves, X.shape[1]))
    best_wolf = wolves[0]
    best_fitness = 0
    for iteration in range(max_iter):
        fitness_scores = Parallel(n_jobs=-1)(
            delayed(fitness_function)(wolves[i], X, y) for i in range(num_wolves)
        )
        best_index = np.argmax(fitness_scores)
        if fitness_scores[best_index] > best_fitness:
            best_fitness = fitness_scores[best_index]
            best_wolf = wolves[best_index]
        wolves = update_wolves(wolves, best_wolf)
        if iteration > 10 and fitness_scores[best_index] == best_fitness:
            print(f"Early stopping at iteration {iteration}")
            break
    best_features = np.where(best_wolf == 1)[0]
    return best_features[:num_features]