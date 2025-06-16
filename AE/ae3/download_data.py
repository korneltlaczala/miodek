from ucimlrepo import fetch_ucirepo 

auto_mpg = fetch_ucirepo(id=9) 

X_mpg = auto_mpg.data.features 
y_mpg = auto_mpg.data.targets 
print(auto_mpg.variables) 
X_train_mpg = X_mpg.to_numpy()
y_train_mpg = y_mpg.to_numpy().reshape(-1, 1)

print(X_train_mpg.shape, y_train_mpg.shape)
