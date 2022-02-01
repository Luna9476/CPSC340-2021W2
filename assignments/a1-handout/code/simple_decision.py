# You don't need to use this file, but you could write your predict()
# function here if you felt like it.
def predict(X):
    if (X[0] > -80.305106):
        return 0
    else:
        if(X[1] > 37.669007):
            return 0
        else:
            return 1
        

