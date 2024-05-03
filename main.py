import pandas as pd
import sklearn
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn import linear_model,metrics, model_selection

url = "https://raw.githubusercontent.com/jonnytsunami23/C964-Capstone/main/possum.csv"
names = ['Population', 'Sex', 'Age', 'Head Length', 'Skull Width', 'Total Length', 'Tail Length', 'Foot Length', 'Ear Length', 'Eye', 'Chest Girth', 'Belly Girth']
df = pd.read_csv(url,names = names)
print(df)

machine_model = linear_model.LogisticRegression(max_iter=3000)

y = df.values[:,0]
x =df.values[:,2:12]

x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=.3)

machine_model.fit(x_train,y_train)

y_pred = machine_model.predict(x_test)

print(metrics.accuracy_score(y_test ,y_pred))

df.hist()
scatter_matrix(df)
metrics.ConfusionMatrixDisplay.from_estimator(machine_model,x_test,y_test)
pyplot.show()




while True:
    age = input("Enter possum Age: ")

    try:
        age = int(age)
        break
    except ValueError:
        print("Invalid! Enter a Number!")


while True:
    headLength = input("Enter possum Head Length: ")

    try:
        headLength = float(headLength)
        break
    except ValueError:
        print("Invalid! Enter a Number!")


while True:
    skullWidth = input("Enter possum Skull Width: ")

    try:
        skullWidth = float(skullWidth)
        break
    except ValueError:
        print("Invalid! Enter a Number!")


while True:
    totalLength = input("Enter possum Total Length: ")

    try:
        totalLength = float(totalLength)
        break
    except ValueError:
        print("Invalid! Enter a Number!")



while True:
    tailLength = input("Enter possum Tail Length: ")

    try:
        tailLength = float(tailLength)
        break
    except ValueError:
        print("Invalid! Enter a Number!")



while True:
    footLength = input("Enter possum Foot Length: ")

    try:
        footLength = float(footLength)
        break
    except ValueError:
        print("Invalid! Enter a Number!")


while True:
    earLength = input("Enter possum Ear Length: ")

    try:
        earLength = float(earLength)
        break
    except ValueError:
        print("Invalid! Enter a Number!")



while True:
    eye = input("Enter possum Eye Canthus Length: ")

    try:
        eye = float(eye)
        break
    except ValueError:
        print("Invalid! Enter a Number!")


while True:
    chestGirth = input("Enter possum Chest Girth: ")

    try:
        chestGirth = float(chestGirth)
        break
    except ValueError:
        print("Invalid! Enter a Number!")


while True:
    bellyGirth = input("Enter possum Belly Girth: ")

    try:
        bellyGirth = float(bellyGirth)
        break
    except ValueError:
        print("Invalid! Enter a Number!")


result = machine_model.predict([[age, headLength, skullWidth, totalLength, tailLength, footLength, earLength, eye, chestGirth, bellyGirth]])
print("Possum is from " + result)