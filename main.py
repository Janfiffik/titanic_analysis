import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

pd.set_option('display.max_columns', None)

titanic_data = pd.read_csv("data/titanic.csv")

titanic_data_type = type(titanic_data)
print(f"Type of file: \n{titanic_data_type}\n")

titanic_data_shape = titanic_data.shape
print(f"Shape of file titanic_data:\n{titanic_data_shape}\n")

titanic_data_info = titanic_data.info
print(f"Info about titanic_data:\n{titanic_data_info}\n")

titanic_first_5 = titanic_data.head(5)
print(f"First 5 rows from titanic_data:\n{titanic_first_5}\n")

titanic_last_5 = titanic_data.tail(5)
print(f"Last 5 rows from titanic_data:\n{titanic_last_5}\n")

titanic_data_describe = titanic_data.describe()
print(f"Descriptive statistic on numeric columns: \n{titanic_data_describe}\n")

titanic_data_categorical_columns = titanic_data.dtypes[titanic_data.dtypes == "object"].index
print(f"Categorical columns:\n{titanic_data_categorical_columns}\n")

titanic_data_categorical_describe = titanic_data[titanic_data_categorical_columns].describe()
print(f"Describe methode on categorical columns: \n{titanic_data_categorical_describe}\n")

titanic_data_columns_names = titanic_data.columns
print(f"Column names of titanic_data:\n{titanic_data_columns_names}")

# -------------------Altering titanic_data---------------------
print(f"\n-------------------Altering titanic_data---------------------\n")

titanic_data.index = titanic_data['Name']  # set names as indexes
del titanic_data['Name']                   # deleting column name because it is duplicate
titanic_data_new_index = titanic_data.index[0:5]
print(f"New indexes in titanic_data:\n{titanic_data_new_index}")

# deleting useless columns
del titanic_data['PassengerId']  # Remove column passenger id
del titanic_data['Ticket']       # remove Ticket column

titanic_data_unique_cabin = titanic_data['Cabin'].describe()
print(f"Number of unique cabins: \n{titanic_data_unique_cabin}")

titanic_data_survived = pd.Categorical(titanic_data['Survived'])
print(f"List of values from column Survived:\n{titanic_data_survived}\n")

# Changing values 0,1 in column Survived to categorical data type: "Died", "Survived"
titanic_data_survived = titanic_data_survived.rename_categories(["Died", "Survived"])
print(f"List of values from column Survived after changing 0, 1 to categorical:\n{titanic_data_survived}\n")

titanic_data_PcClass = pd.Categorical(titanic_data['Pclass'], ordered=True)
print(f"List of old values from column Pclass:\n{titanic_data_PcClass}")

# Changing default values to categorical class1, class2, class3 in column Pclass
titanic_data_PcClass = titanic_data_PcClass.rename_categories(["Class1", "Class2", "Class3"])
print(f"List of values from column Pclass after changing from default to categorical:\n{titanic_data_PcClass}\n")

# Checking for unique cabin names
titanic_data_unique_cabin = titanic_data["Cabin"].unique()
print(f"Unique name of cabins in titanic_data['Cabin']:\n{titanic_data_unique_cabin}\n")

# Changing data to strings
char_cabin = titanic_data['Cabin'].astype(str)                 # Convert data to string.
new_Cabin = np.array([cabin[0] for cabin in char_cabin])       # Take first letter
new_Cabin = pd.Categorical(new_Cabin)
print(f"New variable created from old on by extracting first letters:\n{new_Cabin}\n")

# Changing data in columns for new one:
titanic_data["Cabin"] = new_Cabin
titanic_data["Survived"] = titanic_data_survived
titanic_data["Pclass"] = titanic_data_PcClass
print(f"Titanic_data after changing old values for new one: \n{titanic_data.head(5)}\n")

# ---------column descriptions-----------------
# Name:      names of the passengers it's index same time.
# Survived:  it holds values Died, Survived for description if passenger survive or died on Titanic
# Pclass:    It is class of the passenger. Values are Class1 > Class2 > Class3
# Sex:       Gender of person
# Age:       Age
# SibSp:     Siblings, Spouses on ship
# Parch:     Parents, Children on ship
# Fare:      Price of the ticket
# Cabin:     Cabin specific symbol on which part of the ship was located
# Embarked:  In which port person enter to the ship (C = Cherbourg; Q = Queenstown; S = Southampton)

print("-------------Finding values Na, Outliers or other Strange Values---------------------")

titanic_data_columns_names = titanic_data.columns
for i in titanic_data_columns_names:
    missing = np.where(titanic_data[i].isnull() == True)
    print(f"Nan values in column: {i}\n{len(missing[0])}\n")

# Highest number of Nan values is in column Age. It's not good idea to delete all missing values.
# Its good idea to change all missing values to mean or median value. Finding mean with histogram
histogram = titanic_data.hist(column="Age",     # Column to plot
                              figsize=(9, 6),   # Plot size
                              bins=70)          # Number of histogram bins

# From the histogram we can see distribution of ages is higher between 20 and 30 years.
# So filling in missing values with a central number like mean or median wouldn't be bad idea

new_age_var = np.where(titanic_data["Age"].isnull(),  # Logical check return True or False
                       28,                            # If check is True Nan vale is set to 28
                       titanic_data["Age"])           # Value is not changed if check is False

titanic_data["Age"] = new_age_var
print(f"Filling Nan values in column Age: \n {titanic_data['Age'].describe()}\n")

# ---------Histogram with filled Nan values in column Age--------------
titanic_data.hist(column="Age",
                  figsize=(9, 6),
                  bins=70)

# ---------------Investigating "Fare" variable-------------------------

titanic_data_fare_describe = titanic_data.Fare.describe()
print(f"Description for column Fare in titanic_data: \n{titanic_data_fare_describe}\n")

titanic_data.Fare.plot(kind="box",
                       figsize=(6, 9)
                       )

# Who pays the most expansive ticket
boolean_index_fare = np.where(titanic_data["Fare"] == max(titanic_data["Fare"]))
print(f"Most expansive ticket:\n {titanic_data.iloc[boolean_index_fare]}\n")

# -----------------------new family variable----------------------------------------------------------
family_var = titanic_data["SibSp"] + titanic_data["Parch"]
titanic_data["Family"] = family_var

#  -------Who had most family members on board--------------------------------------------------------
boolean_index_family = np.where(titanic_data["Family"] == max(titanic_data["Family"]))
print(f"The person with most family members on board: \n {titanic_data.iloc[boolean_index_family]}\n")

# Gender vs survived/died
survived_sex = pd.crosstab(index=titanic_data["Survived"],
                           columns=titanic_data["Sex"]
                           )
survived_sex.index = ["died", "survived"]
survived_sex.plot(kind="bar", stacked=True)

# Classes vs survived/died
survived_pclass = pd.crosstab(index=titanic_data["Survived"],
                              columns=titanic_data["Pclass"],
                              margins=True)
survived_pclass.columns = ["class1", "class2", "class3", "rowtotal"]
survived_pclass.index = ["died", "survived", "coltotal"]
survived_pclass.plot(kind="bar", stacked=False)

plt.show()

