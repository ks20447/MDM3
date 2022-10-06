# Important: Only push commits in the test branch to avoid breaking master file
# Always git pull from master first, then switch to branch to commit
import simpy as sp
import random as rn
import numpy as np

TOTAL_BUYERS = 0    # Counter for the total buyers generated
MAX_BUYERS = 10     # The maximum number of buyers that can exist at any time


def buyer(num, income, credit, documents, defaulting, status, asset):
    # Buyer function used to assign attributes
    # This will be passed into the finance_documents/lenders classes
    # Will require an env variable also
    global TOTAL_BUYERS
    TOTAL_BUYERS += 1
    print(f"Buyer {num}; income £{income}k; creditworthiness {credit}%; documents {documents}; defaulting {defaulting}%"
          f"; current status \"{status}\"; desired asset £{asset}k ")


def income_generate():
    income = round(max(1, np.random.normal(10, 4)), 1)  # to be swapped for actual data distribution
    return income


def credit_generate():
    creditworthiness = np.random.randint(0, 100)    # to be swapped for credit score machine learning
    defaulting = np.random.randint(0, 10)           # to be swapped for defaulting score machine learning
    return creditworthiness, defaulting


def documents_generate():
    documents = 1111    # to be swapped with documentation algorithm
    return documents


def asset_generation(value):
    print(value)
    asset = round(value*0.1, 1)     # to be swapped with asset generation procedure
    return asset


if __name__ == '__main__':
    print('Start successful')
    for i in range(MAX_BUYERS):
        buyer_income = income_generate()
        [buyer_credit, buyer_defaulting] = credit_generate()
        buyer_documents = documents_generate()
        buyer_asset = asset_generation(buyer_income)
        buyer(i, buyer_income, buyer_credit, buyer_documents, buyer_defaulting, "Applying", buyer_asset)

    print(TOTAL_BUYERS)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
