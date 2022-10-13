# Important: Only push commits in the test branch to avoid breaking master file
# Always git pull from master first, then switch to branch to commit
import simpy as sp
import random as rn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TOTAL_BUYERS = 0        # Counter for the total buyers generated
MAX_BUYERS = 20         # The maximum number of buyers that can exist at any time
ALL_BUYERS = []         # List of buyer objects
TOTAL_DEFAULTERS = 0    # Counter for number of defaulters


class Buyer:
    def __init__(self, num, income, credit, documents, defaulting, status, asset, loan, duration, rate, monthly):
        global TOTAL_BUYERS
        TOTAL_BUYERS += 1
        self.num = num
        self.income = income
        self.credit = credit
        self.documents = documents
        self.defaulting = defaulting
        self.status = status
        self.asset = asset
        self.loan = loan
        self.duration = duration
        self.rate = rate
        self.monthly = monthly


def income_generate():
    # randomly chooses an income value based on ONS income distribution data in the UK
    income_table = pd.read_table('IncomeDistribution.txt')
    income_band = np.array(income_table[income_table.columns[0]])
    count = np.array(income_table[income_table.columns[1]])
    income_data = []
    for i in range(len(income_band)):
        income_data.append([income_band[i]] * count[i])
    income_data = np.concatenate(income_data)
    income = rn.choice(income_data)
    return income


def credit_generate():
    creditworthiness = np.random.randint(0, 100)    # to be swapped for credit score machine learning
    defaulting = float(np.random.randint(0, 10)/1000)      # to be swapped for defaulting score machine learning
    return creditworthiness, defaulting


def documents_generate():
    documents = 1111    # to be swapped with documentation algorithm
    return documents


def asset_generation(value):
    asset = round(value*0.1, 1)     # to be swapped with asset generation procedure
    return asset


def credit_check(buyers_list):
    global TOTAL_BUYERS
    new_buyers = []
    for i in range(len(buyers_list)):
        temp_buyer = buyers_list[i]
        buyer_num = temp_buyer.num
        if temp_buyer.credit < 30:
            print(f"Credit score is too low. Buyer {buyer_num} removed")
            TOTAL_BUYERS -= 1
        else:
            new_buyers.append(ALL_BUYERS[buyer_num])
    if not new_buyers:
        print(f"No buyers eligible")
        TOTAL_BUYERS = 0
    return new_buyers


def setup_simulation(num_of_buyers):
    for i in range(num_of_buyers):
        buyer_income = income_generate()
        [buyer_credit, buyer_defaulting] = credit_generate()
        buyer_documents = documents_generate()
        buyer_asset = asset_generation(buyer_income)
        ALL_BUYERS.append(Buyer(i, buyer_income, buyer_credit, buyer_documents, buyer_defaulting, "Applying",
                                buyer_asset, 0, 0, 0, 0))


def loan_generate(asset_value):
    if asset_value < 2:
        loan = 15
        loan_duration = 24
        loan_rate = 0.09
    elif 2 < asset_value < 4.0:
        loan = 30
        loan_duration = 36
        loan_rate = 0.07
    else:
        loan = 60
        loan_duration = 48
        loan_rate = 0.05
    monthly_payments = round((loan_rate / 12) * (1 / (1 - (1 + loan_rate / 12) ** (-loan_duration))) * loan, 3)
    loan_total = monthly_payments * loan_duration
    return loan_total, loan_duration, loan_rate, monthly_payments


setup_simulation(MAX_BUYERS)
print('Simulation Begins')
ALL_BUYERS = credit_check(ALL_BUYERS)

for i in range(len(ALL_BUYERS)):
    current_buyer = ALL_BUYERS[i]
    current_buyer.loan, current_buyer.duration, current_buyer.rate, current_buyer.monthly = \
        loan_generate(current_buyer.asset)
    print(f"Buyer {current_buyer.num} - Loan total: £{round(current_buyer.loan, 2)}k, monthly payments: "
          f"£{round(current_buyer.monthly, 2)}k for {current_buyer.duration} months")
    month_counter = 0
    while current_buyer.loan > 0:
        month_counter += 1
        if rn.random() < current_buyer.defaulting:
            print(f"Buyer {current_buyer.num} failed to pay on month: {month_counter}. Total loan remaining: "
                  f"£{current_buyer.loan}k")
            TOTAL_DEFAULTERS += 1
            break
        current_buyer.loan = round(current_buyer.loan - current_buyer.monthly)
        current_buyer.duration -= 1
    if current_buyer.loan == 0:
        print(f"Buyer {current_buyer.num} successfully finished paying loan")
print(f"Simulation finished - Number of Rejected Buyers: {MAX_BUYERS - TOTAL_BUYERS}, "
      f"Number of Defaulters: {TOTAL_DEFAULTERS}")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
