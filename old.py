# This is a sample Python script.
import random as rn
import numpy as np
import pandas as pd

ALL_BUYERS = []
ALL_REJECTS = []
ALL_LENDERS = []
NUM_BUYERS = 0  # Simulation parameters and counters
NUM_REJECTS = 0
ALL_CUSTOMERS = 10
NUM_LENDERS = 0
MAX_LENDERS = 3
num_accepted = 0
num_rejected = 0
all_buyers_dict = []
fields = ['num', 'income', 'house_owned', 'credit', 'married', 'gender', 'age', 'employed_time', 'fam_memb']

class Customer:
    def __init__(self, num, income, credit_band):

        self.num = num
        self.income = income
        self.house_owned = rn.randint(0, 1)  # 0 denotes no house, 1 means owns at least one house
        self.credit = credit_band  # Currently randomised
        self.married = rn.randint(0, 1)  # 0 denotes not married, 1 means married
        self.gender = rn.randint(0, 1)  # 0 denotes female, 1 means male
        self.age = rn.randint(6570, 29200)  # minimum age is 18, maximum is 80
        if self.age > 23725:
            self.employed_time = rn.randint(0,
                                            17155)  # minimum never worked, max worked his entire life since 18 up to age of retirement of 65
        elif self.age <= 23725:
            self.employed_time = rn.randint(0, self.age - 6570)
        self.fam_memb = rn.randint(0, 6)
        self.asset = rn.choice([15000, 30000, 60000])  # Currently randomised
        self.allowance = (self.income * 0.3) / 12  # Allowance based off 50/30/20 rule of income spending
        list = [True, False]
        weights = [95, 5]
        num_docs = 3
        num_docs_approved = 0
        global num_accepted, num_rejected
        for i in (range(num_docs)):
            possession_docs = rn.choices(list, weights, k=1)
            if possession_docs == [True]:
                num_docs_approved += 1
            else:
                break
        if num_docs_approved == 3:
            self.eligible = True
            num_accepted += 1
        else:
            self.eligible = False
            num_rejected += 1


#class Buyer:
    #def __init__(self,num, income, credit_band, asset, allowance):
        #self.num = num
        #self.income = income
        #self.credit = credit_band
        #self.asset = asset
        #self.allowance = allowance


class Lender:
    def __init__(self, num, credit_limit):
        self.num = num
        self.credit_limit = credit_limit
        self.interest_rates = [9, 8, 7, 6, 5]
        self.durations = [48, 36, 24, 12]

    # Function for lender to check credit band against their specific credit threshold
    def credit_check(self, buyer):
        if buyer.credit < self.credit_limit:
            print(f"Buyer {buyer.num} (Lender {self.num}): Rejected")
            buyer.eligible = False

    # Function for lender to generate offer to the buyer (currently maximises monthly payments/minimises duration)
    def monthly_offer(self, buyer):
        principle = buyer.asset
        allowance = buyer.allowance
        rate = self.interest_rates[buyer.credit]
        r = rate / (12 * 100)
        durations = self.durations
        for i in range(len(durations)):
            possible_payments = (principle * (r * ((1 + r) ** durations[i])) / (((1 + r) ** durations[i]) - 1))
            if possible_payments > allowance and i > 0:
                monthly_payments = (principle * (r * ((1 + r) ** durations[i - 1])) / (((1 + r) ** durations[i - 1]) - 1))
                duration = durations[i - 1]
                break
            elif possible_payments > allowance and i == 0:
                print(f"Buyer {buyer.num} (Lender {self.num}): Not enough income for asset loan")
                monthly_payments = 0
                duration = 0
                buyer.eligible = False
                break
        return rate, monthly_payments, duration

def income_generate():
    income_table = pd.read_table('IncomeDistribution.txt')
    income_band = np.array(income_table[income_table.columns[0]])
    count = np.array(income_table[income_table.columns[1]])
    income_data = []
    for i in range(len(income_band)):
        income_data.append([income_band[i]] * count[i])
    income_data = np.concatenate(income_data)
    income = rn.choice(income_data)
    return income
# Press the green button in the gutter to run the script.
def simulation_setup(num_customers, num_lenders):
    global ALL_BUYERS, ALL_LENDERS, NUM_BUYERS, NUM_LENDERS, NUM_REJECTS, fields, all_buyers_dict
    for i in range(num_customers):
        customer_income = income_generate() * 1000
        customer_band = rn.choice([0, 1, 2, 3, 4])  # This will be swapped for credit score process
        if(Customer(i, customer_income, customer_band).eligible == True):
            ALL_BUYERS.append(Customer(i, customer_income, customer_band))
            all_buyers_dict.append(Customer(i, customer_income, customer_band).__dict__)
            NUM_BUYERS += 1
        else:
            ALL_REJECTS.append(Customer(i, customer_income, customer_band))
            NUM_REJECTS += 1
    for j in range(num_lenders):
        lender_check = rn.choice([0, 1])
        ALL_LENDERS.append(Lender(j, lender_check))
        NUM_LENDERS += 1


if __name__ == '__main__':
    simulation_setup(ALL_CUSTOMERS, MAX_LENDERS)
    df = pd.DataFrame(all_buyers_dict, columns = fields)
    df.to_csv('buyers_dataframe.csv')
    for current_buyer in ALL_BUYERS:
        for current_lender in ALL_LENDERS:
            current_lender.credit_check(current_buyer)
            #if current_buyer.eligible:
            offer_rate, offer_monthly, offer_duration = current_lender.monthly_offer(current_buyer)
            if current_buyer.eligible:
                print(
                    f"Buyer {current_buyer.num} (Lender {current_lender.num}): Loan principle of £{current_buyer.asset} with an "
                    f"interest rate of {offer_rate}% for {offer_duration} months."
                    f" Total Monthly payments: £{offer_monthly:.2f}")
