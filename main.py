import random as rn
from numpy.random import choice
import time
from colorama import Fore, Style
from LogisticRegression import *
from Results import *

rn.seed(10)
np.random.seed(10)

# Simulation parameters and counters
ALL_BUYERS = []
ALL_LENDERS = []
TOTAL_CUSTOMERS = 20000
NUM_BUYERS = 0
ALL_ASSETS = np.array([5000, 6000, 7000, 8000, 9000, 10000, 12500, 15000, 17500, 20000])
STANDARD_RATES = np.array([9, 9.5, 10, 10.5, 11, 12, 13, 14, 15, 16])
RATES_DICT = {
    "Large": STANDARD_RATES,
    "Medium": STANDARD_RATES + 2,
    "Small": STANDARD_RATES + 4
            }
DELAY_DICT = {
    "Large": [2.1, 0.2],
    "Medium": [1.9, 0.2],
    "Small": [1.8, 0.2]
            }
SIMULATION_TIME = 120
NUM_SIMS = 1
NUM_ITER = 1
MARKET_VALUE = 5000000


class Customer:
    def __init__(self, income):
        self.Client_Income = income
        self.House_Own = choice((0, 1), p=[0.398, 0.602])  # 0 denotes no house, 1 means owns at least one house
        self.Credit_Amount = find_nearest(ALL_ASSETS, income)
        self.Client_Marital_Status = choice(['M', 'W', 'S', 'D'], p=[0.5105, 0.0571, 0.3350, 0.0974])
        self.Client_Gender = choice(['Male', 'Female'], p=[0.50, 0.50])
        self.Age_Days = choice(choice([range(18, 20), range(20, 24), range(25, 29), range(30, 34),
                                       range(35, 44), range(45, 54), range(55, 64)],
                                      p=[0.0626, 0.1095, 0.1049, 0.1064, 0.2332, 0.2254, 0.1580])) * 365
        self.Client_Family_Members = choice([0, 1, 2, 3, 4, 5, 6],
                                            p=[0.2727, 0.3333, 0.1616, 0.1414, 0.0606, 0.0202, 0.0102])
        self.wait = np.random.normal(14, 6, None)
        if 6570 <= self.Age_Days <= 7300:
            self.Employed_Days = 292
        elif 7300 < self.Age_Days <= 8760:
            self.Employed_Days = 475
        elif 8760 < self.Age_Days <= 12410:
            self.Employed_Days = 1168
        elif 12410 < self.Age_Days <= 16060:
            self.Employed_Days = 1935
        elif 16060 < self.Age_Days <= 19710:
            self.Employed_Days = 2847
        elif 19710 < self.Age_Days <= 23360:
            self.Employed_Days = 3760
        possession_docs = rn.choices([True, False], [95, 5], k=3)
        if all(possession_docs):
            self.eligible = True
        else:
            self.eligible = False


# Buyer object and attributes
class Buyer:
    def __init__(self, num, income, credit_band, asset, defaulting, month):
        self.num = num
        self.income = income
        self.credit = credit_band  # Currently randomised
        self.asset = asset
        self.offered_asset = 0
        self.allowance = (self.income * 0.4) / 12
        self.duration = rn.choice([12, 24, 36, 48])
        self.status = "Applying"
        self.eligible = True
        self.preferred_duration = True
        self.preferred_asset = True
        self.offer = []
        self.defaulting = (1 - defaulting) / 12
        self.start_month = month
        self.time_delay = np.random.normal(2, 0.5)

    def pay_loan(self, counter, lender):
        if self.offered_asset != 0:
            self.asset = self.offered_asset
        monthly = self.offer[1]
        total = monthly * self.duration - ((counter - self.start_month) * monthly)
        if round(total, 2) <= 0:
            self.status = "Finished"
        else:
            if rn.random() < self.defaulting:
                print(f"Buyer {self.num} (Lender {lender.num}) defaulted. "
                      f"Total remaining: ??{round(total, 2)} ")
                collateral = car_depreciation(self.asset, counter - self.start_month)
                lender.loss += (total - collateral)
                lender.collateral_sum += collateral
                lender.collateral_timeline[counter] = collateral
                self.status = 'Defaulted'
            else:
                lender.monthly_sum += monthly


# Lender object and attributes
class Lender:
    def __init__(self, num, credit_limit, max_duration, market_share, size, collusion):
        self.num = num
        self.credit_limit = credit_limit
        self.max_duration = max_duration
        self.monthly_sum = 0     # (market_share * 0.5) * 1.09**2.5
        self.profit_timeline = []
        self.loss = 0
        self.collateral_sum = 0
        self.collateral_timeline = np.zeros(SIMULATION_TIME)
        self.max_loan = market_share
        self.loan_out_sum = 0
        self.current_lending = 0    # market_share * 0.5
        self.rates = RATES_DICT[size] + collusion*rn.random()
        self.delay = np.random.normal(DELAY_DICT[size][0], DELAY_DICT[size][1])

    # Function for lender to check credit band against their specific credit threshold
    def credit_check(self, buyer):
        if buyer.credit > self.credit_limit:
            print(f"Buyer {buyer.num} (Lender {self.num}): Rejected for credit")
            buyer.eligible = False
            buyer.status = "Rejected"
        else:
            buyer.status = 'Approved'

    def monthly_offer(self, buyer):
        principle = buyer.asset
        allowance = buyer.allowance
        rate = self.rates[buyer.credit]
        r = rate / (12 * 100)
        duration = buyer.duration
        monthly_payments = principle * (r * ((1 + r) ** duration)) / (((1 + r) ** duration) - 1)
        if monthly_payments > allowance:
            print(f"Buyer {buyer.num} (Lender {self.num}): Cannot afford loan")
            buyer.eligible = False
        return rate, monthly_payments, duration


# Randomly assigns an income value based on ONS income distribution data in the UK
def income_generate():
    data_average = 13750
    data_std_dev = 7566
    sigma_log_normal = np.sqrt(np.log(1 + (data_std_dev / data_average) ** 2))
    mean_log_normal = np.log(data_average) - sigma_log_normal ** 2 / 2
    income = np.random.lognormal(mean=mean_log_normal, sigma=sigma_log_normal, size=1)
    income = round(income[0])
    return income


def customer_setup(num_customers):
    accepted_customers = []
    fields = ['Client_Income', 'House_Own', 'Credit_Amount', 'Client_Marital_Status', 'Client_Gender', 'Age_Days',
              'Employed_Days', 'Client_Family_Members']
    for i in range(num_customers):
        customer_income = income_generate()
        customer = Customer(customer_income)
        if customer.eligible:
            accepted_customers.append(customer.__dict__)
    df = pd.DataFrame(accepted_customers, columns=fields)
    df.to_csv("customers_dataframe.csv")


def buyer_setup(index, t, buyers_dataframe):
    global ALL_BUYERS, NUM_BUYERS
    num_buyers = len(buyers_dataframe)
    for i in range(num_buyers):
        buyer_num = i + (index * num_buyers)
        buyer_income = buyers_dataframe.iloc[i, 1]
        buyer_band = buyers_dataframe.iloc[i, -1]
        buyer_asset = buyers_dataframe.iloc[i, 3]
        buyer_default = buyers_dataframe.iloc[i, -2]
        ALL_BUYERS.append(Buyer(buyer_num, buyer_income, buyer_band, buyer_asset, buyer_default, t))
        NUM_BUYERS += 1


def lender_setup(sim_num, market_share):
    global ALL_LENDERS
    ALL_LENDERS.append(Lender(0, 4, 48, market_share * 0.4, "Large", 0))
    ALL_LENDERS.append(Lender(1, 6, 48, market_share * 0.25, "Medium", 1))
    ALL_LENDERS.append(Lender(2, 6, 48, market_share * 0.15, "Medium", 0))
    ALL_LENDERS.append(Lender(3, 10, 48, market_share * 0.075, "Small", 1))
    ALL_LENDERS.append(Lender(4, 10, 48, market_share * 0.065, "Small", 1))
    ALL_LENDERS.append(Lender(5, 10, 48, market_share * 0.065, "Small", 0))
    ALL_LENDERS.append(Lender(6, 10, 48, market_share * 0.05, "Small", 1))


def ret_2nd_ele(tuple_1):
    return tuple_1[1]


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def car_depreciation(asset, month_default):
    if month_default < 13:
        r = 25 / 12
        collateral_val = (asset * (1 - (r / 100)) ** month_default)
    else:
        asset_at2 = (asset * (1 - (25 / 1200)) ** 11)
        month_default -= 12
        r = 15.6 / 12
        collateral_val = (asset_at2 * (1 - (r / 100)) ** month_default) * 0.8
    return collateral_val


def generate_offers(all_buyers, all_lenders):
    for current_buyer in all_buyers:
        for current_lender in all_lenders:
            current_lender.credit_check(current_buyer)
            if abs(current_lender.max_loan) > abs(current_lender.current_lending):
                if current_buyer.eligible:
                    offer_rate, offer_monthly, offer_duration = current_lender.monthly_offer(current_buyer)
                if current_buyer.eligible:
                    if current_buyer.offered_asset == 0:
                        asset = current_buyer.asset
                    else:
                        asset = current_buyer.offered_asset
                    print(f"Buyer {current_buyer.num} (Lender {current_lender.num}): "
                          f"Loan principle of ??{asset} with an "
                          f"interest rate of {offer_rate:.1f}% for {offer_duration} months."
                          f" Total Monthly payments: ??{offer_monthly:.2f}")
                    current_buyer.offer.append((current_lender.num, offer_monthly, current_lender.delay))
                    current_buyer.status = "Paying"
                current_buyer.eligible = True
            else:
                print(f"Max loans exceeded")
        temp_offers = []
        for offers in current_buyer.offer:
            if offers[2] < current_buyer.time_delay:
                temp_offers.append(offers)
            else:
                print(f"Rejected Offer: Lender {offers[0]}. Verification delay too large")
        current_buyer.offer = temp_offers
        if current_buyer.offer:
            accepted_offer = offer_accept(current_buyer.offer)
            print(f"Accepted offer: Lender {accepted_offer[0]}. Monthly payments of ??{round(accepted_offer[1], 2)}.")
            current_buyer.offer = accepted_offer
            lender = all_lenders[accepted_offer[0]]
            lender.loan_out_sum -= current_buyer.asset
        print(f"")


def offer_accept(offers):
    monthly = []
    for offer in offers:
        monthly.append(offer[1])
    minimum = min(monthly)
    offer_list = [i for i, v in enumerate(monthly) if v == minimum]
    x = np.random.choice(offer_list)
    accepted_offer = offers[x]
    return accepted_offer


def loan_payments(month, all_buyers, all_lenders):
    for payee in all_buyers:
        if payee.offer:
            lender = all_lenders[payee.offer[0]]
            payee.pay_loan(month, lender)
    for lender in all_lenders:
        lender.current_lending = lender.loan_out_sum + lender.monthly_sum
        print(f"Lender {lender.num}: Profit/Loss/Collateral" +
              Fore.GREEN + f" +??{round(lender.monthly_sum, 2):,}" + Style.RESET_ALL + f" / " +
              Fore.RED + f"-??{round(lender.loss, 2):,}" + Style.RESET_ALL +
              f" / ??{round(lender.collateral_sum, 2):,}")
        print(f"Total amount borrowed by customers: ??{lender.loan_out_sum:,}")
        print(f"Current lending: ??{round(lender.current_lending, 2):,}")
        print(f"")
        lender.profit_timeline.append(lender.monthly_sum)


def simulation(sim_time, num_customers):
    global ALL_BUYERS, ALL_LENDERS
    # revenue = np.zeros((NUM_SIMS, sim_time))
    # ratios = np.zeros((NUM_SIMS, sim_time))
    # final_revenues = np.zeros(NUM_SIMS)
    # average_all = np.zeros((NUM_ITER, NUM_SIMS))
    # final_ratios = np.zeros(NUM_SIMS)
    for j in range(NUM_ITER):
        final_loaned_out = np.zeros((NUM_SIMS, 7))
        final_monthly_sum = np.zeros((NUM_SIMS, 7))
        final_collateral_sum = np.zeros((NUM_SIMS, 7))
        final_profit = np.zeros((NUM_SIMS, 7))
        for i in range(NUM_SIMS):
            ALL_BUYERS, ALL_LENDERS = [], []
            lender_setup(i, MARKET_VALUE)
            index = 0
            customer_setup(num_customers)
            xd = predict_from_generated_customer("customers_dataframe.csv", "Train_Dataset.csv")
            xd = xd.loc[xd["Band"] != 10]
            xd.to_csv("buyers_dataframe.csv")
            buyers_dataframe = pd.read_csv("buyers_dataframe.csv")
            # num_buyers = int(len(buyers_dataframe) / sim_time)
            num_buyers = 200
            for t in range(sim_time):
                print(Fore.BLUE + f"Month {t} (Iteration {i})" + Style.RESET_ALL)
                if t % 1 == 0:
                    buyer_setup(index, t, buyers_dataframe[num_buyers * index:num_buyers + (num_buyers * index)])
                    current_buyers = ALL_BUYERS[num_buyers * index:num_buyers + (num_buyers * index)]
                    generate_offers(current_buyers, ALL_LENDERS)
                    index += 1
                    print(f"")
                loan_payments(t, ALL_BUYERS, ALL_LENDERS)
            for k in range(len(ALL_LENDERS)):
                final_loaned_out[i][k] = abs(ALL_LENDERS[k].loan_out_sum)
                final_monthly_sum[i][k] = ALL_LENDERS[k].monthly_sum
                final_collateral_sum[i][k] = ALL_LENDERS[k].collateral_sum
                final_profit[i][k] = ALL_LENDERS[k].loan_out_sum + ALL_LENDERS[k].monthly_sum + \
                                     ALL_LENDERS[k].collateral_sum
                # final_collateral[i][k] = ALL_LENDERS[k].collateral_sum
        plot_performance(final_loaned_out, final_monthly_sum, final_collateral_sum, final_profit)
        #     revenue[i] = ALL_LENDERS[0].profit_timeline
        #     ratios[i] = ALL_LENDERS[0].profit_timeline / ALL_LENDERS[0].current_loan
        #     final_revenues[i] = revenue[i, -1]
        #     final_ratios[i] = ratios[i, -1]
        # plot_finals(final_revenues, final_ratios, NUM_SIMS)


if __name__ == '__main__':
    start = time.perf_counter()
    simulation(SIMULATION_TIME, TOTAL_CUSTOMERS)
    end = time.perf_counter()
    print(f"Simulation finished in {round(end - start, 2)}s")
