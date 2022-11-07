import numpy as np
from matplotlib import pyplot as plt


def plot_profit(lenders, sim_time):
    for lender in lenders:
        t = np.linspace(0, sim_time-1, sim_time)
        plt.plot(t, lender.profit_timeline, color=lender.colour, label=f"Bank: {lender.num}")
    plt.xlabel('Time in months')
    plt.ylabel('Profit £')
    plt.title('Profit timeline of lenders')
    plt.legend()
    plt.show()


def plot_collateral(lenders, sim_time):
    for lender in lenders:
        t = np.linspace(0, sim_time-1, sim_time)
        total = np.array(np.cumsum(lender.collateral_timeline))
        plt.plot(t, total, color=lender.colour, label=f"Collateral collected by bank {lender.num}")
        plt.bar(t, lender.collateral_timeline, color=lender.colour, edgecolor='k',
                label=f"Shows the individual defaults by bank {lender.num}")
    plt.xlabel('Time in months')
    plt.ylabel('Value in $')
    plt.title('Collateral collected by lenders')
    plt.legend()
    plt.show()


def plot_all_timelines(profit_lending, profit_absolute, sim_time):
    fig, axs = plt.subplots(1, 2)
    fig.suptitle('10 year lender timeline of increasing interest rates')
    for i in range(len(profit_lending)):
        t = np.linspace(0, sim_time-1, sim_time)
        axs[0].plot(t, profit_lending[i], label=f"Prime Rate: {i + 5}%")
        axs[1].plot(t, profit_absolute[i], label=f"Prime Rate: {i + 5}%")
    axs[0].set(xlabel='Time (Months)', ylabel='Profit / Lending')
    axs[1].set(xlabel='Time (Months)', ylabel='Total Profit £')
    axs[0].legend(loc=2)
    axs[0].grid()
    axs[1].grid()
    plt.show()


# def plot_pie(colour_list):
#     lender_portion = []
#     all_lenders_list = ['Bank 0', 'Bank 1']
#     for lender in ALL_LENDERS:
#         lender_portion.append(lender.current_loan)
#     plt.subplot(1, 3, 3)
#     plt.pie(lender_portion, labels=all_lenders_list, colors=colour_list)
#     plt.title('Shows the share of the market of the different banks')
#     plt.legend()
#     plt.show()
