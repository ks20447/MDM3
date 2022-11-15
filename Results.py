import numpy as np
from matplotlib import pyplot as plt


def plot_profit(lenders, sim_time):
    for lender in lenders:
        t = np.linspace(0, sim_time-1, sim_time)
        plt.plot(t, lender.profit_timeline, color=lender.colour, label=f"Bank: {lender.num}")
    plt.xlabel('Time in months')
    plt.ylabel('Revenue £')
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


# def plot_all_timelines(profit_lending, profit_absolute, sim_time):
#     fig, axs = plt.subplots(1, 2)
#     fig.suptitle('10 year lender timeline of increasing interest rates')
#     font_size = 14
#     t = np.linspace(0, sim_time-1, sim_time)
#     axs[0].scatter(t, profit_lending, maker='*')
#     axs[1].scatter(t, profit_absolute, marker='*')
#     axs[0].set_xlabel('Time (Months)', fontsize=font_size)
#     axs[0].set_ylabel('Revenue / Lending', fontsize=font_size)
#     axs[1].set_xlabel('Time (Months)', fontsize=font_size)
#     axs[1].set_ylabel('Total Revenue £', fontsize=font_size)
#     axs[0].grid()
#     axs[1].grid()
#     plt.show()


def plot_finals(final_revenue, final_ratios, num_sims):
    fig, axs = plt.subplots(1, 2)
    font_size = 14
    fig.suptitle('Final Outcomes for a Large Bank per simulation, with Increasing Base Interest Rates')
    interest_increase = np.linspace(0, 10, num_sims)
    axs[0].scatter(interest_increase, final_revenue, marker='*')
    axs[0].set_xlabel('Interest Rate Increase (+X%)', fontsize=font_size)
    axs[0].set_ylabel('Total Revenue £', fontsize=font_size)
    axs[0].grid()
    axs[1].scatter(interest_increase, final_ratios, marker='*')
    axs[1].set_xlabel('Interest Rate Increase (+X%)', fontsize=font_size)
    axs[1].set_ylabel('Revenue / Lending Ratio', fontsize=font_size)
    axs[1].grid()
    # plt.title("Average Final Profit for a large bank with increasing interest rates per simulation, "
    #           "averaged over 10 total iterations")
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

def plot_performance(lender_revenues, lender_loss, lender_collateral):
    average_revenue = sum(lender_revenues)/len(lender_revenues)
    average_loss = sum(lender_loss)/len(lender_loss)
    average_collateral = sum(lender_collateral)/len(lender_collateral)
    x = np.linspace(1, len(average_revenue), len(average_revenue))
    plt.bar(x, average_revenue)
    plt.bar(x, average_collateral, color='k')
    plt.bar(x, average_loss, color='r')
    plt.xlabel("Lenders")
    plt.ylabel("Amount £")
    plt.legend(["Profit", "Collateral", "Loss"])
    plt.title("Average Profit, Loss and Collected Collateral for 7 different Banks")
    plt.show()


