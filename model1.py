#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 15:44:05 2022

@author: conradodriscoll

"""
import numpy as np
from scipy.stats import bernoulli, binom
import pandas as pd

def pmt(principle,rate,duration):
    
        r = rate / (12 * 100)
        monthly_payments = principle * (r * ((1 + r) ** duration)) / (((1 + r) ** duration) - 1)
        return monthly_payments
class Borrower_v1:
    
    def __init__(self, customer_number, asset_type, risk_level, duration):
        self.customer_number = customer_number
        self.asset_type = asset_type
        
        self.risk_level = risk_level
        self.duration = duration
        #Original Value of the car ; Amount paid dependent on car type 
        self.original_value = self.update_value_from_dicts(self.asset_type)
        self.loan_value = self.update_value_from_dicts(self.asset_type)
        
        #Age of the car (calculated from purchase date) - Important for depreciation
        self.age = 0    
        self.interest_rate = self.update_loan_and_income_from_dicts(self.risk_level)[0]
        self.default_rate = self.update_loan_and_income_from_dicts(self.risk_level)[1]
        self.annual_income = self.update_loan_and_income_from_dicts(self.risk_level)[2]
        self.loan_fee_due = self.update_fee_due(self.duration)
        self.monthly_payments = self.calc_monthly_payments_due()
        self.asset_present_value = 0
        self.monthly_depreciation = self.calc_monthly_depreciation()
        self.total_current_payments = 0
        self.total_expected_payments = (self.monthly_payments * self.duration) + self.loan_fee_due
        self.default_status = 0
    def update_value_from_dicts(self, asset_type):
        asset_dict = {
            "Used" : 15000,
            "New" : 30000,
            "Luxury" : 60000,
            }
        original_value = asset_dict.get(asset_type)
        return original_value
    
    def update_loan_and_income_from_dicts(self,risk_level):        
        
        #Dictionary Containing Tranches of Loan as key with values as 1x2 array of [interest_rate,default_probability] (both annual)
        risk_dict = {
            "Sub-Prime" : [9,1,25000],
            "Prime" : [7,0.1,50000],
            "Super-Prime" : [5,0.05,100000]    
            }
        interest_rate = risk_dict.get(risk_level)[0]
        default_rate = risk_dict.get(risk_level)[1]
        annual_income = risk_dict.get(risk_level)[2]
                                                  
        return interest_rate, default_rate, annual_income
        
    def update_fee_due(self, duration):
        
        loan_fee_dict = {
            24:100,
            36:250,
            48:500
            }
        loan_fee_due = loan_fee_dict.get(self.duration)
        return loan_fee_due
    
    def calc_monthly_payments_due(self):
        principal = self.loan_value
        interest_rate = self.interest_rate
        duration = self.duration
        monthly_payment = pmt(principal, interest_rate, duration)
        return monthly_payment
    
    def calc_monthly_depreciation(self):
        # Linear Depreciation
        asset_value_depreciation_dict = {
            "Used" : 3000 ,
            "New" : 15000,
            "Luxury" : 30000
            }
        five_year_value = asset_value_depreciation_dict.get(self.asset_type)
        dy = self.original_value - five_year_value
        dx = 60
        monthly_depreciation = dy/dx
        return monthly_depreciation
        
    def calc_present_value(self,time):
        self.asset_present_value = self.original_value - (time*self.monthly_depreciation)
    
    def pay_monthly_fee(self):
        self.total_current_payments += self.monthly_payments
        
    def pay_loan_fee(self):
        self.total_current_payments += self.loan_fee_due
        
class Lender_v1 :
    
    def __init__(self, credit_available):
        self.credit_available = credit_available
        self.fee_income = 0
        self.payment_income = 0
        self.assets_valued = 0 
        self.loan_book = None
        
    def lend_out(self, borrower):
        self.credit_available -= borrower.loan_value

        self.fee_income += borrower.loan_fee_due
        
    def take_monthly_payments(self, borrower):
        self.payment_income += borrower.monthly_payments
    
    def take_fee_payments(self, borrower):
        self.fee_income += borrower.loan_fee_due
            
    #Function to add loan to book at the start of the loan:
    def update_loan_book_no_default(self,borrower):
        data = {'Customer Number' : [borrower.customer_number],
                'Annual Income' : [borrower.annual_income],
                'Original Loan Value' : [borrower.loan_value],
                'Interest Rate' : [borrower.interest_rate],
                'Total Payments TD' : [borrower.total_current_payments],
                'Total Expected Payments' : [borrower.total_expected_payments],
                'Total Outstanding' : [borrower.total_expected_payments - borrower.total_current_payments],
                'Current Asset Value' : [borrower.asset_present_value],
                'Recoverable Asset Value' : [borrower.asset_present_value * 0.8],
                'Current Exposure' : [borrower.total_expected_payments - borrower.total_current_payments - (borrower.asset_present_value * 0.8)],
                'Recovered Value' : [0],
                'Default' : [borrower.default_status]
                }
        df = pd.DataFrame(data)
        print(df)
        return df
        
    def update_loan_book_default(self,borrower):
        data = {'Customer Number' : [borrower.customer_number],
                'Annual Income' : [borrower.annual_income],
                'Original Loan Value' : [borrower.loan_value],
                'Interest Rate' : [borrower.interest_rate],
                'Total Payments TD' : [borrower.total_current_payments],
                'Total Expected Payments' : [borrower.total_expected_payments],
                'Total Outstanding' : [borrower.total_expected_payments - borrower.total_current_payments],
                'Current Asset Value' : [borrower.asset_present_value],
                'Recoverable Asset Value' : [borrower.asset_present_value * 0.8],
                'Current Exposure More is Worse' : [(borrower.total_expected_payments - borrower.total_current_payments) - (borrower.asset_present_value * 0.8)],
                'Default' : [borrower.default_status],
                'Recovered Value' : [borrower.asset_present_value * 0.8],
                'PL' : [(borrower.total_current_payments - borrower.loan_value) + (borrower.asset_present_value * 0.8) ]
                }
        df = pd.DataFrame(data)
        print(df)
        return df
        
    def update_loan_book_no_default_end(self,borrower):
        data = {'Customer Number' : [borrower.customer_number],
                'Annual Income' : [borrower.annual_income],
                'Original Loan Value' : [borrower.loan_value],
                'Interest Rate' : [borrower.interest_rate],
                'Total Payments TD' : [borrower.total_current_payments],
                'Total Expected Payments' : [borrower.total_expected_payments],
                'Total Outstanding' : [borrower.total_expected_payments - borrower.total_current_payments],
                'Current Asset Value' : [borrower.asset_present_value],
                'Recoverable Asset Value' : [borrower.asset_present_value * 0.8],
                'Current Exposure' : [0],
                'Default' : [borrower.default_status],
                'Recovered Value' : [0],
                'PL' : [borrower.total_current_payments - borrower.loan_value]
                }
        df = pd.DataFrame(data)
        print(df)
        return df
    def update_loan_book_default_end(self,borrower):
        data = {'Customer Number' : [borrower.customer_number],
                'Annual Income' : [borrower.annual_income],
                'Original Loan Value' : [borrower.loan_value],
                'Interest Rate' : [borrower.interest_rate],
                'Total Payments TD' : [borrower.total_current_payments],
                'Total Expected Payments' : [borrower.total_expected_payments],
                'Total Outstanding' : [borrower.total_expected_payments - borrower.total_current_payments],
                'Current Asset Value' : [borrower.asset_present_value],
                'Recoverable Asset Value' : [borrower.asset_present_value * 0.8],
                'Current Exposure More is Worse' : [(borrower.total_expected_payments - borrower.total_current_payments) - (borrower.asset_present_value * 0.8)],
                'Default' : [borrower.default_status],
                'Recovered Value' : [borrower.asset_present_value * 0.8],
                'PL' : [(borrower.total_current_payments - borrower.loan_value) + (borrower.asset_present_value * 0.8) ]
                }
        df = pd.DataFrame(data)
        print(df)
        return df
        
        
        
        

    
Morty = Borrower_v1(1,"Luxury", "Sub-Prime", 48)

Bank_O_Rick = Lender_v1(1000000)


def run_market(lender, borrower):
    duration_months = borrower.duration
    counter = 0
    counter_limit = duration_months
    #Bernoulli probability : outcome of 1 = default
    bernoulli_probability = borrower.default_rate / 1200
    while counter <= counter_limit:
        if counter == 0:
            lender.lend_out(borrower)
            lender.take_fee_payments(borrower)
            borrower.pay_loan_fee()
            lender.loan_book = lender.update_loan_book_no_default(borrower)
            counter += 1
        elif counter == counter_limit:
            default_outcome = bernoulli.rvs(bernoulli_probability)
            if default_outcome == 0:
                lender.take_monthly_payments(borrower)
                borrower.pay_monthly_fee()
                borrower.calc_present_value(counter)
                lender.loan_book = lender.update_loan_book_no_default_end(borrower)
                counter += 1
                return lender.loan_book
            elif default_outcome == 1:
                borrower.default_status = 1
                borrower.calc_present_value(counter)
                lender.loan_book = lender.update_loan_book_default_end(borrower)
                counter += 1
                return lender.loan_book
            
        else:
            
            default_outcome = bernoulli.rvs(bernoulli_probability)
            if default_outcome == 0:
                lender.take_monthly_payments(borrower)
                borrower.pay_monthly_fee()
                borrower.calc_present_value(counter)
                lender.loan_book = lender.update_loan_book_no_default(borrower)
                counter += 1
            elif default_outcome == 1:
                borrower.default_status = 1
                borrower.calc_present_value(counter)
                lender.loan_book = lender.update_loan_book_default(borrower)
                counter += 1
                return lender.loan_book
            
                
                
            
            
        
BalanceSheet_Sim = run_market(Bank_O_Rick, Morty)
        