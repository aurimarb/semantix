#using pandas, matplotlib.pyplot, sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import KNN_index
import statistics

pd.options.mode.chained_assignment = None

dataset = pd.read_csv('bank-full.csv',sep=';')
dataset_yes = dataset[dataset.y == 'yes']
dataset_loan = dataset[dataset.loan == 'yes']
dataset_housing = dataset[dataset.housing == 'yes']
dataset_default = dataset[dataset.default == 'no']
data_previous = dataset.iloc[:,15:]



#Emprego que mais pega empréstimo e de que tipo
loan_jobs = dataset_loan.job.value_counts()/dataset.job.value_counts()
housing_jobs = dataset_housing.job.value_counts()/dataset.job.value_counts()

jobs = loan_jobs + housing_jobs
index_jobs = jobs[jobs == max(jobs)].index[0]

max_loan = loan_jobs[index_jobs]
max_housing = housing_jobs[index_jobs]
print("Tipo de trabalho que mais pega empréstimo e qual tipo:")
if  max_loan > max_housing:
    print(index_jobs,"personal")
elif max_loan < max_housing:
    print(index_jobs,"housing")
else:
    print(index_jobs,"ambos")



#taxa de sucesso pela quantidade de contatos
print('\n\n')

campaign_yes = dataset_yes.campaign.value_counts().to_frame()
campaign = dataset.campaign.value_counts().to_frame()
rate_campaign = campaign_yes/campaign
rate_campaign = rate_campaign.fillna(0)
media_lig = statistics.median(dataset_yes.campaign)
max_lig = statistics.pstdev(dataset_yes.campaign) + media_lig
print("Média de ligações:",media_lig)
print("Máximo de ligações:",max_lig)


fig, ax1 = plt.subplots()
ax1.plot(rate_campaign.index,rate_campaign.values,'r')
ax1.set_xlabel('number of previous contact')

ax1.set_ylabel('rate of success', color='r')
ax1.tick_params('y', colors='r')

ax2 = ax1.twinx()
ax2.plot(campaign.index,campaign.values,'b')
ax2.set_ylabel('total number of previous contact', color='b')
ax2.tick_params('y', colors='b')

fig.tight_layout()
fig.savefig('rate_previous.png')

#resultado da campanha anterior
print('\n\n')
data_previous_know = data_previous[data_previous.poutcome != "unknown"]
print("Taxa conhecida da campanha anterior:",len(data_previous_know.poutcome)/len(data_previous.poutcome))
KNN_index.KNN_index(data_previous_know)


#Crédito seguro
print('\n\n')
labels = ["young","adult","old"]
dataset_default.age = pd.cut(dataset_default.age, [10,25,55,105], right=False, labels=labels)

labels = ["debit","poor","middle-poor","middle","middle-high","high"]
dataset_default.balance = pd.cut(dataset_default.balance, [min(dataset.balance),0,7035,20100,40200,80000,max(dataset.balance)], right=False, labels=labels)

dataset_default_yes = dataset_default[dataset_default.y == 'yes']

def_max_dp = 0
def_max_dp_ind = 0
for i in dataset_housing.columns[0:8]:
    default = dataset_default_yes[i].value_counts()
    if def_max_dp < statistics.pstdev(default):
        def_max_dp = statistics.pstdev(default)
        def_max_dp_ind = [default.name,default[default == max(default)].index[0]]
print("O que mais influência para ter crédito seguro:",def_max_dp_ind)


#característica mais marcante de quem tem empréstimo imobiliária
print('\n\n')

ida_med = statistics.median(dataset_housing.age)
ida_dp = statistics.pstdev(dataset_housing.age)

rate_emp = dataset_housing.job.value_counts()/len(dataset_housing.job)
emp_max = max(rate_emp)
emp_max_ind = rate_emp[rate_emp == emp_max].index[0]

rate_civil =  dataset_housing.marital.value_counts()/len(dataset_housing.marital)
civil_max = max(rate_civil)
civil_max_ind = rate_civil[rate_civil == civil_max].index[0]

rate_edu =  dataset_housing.education.value_counts()/len(dataset_housing.education)
edu_max = max(rate_edu)
edu_max_ind = rate_edu[rate_edu == edu_max].index[0]

ren_med = statistics.median(dataset_housing.balance)
ren_dp = statistics.pstdev(dataset_housing.balance)

rate_loan = dataset_housing.loan.value_counts()/len(dataset_housing.loan)
loan_max = max(rate_loan)
loan_max_ind = rate_loan[rate_loan == loan_max].index[0]

rate_def =  dataset_housing.default.value_counts()/len(dataset_housing.default)
def_max = max(rate_def)
def_max_ind = rate_def[rate_def == def_max].index[0]

print("Características gerais de quem tem empréstimo imobiliário:")
print("Idade média:", ida_med,"; Desvio padrão:",ida_dp)
print("Emprego com maior taxa:", emp_max_ind,"; Taxa:", emp_max)
print("Estado civil com maior taxa:", civil_max_ind, "; Taxa:", civil_max )
print("Nível educaional com maior taxa:", edu_max_ind, "; Taxa:", edu_max)
print("Renda média:", ren_med,"; Desvio padrão:", ren_dp)
print("Maior taxa de empréstimo pessoal:", loan_max_ind, "; Taxa:", loan_max )
print("Maior taxa de divida de cartão:", def_max_ind, "; Taxa:", def_max)
