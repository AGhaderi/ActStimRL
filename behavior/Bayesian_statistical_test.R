# read all datadrame
behAll = read.csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/AllBehData/NoNanBehAll.csv')
library(lmerTest)


install.packages("brms")
library(brms)

##########
model_bayes_wonamount <- brm(wonAmount ~ group + session + block + (1 | sub_ID), data = behAll, family = gaussian())
summary(model_bayes_wonamount)

#############
model_bayes_choicecorrect <- brm(correctChoice ~ group + session + block + (1 | sub_ID), data = behAll, family = gaussian())
summary(model_bayes_choicecorrect)

######
#behAll_agent = read.csv('/mnt/scratch/projects/7TPD/amin/simulation/agent/left-right-task-design-true-param.csv')
#model_bayes_wonAmount_agent <- brm(wonAmount_agent ~ group + session + block + (1 | sub_ID), data = behAll_agent, family = gaussian())
#summary(model_bayes_wonAmount_agent)




########################## Behavioral data
library(brms)
# read all datadrame
behAll = read.csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/AllBehData/NoNanBehAll.csv')
# model
model_bayes_wonamount <- brm(wonAmount ~ group + session + block + (1 | sub_ID), data = behAll, family = gaussian())
summary(model_bayes_wonamount)


########### high reward option
library(lmerTest)
library(brms)

behAll = read.csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/AllBehData/NoNanBehAll_RelIrrelHighReward_Groupby.csv')

model_bayes_wonamount <- brm(relevantVrIrrelevantHighRewardOption ~ patient + medication  + block  + patient*block + (1 | sub_ID), data = behAll, family = gaussian())

summary(model_bayes_wonamount)



########################## Clinical evaluation with latent parameter
library(brms)
# read csv parameter and clinical evaluation
data = read.csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/AllBehData/Clinical_evaluation/clinical_eval_parameter.csv')
# extract PD and Act
data_PD_OFF_Act = data[data['patient']=='PD' & data['block']=='Act' & data['medication']=='OFF',]
data_PD_ON_Act = data[data['patient']=='PD' & data['block']=='Act' & data['medication']=='ON',]
# extract PD and Stim
data_PD_OFF_Stim = data[data['patient']=='PD' & data['block']=='Stim' & data['medication']=='OFF',]
data_PD_ON_Stim = data[data['patient']=='PD' & data['block']=='Stim' & data['medication']=='ON',]

# extract HC
data_HC_Act = data[data['patient']=='HC' & data['block']=='Act',]
data_HC_Stim = data[data['patient']=='HC' & data['block']=='Stim',]

# mixed model, since we have just one sample for each subject we just use linear model rather than mixed linear model
model_PD_OFF_Act_beh_relevant<- brm(relevantHighRewardOption ~ age+sex+disease_duration + NMSS +total_UPDRS, data = data_PD_OFF_Act, family = gaussian())
model_PD_OFF_Act_beh_irrelevant<- brm(irrelevantHighRewardOption ~ age+sex+disease_duration + NMSS + total_UPDRS, data = data_PD_OFF_Act, family = gaussian())

print(model_PD_OFF_Act_beh_relevant, digits = 6)
print(model_PD_OFF_Act_beh_irrelevant, digits = 6)

model_PD_OFF_Act_beh_relevant<- brm(relevantHighRewardOption ~ age+sex + MoCA + LARS+ BDI, data = data_PD_OFF_Act, family = gaussian())
model_PD_OFF_Act_beh_irrelevant<- brm(irrelevantHighRewardOption ~ age+sex + MoCA + LARS + BDI, data = data_PD_OFF_Act, family = gaussian())











bf1<- bf(relevantHighRewardOption ~ age+sex + MoCA + LARS+ BDI)
bf2<- bf(irrelevantHighRewardOption ~ age+sex+ MoCA + LARS+ BDI)

fit <- brm(
  bf1 + bf2 + set_rescor(TRUE),
  data = data_PD_OFF_Act
)




model_PD_OFF_Act_mean<- brm(weight_parameter_mean ~  time_symptomns+ MoCA +LARS + total_UPDRSOFF , data = data_PD_OFF_Act, family = gaussian())
model_PD_ON_Act_mean<- brm(weight_parameter_mean ~  time_symptomns + MoCA +LARS + total_UPDRSON , data = data_PD_ON_Act, family = gaussian())


model_PD_OFF_Act_map<- brm(weight_parameter_map ~  time_symptomns+ MoCA +LARS + total_UPDRSOFF , data = data_PD_OFF_Act, family = gaussian())
model_PD_ON_Act_map<- brm(weight_parameter_map ~  time_symptomns + MoCA +LARS + total_UPDRSON , data = data_PD_ON_Act, family = gaussian())

#print(model_PD_ON_Act_map, digits = 6)



 