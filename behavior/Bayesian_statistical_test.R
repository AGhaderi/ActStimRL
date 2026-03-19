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



########################## Clinical evaluation
library(brms)
# read csv parameter and clinical evaluation
param_CE = read.csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/AllBehData/Clinical_evaluation/parameter_clinical_evaluation.csv')
# extract PD
param_CE_PD = param_CE[param_CE['group']=='PD',]
# mixed model, since we have just one sample for each subject we just use linear model rather than mixed linear model
model_param_CE_PD<- brm(map_med_weighting ~ age + sex + disease_duration + time_symptomns +  NMSS + MoCA +BDI+LARS + med_UPDRS , data = param_CE_PD, family = gaussian())


summary(model_param_CE_PD)

# effect of group over weighting model
model_param_CE <- brm(map_mean_weighting ~ age + sex + group + MoCA +BDI+LARS, data = param_CE, family = gaussian())
summary(model_param_CE)



########### high reward option
library(lmerTest)
library(brms)

behAll = read.csv('/mnt/projects/7TPD/bids/derivatives/fMRI_DA/AllBehData/NoNanBehAll_RelIrrelHighReward_Groupby.csv')

model_bayes_wonamount <- brm(relevantVrIrrelevantHighRewardOption ~ patient + medication  + block  + patient*block + (1 | sub_ID), data = behAll, family = gaussian())

summary(model_bayes_wonamount)


