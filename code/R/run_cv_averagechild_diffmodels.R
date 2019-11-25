library(tools)
#library(Hmisc)
library(glue)
library(broom)
#library(broom.mixed)
#library(langcog)
library(stringr)
library(tidyverse)
library(lme4)
library(modelr)
library(purrr)


load("../../data/aoa_predictors/model_data_imputed_avg_child.RData")
load("../../data/aoa_predictors/uni_model_data_avg_child.RData")

predictors <- c("avg_surprisal","frequency", "MLU", "final_frequency", "solo_frequency", "num_phons", "concreteness", "valence", "arousal", "babiness")

#full_set = ~ (age | item) + age * frequency + age * MLU + age * final_frequency + age * solo_frequency + age * num_phons + age * concreteness + age * valence + age * arousal + age * babiness + lexical_category * frequency + lexical_category * MLU + lexical_category * final_frequency + lexical_category * solo_frequency + lexical_category *  num_phons + lexical_category * concreteness + lexical_category * valence + lexical_category * arousal + lexical_category * babiness

#freq_only = ~ (age | item) + age * frequency + lexical_category * frequency

#freq_MLU = ~ (age | item) + age * frequency + age * MLU + lexical_category * frequency + lexical_category * MLU

full_surp = ~ (age | item) + age * avg_surprisal + age * frequency + age * MLU + age * final_frequency + age * solo_frequency + age * num_phons + age * concreteness + age * valence + age * arousal + age * babiness + lexical_category * avg_surprisal + lexical_category * frequency + lexical_category * MLU + lexical_category * final_frequency + lexical_category * solo_frequency + lexical_category * num_phons + lexical_category * concreteness + lexical_category * valence + lexical_category * arousal + lexical_category * babiness

freq_surp = ~ (age | item) + age * frequency + age * avg_surprisal + lexical_category * frequency + lexical_category * avg_surprisal

#freq_MLU_surp = ~ (age | item) + age * frequency + age * MLU + age * avg_surprisal + lexical_category * frequency + lexical_category * MLU + lexical_category * avg_surprisal

full_set = ~ (age | item) + age * frequency + age * MLU + age * final_frequency + age * solo_frequency + age * num_phons + age * concreteness + age * valence + age * arousal + age * babiness + lexical_category * frequency + lexical_category * MLU + lexical_category * final_frequency + lexical_category * solo_frequency + lexical_category *  num_phons + lexical_category * concreteness + lexical_category * valence + lexical_category * arousal + lexical_category * babiness

freq_only = ~ (age | item) + age * frequency + lexical_category * frequency

full_surp_only = ~ (age | item) + age * avg_surprisal + age * MLU + age * final_frequency + age * solo_frequency + age * num_phons + age * concreteness + age * valence + age * arousal + age * babiness + lexical_category * avg_surprisal + lexical_category * MLU + lexical_category * final_frequency + lexical_category * solo_frequency + lexical_category * num_phons + lexical_category * concreteness + lexical_category * valence + lexical_category * arousal + lexical_category * babiness

surp_only = ~ (age | item) + age * avg_surprisal + lexical_category * avg_surprisal


formulae <- formulas(~prop, full_set, freq_only, full_surp, freq_surp, surp_only, full_surp_only)



fit_models <- function(data, formulae, contrasts = NULL) {
  models <- "no model"
  print("run model")
  try(models <- fit_with(data, glmer, formulae, family = "binomial",
                         weights = data$total, contrasts = contrasts))
  return(models)
}

error_analysis <- function(model, data){
  results = tibble(
    mse_ = mse(model, data),
    rmse_ = rmse(model, data),
    #rsquare_ = rsquare(model, data),
    mae_ = mae(model, data)
    #  probs =  c(0.05, 0.25, 0.5, 0.75, 0.95),
    #  qae_ = qae(model, data)
    #  mape_ = mape(model, data),
    #  rsae_ = rsae(model, data)
  )
  return(results)
}

run_crossv <- function(split_data){
  group = unique(split_data$group)
  print(paste("running models for", group))
  name = paste("../../data/aoa_predictors/",
               gsub(" ", "_", group, fixed = TRUE),
               "_cv_models_data10_nofreq.RData", sep="")
  
  kfold5_data <- crossv_kfold(split_data, k=10)
  models_kfold_try<- kfold5_data %>% 
    mutate(models = train %>% map( ~ fit_models(., formulae)))
  #Remove failed models
  models_kfold <- models_kfold_try %>% filter(models!="no model")
  sep_models_kfold <- models_kfold %>% 
    mutate(full_set = models_kfold$models %>% map(~ .$"full_set"),
           freq_only = models_kfold$models %>% map(~ .$"freq_only"),
           freq_MLU = models_kfold$models %>% map(~ .$"freq_MLU"),
           full_surp = models_kfold$models %>% map(~ .$"full_surp"),
           freq_surp = models_kfold$models %>% map(~ .$"freq_surp"),
           freq_MLU_surp = models_kfold$models %>% map(~ .$"freq_MLU_surp")
    ) %>% 
    select(train, test, .id, 
           full_set, 
           freq_only, 
           freq_MLU, 
           full_surp, 
           freq_surp, 
           freq_MLU_surp)
  
  save(sep_models_kfold, file = name)
  
  get_avg_errs <- function(name){
    errs_<- map2(sep_models_kfold[[name]], 
                 sep_models_kfold$test, error_analysis) %>% 
      reduce(rbind) 
    model_names = c("mse_", "rmse_", "mae_")
    avgs = model_names %>% map(~mean(errs_[[.]]))
    result <- data.frame(avgs)
    colnames(result) = model_names
    result <- result %>% mutate(model= name)
  }
  
  if(nrow(sep_models_kfold)>0){
    model_names = c("full_set", "freq_only", "full_surp", "freq_surp", "surp_only", "full_surp_only")
    #model_names = c("full_surp", "freq_surp", "freq_MLU_surp")
    errs_<- map(model_names, get_avg_errs) %>% reduce(rbind)
  }else{
    errs_ <- tibble(
      mse_ = NA,
      rmse_ = NA,
      mae_ = NA,
      model = NA
    )
  }
  
  results <- errs_ %>% 
    mutate(group = group,
           child_name = unique(split_data$child_name),
           measure = unique(split_data$measure),
           kfolds = nrow(sep_models_kfold)
    )
  
  return(results)
}

run_cv_by_childname <- function(child){
  group_data <- uni_model_data %>%
    filter(child_name==child) %>% 
    mutate(group = paste(child_name, measure),
           lexical_category = lexical_category %>% fct_relevel("other")) %>%
    select(child_name, measure, group, lexical_category, item = uni_lemma,
           prop,total, age, !!predictors) %>%
    mutate(item = as.factor(item)) %>% 
    group_by(group) %>%
    nest()
  
  cv_errs_data <- group_data %>% 
    unnest() %>%
    group_by(group) %>% 
    split( .$group) %>% 
    map(~ run_crossv(split_data = .)) %>% 
    reduce(rbind)
  
  name = paste("../../data/aoa_predictors/",
               gsub(" ", "_", child, fixed = TRUE),
               "_cv_errs_data10_nofreq.RData", sep="")
  
  
  save(cv_errs_data, file = name)
}

uni_model_data$child_name %>% unique() %>% map(run_cv_by_childname)
