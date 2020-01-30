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


load("../../../data/aoa_predictors/model_data_imputed_avg_child.RData")
load("../../../data/aoa_predictors/uni_model_data_avg_child.RData")

predictors <- c("avg_surprisal","frequency", "MLU", "final_frequency", "solo_frequency", "num_phons", "concreteness", "valence", "arousal", "babiness")

### LMs predicting prop
full_surp = ~ age * avg_surprisal + age * frequency + age * MLU + age * final_frequency + age * solo_frequency + age * num_phons + age * concreteness + age * valence + age * arousal + age * babiness + lexical_category * avg_surprisal + lexical_category * frequency + lexical_category * MLU + lexical_category * final_frequency + lexical_category * solo_frequency + lexical_category * num_phons + lexical_category * concreteness + lexical_category * valence + lexical_category * arousal + lexical_category * babiness

freq_surp = ~ age * frequency + age * avg_surprisal + lexical_category * frequency + lexical_category * avg_surprisal

#freq_MLU_surp = ~ (age | item) + age * frequency + age * MLU + age * avg_surprisal + lexical_category * frequency + lexical_category * MLU + lexical_category * avg_surprisal

full_set = ~ age * frequency + age * MLU + age * final_frequency + age * solo_frequency + age * num_phons + age * concreteness + age * valence + age * arousal + age * babiness + lexical_category * frequency + lexical_category * MLU + lexical_category * final_frequency + lexical_category * solo_frequency + lexical_category *  num_phons + lexical_category * concreteness + lexical_category * valence + lexical_category * arousal + lexical_category * babiness

freq_only = ~ age * frequency + lexical_category * frequency

full_surp_only = ~ age * avg_surprisal + age * MLU + age * final_frequency + age * solo_frequency + age * num_phons + age * concreteness + age * valence + age * arousal + age * babiness + lexical_category * avg_surprisal + lexical_category * MLU + lexical_category * final_frequency + lexical_category * solo_frequency + lexical_category * num_phons + lexical_category * concreteness + lexical_category * valence + lexical_category * arousal + lexical_category * babiness

surp_only = ~ age * avg_surprisal + lexical_category * avg_surprisal

null_model = ~ 1

formulae <- formulas(~prop, full_set, freq_only, full_surp, freq_surp, surp_only, full_surp_only)

#formulae <- formulas(~prop, full_set)


run_crossv <- function(split_data){
  group = unique(split_data$group)
  print(paste("running models for", group))
  name = paste("../../../data/aoa_predictors/",
               gsub(" ", "_", group, fixed = TRUE),
               "_cv_models_data10_lms.RData", sep="")
  
  kfold10_data <- crossv_kfold((ungroup(split_data)), k=10)
  
  fit_models <- function(fold, formulae, contrasts = NULL) {
    models <- "no model"
    print("run model")
    train_idx <- kfold10_data[fold,1][[1]][[1]]$idx
    test_idx <- kfold10_data[fold,2][[1]][[1]]$idx
    data <- split_data[train_idx,]
    try(models <- fit_with(data, lm, formulae))
    
    result = tibble(
      train = list(train_idx),
      test = list(test_idx),
      models = models)
    
    return(result)
    
  }
  
  
  models_kfold_try<- c(1:10) %>% map( ~ fit_models(., formulae)) %>% reduce(rbind)
  #Remove failed models
  
  
  model_name <- tibble(
    model_name = as_factor(rep(c("full_set", "freq_only", "full_surp", "freq_surp", "surp_only", "full_surp_only"), 10)))
  
  models_kfold <- cbind(models_kfold_try,  model_name)
  
  mse_calc <- function(n){
    test_data = split_data[models_kfold$test[[n]],]
    Y = test_data$prop
    Y_pred = predict(models_kfold$models[[n]],  test_data, type="response")
    mse_ = mean((Y - Y_pred)^2)
    return(as.numeric(mse_))
  }
  
  #models_kfold_try <- sep_models_kfold  %>% gather( key= model_name, value = "models", full_set, freq_only, full_surp, freq_surp, surp_only, full_surp_only)
  
  sep_models_kfold <- models_kfold %>% 
    mutate(mse_ = c(1:60) %>% map(~ mse_calc(.)))
  
  save(sep_models_kfold, file = name)
  
  results <- sep_models_kfold %>% 
    transform(mse_ = as.numeric(mse_)) %>% 
    group_by(model_name) %>%
    summarise(mean = mean(mse_), sd = sd(mse_)) %>% 
    mutate(group=group)
  
  
  
  
  return(results)
}

run_cv_by_childname <- function(child){
  group_data <- uni_model_data %>%
    filter(child_name==child & measure=="produces") %>% 
    mutate(group = paste(child_name, measure),
           lexical_category = lexical_category %>% fct_relevel("other")) %>%
    select(child_name, measure, group, lexical_category, item = uni_lemma,
           prop,total, age, !!predictors) %>%
    mutate(item = as.factor(item)) %>% 
    group_by(group)
  
  cv_errs_data <- group_data %>% 
    map(~ run_crossv(split_data = .)) %>% 
    reduce(rbind)
  
  name = paste("../../../data/aoa_predictors/",
               gsub(" ", "_", child, fixed = TRUE),
               "_cv_errs_data_lms.RData", sep="")
  
  
  save(cv_errs_data, file = name)
}

uni_model_data$child_name %>% unique() %>% map(run_cv_by_childname)



### plot predictions function 
plot_predicts <- function(n){
  print(sep_models_kfold$model_name[[n]])
  test_data = split_data[sep_models_kfold$test[[n]],]
  Y = test_data$prop
  Y_pred = predict(sep_models_kfold$models[[n]],  test_data, type="response")
  plot(Y_pred, Y)
}



### fit to all data
all_data_models <- fit_with(group_data, lm, formulae)
plot(predict(all_data_models$full_set), group_data$prop, type="response")
summary(all_data_models$freq_surp)

### error analysis by words
error_analysis_byword <- function(model, data){
  
  get_mse <- function(data_word){
    return(tibble(
      item = unique(data_word$item), 
      measure = mean((data_word$prop -  predict(model, data_word, type="response"))^2)
    )
    )
  }
  
  data = data %>% 
    group_by(item)
  results = data %>%
    split( .$item) %>% 
    map(get_mse) %>% 
    reduce(rbind)
  
  return(results)
}

get_errs_byword <- function(model){
  get_mean<-function(data){
    return(tibble(
      item = unique(data$item), 
      measure = mean(data$measure)))
  }
  errs_ <- error_analysis_byword(model = all_data_models$model, data = group_data) %>% 
    mutate(model_name = model)
  
  return(errs_)
}

load("../../../data/aoa_predictors/Average_child_produces_cv_models_data5_nofreq.RData")  
errs_produces_byword_all_data<- map(c("full_set", "freq_only", "full_surp", "freq_surp", "surp_only", "full_surp_only"), get_errs_byword) %>% reduce(rbind) 

results =  errs_produces_byword %>%
  group_by(model_name, item) %>% 
  summarise(mean= mean(measure))



### LMs predicting AoA 

### step 1: get aoa predictions by word
