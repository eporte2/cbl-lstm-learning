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


load("../../data/aoa_predictors/model_data.RData")
load("../../data/aoa_predictors/uni_joined.RData")
uni_joined_eng <- uni_joined %>% filter(language == "English (American)")

my_get_transcript_results <- function(file){
  df = read.csv(file)
  #Extract child name from file name and add variable.
  file_name = strsplit(file_path_sans_ext(file), "/")[[1]]
  file_name = file_name[(length(file_name))]
  child_name = strsplit(file_name,".aoa_result", fixed=TRUE)[[1]]
  df = df %>%
    mutate(child_name = child_name)
  return(df)
}

my_get_prod_results <- function(result_dir){
  files = list.files(path=result_dir, pattern="*.csv", full.names=TRUE, recursive=FALSE)
  df.prod_results = files %>% map(my_get_transcript_results) %>% reduce(rbind)
  return(df.prod_results)
}

result_dir = "../../data/results/aoa_surprisals/"
model_surprisals = my_get_prod_results(result_dir)
model_surprisals = model_surprisals %>% filter(child_name!="Thomas" && child_name!="Adam")

surp_model_data <- model_surprisals %>% 
  select(uni_lemma, avg_surprisal, child_name) %>% 
  mutate(uni_lemma = as.character(uni_lemma)) %>% 
  left_join(model_data) %>% 
  group_by(child_name) %>%
  # mutate_at(vars(!!predictors), funs(as.numeric(scale(.)))) %>%
  nest()

predictors <- c("avg_surprisal","frequency", "MLU", "final_frequency", "solo_frequency", "num_phons", "concreteness", "valence", "arousal", "babiness")

pred_sources <- list(
  c("frequency", "MLU", "final_frequency", "solo_frequency"),
  c("valence", "arousal"),
  "concreteness", "babiness", "num_phons", "avg_surprisal"
)

fit_predictor <- function(pred, d) {
  xs <- pred_sources %>% discard(~pred %in% .x) %>% unlist()
  x_str <- xs %>% paste(collapse = " + ")
  lm(as.formula(glue("{pred} ~ {x_str}")), data = d) %>%
    augment(newdata = d) %>%
    select(uni_lemma, lexical_category, .fitted)
}

max_steps <- 20
iterate_predictors <- function(lang_data) {
  missing <- lang_data %>%
    gather(predictor, value, !!predictors) %>%
    mutate(missing = is.na(value)) %>%
    select(-value) %>%
    spread(predictor, missing)
  predictor_order <- lang_data %>%
    gather(predictor, value, !!predictors) %>%
    group_by(predictor) %>%
    summarise(num_na = sum(is.na(value))) %>%
    filter(num_na != 0) %>%
    arrange(num_na) %>%
    pull(predictor)
  imputation_data <- lang_data %>%
    mutate_at(vars(!!predictors),
              funs(as.numeric(Hmisc::impute(., fun = "random"))))
  for (i in 0:max_steps) {
    pred <- predictor_order[(i %% length(predictor_order)) + 1]
    imputation_fits <- fit_predictor(pred, imputation_data)
    imputation_data <- missing %>%
      select(uni_lemma, lexical_category, !!pred) %>%
      rename(missing = !!pred) %>%
      right_join(imputation_data) %>%
      left_join(imputation_fits) %>%
      mutate_at(vars(pred), funs(if_else(is.na(missing), .fitted, .))) %>%
      select(-.fitted, -missing)
  }
  return(imputation_data)
}

model_data_imputed <- surp_model_data %>%
  mutate(imputed = map(data, iterate_predictors))

uni_model_data <- model_data_imputed %>%
  select(-data) %>%
  unnest() %>%
  group_by(child_name) %>%
  mutate_at(vars(predictors), funs(as.numeric(scale(.)))) %>%
  right_join(uni_joined_eng %>% select(measure, uni_lemma, age, num_true,
                                       num_false)) %>%
  filter(!is.na(child_name)) %>% 
  group_by(child_name, measure) %>%
  mutate(unscaled_age = age, age = scale(age),
         total = as.double(num_true + num_false), prop = num_true / total)

save(model_data_imputed, file = "../../data/aoa_predictors/model_data_imputed.RData")
save(uni_model_data, file = "../../data/aoa_predictors/uni_model_data.RData")

full_set = ~ (age | item) + age * frequency + age * MLU + age * final_frequency + age * solo_frequency + age * num_phons + age * concreteness + age * valence + age * arousal + age * babiness + lexical_category * frequency + lexical_category * MLU + lexical_category * final_frequency + lexical_category * solo_frequency + lexical_category *  num_phons + lexical_category * concreteness + lexical_category * valence + lexical_category * arousal + lexical_category * babiness

freq_only = ~ (age | item) + age * frequency + lexical_category * frequency

freq_MLU = ~ (age | item) + age * frequency + age * MLU + lexical_category * frequency + lexical_category * MLU

full_surp = ~ (age | item) + age * avg_surprisal + age * frequency + age * MLU + age * final_frequency + age * solo_frequency + age * num_phons + age * concreteness + age * valence + age * arousal + age * babiness + lexical_category * avg_surprisal + lexical_category * frequency + lexical_category * MLU + lexical_category * final_frequency + lexical_category * solo_frequency + lexical_category * num_phons + lexical_category * concreteness + lexical_category * valence + lexical_category * arousal + lexical_category * babiness

freq_surp = ~ (age | item) + age * frequency + age * avg_surprisal + lexical_category * frequency + lexical_category * avg_surprisal

freq_MLU_surp = ~ (age | item) + age * frequency + age * MLU + age * avg_surprisal + lexical_category * frequency + lexical_category * MLU + lexical_category * avg_surprisal

#formulae <- formulas(~prop, full_set, freq_only, freq_MLU, full_surp, freq_surp, freq_MLU_surp)
formulae <- formulas(~prop, full_surp, freq_surp, freq_MLU_surp)

fit_models <- function(data, formulae, contrasts = NULL) {
  print("run model")
  fit_with(data, glmer, formulae, family = "binomial",
           weights = data$total, contrasts = contrasts) 
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
               "_cv_models_data.RData", sep="")
  
  kfold5_data <- crossv_kfold(split_data, k=5)
  models_kfold <- kfold5_data %>% 
    mutate(models = train %>% map( ~ fit_models(., formulae)))
  sep_models_kfold <- models_kfold %>% 
    mutate(#full_set = models_kfold$models %>% map(~ .$"full_set"),
           #freq_only = models_kfold$models %>% map(~ .$"freq_only"),
           #freq_MLU = models_kfold$models %>% map(~ .$"freq_MLU"),
           full_surp = models_kfold$models %>% map(~ .$"full_surp"),
           freq_surp = models_kfold$models %>% map(~ .$"freq_surp"),
           freq_MLU_surp = models_kfold$models %>% map(~ .$"freq_MLU_surp")
    ) %>% 
    select(train, test, .id, 
           #full_set, 
           #freq_only, 
           #freq_MLU, 
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
  
  #model_names = c("full_set", "freq_only", "freq_MLU", "full_surp", "freq_surp", "freq_MLU_surp")
  model_names = c("full_surp", "freq_surp", "freq_MLU_surp")
  errs_<- map(model_names, get_avg_errs) %>% reduce(rbind)
  
  results <- errs_ %>% 
    mutate(group = group,
           child_name = unique(split_data$child_name),
           measure = unique(split_data$measure)
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
               "_cv_errs_data.RData", sep="")
  
  
  save(cv_errs_data, file = name)
}

uni_model_data$child_name %>% unique() %>% map(run_cv_by_childname)
