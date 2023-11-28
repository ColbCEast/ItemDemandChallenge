# Run for ALL Models

setwd("C:/Users/colby/OneDrive/Desktop/STAT 348/ItemDemandChallenge")

library(tidymodels)
library(timetk)
library(vroom)
library(patchwork)
library(ranger)
library(naivebayes)
library(discrim)
library(kernlab)
library(bonsai)
library(lightgbm)
library(dbarts)
library(timeDate)
library(modeltime)
library(kknn)

train_data <- vroom("train.csv")

test_data <- vroom("test.csv")

# Exploratory Data Analysis

range(train_data$store)
range(train_data$item)

item1 <- train_data %>%
  filter(store == 2, item == 28)

item2 <- train_data %>%
  filter(store == 7, item == 5)

item3 <- train_data %>%
  filter(store == 5, item == 46)

item4 <- train_data %>%
  filter(store == 1, item == 18)

plot1 <- item1 %>%
  pull(sales) %>%
  forecast::ggAcf(.)

plot2 <- item2 %>%
  pull(sales) %>%
  forecast::ggAcf(., lag.max = 365)

plot3 <- item3 %>%
  pull(sales) %>%
  forecast::ggAcf(., lag.max = 2*365)

plot4 <- item4 %>%
  pull(sales) %>%
  forecast::ggAcf(., lag.max = 2 * 30)

patchwork_plot <- (plot1 + plot2) / (plot3 + plot4)

#ggsave("time_series_eda.jpg", plot = patchwork_plot)


# Feature Engineering

train_data1 <- train_data %>%
  filter(store == 4, item == 28)

engineer_recipe <- recipe(sales~., data = train_data1) %>%
  step_date(date, features = "dow") %>%
  step_date(date, features = "month") %>%
  step_mutate(weekend = ifelse(date_dow == c("Sat", "Sun"), 1, 0)) %>%
  step_date(date, features = "doy") %>%
  step_range(date_doy, min = 0, max = pi) %>%
  step_mutate(cosDOY = cos(date_doy)) %>%
  step_rm(date_doy)

rf_model <- rand_forest(mtry = tune(),
                        min_n = tune(),
                        trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("regression")

rf_wf <- workflow() %>%
  add_recipe(engineer_recipe) %>%
  add_model(rf_model)

rf_tune_grid <- grid_regular(mtry(range = c(1,10)),
                             min_n(),
                             levels = 5)

folds <- vfold_cv(train_data, v = 5, repeats = 1)

cv_results <- rf_wf %>%
  tune_grid(resamples = folds,
            grid = rf_tune_grid,
            metrics = metric_set(smape))

best_tune <- cv_results %>%
  select_best("smape")

collect_metrics(cv_results) %>%
  filter(best_tune) %>%
  pull(mean)

# KNN

knn_model <- nearest_neighbor(neighbors = tune()) %>%
  set_mode("regression") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_model(knn_model) %>%
  add_recipe(engineer_recipe)

knn_tune_grid <- grid_regular(neighbors(),
                              levels = 5)

folds <- vfold_cv(train_data, v = 5, repeats = 1)

cv_results_knn <- knn_wf %>%
  tune_grid(resamples = folds,
            grid = knn_tune_grid,
            metrics = metric_set(smape))

best_tune_knn <- cv_results_knn %>%
  select_best("smape")

collect_metrics(cv_results_knn) %>%
  filter(best_tune_knn) %>%
  pull(mean)


# Exponential Smoothing

train_data_item1 <- train_data %>%
  filter(store == 2, item == 15)

train_data_item2 <- train_data %>%
  filter(store == 7, item == 34)

test_data_item1 <- test_data %>%
  filter(store == 2, item == 15)

test_data_item2 <- test_data %>%
  filter(store == 7, item == 34)

cv_split1 <- time_series_split(train_data_item1, assess = "3 months", cumulative = TRUE)
cv_split1 %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(date, sales, .interactive = FALSE)

cv_split2 <- time_series_split(train_data_item2, assess = "3 months", cumulative = TRUE)
cv_split2 %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(date, sales, .interactive = FALSE)

ss_model1 <- exp_smoothing() %>%
  set_engine("ets") %>%
  fit(sales~date, data = training(cv_split1))

cv_results1 <- modeltime_calibrate(ss_model1,
                                  new_data = testing(cv_split1))

item1_cv_plot <- cv_results1 %>%
  modeltime_forecast(
    new_data = testing(cv_split1),
    actual_data = train_data_item1
  ) %>%
  plot_modeltime_forecast(.interactive = TRUE)

cv_results1 %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(.interactive = FALSE)

ss_fullfit1 <- cv_results1 %>%
  modeltime_refit(data = train_data_item1)

ss_preds1 <- ss_fullfit1 %>%
  modeltime_forecast(h = "3 months") %>%
  rename(date = .index, sales = .value) %>%
  select(date, sales) %>%
  full_join(., y = test_data_item1, by = "date") %>%
  select(id, sales)

item1_forecast_plot <- ss_fullfit1 %>%
  modeltime_forecast(h = "3 months", actual_data = train_data_item1) %>%
  plot_modeltime_forecast(.interactive = FALSE)

ss_model2 <- exp_smoothing() %>%
  set_engine("ets") %>%
  fit(sales~date, data = training(cv_split2))

cv_results2 <- modeltime_calibrate(ss_model2,
                                   new_data = testing(cv_split2))

item2_cv_plot <- cv_results2 %>%
  modeltime_forecast(
    new_data = testing(cv_split2),
    actual_data = train_data_item2
  ) %>%
  plot_modeltime_forecast(.interactive = TRUE)

cv_results2 %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(.interactive = FALSE)

ss_fullfit2 <- cv_results2 %>%
  modeltime_refit(data = train_data_item2)

ss_preds2 <- ss_fullfit2 %>%
  modeltime_forecast(h = "3 months") %>%
  rename(date = .index, sales = .value) %>%
  select(date, sales) %>%
  full_join(., y = test_data_item2, by = "date") %>%
  select(id, sales)

item2_forecast_plot <- ss_fullfit2 %>%
  modeltime_forecast(h = "3 months", actual_data = train_data_item2) %>%
  plot_modeltime_forecast(.interactive = FALSE)

class_plots <- plotly::subplot(item1_cv_plot,item2_cv_plot,item1_forecast_plot,item2_forecast_plot, nrows = 2)


## SARIMA Models

train_data_item1 <- train_data %>%
  filter(store == 2, item == 15)

train_data_item2 <- train_data %>%
  filter(store == 7, item == 34)

test_data_item1 <- test_data %>%
  filter(store == 2, item == 15)

test_data_item2 <- test_data %>%
  filter(store == 7, item == 34)

cv_split1 <- time_series_split(train_data_item1, assess = "3 months", cumulative = TRUE)
cv_split1 %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(date, sales, .interactive = FALSE)

cv_split2 <- time_series_split(train_data_item2, assess = "3 months", cumulative = TRUE)
cv_split2 %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(date, sales, .interactive = FALSE)

arima_recipe_item1 <- recipe(sales~., data = train_data_item1) %>%
  step_date(date, features = "dow") %>%
  step_date(date, features = "month") %>%
  step_mutate(weekend = ifelse(date_dow == c("Sat", "Sun"), 1, 0)) %>%
  step_date(date, features = "doy") %>%
  step_range(date_doy, min = 0, max = pi) %>%
  step_mutate(cosDOY = cos(date_doy)) %>%
  step_rm(date_doy)

arima_recipe_item2 <- recipe(sales~., data = train_data_item2) %>%
  step_date(date, features = "dow") %>%
  step_date(date, features = "month") %>%
  step_mutate(weekend = ifelse(date_dow == c("Sat", "Sun"), 1, 0)) %>%
  step_date(date, features = "doy") %>%
  step_range(date_doy, min = 0, max = pi) %>%
  step_mutate(cosDOY = cos(date_doy)) %>%
  step_rm(date_doy)

arima_model <- arima_reg(seasonal_period = 365,
                         non_seasonal_ar = 5,
                         non_seasonal_ma = 5,
                         seasonal_ar = 2,
                         seasonal_ma = 2,
                         non_seasonal_differences = 2,
                         seasonal_differences = 2) %>%
  set_engine("auto_arima")

arima_wf_item1 <- workflow() %>%
  add_recipe(arima_recipe_item1) %>%
  add_model(arima_model) %>%
  fit(data = training(cv_split1))

arima_wf_item2 <- workflow() %>%
  add_recipe(arima_recipe_item2) %>%
  add_model(arima_model) %>%
  fit(data = training(cv_split2))

cv_results1 <- modeltime_calibrate(arima_wf_item1,
                                   new_data = testing(cv_split1))

item1_cv_plot <- cv_results1 %>%
  modeltime_forecast(
    new_data = testing(cv_split1),
    actual_data = train_data_item1
  ) %>%
  plot_modeltime_forecast(.interactive = TRUE)

cv_results1 %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(.interactive = FALSE)

arima_fullfit1 <- cv_results1 %>%
  modeltime_refit(data = train_data_item1)

item1_forecast_plot <- arima_fullfit1 %>%
  modeltime_forecast(new_data = test_data_item1,
                     actual_data = train_data_item1) %>%
  plot_modeltime_forecast(.interactive = FALSE)

cv_results2 <- modeltime_calibrate(arima_wf_item2,
                                   new_data = testing(cv_split2))

item2_cv_plot <- cv_results2 %>%
  modeltime_forecast(
    new_data = testing(cv_split2),
    actual_data = train_data_item2
  ) %>%
  plot_modeltime_forecast(.interactive = TRUE)

cv_results2 %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(.interactive = FALSE)

arima_fullfit2 <- cv_results2 %>%
  modeltime_refit(data = train_data_item2)

item2_forecast_plot <- arima_fullfit2 %>%
  modeltime_forecast(new_data = test_data_item2,
                     actual_data = train_data_item2) %>%
  plot_modeltime_forecast(.interactive = FALSE)

class_plots_arima <- plotly::subplot(item1_cv_plot,item2_cv_plot,item1_forecast_plot,item2_forecast_plot, nrows = 2)
class_plots_arima
