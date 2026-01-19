## +++++++++++++++++++++++++++++++++++++++++
## ++ Aggregate & plot modeled and         ++
## ++  observed data by income class       ++
## +++++++++++++++++++++++++++++++++++++++++
## Last updated: December 2025

## ++++++++++++++++++++++++++++++++++++
## A. Read in packages, functions, and data
## ++++++++++++++++++++++++++++++++++++

# load packages
library(readr)
library(dplyr)
library(purrr)
library(stringr)
library(tidyverse)
library(ggplot2)
library(GGally)
library(scatterplot3d)
library(RColorBrewer)
library(bbmle)
library(stats4)
library(DBI)
library(RSQLite)
#install.packages("visreg")
library(visreg)
library(patchwork)


###  Read in and process water billing dataset ###
# set working directory
#setwd("/Users/jenniferskerker/Documents/GradSchool/Research/Equity/Model/DCC_Demand_Estimation/data/")

# Read data from SQLite3 database
con <- dbConnect(RSQLite::SQLite(), ":memory:", dbname = 'DCC_DB.db') # create database connection
cols_to_read <- paste('account', 'Date',  'rmon', 'ryr', 'bill_length ', 'totwuse',
                      'rate_str_change', 'p_1_all', 'p_2_all', 'p_3_all', 'p_4_all', 'p_5_all', 
                      'drought_surcharge', 'curtail', 'curtail_inv', 'YSD',
                      't12_bound', 't23_bound', 't34_bound', 't45_bound',
                      'bathrooms', 'main_area', 'pool', 'tax_value',
                      'MHI',
                      'precip_mm', 'AET_mm', 'meanT_C', 
                      'pen', 'exu',
                      sep=',') # columns to read
data <- dbGetQuery(con, paste("SELECT ", cols_to_read, 
                              " FROM sf_data 
                               WHERE (flat_fee > 0)  ", # and (drought_surcharge >= 0)
                              sep='')) # send query and save results into a dataframe # (quality = 1) LIMIT 10000
dbDisconnect(con) # Disconnect from the database

data_backup <- data

# add bill length filters on-- 28-31 days
data <- data[data$bill_length >= 28 & data$bill_length <= 32, ]

# filter out negative penalty values
data <- data[data$pen >= 0, ] # filters out one datapoint 

# calculate bathrooms squared values
data$bath_squared <- data$bathrooms^2

# use cut function to convert water use volumes to tiers
tiers <- c(-Inf, data$t12_bound[1], data$t23_bound[1], data$t34_bound[1], data$t45_bound[1], Inf)
categories <- c(1, 2, 3, 4, 5)
Q <- data$totwuse
tier_num_pred = as.numeric(unclass(cut(Q, breaks=tiers, labels=categories))) # convert water use values to tiers and then convert to numerical

# predict marginal price based on predicted tier
tier_prices = matrix(c(data$p_1_all, data$p_2_all, data$p_3_all, data$p_4_all, data$p_5_all), ncol=5)
rows = 1:length(data$totwuse)

# Get predicted marginal prices: Extract the specified values for each row and column to 
data$marg_price_pred <- mapply(function(row_idx, col_idx) tier_prices[row_idx, col_idx], rows, tier_num_pred)

# Read in direct bias correction factors
### Read in income estimates & merge with water billing dataset ###
df_income <- read_csv("/Users/jenniferskerker/Documents/GradSchool/Research/Equity/Model/Santa_Cruz_WRM_updated/data/dcc_data/HH_income_qm_cases.csv")
df_merged <- inner_join(data, df_income, by="account")


# Read in DCC model coefficients
path <- '/Users/jenniferskerker/Documents/GradSchool/Research/Equity/Model/DCC_Demand_Estimation/outputs/'
filename <- 'dccfit_alldata_curtail_v1' 
DCCfit1 <- readRDS(paste0(path, filename, '.rds'))
DCCcoefs <- coef(DCCfit1)

# get predicted values- model results
df_merged$pred_logQ_dcc = (DCCcoefs['beta0'] + DCCcoefs['betaPAl']*log(df_merged$marg_price_pred)+DCCcoefs['betaTax']*log(df_merged$tax_value) + DCCcoefs['betaMA']*df_merged$main_area + 
                        DCCcoefs['betaBath']*df_merged$bathrooms + DCCcoefs['betaBathSq']*df_merged$bath_squared  +  DCCcoefs['betaPool']*df_merged$pool + 
                        DCCcoefs['betaPrecip']*df_merged$precip_mm + DCCcoefs['betaTemp']*df_merged$meanT_C + DCCcoefs['betaAET']*df_merged$AET_mm +
                        DCCcoefs['betaCurtail']*df_merged$curtail)

df_merged$predQ_dcc <- exp(df_merged$pred_logQ_dcc)

# calculate residuals and direct bias correction factor
df_merged$residuals_real <- df_merged$totwuse - df_merged$predQ_dcc

# version 2 with count as well
avg_resids <- df_merged %>%
  group_by(account) %>%
  summarise(
    #mean_residuals_log = mean(residuals_log, na.rm = TRUE),
    mean_residuals_real = mean(residuals_real, na.rm = TRUE),
    mean_totwuse = mean(totwuse, na.rm = TRUE),
    mean_pred_logQ = mean(pred_logQ_dcc, na.rm = TRUE),
    mean_pred = mean(predQ_dcc, na.rm = TRUE),
    count = n()
  )
filtered_df <- avg_resids %>% filter(count >= 12)

# merge df_merged and filtered_df
df <- inner_join(df_merged, filtered_df, by="account")
df$predQ_dbc <- df$predQ_dcc + df$residuals_real

# Group by income class
df_avg <- df_merged %>%
  group_by(map_inc_1) %>%
  summarise(across(c(totwuse, predQ_dcc), mean, na.rm = TRUE))

# get statistics
df_stats <- df%>%
  group_by(map_inc_1) %>%
  summarise(
    across(
      c(totwuse, predQ_dcc, predQ_dbc),
      list(
        mean   = ~mean(.x, na.rm = TRUE),
        median = ~median(.x, na.rm = TRUE),
        q25    = ~quantile(.x, 0.25, na.rm = TRUE),
        q75    = ~quantile(.x, 0.75, na.rm = TRUE)
      )
    ),
    .groups = "drop"
  )


# calculate r2- no dbc
fit <- lm(df_stats$totwuse_mean ~ df_stats$predQ_dcc_mean)
r2 <- summary(fit)$r.squared


fit_dbc <- lm(df_stats$totwuse_mean ~ df_stats$predQ_dbc_mean)
r2_dbc <- summary(fit_dbc)$r.squared

# no dbc
p1 <- ggplot(df_stats, aes(x = totwuse_mean, y = predQ_dcc_mean, color=map_inc_1)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  coord_fixed(xlim = c(5, 7), ylim = c(5, 7)) +
  labs(
    x = "Modeled value (ccf)",
    y = "Measured value (ccf)",
    color = "Income",
    title = "No Error Term"
  ) +
  annotate(
    "text",
    x = 5.5, y = 6.5,
    label = paste0("R² = ", round(r2, 2)),
    hjust = 0
  )

p1





## loop through income estimates to get r2 values
income_vars <- paste0("map_inc_", 1:10)

# put data together into one df
df_stats_all <- map_dfr(income_vars, function(inc_var) {
  
  df %>%
    group_by(income_group = .data[[inc_var]]) %>%
    summarise(
      across(
        c(totwuse, predQ_dcc, predQ_dbc),
        list(
          mean   = ~mean(.x, na.rm = TRUE),
          median = ~median(.x, na.rm = TRUE),
          q25    = ~quantile(.x, 0.25, na.rm = TRUE),
          q75    = ~quantile(.x, 0.75, na.rm = TRUE)
        )
      ),
      .groups = "drop"
    ) %>%
    mutate(income_map = inc_var)
})

# get r2 values
r2_by_income <- df_stats_all %>%
  group_by(income_map) %>%
  summarise(
    r2 = summary(
      lm(totwuse_mean ~ predQ_dcc_mean)
    )$r.squared,
    .groups = "drop"
  )

mean_r2 <- mean(r2_by_income$r2, na.rm = TRUE)
