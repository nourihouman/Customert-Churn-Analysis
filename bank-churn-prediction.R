
library(pROC)
library(tidyverse)
library(ggplot2)
library(themis)
library(randomForest)
library(caret)



#Load data
df <- read.csv("Customer-Churn-Records.csv")


# Drop useless columns
df <- df %>% select(-RowNumber, -CustomerId, -Surname)

# Convert to factors
df <- df %>%
  mutate(
    Geography     = as.factor(Geography),
    Gender        = as.factor(Gender),
    Card.Type     = as.factor(Card.Type),
    HasCrCard     = as.factor(HasCrCard),
    IsActiveMember = as.factor(IsActiveMember),
    Complain      = as.factor(Complain),
    Exited        = as.factor(Exited)  # 0 = stayed, 1 = churned
  )

# Check for missing values
sum(is.na(df))

# Final check
str(df)
summary(df)

colnames(df)



# Churn distribution
ggplot(df, aes(x = Exited, fill = Exited)) +
  geom_bar() +
  scale_fill_manual(values = c("0" = "#2ecc71", "1" = "#e74c3c")) +
  labs(title = "Churn Distribution", x = "Exited (0 = Stayed, 1 = Churned)", y = "Count") +
  theme_minimal()


# Churn by Geography
ggplot(df, aes(x = Geography, fill = Exited)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("0" = "#2ecc71", "1" = "#e74c3c")) +
  labs(title = "Churn by Geography", x = "Country", y = "Count") +
  theme_minimal()

# Churn by Gender
ggplot(df, aes(x = Gender, fill = Exited)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("0" = "#2ecc71", "1" = "#e74c3c")) +
  labs(title = "Churn by Gender", x = "Gender", y = "Count") +
  theme_minimal()


# Age distribution by Churn
ggplot(df, aes(x = Age, fill = Exited)) +
  geom_histogram(bins = 40, alpha = 0.7, position = "identity") +
  scale_fill_manual(values = c("0" = "#2ecc71", "1" = "#e74c3c")) +
  labs(title = "Age Distribution by Churn", x = "Age", y = "Count") +
  theme_minimal()

# Balance by Churn
ggplot(df, aes(x = Exited, y = Balance, fill = Exited)) +
  geom_boxplot(alpha = 0.7) +
  scale_fill_manual(values = c("0" = "#2ecc71", "1" = "#e74c3c")) +
  labs(title = "Balance Distribution by Churn", x = "Exited", y = "Balance") +
  theme_minimal()

# Complain vs Exited - the suspicious one!
ggplot(df, aes(x = Complain, fill = Exited)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("0" = "#2ecc71", "1" = "#e74c3c")) +
  labs(title = "Complain vs Churn", x = "Complained (0 = No, 1 = Yes)", y = "Count") +
  theme_minimal()

# Credit Score by Churn
ggplot(df, aes(x = Exited, y = CreditScore, fill = Exited)) +
  geom_boxplot(alpha = 0.7) +
  scale_fill_manual(values = c("0" = "#2ecc71", "1" = "#e74c3c")) +
  labs(title = "Credit Score by Churn", x = "Exited", y = "Credit Score") +
  theme_minimal()

# NumOfProducts by Churn
ggplot(df, aes(x = as.factor(NumOfProducts), fill = Exited)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("0" = "#2ecc71", "1" = "#e74c3c")) +
  labs(title = "Number of Products by Churn", x = "Num Of Products", y = "Count") +
  theme_minimal()

# Satisfaction Score by Churn
ggplot(df, aes(x = as.factor(Satisfaction.Score), fill = Exited)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("0" = "#2ecc71", "1" = "#e74c3c")) +
  labs(title = "Satisfaction Score by Churn", x = "Satisfaction Score", y = "Count") +
  theme_minimal()


# Drop Complain (data leakage!)
df_ml <- df %>% select(-Complain)

# Split data 80/20
set.seed(42)

trainIndex <- createDataPartition(df_ml$Exited, p = 0.8, list = FALSE)
train <- df_ml[trainIndex, ]
test  <- df_ml[-trainIndex, ]

cat("Before SMOTE:\n")
table(train$Exited)


train_balanced <- recipe(Exited ~ ., data = train) %>%
  step_smotenc(Exited, over_ratio = 1) %>%
  prep() %>%
  bake(new_data = NULL)

cat("After SMOTE:\n")
table(train_balanced$Exited)




# Rename levels in BOTH train_balanced and test
levels(train_balanced$Exited) <- c("No", "Yes")
levels(test$Exited)           <- c("No", "Yes")


set.seed(42)
log_model <- train(
  Exited ~ .,
  data = train_balanced,
  method = "glm",
  family = "binomial",
  trControl = trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  ),
  metric = "ROC"
)

# Predict
pred <- predict(log_model, newdata = test)

# Confusion Matrix
confusionMatrix(pred, test$Exited, positive = "Yes")

# AUC
pred_prob <- predict(log_model, newdata = test, type = "prob")[, "Yes"]
roc_obj <- roc(test$Exited, pred_prob)
cat("AUC:", auc(roc_obj), "\n")
plot(roc_obj, col = "#e74c3c", main = "ROC Curve - Logistic Regression")





# Train Random Forest
set.seed(42)
rf_model <- train(
  Exited ~ .,
  data = train_balanced,
  method = "rf",
  trControl = trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  ),
  metric = "ROC",
  tuneGrid = expand.grid(mtry = c(2, 4, 6, 8))  # tuning parameter
)

# Best mtry
print(rf_model)

# Predict on test set
pred_rf <- predict(rf_model, newdata = test)

# Confusion Matrix
confusionMatrix(pred_rf, test$Exited, positive = "Yes")

# AUC
pred_rf_prob <- predict(rf_model, newdata = test, type = "prob")[, "Yes"]
roc_rf <- roc(test$Exited, pred_rf_prob)
cat("AUC:", auc(roc_rf), "\n")
plot(roc_rf, col = "#3498db", main = "ROC Curve - Random Forest")

# Feature Importance
varImpPlot(rf_model$finalModel, main = "Feature Importance - Random Forest")





#tuned XGBoost with random search
set.seed(42)
xgb_model_tuned <- train(
  Exited ~ .,
  data = train_balanced,
  method = "xgbTree",
  trControl = trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary,
    search = "random"       # random search
  ),
  metric = "ROC",
  tuneLength = 20,          # tries 20 random combinations
  verbosity = 0
)

# Best parameters found
print(xgb_model_tuned)
cat("Best AUC (CV):", max(xgb_model_tuned$results$ROC), "\n")

# Predict on test set
pred_xgb_tuned <- predict(xgb_model_tuned, newdata = test)

# Confusion Matrix
confusionMatrix(pred_xgb_tuned, test$Exited, positive = "Yes")

# AUC
pred_xgb_tuned_prob <- predict(xgb_model_tuned, newdata = test, type = "prob")[, "Yes"]
roc_xgb_tuned <- roc(test$Exited, pred_xgb_tuned_prob)
cat("AUC:", auc(roc_xgb_tuned), "\n")
plot(roc_xgb_tuned, col = "#2ecc71", main = "ROC Curve - XGBoost Tuned")

# Feature Importance
xgb_tuned_imp <- varImp(xgb_model_tuned)
plot(xgb_tuned_imp, main = "Feature Importance - XGBoost Tuned")







# Plot the first ROC curve (Logistic Regression)
plot(roc_obj, 
     col = "#e74c3c", 
     lwd = 2,
     main = "ROC Curve Comparison - All Models")

# Add Random Forest
lines(roc_rf, 
      col = "#3498db", 
      lwd = 2)

# Add XGBoost Basic
lines(roc_xgb, 
      col = "#2ecc71", 
      lwd = 2)

# Add XGBoost Tuned
lines(roc_xgb_tuned, 
      col = "#f39c12", 
      lwd = 2)

# Add diagonal baseline
abline(a = 1, b = -1, 
       lty = 2, 
       col = "gray")

# Add legend with AUC values
legend("bottomright",
       legend = c(
         paste("Logistic Regression  AUC =", round(auc(roc_obj), 3)),
         paste("Random Forest        AUC =", round(auc(roc_rf), 3)),
         paste("XGBoost Basic        AUC =", round(auc(roc_xgb), 3)),
         paste("XGBoost Tuned        AUC =", round(auc(roc_xgb_tuned), 3))
       ),
       col = c("#e74c3c", "#3498db", "#2ecc71", "#f39c12"),
       lwd = 2,
       cex = 0.85,
       bg = "white")

