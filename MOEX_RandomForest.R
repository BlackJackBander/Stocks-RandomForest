library(jsonlite)
library(tseries)
library(randomForest)
library(recipes)
library(ggplot2)
library(dplyr)
library(lubridate)

getMOEX <- function(ticker) {
  url <- paste0('https://iss.moex.com/iss/engines/stock/markets/shares/securities/', ticker, '/candles.json?from=2024-01-01&till=', Sys.Date(), '&interval=24')
  ticker_data <- fromJSON(url)
  test <- ticker_data$candles$data
  colnames(test) <- ticker_data[["candles"]][["columns"]]
  rownames(test) <- as.character(as.Date(as.POSIXlt(test[, 'end'], format="%Y-%m-%d")))
  return(test)
}

i <- 'HYDR'

# Получаем данные и преобразуем в data.frame
ch <- as.data.frame(getMOEX(i))
chartName <- i
# Преобразуем числовые колонки из character в numeric
numeric_cols <- c("open", "close", "high", "low", "value", "volume")
ch[numeric_cols] <- lapply(ch[numeric_cols], as.numeric)

# Сохраняем даты для будущего использования
dates <- as.Date(rownames(ch))

# Создаем рецепт для предобработки данных с импутацией через KNN 
recipe_spec <- recipe(close ~ ., data = ch) %>%
  step_rm(begin, end) %>%  # Удаляем временные метки
  step_impute_knn(all_predictors(), neighbors = 6)  # Импутация через 6 ближайших соседей 

# Применяем рецепт
prepped_data <- prep(recipe_spec, training = ch) %>%
  bake(new_data = ch)

# Добавляем даты обратно в данные
prepped_data$date <- dates

set.seed(42)
# вот тут есть ошибка, т.к. мы создаем случайную выборку.
# для задач классификации это отлично подходит, но мы работаем с временными рядами, т.о. нарушаем структуру.
# изменим на хронологический отбор
trainIndex <- sample(1:nrow(prepped_data), 0.8 * nrow(prepped_data))
trainData <- prepped_data[trainIndex, ]
testData <- prepped_data[-trainIndex, ]

# Хронологическое разделение на train/test (80%/20%)
train_ratio <- 0.8
n_total <- nrow(prepped_data)
n_train <- floor(train_ratio * n_total)

# Создаем тренировочную и тестовую выборки
trainData <- prepped_data[1:n_train, ]
testData <- prepped_data[(n_train + 1):n_total, ]

# Граничная дата для разделения
split_date <- trainData$date[n_train]


# Обучаем модель Random Forest
#rf_model <- randomForest(
#  close ~ . -date,  # Исключаем дату из предикторов
#  data = trainData,
#  ntree = 500,
#  importance = TRUE,
#  proximity = TRUE
#)

rf_model <- randomForest(
  close ~ . - value,  # Исключаем дату и value из предикторов
  data = trainData,
  ntree = 500,
  importance = TRUE,
  proximity = TRUE
)

print(rf_model)
plot(rf_model)


# Продолжение вашего кзова...

# Сортируем данные по дате (на всякий случай)
prepped_data <- prepped_data[order(prepped_data$date), ]
# Продолжение вашего кода...

# Создаем функцию для создания лаговых признаков
create_lag_features <- function(data, n_lags = 5) {
  data <- data[order(data$date), ]
  for (i in 1:n_lags) {
    data[[paste0("close_lag", i)]] <- lag(data$close, i)
    data[[paste0("volume_lag", i)]] <- lag(data$volume, i)
  }
  return(na.omit(data))
}

# Создаем лаговые признаки
prepped_data_lags <- create_lag_features(prepped_data, n_lags = 5)

# Хронологическое разделение на train/test
n_total <- nrow(prepped_data_lags)
n_train <- floor(0.8 * n_total)

trainData <- prepped_data_lags[1:n_train, ]
testData <- prepped_data_lags[(n_train + 1):n_total, ]
split_date <- trainData$date[n_train]

# Обучаем модель с лаговыми признаками
rf_model <- randomForest(
  close ~ . - date - value - open - high - low,  # Используем только лаговые признаки
  data = trainData,
  ntree = 500,
  importance = TRUE
)

# Функция для прогнозирования на n дней вперед
forecast_future <- function(model, last_data, n_days = 5) {
  forecasts <- numeric(n_days)
  current_data <- last_data
  
  for (i in 1:n_days) {
    # Предсказываем следующее значение
    next_pred <- predict(model, newdata = current_data)
    forecasts[i] <- next_pred
    
    # Обновляем данные для следующего прогноза
    if (i < n_days) {
      # Сдвигаем лаги
      for (j in 5:2) {
        current_data[[paste0("close_lag", j)]] <- current_data[[paste0("close_lag", j-1)]]
        current_data[[paste0("volume_lag", j)]] <- current_data[[paste0("volume_lag", j-1)]]
      }
      current_data$close_lag1 <- next_pred
      # Для volume используем среднее значение (можно улучшить)
      current_data$volume_lag1 <- mean(last_data$volume_lag1, na.rm = TRUE)
    }
  }
  
  return(forecasts)
}

# Получаем последние данные для прогнозирования
last_known_data <- testData[nrow(testData), ]

# Прогнозируем на 5 дней вперед
future_days <- 5
future_predictions <- forecast_future(rf_model, last_known_data, future_days)

# Создаем даты для прогноза
last_date <- max(prepped_data_lags$date)
future_dates <- last_date + 1:future_days

# Обновляем полный датафрейм с прогнозом
full_results <- rbind(
  data.frame(
    date = trainData$date,
    actual = trainData$close,
    predicted = NA,
    set = "train",
    type = "Исторические данные"
  ),
  data.frame(
    date = testData$date,
    actual = testData$close,
    predicted = predict(rf_model, newdata = testData),
    set = "test",
    type = "Тестовые данные"
  ),
  data.frame(
    date = future_dates,
    actual = NA,
    predicted = future_predictions,
    set = "future",
    type = "Прогноз на 5 дней"
  )
)

# Вычисляем метрики для тестового периода
test_actual <- testData$close
test_pred <- predict(rf_model, newdata = testData)

mae <- mean(abs(test_actual - test_pred))
rmse <- sqrt(mean((test_actual - test_pred)^2))
mape <- mean(abs((test_actual - test_pred) / test_actual)) * 100
r_squared <- 1 - (sum((test_actual - test_pred)^2) / 
                    sum((test_actual - mean(test_actual))^2))

# Информация о прогнозе
forecast_text <- paste(
  sprintf("Прогноз на 5 дней вперед:\n"),
  sprintf("%s: %.2f руб\n", future_dates[1], future_predictions[1]),
  sprintf("%s: %.2f руб\n", future_dates[2], future_predictions[2]),
  sprintf("%s: %.2f руб\n", future_dates[3], future_predictions[3]),
  sprintf("%s: %.2f руб\n", future_dates[4], future_predictions[4]),
  sprintf("%s: %.2f руб\n", future_dates[5], future_predictions[5]),
  sprintf("Тренд: %s", 
          ifelse(future_predictions[5] > future_predictions[1], "↗ Рост", "↘ Спад"))
)

# Создаем основной график с прогнозом
p <- ggplot(full_results, aes(x = date)) +
  # Исторические данные
  geom_line(data = subset(full_results, set %in% c("train", "test")),
            aes(y = actual, color = "Фактические значения"), 
            linewidth = 0.8, alpha = 0.8) +
  # Предсказания на тестовом периоде
  geom_line(data = subset(full_results, set == "test"),
            aes(y = predicted, color = "Предсказания (тест)"), 
            linewidth = 0.8, linetype = "solid", alpha = 0.8) +
  # Прогноз на будущее
  geom_line(data = subset(full_results, set == "future"),
            aes(y = predicted, color = "Прогноз на 5 дней"), 
            linewidth = 1.2, linetype = "dashed", alpha = 0.9) +
  geom_point(data = subset(full_results, set == "future"),
             aes(y = predicted, color = "Прогноз на 5 дней"), 
             size = 3, shape = 18) +
  # Вертикальные линии разделения
  geom_vline(xintercept = as.numeric(split_date), 
             linetype = "dashed", color = "red", alpha = 0.8, linewidth = 1) +
  geom_vline(xintercept = as.numeric(last_date), 
             linetype = "dashed", color = "blue", alpha = 0.8, linewidth = 1) +
  # Настройки внешнего вида
  labs(
    title = "Прогноз цен закрытия {i} на 5 дней вперед",
    subtitle = "Random Forest с использованием лаговых признаков",
    x = "Дата",
    y = "Цена закрытия (руб)",
    color = "Легенда"
  ) +
  scale_color_manual(
    values = c(
      "Фактические значения" = "#3498db",
      "Предсказания (тест)" = "#27ae60",
      "Прогноз на 5 дней" = "#e74c3c"
    )
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray50"),
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1)
  ) +
  # Аннотации
  annotate("text", x = split_date, y = min(full_results$actual, na.rm = TRUE) * 0.95,
           label = paste("Начало теста:\n", split_date),
           hjust = 1.1, vjust = 0, size = 3, color = "red") +
  annotate("text", x = last_date, y = min(full_results$actual, na.rm = TRUE) * 0.95,
           label = paste("Начало прогноза:\n", last_date),
           hjust = -0.1, vjust = 0, size = 3, color = "blue") +
  annotate("label", 
           x = min(full_results$date, na.rm = TRUE), 
           y = max(full_results$actual, na.rm = TRUE) * 0.95,
           label = paste(
             sprintf("Метрики на тестовом периоде:\n"),
             sprintf("MAE: %.2f руб\n", mae),
             sprintf("RMSE: %.2f руб\n", rmse),
             sprintf("MAPE: %.1f%%\n", mape),
             sprintf("R²: %.3f", r_squared)
           ),
           hjust = 0, vjust = 1,
           size = 3.2,
           fill = "white",
           alpha = 0.8) +
  annotate("label", 
           x = max(full_results$date, na.rm = TRUE), 
           y = max(full_results$actual, na.rm = TRUE) * 0.95,
           label = forecast_text,
           hjust = 1, vjust = 1,
           size = 3.2,
           fill = "#ffebee",
           alpha = 0.9)

print(p)

# Детальный график прогноза
ggplot(subset(full_results, date >= last_date - 10), aes(x = date)) +
  geom_line(aes(y = actual, color = "Фактические значения"), linewidth = 1.2) +
  geom_line(aes(y = predicted, color = "Предсказанные значения"), 
            linewidth = 1.2, linetype = "solid") +
  geom_point(aes(y = actual), size = 3, color = "#3498db") +
  geom_point(aes(y = predicted), size = 3, shape = 17, color = "#e74c3c") +
  geom_vline(xintercept = as.numeric(last_date), 
             linetype = "dashed", color = "red", linewidth = 1) +
  labs(
    title = "Детальный вид: последние данные и прогноз",
    subtitle = "Переход от известных данных к прогнозу",
    x = "Дата",
    y = "Цена закрытия (руб)"
  ) +
  scale_color_manual(
    values = c(
      "Фактические значения" = "#3498db",
      "Предсказанные значения" = "#e74c3c"
    )
  ) +
  theme_minimal()

# Выводим информацию о прогнозе
cat("\n=== ПРОГНОЗ НА 5 ДНЕЙ ВПЕРЕД ===\n")
cat("Последняя известная дата:", as.character(last_date), "\n")
cat("Последняя известная цена:", tail(testData$close, 1), "руб\n\n")

for (i in 1:future_days) {
  cat(sprintf("%s: %.2f руб", as.character(future_dates[i]), future_predictions[i]))
  if (i > 1) {
    change <- future_predictions[i] - future_predictions[i-1]
    change_percent <- (change / future_predictions[i-1]) * 100
    cat(sprintf(" (%+.2f руб, %+.1f%%)", change, change_percent))
  }
  cat("\n")
}

cat(sprintf("\nОбщее изменение: %+.2f руб (%+.1f%%)",
            future_predictions[5] - future_predictions[1],
            (future_predictions[5] - future_predictions[1]) / future_predictions[1] * 100))
cat("\nТренд:", ifelse(future_predictions[5] > future_predictions[1], "ВОСХОДЯЩИЙ", "НИСХОДЯЩИЙ"))
